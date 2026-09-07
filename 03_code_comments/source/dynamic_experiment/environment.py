from __future__ import annotations

from contextlib import contextmanager
import math

import numpy as np

from channel_env.channel_mec_env import ChannelAwareMEC
from qeco_runtime.MEC_Env import MEC

from .traces import EpisodeTrace


@contextmanager
def _preserve_global_numpy_rng():
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


class TraceMEC(ChannelAwareMEC):
    """Opt-in channel MEC driven entirely by an immutable episode trace.

    Dynamic activity controls arrivals only. Existing local, transmission, and
    edge work remains under the inherited service discipline. In particular,
    this wrapper intentionally does not redesign retained transmission capacity
    for already in-flight work.
    """

    def __init__(self, trace: EpisodeTrace, num_edges: int = 3, max_delay: int = 10):
        if num_edges != 3:
            raise ValueError("TraceMEC currently requires the fixed three-edge topology")
        if max_delay <= 0 or max_delay > trace.total_steps:
            raise ValueError("max_delay must lie in [1, trace.total_steps]")
        if np.any(trace.task_sizes[-max_delay:, :] != 0):
            raise ValueError("the final max_delay trace slots must contain no arrivals")
        self.trace = trace
        with _preserve_global_numpy_rng():
            super().__init__(
                trace.pool_users,
                num_edges,
                trace.total_steps,
                1,
                max_delay,
            )
        if self.n_ap != 6 or self.n_channel != 4:
            raise ValueError("TraceMEC requires six APs and four channels per AP")
        self._install_physical_trace(0)

    def _install_physical_trace(self, gain_index: int) -> None:
        self.ue_energy_state = self.trace.ue_energy.copy()
        self.ap_fronthaul_delay = self.trace.fronthaul.copy()
        self.access_channel_gains = self.trace.channel_gains[gain_index].copy()
        self.channel_gains = self.access_channel_gains

    def reset_trace(self):
        self._install_physical_trace(0)
        with _preserve_global_numpy_rng():
            base_obs, base_lstm = self._call_base_with_base_dimensions(
                lambda sizes, densities: MEC.reset(self, sizes, densities),
                self.trace.task_sizes,
                self.trace.task_densities,
            )
        self.access_occupancy_observe = np.zeros((self.n_ap, self.n_channel))
        self._install_physical_trace(0)
        self._build_user_ap_clusters()
        self.last_selected_edges = -np.ones(self.n_ue, dtype=int)
        self.last_selected_aps = -np.ones(self.n_ue, dtype=int)
        self.last_selected_channels = -np.ones(self.n_ue, dtype=int)
        self._repair_edge_backlog(base_obs)
        return self._augment_observation(base_obs), self._augment_lstm_state(base_lstm)

    def step(self, actions):
        step_index = self.time_count
        if step_index >= self.trace.total_steps:
            raise RuntimeError("episode trace is already complete")
        self._install_physical_trace(step_index)
        base_action, selected_edges = self.decode_edge_actions(actions)
        selected_aps, selected_channels, step_occupancy = self.allocate_access_resources(base_action)
        self.last_selected_edges = selected_edges
        self.last_selected_aps = selected_aps
        self.last_selected_channels = selected_channels
        self._apply_access_capacities(base_action, selected_aps, selected_channels, step_occupancy)
        with _preserve_global_numpy_rng():
            base_obs, base_lstm, done = self._call_base_with_base_dimensions(
                lambda decoded: MEC.step(self, decoded), base_action
            )
        self.access_occupancy_observe = step_occupancy
        self._install_physical_trace(step_index + 1)
        self._build_user_ap_clusters()
        self._repair_edge_backlog(base_obs)
        if done:
            self._assert_terminal_classification()
        return self._augment_observation(base_obs), self._augment_lstm_state(base_lstm), done

    def _outstanding_edge_bits(self) -> np.ndarray:
        outstanding = np.zeros((self.n_ue, self.n_edge), dtype=float)
        for ue_index in range(self.n_ue):
            for edge_index in range(self.n_edge):
                queued = self.edge_computation_queue[ue_index][edge_index]
                for task in tuple(queued.queue):
                    if "REMAIN" in task:
                        size = float(task["REMAIN"])
                    elif "SIZE" in task:
                        size = float(task["SIZE"])
                    else:
                        raise ValueError("queued edge task must contain REMAIN or SIZE")
                    if math.isfinite(size) and size > 0:
                        outstanding[ue_index, edge_index] += size
                remainder = float(self.edge_process_task[ue_index][edge_index]["REMAIN"])
                if math.isfinite(remainder) and remainder > 0:
                    outstanding[ue_index, edge_index] += remainder
        return outstanding

    def _repair_edge_backlog(self, base_obs: np.ndarray) -> None:
        self.b_edge_comp = self._outstanding_edge_bits()
        rows_with_arrivals = base_obs[:, 0] != 0
        base_obs[rows_with_arrivals, 3 : 3 + self.n_edge] = self.b_edge_comp[rows_with_arrivals]

    def _assert_terminal_classification(self) -> None:
        arrived = self.arrive_task_size != 0
        classified = self.process_delay > 0
        if not np.array_equal(arrived, classified):
            missing = int(np.count_nonzero(arrived & ~classified))
            unexpected = int(np.count_nonzero(~arrived & classified))
            raise AssertionError(
                f"terminal task classification mismatch: missing={missing}, unexpected={unexpected}"
            )
        flags = self.unfinish_task[arrived]
        if np.any((flags != 0) & (flags != 1)):
            raise AssertionError("terminal unfinished flags must be binary")

    def context(self) -> np.ndarray:
        mask_index = min(self.time_count, self.trace.total_steps - 1)
        active_count = int(np.count_nonzero(self.trace.active_mask[mask_index]))
        normalized_users = active_count / float(self.trace.pool_users)
        edge_backlog = float(np.clip(np.mean(np.sum(self.b_edge_comp, axis=0)) / 12.0, 0.0, 1.0))
        raw_rates = self._channel_rate(self.access_channel_gains)
        best_rates = np.max(raw_rates, axis=(1, 2))
        normalized_rates = np.asarray([self._normalized_rate(rate) for rate in best_rates])
        access_pressure = float(np.clip(1.0 - np.mean(normalized_rates), 0.0, 1.0))
        return np.asarray([normalized_users, edge_backlog, access_pressure], dtype=float)
