from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiment_support import SharedExperiment
from qeco_runtime.MEC_Env import MEC


@dataclass(frozen=True)
class AccessConfig:
    n_channel: int = SharedExperiment.CHANNEL_COUNT
    n_access_point: int = SharedExperiment.NUM_ACCESS_POINTS
    ap_cluster_size: int = SharedExperiment.AP_CLUSTER_SIZE
    bandwidth_mhz: float = SharedExperiment.CHANNEL_BANDWIDTH_MHZ
    noise_power: float = SharedExperiment.CHANNEL_NOISE_POWER
    tx_power: float = SharedExperiment.CHANNEL_TX_POWER
    min_channel_gain: float = SharedExperiment.CHANNEL_MIN_GAIN
    max_channel_gain: float = SharedExperiment.CHANNEL_MAX_GAIN
    capacity_scale: float = SharedExperiment.CHANNEL_CAPACITY_SCALE
    min_fronthaul_delay: float = SharedExperiment.AP_FRONTHAUL_MIN_DELAY
    max_fronthaul_delay: float = SharedExperiment.AP_FRONTHAUL_MAX_DELAY


class ChannelAwareMEC(MEC):
    """QECO MEC environment with user-centric AP cluster access.

    Physical path:

    user -> AP cluster -> channel/resource -> edge server

    The QECO action remains local or edge selection. For each selected edge,
    the environment traverses a user-specific AP cluster and chooses an
    AP-channel resource node before applying the transmission capacity.
    """

    def __init__(
        self,
        num_ue,
        num_edge,
        num_time,
        num_component,
        max_delay,
        access_config: AccessConfig | None = None,
    ):
        self.access_config = access_config or AccessConfig()
        self.n_channel = self.access_config.n_channel
        self.n_ap = self.access_config.n_access_point
        self.ap_cluster_size = min(max(self.access_config.ap_cluster_size, num_edge), self.n_ap)
        super().__init__(num_ue, num_edge, num_time, num_component, max_delay)
        self.base_n_actions = self.n_actions
        self.base_n_features = self.n_features
        self.base_n_lstm_state = self.n_lstm_state
        self.n_actions = self.base_n_actions
        self.n_features = self.base_n_features + self.n_edge
        self.n_lstm_state = self.base_n_lstm_state + self.n_edge
        self.action_map = self._build_action_map()
        self.ap_to_edge = self._build_ap_to_edge()
        self.ap_fronthaul_delay = self._build_ap_fronthaul_delay()
        self.access_channel_gains = np.zeros((self.n_ue, self.n_ap, self.n_channel))
        self.channel_gains = self.access_channel_gains
        self.access_occupancy_observe = np.zeros((self.n_ap, self.n_channel))
        self.ap_clusters = np.zeros((self.n_ue, self.ap_cluster_size), dtype=int)
        self.state_map = {}
        self.last_selected_edges = -np.ones(self.n_ue, dtype=int)
        self.last_selected_aps = -np.ones(self.n_ue, dtype=int)
        self.last_selected_channels = -np.ones(self.n_ue, dtype=int)

    def reset(self, arrive_task_size, arrive_task_dens):
        base_obs, base_lstm = self._call_base_with_base_dimensions(
            super().reset,
            arrive_task_size,
            arrive_task_dens,
        )
        self._sample_access_channel_gains()
        self.access_occupancy_observe = np.zeros((self.n_ap, self.n_channel))
        self._build_user_ap_clusters()
        self.last_selected_edges = -np.ones(self.n_ue, dtype=int)
        self.last_selected_aps = -np.ones(self.n_ue, dtype=int)
        self.last_selected_channels = -np.ones(self.n_ue, dtype=int)
        return self._augment_observation(base_obs), self._augment_lstm_state(base_lstm)

    def step(self, action):
        base_action, selected_edges = self.decode_edge_actions(action)
        selected_aps, selected_channels, step_occupancy = self.allocate_access_resources(base_action)
        self.last_selected_edges = selected_edges
        self.last_selected_aps = selected_aps
        self.last_selected_channels = selected_channels
        self._apply_access_capacities(base_action, selected_aps, selected_channels, step_occupancy)
        base_obs, base_lstm, done = self._call_base_with_base_dimensions(super().step, base_action)
        self.access_occupancy_observe = step_occupancy
        self._sample_access_channel_gains()
        self._build_user_ap_clusters()
        return self._augment_observation(base_obs), self._augment_lstm_state(base_lstm), done

    def _call_base_with_base_dimensions(self, method, *args, **kwargs):
        current_n_features = self.n_features
        current_n_lstm_state = self.n_lstm_state
        try:
            self.n_features = self.base_n_features
            self.n_lstm_state = self.base_n_lstm_state
            return method(*args, **kwargs)
        finally:
            self.n_features = current_n_features
            self.n_lstm_state = current_n_lstm_state

    def _build_action_map(self):
        action_map = {0: {"type": "local", "edge": None}}
        for edge_index in range(self.n_edge):
            action_map[edge_index + 1] = {"type": "edge", "edge": edge_index}
        return action_map

    def _build_ap_to_edge(self):
        return np.asarray(
            [min((ap_index * self.n_edge) // max(self.n_ap, 1), self.n_edge - 1) for ap_index in range(self.n_ap)],
            dtype=int,
        )

    def _build_ap_fronthaul_delay(self):
        cfg = self.access_config
        return np.random.uniform(cfg.min_fronthaul_delay, cfg.max_fronthaul_delay, size=self.n_ap)

    def decode_edge_actions(self, action):
        base_action = np.zeros(self.n_ue, dtype=int)
        selected_edges = -np.ones(self.n_ue, dtype=int)
        for ue_index, raw_action in enumerate(action):
            raw_action = int(raw_action)
            if raw_action <= 0:
                base_action[ue_index] = 0
                continue
            edge_index = min(max(raw_action - 1, 0), self.n_edge - 1)
            base_action[ue_index] = edge_index + 1
            selected_edges[ue_index] = edge_index
        return base_action, selected_edges

    def _sample_access_channel_gains(self):
        cfg = self.access_config
        self.access_channel_gains = np.random.uniform(
            low=cfg.min_channel_gain,
            high=cfg.max_channel_gain,
            size=(self.n_ue, self.n_ap, self.n_channel),
        )
        self.channel_gains = self.access_channel_gains

    def _channel_rate(self, gain):
        cfg = self.access_config
        snr = cfg.tx_power * gain / max(cfg.noise_power, 1e-12)
        return cfg.capacity_scale * np.log2(1.0 + snr) * cfg.bandwidth_mhz

    def _normalized_rate(self, rate):
        min_rate = self._channel_rate(self.access_config.min_channel_gain)
        max_rate = self._channel_rate(self.access_config.max_channel_gain)
        return float(np.clip((rate - min_rate) / max(max_rate - min_rate, 1e-9), 0.0, 1.0))

    def _ap_link_score(self, ue_index, ap_index):
        best_gain = float(np.max(self.access_channel_gains[ue_index, ap_index, :]))
        return self._normalized_rate(self._channel_rate(best_gain))

    def _build_user_ap_clusters(self):
        clusters = np.zeros((self.n_ue, self.ap_cluster_size), dtype=int)
        for ue_index in range(self.n_ue):
            selected = []
            for edge_index in range(self.n_edge):
                edge_aps = np.where(self.ap_to_edge == edge_index)[0]
                if edge_aps.size == 0:
                    continue
                best_ap = max(edge_aps, key=lambda ap_index: self._ap_link_score(ue_index, int(ap_index)))
                selected.append(int(best_ap))

            ranked_aps = sorted(range(self.n_ap), key=lambda ap_index: self._ap_link_score(ue_index, ap_index), reverse=True)
            for ap_index in ranked_aps:
                if len(selected) >= self.ap_cluster_size:
                    break
                if ap_index not in selected:
                    selected.append(ap_index)

            clusters[ue_index, :] = np.asarray(selected[: self.ap_cluster_size], dtype=int)
        self.ap_clusters = clusters

    def _candidate_aps_for_edge(self, ue_index, edge_index):
        cluster = [int(ap_index) for ap_index in self.ap_clusters[ue_index]]
        candidates = [ap_index for ap_index in cluster if self.ap_to_edge[ap_index] == edge_index]
        if candidates:
            return candidates
        return [int(ap_index) for ap_index in np.where(self.ap_to_edge == edge_index)[0]]

    def _effective_access_rate(self, ue_index, ap_index, channel_index, occupancy_count=0.0):
        gain = self.access_channel_gains[ue_index, ap_index, channel_index]
        raw_rate = self._channel_rate(gain)
        shared_rate = raw_rate / max(float(occupancy_count), 1.0)
        fronthaul_factor = 1.0 / (1.0 + self.ap_fronthaul_delay[ap_index])
        return shared_rate * fronthaul_factor

    def access_node_scores(self, ue_index, edge_index, extra_occupancy=None):
        occupancy = self.access_occupancy_observe.copy()
        if extra_occupancy is not None:
            occupancy = occupancy + extra_occupancy

        scores = []
        for ap_index in self._candidate_aps_for_edge(ue_index, edge_index):
            for channel_index in range(self.n_channel):
                projected_occupancy = occupancy[ap_index, channel_index] + 1.0
                effective_rate = self._effective_access_rate(ue_index, ap_index, channel_index, projected_occupancy)
                scores.append(
                    {
                        "ap": ap_index,
                        "channel": channel_index,
                        "score": self._normalized_rate(effective_rate),
                        "effective_rate": float(effective_rate),
                    }
                )
        return scores

    def select_access_node(self, ue_index, edge_index, extra_occupancy=None):
        scores = self.access_node_scores(ue_index, edge_index, extra_occupancy)
        if not scores:
            return -1, -1
        selected = max(scores, key=lambda item: item["score"])
        return int(selected["ap"]), int(selected["channel"])

    def allocate_access_resources(self, base_action):
        selected_aps = -np.ones(self.n_ue, dtype=int)
        selected_channels = -np.ones(self.n_ue, dtype=int)
        step_occupancy = np.zeros((self.n_ap, self.n_channel))
        for ue_index, edge_action in enumerate(base_action):
            if edge_action <= 0:
                continue
            edge_index = int(edge_action - 1)
            ap_index, channel_index = self.select_access_node(ue_index, edge_index, step_occupancy)
            if ap_index < 0 or channel_index < 0:
                continue
            selected_aps[ue_index] = ap_index
            selected_channels[ue_index] = channel_index
            step_occupancy[ap_index, channel_index] += 1
        return selected_aps, selected_channels, step_occupancy

    def _apply_access_capacities(self, base_action, selected_aps, selected_channels, step_occupancy):
        for ue_index in range(self.n_ue):
            edge_action = int(base_action[ue_index])
            if edge_action <= 0:
                continue
            edge_index = edge_action - 1
            ap_index = selected_aps[ue_index]
            channel_index = selected_channels[ue_index]
            if ap_index < 0 or channel_index < 0:
                continue
            occupancy_count = max(step_occupancy[ap_index, channel_index], 1.0)
            effective_rate = self._effective_access_rate(ue_index, ap_index, channel_index, occupancy_count)
            self.tran_cap_ue[ue_index, edge_index] = effective_rate * self.duration

    def _edge_access_scores(self, base_obs):
        scores = np.zeros((self.n_ue, self.n_edge))
        for ue_index in range(self.n_ue):
            if base_obs[ue_index, 0] == 0:
                continue
            for edge_index in range(self.n_edge):
                node_scores = self.access_node_scores(ue_index, edge_index)
                scores[ue_index, edge_index] = max((item["score"] for item in node_scores), default=0.0)
        return scores

    def _build_state_map(self, base_obs, edge_access_scores):
        access_points = {}
        for ap_index in range(self.n_ap):
            channels = {}
            for channel_index in range(self.n_channel):
                gains = self.access_channel_gains[:, ap_index, channel_index]
                channels[channel_index] = {
                    "occupancy": float(self.access_occupancy_observe[ap_index, channel_index]),
                    "average_gain": float(np.mean(gains)),
                    "average_rate": float(np.mean([self._channel_rate(gain) for gain in gains])),
                }
            access_points[ap_index] = {
                "edge": int(self.ap_to_edge[ap_index]),
                "fronthaul_delay": float(self.ap_fronthaul_delay[ap_index]),
                "channels": channels,
            }

        servers = {}
        for edge_index in range(self.n_edge):
            servers[edge_index] = {
                "connected_aps": [int(ap_index) for ap_index in np.where(self.ap_to_edge == edge_index)[0]],
                "edge_backlog_by_user": [
                    float(base_obs[ue_index, 3 + edge_index]) if base_obs[ue_index, 0] != 0 else 0.0
                    for ue_index in range(self.n_ue)
                ],
                "access_score_by_user": [
                    float(edge_access_scores[ue_index, edge_index])
                    for ue_index in range(self.n_ue)
                ],
            }

        users = {}
        for ue_index in range(self.n_ue):
            users[ue_index] = {
                "task_size": float(base_obs[ue_index, 0]),
                "ue_comp_queue": float(base_obs[ue_index, 1]),
                "ue_tran_queue": float(base_obs[ue_index, 2]),
                "energy_state": float(base_obs[ue_index, self.base_n_features - 1]),
                "ap_cluster": [int(ap_index) for ap_index in self.ap_clusters[ue_index]],
            }
        return {
            "users": users,
            "access_points": access_points,
            "servers": servers,
            "action_map": self.action_map,
        }

    def _augment_observation(self, base_obs):
        augmented = np.zeros((self.n_ue, self.n_features))
        edge_access_scores = self._edge_access_scores(base_obs)
        self.state_map = self._build_state_map(base_obs, edge_access_scores)
        for ue_index in range(self.n_ue):
            augmented[ue_index, :] = np.hstack([base_obs[ue_index, :], edge_access_scores[ue_index, :]])
        return augmented

    def _augment_lstm_state(self, base_lstm):
        augmented = np.zeros((self.n_ue, self.n_lstm_state))
        edge_access_occupancy = np.zeros(self.n_edge)
        for edge_index in range(self.n_edge):
            edge_aps = np.where(self.ap_to_edge == edge_index)[0]
            edge_access_occupancy[edge_index] = float(np.sum(self.access_occupancy_observe[edge_aps, :]))
        for ue_index in range(self.n_ue):
            augmented[ue_index, :] = np.hstack([base_lstm[ue_index, :], edge_access_occupancy])
        return augmented
