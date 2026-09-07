from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from experiment_support import SharedExperiment
from qeco_runtime.Config import Config


TRACE_GENERATOR_VERSION = "dynamic-trace-v1"
_N_ACCESS_POINTS = 6
_N_CHANNELS = 4


def _readonly_array(value, dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class EpisodeTrace:
    active_mask: np.ndarray
    task_sizes: np.ndarray
    task_densities: np.ndarray
    ue_energy: np.ndarray
    fronthaul: np.ndarray
    channel_gains: np.ndarray
    generator_version: str = TRACE_GENERATOR_VERSION

    def __post_init__(self) -> None:
        fields = {
            "active_mask": _readonly_array(self.active_mask, bool),
            "task_sizes": _readonly_array(self.task_sizes, float),
            "task_densities": _readonly_array(self.task_densities, float),
            "ue_energy": _readonly_array(self.ue_energy, float),
            "fronthaul": _readonly_array(self.fronthaul, float),
            "channel_gains": _readonly_array(self.channel_gains, float),
        }
        for name, value in fields.items():
            object.__setattr__(self, name, value)

        if self.active_mask.ndim != 2:
            raise ValueError("active_mask must have shape [steps, users]")
        steps, users = self.active_mask.shape
        expected = (steps, users)
        if self.task_sizes.shape != expected or self.task_densities.shape != expected:
            raise ValueError("task arrays must match active_mask shape")
        if self.ue_energy.shape != (users,):
            raise ValueError("ue_energy must have shape [users]")
        if self.fronthaul.shape != (_N_ACCESS_POINTS,):
            raise ValueError("fronthaul must have shape [6]")
        if self.channel_gains.shape != (steps + 1, users, _N_ACCESS_POINTS, _N_CHANNELS):
            raise ValueError("channel_gains must have shape [steps + 1, users, 6, 4]")
        if np.any(self.task_sizes[~self.active_mask] != 0) or np.any(
            self.task_densities[~self.active_mask] != 0
        ):
            raise ValueError("inactive users cannot receive task arrivals")
        if np.any((self.task_sizes == 0) != (self.task_densities == 0)):
            raise ValueError("task size and density must be zero together")
        if not all(np.all(np.isfinite(value)) for value in fields.values() if value.dtype != bool):
            raise ValueError("trace values must be finite")

    @property
    def total_steps(self) -> int:
        return int(self.active_mask.shape[0])

    @property
    def pool_users(self) -> int:
        return int(self.active_mask.shape[1])

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.generator_version.encode("utf-8"))
        for name in (
            "active_mask",
            "task_sizes",
            "task_densities",
            "ue_energy",
            "fronthaul",
            "channel_gains",
        ):
            value = getattr(self, name)
            header = json.dumps(
                {"name": name, "dtype": value.dtype.str, "shape": value.shape},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(header)
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()


def _stream_rng(seed: int, phase: str, episode: int, stream: str) -> np.random.Generator:
    key = json.dumps(
        [TRACE_GENERATOR_VERSION, int(seed), str(phase), int(episode), stream],
        separators=(",", ":"),
    ).encode("utf-8")
    entropy = np.frombuffer(hashlib.sha256(key).digest(), dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _validate_config(config) -> tuple[int, int, int, tuple[int, ...], str, int, int]:
    pool_users = int(config.pool_users)
    total_steps = int(config.total_steps)
    max_delay = int(config.max_delay)
    load_levels = tuple(int(value) for value in config.load_levels)
    load_mode = str(config.load_mode).lower()
    dwell_min = int(config.dwell_min)
    dwell_max = int(config.dwell_max)
    if pool_users <= 0:
        raise ValueError("pool_users must be positive")
    if max_delay <= 0 or total_steps <= max_delay:
        raise ValueError("total_steps must exceed positive max_delay")
    if not load_levels or any(value < 0 or value > pool_users for value in load_levels):
        raise ValueError("load_levels must lie in [0, pool_users]")
    if load_mode not in {"iid", "persistent"}:
        raise ValueError("load_mode must be 'iid' or 'persistent'")
    if dwell_min <= 0 or dwell_max < dwell_min:
        raise ValueError("dwell bounds must satisfy 0 < dwell_min <= dwell_max")
    return pool_users, total_steps, max_delay, load_levels, load_mode, dwell_min, dwell_max


def _persistent_counts(
    rng: np.random.Generator,
    total_steps: int,
    load_levels: tuple[int, ...],
    dwell_min: int,
    dwell_max: int,
) -> np.ndarray:
    counts = np.empty(total_steps, dtype=int)
    offset = 0
    previous = None
    while offset < total_steps:
        candidates = load_levels
        if previous is not None and len(load_levels) > 1:
            candidates = tuple(value for value in load_levels if value != previous)
        selected = int(rng.choice(candidates))
        dwell = int(rng.integers(dwell_min, dwell_max + 1))
        counts[offset : min(offset + dwell, total_steps)] = selected
        offset += dwell
        previous = selected
    return counts


def _activity_masks(
    rng: np.random.Generator,
    counts: np.ndarray,
    pool_users: int,
    persistent: bool,
) -> np.ndarray:
    masks = np.zeros((len(counts), pool_users), dtype=bool)
    previous = np.empty(0, dtype=int)
    previous_count = -1
    for step, count_value in enumerate(counts):
        count = int(count_value)
        if persistent and count == previous_count:
            selected = previous
        elif persistent:
            retained_count = min(len(previous), count)
            retained = (
                rng.choice(previous, size=retained_count, replace=False)
                if retained_count
                else np.empty(0, dtype=int)
            )
            needed = count - retained_count
            if needed:
                available = np.setdiff1d(np.arange(pool_users), retained, assume_unique=False)
                added = rng.choice(available, size=needed, replace=False)
                selected = np.concatenate((retained, added))
            else:
                selected = retained
        else:
            selected = rng.choice(pool_users, size=count, replace=False)
        masks[step, selected] = True
        previous = np.asarray(selected, dtype=int)
        previous_count = count
    return masks


def make_trace(config, seed: int, phase: str, episode: int) -> EpisodeTrace:
    """Create a policy-independent episode trace from disjoint keyed RNG streams."""

    pool, steps, max_delay, levels, mode, dwell_min, dwell_max = _validate_config(config)
    schedule_rng = _stream_rng(seed, phase, episode, "load-schedule")
    membership_rng = _stream_rng(seed, phase, episode, "active-membership")
    if mode == "iid":
        counts = schedule_rng.choice(levels, size=steps).astype(int)
    else:
        counts = _persistent_counts(schedule_rng, steps, levels, dwell_min, dwell_max)
    active_mask = _activity_masks(membership_rng, counts, pool, mode == "persistent")

    size_rng = _stream_rng(seed, phase, episode, "task-size")
    arrival_rng = _stream_rng(seed, phase, episode, "task-arrival")
    density_rng = _stream_rng(seed, phase, episode, "task-density")
    task_sizes = size_rng.uniform(Config.TASK_MIN_SIZE, Config.TASK_MAX_SIZE, size=(steps, pool))
    base_profile = np.asarray(SharedExperiment.QECO_ARRIVAL_BASE_PROFILE, dtype=float)
    profile = np.repeat(base_profile, int(np.ceil(steps / len(base_profile))))[:steps]
    user_activity = np.linspace(
        SharedExperiment.QECO_USER_ACTIVITY_MIN,
        SharedExperiment.QECO_USER_ACTIVITY_MAX,
        pool,
    )
    probabilities = np.clip(profile[:, None] * user_activity[None, :], 0.0, 1.0)
    arrivals = (arrival_rng.random((steps, pool)) < probabilities) & active_mask
    arrivals[-max_delay:, :] = False
    task_sizes *= arrivals
    density_indices = density_rng.integers(0, len(Config.TASK_COMP_DENS), size=(steps, pool))
    task_densities = np.asarray(Config.TASK_COMP_DENS, dtype=float)[density_indices] * arrivals

    ue_energy = _stream_rng(seed, phase, episode, "ue-energy").choice(
        np.asarray(Config.UE_ENERGY_STATE, dtype=float), size=pool
    )
    fronthaul = _stream_rng(seed, phase, episode, "fronthaul").uniform(
        SharedExperiment.AP_FRONTHAUL_MIN_DELAY,
        SharedExperiment.AP_FRONTHAUL_MAX_DELAY,
        size=_N_ACCESS_POINTS,
    )
    channel_gains = _stream_rng(seed, phase, episode, "channel-gains").uniform(
        SharedExperiment.CHANNEL_MIN_GAIN,
        SharedExperiment.CHANNEL_MAX_GAIN,
        size=(steps + 1, pool, _N_ACCESS_POINTS, _N_CHANNELS),
    )
    return EpisodeTrace(
        active_mask=active_mask,
        task_sizes=task_sizes,
        task_densities=task_densities,
        ue_energy=ue_energy,
        fronthaul=fronthaul,
        channel_gains=channel_gains,
    )
