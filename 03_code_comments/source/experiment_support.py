from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int_list(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return tuple(int(item) for item in value.replace(";", ",").split(",") if item.strip())


class SharedExperiment:
    NAME = "qeco_adapt_channel_aware_extension"
    RANDOM_SEED = _env_int("QECO_RANDOM_SEED", 42)
    NUM_USERS = _env_int("QECO_NUM_USERS", 800)
    NUM_EDGES = 3
    NUM_COMPONENTS = 1
    MAX_DELAY = 10

    # Experiment family and density mode (qeco-adapt2 extension).
    # EXPERIMENT_FAMILY="" keeps the legacy result layout results/<algorithm>/run_*.
    # EXPERIMENT_FAMILY="adapt2" routes runs to results/adapt2/<density_mode>_density/<algorithm>/run_*.
    EXPERIMENT_FAMILY = _env_str("QECO_EXPERIMENT_FAMILY", "")
    # DENSITY_MODE="fixed": every episode uses NUM_USERS active users (legacy behaviour).
    # DENSITY_MODE="random": NUM_USERS is the agent pool size; each episode activates a
    # sampled subset whose size is drawn from DENSITY_CHOICES (or [DENSITY_MIN, DENSITY_MAX]).
    DENSITY_MODE = _env_str("QECO_DENSITY_MODE", "fixed").lower()
    DENSITY_CHOICES = _env_int_list("QECO_DENSITY_CHOICES", ())
    DENSITY_MIN = _env_int("QECO_DENSITY_MIN", 0)
    DENSITY_MAX = _env_int("QECO_DENSITY_MAX", 0)
    # "random": a fresh random subset of UE indices is active each episode.
    # "prefix": the first k UE indices are active (deterministic membership).
    DENSITY_SUBSET = _env_str("QECO_DENSITY_SUBSET", "random").lower()

    # qeco-adapt2 density regime thresholds, expressed in active users per edge.
    ADAPT2_SPARSE_MAX_USERS_PER_EDGE = _env_float("QECO_ADAPT2_SPARSE_MAX_USERS_PER_EDGE", 50.0)
    ADAPT2_DENSE_MIN_USERS_PER_EDGE = _env_float("QECO_ADAPT2_DENSE_MIN_USERS_PER_EDGE", 350.0)
    # qeco-adapt2 regime -> sub-policy mapping (registered QECO-family adapter names).
    # Sparse default fixed by Phase A-1 (2026-08-20, bear, users=100 seed=42 500ep):
    # general QoE 12.876 > adaptive gate 8.588 > gate only 8.241 > QECO 7.595.
    ADAPT2_SPARSE_POLICY = _env_str("QECO_ADAPT2_SPARSE_POLICY", "qeco-adapt-general")
    ADAPT2_GENERAL_POLICY = _env_str("QECO_ADAPT2_GENERAL_POLICY", "qeco-adapt-general")
    ADAPT2_DENSE_POLICY = _env_str("QECO_ADAPT2_DENSE_POLICY", "qeco-adaptive-gate")

    QECO_EPISODES = 500
    QECO_TIME_SLOTS = 100
    QECO_TASK_ARRIVE_PROB = 0.3
    QECO_ARRIVAL_BASE_PROFILE = (0.18, 0.30, 0.42, 0.24)
    QECO_USER_ACTIVITY_MIN = 0.7
    QECO_USER_ACTIVITY_MAX = 1.3
    COMPARISON_POINTS = QECO_EPISODES

    QECO_ADAPT_BASE_ENERGY_WEIGHT = 1.20
    QECO_ADAPT_USER_EXPONENT = 0.35
    QECO_ADAPT_LOAD_SCALE = 10.0
    QECO_ADAPT_MIN_ENERGY_STATE = 0.35
    QECO_ADAPT_EDGE_BACKLOG_THRESHOLD = 12.0
    QECO_ADAPT_SIZE_THRESHOLD = 1.5

    CHANNEL_COUNT = 4
    CHANNEL_BANDWIDTH_MHZ = 1.0
    CHANNEL_NOISE_POWER = 1e-9
    CHANNEL_TX_POWER = 1.0
    CHANNEL_MIN_GAIN = 1e-7
    CHANNEL_MAX_GAIN = 1e-5
    CHANNEL_CAPACITY_SCALE = 14.0
    NUM_ACCESS_POINTS = NUM_EDGES * 2
    AP_CLUSTER_SIZE = NUM_EDGES
    AP_FRONTHAUL_MIN_DELAY = 0.01
    AP_FRONTHAUL_MAX_DELAY = 0.10

    RESULT_ROOT = Path(_env_str("QECO_RESULT_ROOT", str(ROOT_DIR / "results")))

    @classmethod
    def density_dir_name(cls) -> str:
        return f"{cls.DENSITY_MODE}_density"

    @classmethod
    def result_base_dir(cls) -> Path:
        """Root under which per-algorithm run directories are created."""
        if cls.EXPERIMENT_FAMILY:
            return cls.RESULT_ROOT / cls.EXPERIMENT_FAMILY / cls.density_dir_name()
        return cls.RESULT_ROOT

    @classmethod
    def comparison_base_dir(cls) -> Path:
        """Root under which comparison scenario directories are created."""
        if cls.EXPERIMENT_FAMILY:
            return cls.RESULT_ROOT / cls.EXPERIMENT_FAMILY / "comparisons" / cls.density_dir_name()
        return cls.RESULT_ROOT / "comparisons"

    @classmethod
    def density_as_dict(cls) -> dict:
        return {
            "experiment_family": cls.EXPERIMENT_FAMILY or None,
            "mode": cls.DENSITY_MODE,
            "agent_pool_users": cls.NUM_USERS,
            "choices": list(cls.DENSITY_CHOICES),
            "min": cls.DENSITY_MIN,
            "max": cls.DENSITY_MAX,
            "subset": cls.DENSITY_SUBSET,
            "adapt2": {
                "sparse_max_users_per_edge": cls.ADAPT2_SPARSE_MAX_USERS_PER_EDGE,
                "dense_min_users_per_edge": cls.ADAPT2_DENSE_MIN_USERS_PER_EDGE,
                "sparse_policy": cls.ADAPT2_SPARSE_POLICY,
                "general_policy": cls.ADAPT2_GENERAL_POLICY,
                "dense_policy": cls.ADAPT2_DENSE_POLICY,
            },
        }

    @classmethod
    def as_dict(cls) -> dict:
        return {
            "name": cls.NAME,
            "random_seed": cls.RANDOM_SEED,
            "num_users": cls.NUM_USERS,
            "num_edges": cls.NUM_EDGES,
            "num_components": cls.NUM_COMPONENTS,
            "max_delay": cls.MAX_DELAY,
            "density": cls.density_as_dict(),
            "qeco": {
                "episodes": cls.QECO_EPISODES,
                "time_slots": cls.QECO_TIME_SLOTS,
                "task_arrive_prob": cls.QECO_TASK_ARRIVE_PROB,
                "arrival_profile": {
                    "base_profile": cls.QECO_ARRIVAL_BASE_PROFILE,
                    "user_activity_min": cls.QECO_USER_ACTIVITY_MIN,
                    "user_activity_max": cls.QECO_USER_ACTIVITY_MAX,
                },
                "qeco_adapt": {
                    "base_energy_weight": cls.QECO_ADAPT_BASE_ENERGY_WEIGHT,
                    "user_exponent": cls.QECO_ADAPT_USER_EXPONENT,
                    "load_scale": cls.QECO_ADAPT_LOAD_SCALE,
                    "min_energy_state": cls.QECO_ADAPT_MIN_ENERGY_STATE,
                    "edge_backlog_threshold": cls.QECO_ADAPT_EDGE_BACKLOG_THRESHOLD,
                    "size_threshold": cls.QECO_ADAPT_SIZE_THRESHOLD,
                },
            },
            "channel": {
                "count": cls.CHANNEL_COUNT,
                "bandwidth_mhz": cls.CHANNEL_BANDWIDTH_MHZ,
                "noise_power": cls.CHANNEL_NOISE_POWER,
                "tx_power": cls.CHANNEL_TX_POWER,
                "min_gain": cls.CHANNEL_MIN_GAIN,
                "max_gain": cls.CHANNEL_MAX_GAIN,
                "capacity_scale": cls.CHANNEL_CAPACITY_SCALE,
            },
            "access": {
                "num_access_points": cls.NUM_ACCESS_POINTS,
                "ap_cluster_size": cls.AP_CLUSTER_SIZE,
                "fronthaul_min_delay": cls.AP_FRONTHAUL_MIN_DELAY,
                "fronthaul_max_delay": cls.AP_FRONTHAUL_MAX_DELAY,
            },
            "comparison": {
                "points": cls.COMPARISON_POINTS,
                "axis_name": "episode",
            },
        }


def apply_global_seed(seed: int = SharedExperiment.RANDOM_SEED, use_tensorflow: bool = False) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception as exc:
        raise RuntimeError(f"Failed to apply NumPy random seed {seed}") from exc
    if use_tensorflow:
        try:
            import tensorflow.compat.v1 as tf

            tf.set_random_seed(seed)
        except Exception as exc:
            raise RuntimeError(f"Failed to apply TensorFlow random seed {seed}") from exc


def write_experiment_snapshot(output_dir: Path, extra: dict | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = SharedExperiment.as_dict()
    if extra:
        snapshot["extra"] = extra
    target = output_dir / "experiment_config.json"
    with open(target, "w") as f:
        json.dump(snapshot, f, indent=2)
    return target


def write_progress_checkpoint(
    run_dir: Path,
    algorithm: str,
    episodes_completed: int,
    episodes_total: int,
    latest_metrics: dict[str, float],
) -> Path:
    """Atomically publish a lightweight progress checkpoint for long runs."""
    target = run_dir / "progress.json"
    temporary = run_dir / "progress.json.tmp"
    payload = {
        "algorithm": algorithm,
        "episodes_completed": episodes_completed,
        "episodes_total": episodes_total,
        "latest_metrics": {key: float(value) for key, value in latest_metrics.items()},
        "updated_at_epoch_seconds": time.time(),
    }
    with open(temporary, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(temporary, target)
    return target


def resample_series(values: list[float], target_points: int) -> list[float]:
    if not values:
        return []
    if target_points <= 0:
        raise ValueError("target_points must be positive")
    if len(values) == target_points:
        return [float(value) for value in values]
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if len(arr) > target_points:
        bins = np.array_split(arr, target_points)
        return [float(bin_values.mean()) for bin_values in bins]
    source_x = np.linspace(0.0, 1.0, len(arr))
    target_x = np.linspace(0.0, 1.0, target_points)
    return [float(value) for value in np.interp(target_x, source_x, arr)]


def create_run_dir(algorithm_name: str) -> Path:
    timestamp = time.strftime("run_%Y%m%d_%H%M%S")
    base_dir = SharedExperiment.result_base_dir() / algorithm_name
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_{attempt + 1}"
        run_dir = base_dir / f"{timestamp}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            # Two workers for the same algorithm can start within the same
            # second (e.g. the same adapter at two user counts); take the next
            # suffixed name instead of failing the run.
            continue
    raise FileExistsError(f"Could not allocate a unique run directory under {base_dir}")


# ---------------------------------------------------------------------------
# User density regimes (qeco-adapt2)
# ---------------------------------------------------------------------------

DENSITY_REGIMES = ("sparse", "general", "dense")


def users_per_edge(active_users: int) -> float:
    return float(max(int(active_users), 0)) / float(max(SharedExperiment.NUM_EDGES, 1))


def classify_density_regime(active_users: int) -> str:
    """Map an active-user count to sparse / general / dense using users-per-edge thresholds."""
    density = users_per_edge(active_users)
    if density < SharedExperiment.ADAPT2_SPARSE_MAX_USERS_PER_EDGE:
        return "sparse"
    if density >= SharedExperiment.ADAPT2_DENSE_MIN_USERS_PER_EDGE:
        return "dense"
    return "general"


def adapt2_regime_policy_map() -> dict[str, str]:
    return {
        "sparse": SharedExperiment.ADAPT2_SPARSE_POLICY,
        "general": SharedExperiment.ADAPT2_GENERAL_POLICY,
        "dense": SharedExperiment.ADAPT2_DENSE_POLICY,
    }


class DensitySchedule:
    """Per-episode active-user schedule.

    The schedule is drawn from a dedicated NumPy Generator seeded with the run
    seed, so it is identical across algorithms and independent of the shared
    global RNG stream used by task arrivals, channel gains and exploration.
    """

    def __init__(self, episodes: int, agent_pool_users: int | None = None, seed: int | None = None):
        import numpy as np

        self.episodes = int(episodes)
        self.agent_pool_users = int(agent_pool_users if agent_pool_users is not None else SharedExperiment.NUM_USERS)
        self.seed = int(seed if seed is not None else SharedExperiment.RANDOM_SEED)
        self.mode = SharedExperiment.DENSITY_MODE
        self.subset_mode = SharedExperiment.DENSITY_SUBSET
        if self.mode not in ("fixed", "random"):
            raise ValueError(f"QECO_DENSITY_MODE must be 'fixed' or 'random', got {self.mode!r}.")
        if self.subset_mode not in ("random", "prefix"):
            raise ValueError(f"QECO_DENSITY_SUBSET must be 'random' or 'prefix', got {self.subset_mode!r}.")

        self.choices = self._resolve_choices()
        self._rng = np.random.default_rng(self.seed)
        if self.mode == "fixed":
            self.active_users = [self.agent_pool_users] * self.episodes
        else:
            self.active_users = [int(value) for value in self._rng.choice(self.choices, size=self.episodes)]
        self._active_masks: dict[int, "np.ndarray"] = {}
        for episode, count in enumerate(self.active_users):
            self._active_masks[episode] = self._draw_mask(np, count)

    def _resolve_choices(self) -> tuple[int, ...]:
        pool = self.agent_pool_users
        if self.mode == "fixed":
            return (pool,)
        if SharedExperiment.DENSITY_CHOICES:
            choices = tuple(sorted({int(value) for value in SharedExperiment.DENSITY_CHOICES}))
        elif SharedExperiment.DENSITY_MIN > 0 and SharedExperiment.DENSITY_MAX > 0:
            low = min(SharedExperiment.DENSITY_MIN, SharedExperiment.DENSITY_MAX)
            high = max(SharedExperiment.DENSITY_MIN, SharedExperiment.DENSITY_MAX)
            choices = tuple(range(low, high + 1))
        else:
            raise ValueError(
                "QECO_DENSITY_MODE=random requires QECO_DENSITY_CHOICES (e.g. '100,300,500,800,1200,1500') "
                "or QECO_DENSITY_MIN/QECO_DENSITY_MAX."
            )
        invalid = [value for value in choices if value <= 0 or value > pool]
        if invalid:
            raise ValueError(
                f"Density choices {invalid} must lie in [1, QECO_NUM_USERS={pool}]; "
                "QECO_NUM_USERS is the agent pool size in random density mode."
            )
        return choices

    def _draw_mask(self, np, count: int):
        mask = np.zeros(self.agent_pool_users, dtype=bool)
        if self.subset_mode == "prefix" or self.mode == "fixed":
            mask[:count] = True
            return mask
        active = self._rng.choice(self.agent_pool_users, size=count, replace=False)
        mask[active] = True
        return mask

    def active_count(self, episode: int) -> int:
        return self.active_users[episode]

    def active_mask(self, episode: int):
        return self._active_masks[episode]

    def regimes(self) -> list[str]:
        return [classify_density_regime(count) for count in self.active_users]

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "subset": self.subset_mode,
            "seed": self.seed,
            "agent_pool_users": self.agent_pool_users,
            "choices": list(self.choices),
            "episodes": self.episodes,
            "active_users": list(self.active_users),
            "regimes": self.regimes(),
        }


def apply_active_user_mask(bitarrive_size, bitarrive_dens, active_mask):
    """Zero task arrivals for inactive UEs so they never receive a task in this episode."""
    if active_mask is None:
        return bitarrive_size, bitarrive_dens
    inactive = ~active_mask
    if inactive.any():
        bitarrive_size[:, inactive] = 0.0
        bitarrive_dens[:, inactive] = 0.0
    return bitarrive_size, bitarrive_dens


def save_series(values, path: Path) -> None:
    with open(path, "w") as f:
        for value in values:
            f.write(f"{value}\n")


def save_common_outputs(run_dir: Path, episode_metrics: dict[str, list[float]], metadata: dict) -> None:
    save_series(episode_metrics["qoe"], run_dir / "QoE.txt")
    save_series(episode_metrics["delay"], run_dir / "Delay.txt")
    save_series(episode_metrics["energy"], run_dir / "Energy.txt")
    save_series(episode_metrics["drop"], run_dir / "Drop.txt")
    if "completion_rate" in episode_metrics:
        save_series(episode_metrics["completion_rate"], run_dir / "CompletionRate.txt")
    if "drop_rate" in episode_metrics:
        save_series(episode_metrics["drop_rate"], run_dir / "DropRate.txt")
    if "completed_task" in episode_metrics:
        save_series(episode_metrics["completed_task"], run_dir / "CompletedTask.txt")
    if "arrived_task" in episode_metrics:
        save_series(episode_metrics["arrived_task"], run_dir / "ArrivedTask.txt")
    if "active_users" in episode_metrics:
        save_series(episode_metrics["active_users"], run_dir / "ActiveUsers.txt")
    save_series(range(1, len(episode_metrics["qoe"]) + 1), run_dir / "Episode.txt")

    comparison_dir = run_dir / "comparison_ready"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    save_series(resample_series(episode_metrics["qoe"], SharedExperiment.COMPARISON_POINTS), comparison_dir / "QoE.txt")
    save_series(resample_series(episode_metrics["delay"], SharedExperiment.COMPARISON_POINTS), comparison_dir / "Delay.txt")
    save_series(resample_series(episode_metrics["energy"], SharedExperiment.COMPARISON_POINTS), comparison_dir / "Energy.txt")
    save_series(resample_series(episode_metrics["drop"], SharedExperiment.COMPARISON_POINTS), comparison_dir / "Drop.txt")
    if "completion_rate" in episode_metrics:
        save_series(
            resample_series(episode_metrics["completion_rate"], SharedExperiment.COMPARISON_POINTS),
            comparison_dir / "CompletionRate.txt",
        )
    if "drop_rate" in episode_metrics:
        save_series(resample_series(episode_metrics["drop_rate"], SharedExperiment.COMPARISON_POINTS), comparison_dir / "DropRate.txt")
    if "completed_task" in episode_metrics:
        save_series(
            resample_series(episode_metrics["completed_task"], SharedExperiment.COMPARISON_POINTS),
            comparison_dir / "CompletedTask.txt",
        )
    if "arrived_task" in episode_metrics:
        save_series(
            resample_series(episode_metrics["arrived_task"], SharedExperiment.COMPARISON_POINTS),
            comparison_dir / "ArrivedTask.txt",
        )
    if "active_users" in episode_metrics:
        save_series(
            resample_series(episode_metrics["active_users"], SharedExperiment.COMPARISON_POINTS),
            comparison_dir / "ActiveUsers.txt",
        )
    save_series(range(1, SharedExperiment.COMPARISON_POINTS + 1), comparison_dir / "ComparisonStep.txt")

    with open(run_dir / "common_eval_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def normalize(parameter, minimum, maximum):
    return (parameter - minimum) / (maximum - minimum)


def qoe_function(delay, max_delay, unfinish_task, ue_energy_state, ue_comp_energy, ue_trans_energy, edge_comp_energy, ue_idle_energy):
    energy_cons = ue_comp_energy + ue_trans_energy
    scaled_energy = normalize(energy_cons, 0, 20) * 10
    cost = 2 * ((ue_energy_state * delay) + ((1 - ue_energy_state) * scaled_energy))
    reward = max_delay * 4
    if unfinish_task:
        return -cost
    return reward - cost


def qeco_adapt_reward_function(
    delay,
    max_delay,
    unfinish_task,
    ue_energy_state,
    ue_comp_energy,
    ue_trans_energy,
    edge_comp_energy,
    ue_idle_energy,
    energy_weight,
):
    energy_cons = ue_comp_energy + ue_trans_energy
    scaled_energy = normalize(energy_cons, 0, 20) * 10
    cost = 2 * ((ue_energy_state * delay) + ((1 - ue_energy_state) * (energy_weight * scaled_energy)))
    reward = max_delay * 4
    if unfinish_task:
        return -cost
    return reward - cost


def qeco_adapt_effective_load(num_users: int | None = None) -> float:
    """Effective load per edge, L_eff = N_U * mean(b) * mean(a) / N_E.

    ``num_users`` defaults to the configured NUM_USERS (legacy fixed-density
    behaviour). In random density mode the evaluator passes the episode's
    active-user count so that load-conditioned terms follow the live density.
    """
    users = SharedExperiment.NUM_USERS if num_users is None else int(num_users)
    base_profile_mean = sum(SharedExperiment.QECO_ARRIVAL_BASE_PROFILE) / len(SharedExperiment.QECO_ARRIVAL_BASE_PROFILE)
    user_activity_mean = (SharedExperiment.QECO_USER_ACTIVITY_MIN + SharedExperiment.QECO_USER_ACTIVITY_MAX) / 2.0
    return (
        max(users, 0)
        * base_profile_mean
        * user_activity_mean
        / max(SharedExperiment.NUM_EDGES, 1)
    )


def qeco_adapt_gating_strength(num_users: int | None = None) -> float:
    effective_load = qeco_adapt_effective_load(num_users)
    load_scale = max(SharedExperiment.NUM_EDGES * SharedExperiment.QECO_ADAPT_LOAD_SCALE, 1e-6)
    return effective_load / (effective_load + load_scale)


def qeco_adapt_energy_weight(num_users: int | None = None) -> float:
    scaled_user_factor = 1.0 + qeco_adapt_gating_strength(num_users)
    return SharedExperiment.QECO_ADAPT_BASE_ENERGY_WEIGHT * (
        scaled_user_factor ** SharedExperiment.QECO_ADAPT_USER_EXPONENT
    )


def build_task_arrivals(env, config, np_module):
    bitarrive_size = np_module.random.uniform(env.min_arrive_size, env.max_arrive_size, size=[env.n_time, env.n_ue])
    time_profile = np_module.repeat(
        np_module.asarray(SharedExperiment.QECO_ARRIVAL_BASE_PROFILE, dtype=float),
        int(np_module.ceil(env.n_time / len(SharedExperiment.QECO_ARRIVAL_BASE_PROFILE))),
    )[: env.n_time]
    user_activity = np_module.linspace(
        SharedExperiment.QECO_USER_ACTIVITY_MIN,
        SharedExperiment.QECO_USER_ACTIVITY_MAX,
        env.n_ue,
    )
    task_prob = np_module.clip(time_profile[:, None] * user_activity[None, :], 0.0, 1.0)
    bitarrive_size = bitarrive_size * (np_module.random.uniform(0, 1, size=[env.n_time, env.n_ue]) < task_prob)
    bitarrive_size[-env.max_delay:, :] = np_module.zeros([env.max_delay, env.n_ue])

    bitarrive_dens = np_module.zeros([env.n_time, env.n_ue])
    for i in range(len(bitarrive_size)):
        for j in range(len(bitarrive_size[i])):
            if bitarrive_size[i][j] != 0:
                bitarrive_dens[i][j] = config.TASK_COMP_DENS[np_module.random.randint(0, len(config.TASK_COMP_DENS))]
    return bitarrive_size, bitarrive_dens


def compute_env_metrics(env, np_module):
    qoe_values = []
    delay_values = []
    energy_values = []
    arrived_task_count = 0
    unfinished_task_count = 0
    for time_index in range(env.n_time):
        for ue_index in range(env.n_ue):
            task_size = env.arrive_task_size[time_index, ue_index]
            if task_size == 0:
                continue
            arrived_task_count += 1
            delay = env.process_delay[time_index, ue_index]
            unfinished = env.unfinish_task[time_index, ue_index]
            unfinished_task_count += int(unfinished)
            ue_comp_energy = env.ue_comp_energy[time_index, ue_index]
            ue_tran_energy = env.ue_tran_energy[time_index, ue_index]
            edge_comp_energy = env.edge_comp_energy[time_index, ue_index]
            ue_idle_energy = env.ue_idle_energy[time_index, ue_index]
            qoe_values.append(
                qoe_function(
                    delay,
                    env.max_delay,
                    unfinished,
                    env.ue_energy_state[ue_index],
                    ue_comp_energy,
                    ue_tran_energy,
                    edge_comp_energy,
                    ue_idle_energy,
                )
            )
            delay_values.append(delay)
            energy_values.append(
                ue_comp_energy
                + ue_tran_energy
                + float(np_module.sum(edge_comp_energy))
                + float(np_module.sum(ue_idle_energy))
            )
    completed_task_count = max(arrived_task_count - unfinished_task_count, 0)
    completion_rate = completed_task_count / arrived_task_count if arrived_task_count else 0.0
    drop_rate = unfinished_task_count / arrived_task_count if arrived_task_count else 0.0
    return {
        "qoe": float(np_module.mean(qoe_values)) if qoe_values else 0.0,
        "delay": float(np_module.mean(delay_values)) if delay_values else 0.0,
        "energy": float(np_module.mean(energy_values)) if energy_values else 0.0,
        "drop": float(unfinished_task_count),
        "completion_rate": float(completion_rate),
        "drop_rate": float(drop_rate),
        "completed_task": float(completed_task_count),
        "arrived_task": float(arrived_task_count),
    }
