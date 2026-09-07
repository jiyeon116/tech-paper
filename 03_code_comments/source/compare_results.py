#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from statistics import mean


ROOT_DIR = Path(__file__).resolve().parent
from experiment_support import SharedExperiment  # noqa: E402
from algorithm_adapters import (  # noqa: E402
    ADAPT2_FAMILY_ALGORITHMS,
    available_channel_algorithm_names,
    channel_algorithm_catalog,
)

# Result roots follow the experiment family / density mode selected through the
# environment (QECO_EXPERIMENT_FAMILY, QECO_DENSITY_MODE, QECO_RESULT_ROOT):
#   legacy : results/<algorithm>/run_*                       -> results/comparisons/<scenario>
#   family : results/<family>/<mode>_density/<algorithm>/run_* -> results/<family>/comparisons/<mode>_density/<scenario>
RESULT_ROOT = SharedExperiment.result_base_dir()
COMPARISON_ROOT = SharedExperiment.comparison_base_dir()

METRICS = ("qoe", "delay", "energy", "drop", "completion_rate")
METRIC_FILES = {
    "qoe": "QoE.txt",
    "delay": "Delay.txt",
    "energy": "Energy.txt",
    "drop": "Drop.txt",
    "completion_rate": "CompletionRate.txt",
}
REQUIRED_METRIC_FILES = ("QoE.txt", "Delay.txt", "Energy.txt", "Drop.txt")
COMPLETION_METRIC_FILES = ("CompletionRate.txt", "DropRate.txt")
ALGORITHM_CATALOG = channel_algorithm_catalog()
ALGORITHM_ALIASES = {}
for external_name in available_channel_algorithm_names():
    result_name = external_name.replace("-", "_")
    ALGORITHM_ALIASES[external_name] = result_name
    ALGORITHM_ALIASES[result_name] = result_name
ALGORITHM_ALIASES["twdqn"] = "tang_wong"
DEFAULT_ALGORITHMS = ("tang_wong", "qeco", "qeco_adapt")
LABELS = {
    name.replace("-", "_"): metadata["label"]
    for name, metadata in ALGORITHM_CATALOG.items()
}
BAR_LABELS = {
    algorithm: label.replace(" DQN", "\nDQN").replace("-ADAPT", "-\nADAPT")
    for algorithm, label in LABELS.items()
}
COLORS = {
    "tang_wong": "#E69F00",
    "qeco": "#0072B2",
    "qeco_reward_weight_only": "#D55E00",
    "qeco_gate_only": "#CC79A7",
    "qeco_fixed_weight_fixed_gate": "#56B4E9",
    "qeco_adaptive_weight_fixed_gate": "#F0E442",
    "qeco_fixed_weight_adaptive_gate": "#999999",
    "qeco_adapt": "#009E73",
    "qeco_adapt_full": "#009E73",
    # adapt2 family: general keeps the fixed-gate colour, adaptive gate keeps the
    # QECO-ADAPT colour, adapt2 gets its own accent.
    "qeco_adapt_general": "#F0E442",
    "qeco_adaptive_gate": "#009E73",
    "qeco_adapt2": "#B2182B",
}
COLOR_CYCLE = ("#E69F00", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#999999")
MARKERS = {
    "tang_wong": "^",
    "qeco": "o",
    "qeco_reward_weight_only": "^",
    "qeco_gate_only": "s",
    "qeco_fixed_weight_fixed_gate": "D",
    "qeco_adaptive_weight_fixed_gate": "v",
    "qeco_fixed_weight_adaptive_gate": "P",
    "qeco_adapt": "s",
    "qeco_adapt_full": "X",
    "qeco_adapt_general": "v",
    "qeco_adaptive_gate": "X",
    "qeco_adapt2": "*",
}
MARKER_CYCLE = ("^", "o", "s", "D", "v", "P", "X")
METRIC_DISPLAY = {
    "qoe": {"title": "QoE", "ylabel": "QoE", "direction": "higher is better", "scale": 1.0, "suffix": ""},
    "delay": {"title": "Delay", "ylabel": "Delay", "direction": "lower is better", "scale": 1.0, "suffix": ""},
    "energy": {"title": "Energy", "ylabel": "Energy", "direction": "lower is better", "scale": 1.0, "suffix": ""},
    "drop": {"title": "Dropped Tasks", "ylabel": "Tasks", "direction": "lower is better", "scale": 1.0, "suffix": ""},
    "completion_rate": {
        "title": "Completion Rate",
        "ylabel": "Completion Rate (%)",
        "direction": "higher is better",
        "scale": 100.0,
        "suffix": "%",
    },
}
COMPARISON_METRIC_MAP = {
    "qoe_mean": "qoe",
    "delay_mean": "delay",
    "energy_mean": "energy",
    "drop_mean": "drop",
    "completion_rate_mean": "completion_rate",
}
METRIC_FILE_STEMS = {
    "qoe": "QoE",
    "delay": "Delay",
    "energy": "Energy",
    "drop": "Drop",
    "completion_rate": "CompletionRate",
}
SECTION_RANGES = (
    ("0_300", "1-300", 1, 300),
    ("300_500", "301-500", 301, 500),
)
SELECTED_FOUR_ALGORITHMS = (
    "qeco",
    "qeco_gate_only",
    "qeco_adaptive_weight_fixed_gate",
    "qeco_adapt_full",
)
# adapt2 family representative four: baseline, general, adaptive gate, adapt2.
SELECTED_FOUR_ADAPT2_ALGORITHMS = tuple(name.replace("-", "_") for name in ADAPT2_FAMILY_ALGORITHMS)
SELECTED_FOUR_SETS = (SELECTED_FOUR_ALGORITHMS, SELECTED_FOUR_ADAPT2_ALGORITHMS)
SELECTED_FOUR_METRICS = ("qoe", "completion_rate")


def configure_matplotlib():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qeco_adapt_mplconfig"))
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 15,
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "savefig.facecolor": "white",
        }
    )
    return plt


def normalize_algorithm_name(value: str) -> str:
    normalized = value.replace("-", "_")
    if normalized not in ALGORITHM_ALIASES:
        raise ValueError(f"Unknown algorithm: {value}")
    return ALGORITHM_ALIASES[normalized]


def algorithm_label(algorithm: str) -> str:
    return LABELS.get(algorithm, algorithm.replace("_", "-"))


def algorithm_bar_label(algorithm: str) -> str:
    return BAR_LABELS.get(algorithm, algorithm_label(algorithm))


def algorithm_color(algorithm: str, index: int = 0) -> str:
    return COLORS.get(algorithm, COLOR_CYCLE[index % len(COLOR_CYCLE)])


def algorithm_marker(algorithm: str, index: int = 0) -> str:
    return MARKERS.get(algorithm, MARKER_CYCLE[index % len(MARKER_CYCLE)])


def metric_values(metric: str, values: list[float]) -> list[float]:
    scale = METRIC_DISPLAY[metric]["scale"]
    return [value * scale for value in values]


def metric_value(metric: str, value: float) -> float:
    return value * METRIC_DISPLAY[metric]["scale"]


def repo_relative_path(path: Path | str) -> str | None:
    """Return a POSIX path relative to this repository, or None for external paths."""
    candidate = Path(path).expanduser()
    try:
        relative = candidate.resolve().relative_to(ROOT_DIR)
    except (OSError, ValueError):
        return None
    return relative.as_posix()


def repo_relative_paths(paths: list[Path] | tuple[Path, ...]) -> list[str | None]:
    return [repo_relative_path(path) for path in paths]


def existing_visualizations(output_dir: Path) -> list[Path]:
    """Return already-rendered PNG outputs when a metadata-only refresh is requested."""
    return sorted(path for path in output_dir.glob("*.png") if path.is_file())


def format_metric_value(metric: str, value: float) -> str:
    display = METRIC_DISPLAY[metric]
    scaled_value = metric_value(metric, value)
    if metric == "completion_rate":
        return f"{scaled_value:.1f}%"
    if metric == "drop":
        return f"{scaled_value:.0f}"
    return f"{scaled_value:.2f}"


def format_axis_tick(metric: str, value: float) -> str:
    if metric == "completion_rate":
        return f"{value:.0f}%"
    return f"{value:g}"


def style_axis(axis, metric: str) -> None:
    display = METRIC_DISPLAY[metric]
    axis.set_title(f"{display['title']} ({display['direction']})", loc="left")
    axis.set_ylabel(display["ylabel"])
    if metric == "completion_rate":
        from matplotlib.ticker import FuncFormatter

        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: format_axis_tick(metric, value)))
    axis.grid(True, axis="y", linestyle="-", linewidth=0.7, alpha=0.22)
    axis.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.12)
    axis.tick_params(axis="both", length=0)


def add_reading_guide(fig, text: str) -> None:
    """Place a paper-style reading guide below the plotting area, never over data."""
    fig.text(0.015, -0.035, f"Reading guide: {text}", fontsize=12, color="#333333", ha="left", va="bottom")


def metric_base_dir(run_dir: Path) -> Path:
    if has_metric_files(run_dir):
        return run_dir
    comparison_ready = run_dir / "comparison_ready"
    if has_metric_files(comparison_ready):
        return comparison_ready
    return run_dir


def normalize_density_choices(values) -> list[int]:
    """Normalize density choices exactly as DensitySchedule resolves them."""
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = [item for item in values.replace(";", ",").split(",") if item.strip()]
    else:
        raw_values = list(values)
    return sorted({int(value) for value in raw_values})


def current_density_choices() -> list[int]:
    if SharedExperiment.DENSITY_MODE != "random":
        return []
    if SharedExperiment.DENSITY_CHOICES:
        return normalize_density_choices(SharedExperiment.DENSITY_CHOICES)
    if SharedExperiment.DENSITY_MIN > 0 and SharedExperiment.DENSITY_MAX > 0:
        low = min(SharedExperiment.DENSITY_MIN, SharedExperiment.DENSITY_MAX)
        high = max(SharedExperiment.DENSITY_MIN, SharedExperiment.DENSITY_MAX)
        return list(range(low, high + 1))
    return []


def has_metric_files(path: Path) -> bool:
    return all((path / filename).exists() for filename in REQUIRED_METRIC_FILES) and any(
        (path / filename).exists() for filename in COMPLETION_METRIC_FILES
    )


def run_profile_matches_current(run_dir: Path) -> bool:
    """True when a run was produced under the currently selected scenario.

    One algorithm folder can hold runs from several user counts / seeds (e.g.
    adapt2 Phase A runs the same adapter at users=100 and 300). Matching on the
    live QECO_NUM_USERS / QECO_RANDOM_SEED / density mode keeps `compare` from
    picking a run that ensure_matching_profiles would reject anyway.
    """
    try:
        profile = load_profile(run_dir)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return profile_matches_current(profile)


def profile_matches_current(profile: dict) -> bool:
    return (
        profile.get("num_users") == SharedExperiment.NUM_USERS
        and profile.get("random_seed") == SharedExperiment.RANDOM_SEED
        and (profile.get("experiment_family") or "") == (SharedExperiment.EXPERIMENT_FAMILY or "")
        and (profile.get("density_mode") or "fixed") == SharedExperiment.DENSITY_MODE
        and profile.get("density_choices", []) == current_density_choices()
    )


def latest_completed_run(algorithm: str) -> Path:
    algorithm_dir = RESULT_ROOT / algorithm
    if not algorithm_dir.exists():
        raise FileNotFoundError(f"No result directory for {algorithm}: {algorithm_dir}")
    run_dirs = sorted(path for path in algorithm_dir.iterdir() if path.is_dir() and path.name.startswith("run_"))
    completed = [run_dir for run_dir in reversed(run_dirs) if has_metric_files(metric_base_dir(run_dir))]
    profiled = False
    for run_dir in completed:
        try:
            profile = load_profile(run_dir)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            continue
        profiled = True
        if profile_matches_current(profile):
            return run_dir
    if completed and not profiled and not (SharedExperiment.EXPERIMENT_FAMILY or "") and SharedExperiment.DENSITY_MODE == "fixed":
        # Legacy fallback for old fixed-density runs that predate
        # experiment_config.json. If any completed run has a readable profile,
        # no-match means the requested scenario is absent and must be reported.
        return completed[0]
    if completed:
        raise FileNotFoundError(
            f"No completed metric run found for {algorithm} matching current scenario "
            f"(users={SharedExperiment.NUM_USERS}, seed={SharedExperiment.RANDOM_SEED}, "
            f"family={SharedExperiment.EXPERIMENT_FAMILY or '(legacy)'}, "
            f"density_mode={SharedExperiment.DENSITY_MODE}, "
            f"density_choices={current_density_choices()})."
        )
    raise FileNotFoundError(
        f"No completed metric run found for {algorithm}. "
        f"Run channel_common_eval.py first; dry-run folders are config-only and are ignored."
    )


def resolve_run_dir(algorithm: str, requested: str | None) -> Path:
    if requested:
        run_dir = Path(requested).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        if not has_metric_files(metric_base_dir(run_dir)):
            raise FileNotFoundError(f"Run directory has no metric files: {run_dir}")
        return run_dir
    return latest_completed_run(algorithm)


def read_series(path: Path) -> list[float]:
    values = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))
    return values


def load_metrics(run_dir: Path) -> dict[str, list[float]]:
    base = metric_base_dir(run_dir)
    missing = [filename for filename in REQUIRED_METRIC_FILES if not (base / filename).exists()]
    if missing:
        raise FileNotFoundError(
            f"Run directory is missing required metric files: {', '.join(missing)}"
        )
    metrics = {
        metric: read_series(base / filename)
        for metric, filename in METRIC_FILES.items()
        if (base / filename).exists()
    }
    if "completion_rate" not in metrics:
        drop_rate_path = base / "DropRate.txt"
        if not drop_rate_path.exists():
            raise FileNotFoundError(
                "Run directory must contain CompletionRate.txt or legacy DropRate.txt"
            )
        metrics["completion_rate"] = [1.0 - value for value in read_series(drop_rate_path)]
    return metrics


def load_density_context(run_dir: Path) -> dict | None:
    """Return the per-episode active-user series and regime thresholds recorded by a run.

    Only random-density runs carry this context; fixed-density runs return None.
    """
    base = metric_base_dir(run_dir)
    active_path = run_dir / "ActiveUsers.txt"
    if not active_path.exists():
        active_path = base / "ActiveUsers.txt"
    if not active_path.exists():
        return None
    config_path = run_dir / "experiment_config.json"
    thresholds = {"sparse_max_users_per_edge": None, "dense_min_users_per_edge": None}
    num_edges = None
    if config_path.exists():
        with open(config_path) as f:
            payload = json.load(f)
        density = payload.get("density", {}) or {}
        if density.get("mode") != "random" and (payload.get("extra", {}) or {}).get("density_mode") != "random":
            return None
        thresholds.update({key: (density.get("adapt2", {}) or {}).get(key) for key in thresholds})
        num_edges = payload.get("num_edges")
    active_users = [int(round(value)) for value in read_series(active_path)]
    return {
        "active_users": active_users,
        "num_edges": num_edges,
        "sparse_max_users_per_edge": thresholds["sparse_max_users_per_edge"],
        "dense_min_users_per_edge": thresholds["dense_min_users_per_edge"],
    }


def regime_for_active_users(active_users: int, context: dict) -> str:
    num_edges = context.get("num_edges") or SharedExperiment.NUM_EDGES
    sparse_max = context.get("sparse_max_users_per_edge")
    dense_min = context.get("dense_min_users_per_edge")
    if sparse_max is None:
        sparse_max = SharedExperiment.ADAPT2_SPARSE_MAX_USERS_PER_EDGE
    if dense_min is None:
        dense_min = SharedExperiment.ADAPT2_DENSE_MIN_USERS_PER_EDGE
    density = float(active_users) / float(max(int(num_edges), 1))
    if density < float(sparse_max):
        return "sparse"
    if density >= float(dense_min):
        return "dense"
    return "general"


def write_regime_outputs(
    output_dir: Path,
    aligned: dict[str, dict[str, list[float]]],
    algorithms: tuple[str, ...],
    context: dict,
) -> tuple[Path, Path, Path]:
    """Write active-user schedule and per-regime metric means for random-density comparisons."""
    steps = len(next(iter(aligned.values()))["qoe"])
    active_users = context["active_users"][:steps]
    regimes = [regime_for_active_users(value, context) for value in active_users]

    schedule_path = output_dir / "active_users_timeseries.csv"
    with open(schedule_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "active_users", "regime"])
        writer.writeheader()
        for index in range(steps):
            writer.writerow({"step": index + 1, "active_users": active_users[index], "regime": regimes[index]})

    rows = []
    for regime in ("sparse", "general", "dense", "all"):
        indices = [index for index in range(steps) if regime == "all" or regimes[index] == regime]
        for algorithm in algorithms:
            row = {
                "regime": regime,
                "episodes": len(indices),
                "algorithm": algorithm,
                "label": algorithm_label(algorithm),
            }
            for metric in METRICS:
                values = [aligned[algorithm][metric][index] for index in indices]
                row[f"{metric}_mean"] = overall_mean(values)
            rows.append(row)

    csv_path = output_dir / "comparison_by_regime.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["regime", "episodes", "algorithm", "label", *[f"{metric}_mean" for metric in METRICS]])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{value:.8f}" if isinstance(value, float) else value) for key, value in row.items()})
    json_path = output_dir / "comparison_by_regime.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "thresholds": {
                    "sparse_max_users_per_edge": context.get("sparse_max_users_per_edge"),
                    "dense_min_users_per_edge": context.get("dense_min_users_per_edge"),
                    "num_edges": context.get("num_edges"),
                },
                "regime_counts": {regime: regimes.count(regime) for regime in ("sparse", "general", "dense")},
                "rows": rows,
            },
            f,
            indent=2,
        )
    return schedule_path, csv_path, json_path


def render_active_users(output_dir: Path, context: dict, steps: int) -> list[Path]:
    plt = configure_matplotlib()
    active_users = context["active_users"][:steps]
    x_axis = list(range(1, len(active_users) + 1))
    fig, axis = plt.subplots(figsize=(12.6, 4.6), constrained_layout=True)
    axis.step(x_axis, active_users, where="mid", color="#4D4D4D", linewidth=1.4)
    axis.set_title("Active Users per Episode (random density schedule)")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Active users")
    axis.grid(True, alpha=0.3)
    path = output_dir / "ActiveUsers_Timeseries.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def load_profile(run_dir: Path) -> dict:
    config_path = run_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing experiment_config.json: {config_path}")
    with open(config_path) as f:
        payload = json.load(f)
    extra = payload.get("extra", {})
    density = payload.get("density", {}) or {}
    density_mode = extra.get("density_mode") or density.get("mode") or "fixed"
    density_choices = extra.get("density_choices")
    if density_choices is None:
        density_choices = density.get("choices") or []
    return {
        "random_seed": payload.get("random_seed"),
        "num_users": payload.get("num_users"),
        "num_edges": payload.get("num_edges"),
        "episodes": extra.get("run_episodes") or payload.get("qeco", {}).get("episodes"),
        "max_delay": payload.get("max_delay"),
        "channels": payload.get("channel", {}).get("count"),
        "access_points": payload.get("access", {}).get("num_access_points"),
        "ap_cluster_size": payload.get("access", {}).get("ap_cluster_size"),
        "n_actions": extra.get("n_actions"),
        "n_features": extra.get("n_features"),
        "n_lstm_state": extra.get("n_lstm_state"),
        "action_space": extra.get("action_space"),
        "state_layout": extra.get("state_layout"),
        "experiment_family": extra.get("experiment_family") or density.get("experiment_family"),
        "density_mode": density_mode,
        "density_choices": normalize_density_choices(density_choices) if density_mode == "random" else [],
    }


def ensure_matching_profiles(run_dirs: dict[str, Path]) -> dict:
    profiles = {algorithm: load_profile(run_dir) for algorithm, run_dir in run_dirs.items()}
    anchor_algorithm = next(iter(profiles))
    anchor = profiles[anchor_algorithm]
    compare_keys = (
        "random_seed",
        "num_users",
        "num_edges",
        "max_delay",
        "channels",
        "access_points",
        "ap_cluster_size",
        "n_actions",
        "n_features",
        "n_lstm_state",
        "experiment_family",
        "density_mode",
        "density_choices",
    )
    mismatches = []
    for algorithm, profile in profiles.items():
        for key in compare_keys:
            if profile.get(key) != anchor.get(key):
                mismatches.append(f"{algorithm}.{key}={profile.get(key)} != {anchor_algorithm}.{key}={anchor.get(key)}")
    if mismatches:
        raise ValueError("Selected runs do not share the same environment: " + "; ".join(mismatches))
    return anchor


def aligned_metrics(metrics_by_algorithm: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    min_length = min(len(metrics["qoe"]) for metrics in metrics_by_algorithm.values())
    return {
        algorithm: {
            metric: values[:min_length]
            for metric, values in metrics.items()
        }
        for algorithm, metrics in metrics_by_algorithm.items()
    }


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    averaged = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        averaged.append(running_sum / min(index + 1, window))
    return averaged


def overall_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def scenario_name(profile: dict) -> str:
    if profile.get("density_mode") == "random" and profile.get("density_choices"):
        choices = profile["density_choices"]
        user_part = f"user_{min(choices)}-{max(choices)}_random"
    else:
        user_part = f"user_{profile['num_users']}"
    return (
        f"{user_part}"
        f"_seed_{profile['random_seed']}"
        f"_ep_{profile['episodes']}"
        f"_edge_{profile['num_edges']}"
        f"_ap_{profile['access_points']}"
        f"_channel_{profile['channels']}"
    )


def write_timeseries_csv(output_dir: Path, aligned: dict[str, dict[str, list[float]]], algorithms: tuple[str, ...]) -> Path:
    output_path = output_dir / "comparison_timeseries.csv"
    steps = len(next(iter(aligned.values()))["qoe"])
    fieldnames = ["step"]
    for algorithm in algorithms:
        for metric in METRICS:
            fieldnames.append(f"{algorithm}_{metric}")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(steps):
            row = {"step": index + 1}
            for algorithm in algorithms:
                for metric in METRICS:
                    row[f"{algorithm}_{metric}"] = f"{aligned[algorithm][metric][index]:.8f}"
            writer.writerow(row)
    return output_path


def load_timeseries_csv(input_path: Path) -> tuple[dict[str, dict[str, list[float]]], tuple[str, ...]]:
    """Load a comparison_timeseries.csv without rerunning an experiment."""
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "step" not in fieldnames:
            raise ValueError(f"Timeseries CSV is missing the step column: {input_path}")

        algorithms: list[str] = []
        columns: dict[tuple[str, str], str] = {}
        for fieldname in fieldnames:
            if fieldname == "step":
                continue
            metric = next((item for item in METRICS if fieldname.endswith(f"_{item}")), None)
            if not metric:
                raise ValueError(f"Unexpected timeseries CSV column: {fieldname}")
            algorithm = fieldname[: -(len(metric) + 1)]
            if not algorithm:
                raise ValueError(f"Timeseries CSV column has no algorithm prefix: {fieldname}")
            if algorithm not in algorithms:
                algorithms.append(algorithm)
            columns[(algorithm, metric)] = fieldname

        if not algorithms:
            raise ValueError(f"Timeseries CSV has no algorithm metrics: {input_path}")
        missing_columns = [
            f"{algorithm}_{metric}"
            for algorithm in algorithms
            for metric in METRICS
            if (algorithm, metric) not in columns
        ]
        if missing_columns:
            raise ValueError(f"Timeseries CSV is missing metric columns: {', '.join(missing_columns)}")

        aligned = {algorithm: {metric: [] for metric in METRICS} for algorithm in algorithms}
        for expected_step, row in enumerate(reader, start=1):
            try:
                step = int(row["step"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid step in {input_path}: {row.get('step')!r}") from exc
            if step != expected_step:
                raise ValueError(f"Timeseries CSV steps must be contiguous from 1; got {step} at row {expected_step}")
            for algorithm in algorithms:
                for metric in METRICS:
                    try:
                        aligned[algorithm][metric].append(float(row[columns[(algorithm, metric)]]))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid {algorithm}_{metric} value at step {step} in {input_path}"
                        ) from exc
    if not next(iter(aligned.values()))["qoe"]:
        raise ValueError(f"Timeseries CSV has no data rows: {input_path}")
    return aligned, tuple(algorithms)


def write_overall_comparison(output_dir: Path, aligned: dict[str, dict[str, list[float]]], algorithms: tuple[str, ...]) -> dict:
    payload = {}
    for algorithm in algorithms:
        payload[algorithm] = {
            "label": algorithm_label(algorithm),
            "qoe_mean": overall_mean(aligned[algorithm]["qoe"]),
            "delay_mean": overall_mean(aligned[algorithm]["delay"]),
            "energy_mean": overall_mean(aligned[algorithm]["energy"]),
            "drop_mean": overall_mean(aligned[algorithm]["drop"]),
            "completion_rate_mean": overall_mean(aligned[algorithm]["completion_rate"]),
        }
    with open(output_dir / "comparison_overall.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(output_dir / "comparison_overall.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "label",
                "qoe_mean",
                "delay_mean",
                "energy_mean",
                "drop_mean",
                "completion_rate_mean",
            ]
        )
        for algorithm in algorithms:
            row = payload[algorithm]
            writer.writerow(
                [
                    algorithm,
                    row["label"],
                    f"{row['qoe_mean']:.8f}",
                    f"{row['delay_mean']:.8f}",
                    f"{row['energy_mean']:.8f}",
                    f"{row['drop_mean']:.8f}",
                    f"{row['completion_rate_mean']:.8f}",
                ]
            )
    return payload


def section_values(values: list[float], start_step: int, end_step: int) -> list[float]:
    start_index = max(0, start_step - 1)
    end_index = min(len(values), end_step)
    return values[start_index:end_index]


def write_section_means(output_dir: Path, aligned: dict[str, dict[str, list[float]]], algorithms: tuple[str, ...]) -> tuple[Path, Path]:
    payload = []
    for section_key, section_label, start_step, end_step in SECTION_RANGES:
        for algorithm in algorithms:
            row = {
                "section": section_label,
                "section_key": section_key,
                "algorithm": algorithm,
                "label": algorithm_label(algorithm),
            }
            for metric in METRICS:
                row[metric] = overall_mean(section_values(aligned[algorithm][metric], start_step, end_step))
            payload.append(row)

    csv_path = output_dir / "comparison_section_means.csv"
    json_path = output_dir / "comparison_section_means.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "section_key", "algorithm", "label", *METRICS],
        )
        writer.writeheader()
        writer.writerows(payload)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return csv_path, json_path


def render_timeseries(output_dir: Path, aligned: dict[str, dict[str, list[float]]], algorithms: tuple[str, ...], smoothing_window: int) -> list[Path]:
    plt = configure_matplotlib()
    x_axis = list(range(1, len(next(iter(aligned.values()))["qoe"]) + 1))
    marker_interval = max(1, len(x_axis) // 25)
    created = []

    for metric in METRICS:
        fig, axis = plt.subplots(figsize=(12.6, 7.0), constrained_layout=True)
        for index, algorithm in enumerate(algorithms):
            values = aligned[algorithm][metric]
            raw_values = metric_values(metric, values)
            smoothed_values = metric_values(metric, moving_average(values, smoothing_window))
            axis.plot(
                x_axis,
                raw_values,
                color=algorithm_color(algorithm, index),
                linewidth=0.85,
                alpha=0.18,
                marker=algorithm_marker(algorithm, index),
                markersize=4.4,
                markevery=marker_interval,
                markerfacecolor="white",
                markeredgewidth=0.9,
            )
            axis.plot(
                x_axis,
                smoothed_values,
                label=algorithm_label(algorithm),
                color=algorithm_color(algorithm, index),
                linewidth=2.8,
                marker=algorithm_marker(algorithm, index),
                markersize=8.0,
                markevery=marker_interval,
                markerfacecolor="white",
                markeredgewidth=1.5,
            )[0]
        style_axis(axis, metric)
        axis.set_xlabel("Episode")
        axis.set_xlim(x_axis[0], x_axis[-1] + len(x_axis) * 0.06)
        axis.legend(loc="best", frameon=False)
        add_reading_guide(
            fig,
            f"thick line = {smoothing_window}-episode moving average; faint line = raw episode value; markers = sampled episodes.",
        )
        fig.suptitle(f"{METRIC_DISPLAY[metric]['title']} Timeseries", fontsize=22, fontweight="bold")
        path = output_dir / f"{METRIC_FILE_STEMS[metric]}_Timeseries.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        created.append(path)
    return created


def render_branch_timeseries(output_dir: Path, aligned: dict[str, dict[str, list[float]]], algorithms: tuple[str, ...], smoothing_window: int) -> list[Path]:
    plt = configure_matplotlib()
    steps = len(next(iter(aligned.values()))["qoe"])
    created = []

    for section_key, section_label, start_step, end_step in SECTION_RANGES:
        if start_step > steps:
            continue
        x_axis = list(range(start_step, min(end_step, steps) + 1))
        marker_interval = max(1, len(x_axis) // 14)
        fig, axes = plt.subplots(len(METRICS), 1, figsize=(17.5, 16.0), sharex=True, constrained_layout=True)
        fig.suptitle(
            f"Branch Timeseries: Episode {section_label} (smoothed)",
            fontsize=22,
            fontweight="bold",
        )
        for axis, metric in zip(axes, METRICS):
            for index, algorithm in enumerate(algorithms):
                values = section_values(aligned[algorithm][metric], start_step, end_step)
                raw_values = metric_values(metric, values)
                smoothed_values = metric_values(metric, moving_average(values, smoothing_window))
                axis.plot(
                    x_axis,
                    raw_values,
                    color=algorithm_color(algorithm, index),
                    linewidth=0.7,
                    alpha=0.14,
                    marker=algorithm_marker(algorithm, index),
                    markersize=4.0,
                    markevery=marker_interval,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                )
                axis.plot(
                    x_axis,
                    smoothed_values,
                    label=algorithm_label(algorithm),
                    color=algorithm_color(algorithm, index),
                    linewidth=2.6,
                    marker=algorithm_marker(algorithm, index),
                    markersize=7.4,
                    markevery=marker_interval,
                    markerfacecolor="white",
                    markeredgewidth=1.4,
                )
            style_axis(axis, metric)
        axes[-1].set_xlabel("Episode")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=False)
        add_reading_guide(
            fig,
            f"this panel shows episodes {section_label}; thick line = {smoothing_window}-episode moving average; faint line = raw episode value.",
        )
        path = output_dir / f"branch_timeseries_{section_key}_smoothed.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        created.append(path)
    return created


def selected_four_algorithms(algorithms: tuple[str, ...]) -> tuple[str, ...]:
    available = set(algorithms)
    for candidate_set in SELECTED_FOUR_SETS:
        if all(algorithm in available for algorithm in candidate_set):
            return candidate_set
    return ()


def render_selected_four_timeseries(
    output_dir: Path,
    aligned: dict[str, dict[str, list[float]]],
    algorithms: tuple[str, ...],
    smoothing_window: int,
) -> list[Path]:
    selected_algorithms = selected_four_algorithms(algorithms)
    if not selected_algorithms:
        return []

    plt = configure_matplotlib()
    steps = len(next(iter(aligned.values()))["qoe"])
    created = []

    def render_metric(metric: str, section: tuple[str, str, int, int] | None = None) -> None:
        if section:
            section_key, section_label, start_step, end_step = section
            if start_step > steps:
                return
            x_axis = list(range(start_step, min(end_step, steps) + 1))
            title_suffix = f": Episode {section_label}"
            filename = f"selected4_{METRIC_FILE_STEMS[metric]}_Timeseries_{section_key}.png"
            guide_context = f"episodes {section_label}"
        else:
            x_axis = list(range(1, steps + 1))
            title_suffix = ""
            filename = f"selected4_{METRIC_FILE_STEMS[metric]}_Timeseries.png"
            guide_context = "all episodes"

        marker_interval = max(1, len(x_axis) // 18)
        fig, axis = plt.subplots(figsize=(16.2, 9.3), constrained_layout=True)

        for index, algorithm in enumerate(selected_algorithms):
            if section:
                _, _, start_step, end_step = section
                values = section_values(aligned[algorithm][metric], start_step, end_step)
            else:
                values = aligned[algorithm][metric]

            raw_values = metric_values(metric, values)
            smoothed_values = metric_values(metric, moving_average(values, smoothing_window))
            color = algorithm_color(algorithm, index)
            marker = algorithm_marker(algorithm, index)
            axis.plot(
                x_axis,
                raw_values,
                color=color,
                linewidth=0.8,
                alpha=0.16,
                marker=marker,
                markersize=4.4,
                markevery=marker_interval,
                markerfacecolor="white",
                markeredgewidth=0.9,
            )
            axis.plot(
                x_axis,
                smoothed_values,
                label=algorithm_label(algorithm),
                color=color,
                linewidth=3.0,
                marker=marker,
                markersize=8.2,
                markevery=marker_interval,
                markerfacecolor="white",
                markeredgewidth=1.5,
            )

        if not section and x_axis[0] <= 300 <= x_axis[-1]:
            axis.axvline(300, color="#222222", linestyle="--", linewidth=1.0, alpha=0.42)
            axis.text(
                300,
                0.98,
                "Episode 300 split",
                transform=axis.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=12,
                color="#333333",
            )

        style_axis(axis, metric)
        axis.set_xlabel("Episode")
        axis.set_xlim(x_axis[0], x_axis[-1] + len(x_axis) * 0.06)
        axis.legend(loc="best", frameon=False)
        add_reading_guide(
            fig,
            f"selected four algorithms, {guide_context}; thick line = {smoothing_window}-episode moving average; faint line = raw episode value.",
        )
        fig.suptitle(
            f"Selected Four {METRIC_DISPLAY[metric]['title']} Timeseries{title_suffix}",
            fontsize=22,
            fontweight="bold",
        )
        path = output_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    for metric in SELECTED_FOUR_METRICS:
        render_metric(metric)
        for section in SECTION_RANGES:
            render_metric(metric, section)

    return created


def render_overall_charts(output_dir: Path, overall: dict, algorithms: tuple[str, ...]) -> list[Path]:
    plt = configure_matplotlib()
    created = []

    for overall_metric, metric in COMPARISON_METRIC_MAP.items():
        fig, axis = plt.subplots(figsize=(10.0, 5.5), constrained_layout=True)
        values = [metric_value(metric, overall[algorithm][overall_metric]) for algorithm in algorithms]
        positions = list(range(len(algorithms)))
        bars = axis.barh(
            positions,
            values,
            color=[algorithm_color(algorithm, index) for index, algorithm in enumerate(algorithms)],
            alpha=0.92,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels([algorithm_label(algorithm) for algorithm in algorithms])
        axis.invert_yaxis()
        axis.set_title(f"{METRIC_DISPLAY[metric]['title']} Overall Mean ({METRIC_DISPLAY[metric]['direction']})", loc="left")
        axis.grid(True, axis="x", linestyle="-", linewidth=0.7, alpha=0.22)
        axis.tick_params(axis="both", length=0)
        axis.set_xlabel(METRIC_DISPLAY[metric]["ylabel"])
        x_max = max(values) if values else 1.0
        axis.set_xlim(0, x_max * 1.18 if x_max > 0 else 1.0)
        for bar, value in zip(bars, values):
            label = f"{value:.1f}%" if metric == "completion_rate" else (f"{value:.0f}" if metric == "drop" else f"{value:.2f}")
            axis.annotate(
                label,
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=13,
            )
        add_reading_guide(fig, "bar length and printed value are full-episode means.")
        path = output_dir / f"{METRIC_FILE_STEMS[metric]}_chart.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    return created


def remove_obsolete_outputs(output_dir: Path) -> None:
    obsolete_names = [
        "comparison_finals.csv",
        "comparison_finals.json",
        "comparison_finals.png",
        "comparison_runtime.png",
        "Runtime_chart.png",
        "comparison_timeseries.png",
        "comparison_timeseries_smoothed.png",
        "branch_timeseries_0_300.png",
        "branch_timeseries_300_500.png",
    ]
    for filename in obsolete_names:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def run_comparison(args) -> Path:
    algorithms = tuple(normalize_algorithm_name(algorithm) for algorithm in args.algorithms)
    requested = {
        "tang_wong": args.tang_wong_run,
        "qeco": args.qeco_run,
        "qeco_adapt": args.qeco_adapt_run,
    }
    run_dirs = {algorithm: resolve_run_dir(algorithm, requested.get(algorithm)) for algorithm in algorithms}
    profile = ensure_matching_profiles(run_dirs)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else COMPARISON_ROOT / scenario_name(profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_visualize:
        remove_obsolete_outputs(output_dir)

    metrics = {algorithm: load_metrics(run_dirs[algorithm]) for algorithm in algorithms}
    aligned = aligned_metrics(metrics)

    timeseries_csv = write_timeseries_csv(output_dir, aligned, algorithms)
    overall = write_overall_comparison(output_dir, aligned, algorithms)
    section_csv, section_json = write_section_means(output_dir, aligned, algorithms)
    regime_outputs = {}
    density_context = None
    if profile.get("density_mode") == "random":
        anchor_algorithm = next(iter(run_dirs))
        density_context = load_density_context(run_dirs[anchor_algorithm])
        if density_context is not None:
            schedule_path, regime_csv, regime_json = write_regime_outputs(output_dir, aligned, algorithms, density_context)
            regime_outputs = {
                "active_users_csv": str(schedule_path),
                "active_users_csv_relative": repo_relative_path(schedule_path),
                "by_regime_csv": str(regime_csv),
                "by_regime_csv_relative": repo_relative_path(regime_csv),
                "by_regime_json": str(regime_json),
                "by_regime_json_relative": repo_relative_path(regime_json),
            }
    created_visuals = []
    if not args.no_visualize:
        created_visuals.extend(render_timeseries(output_dir, aligned, algorithms, args.smoothing_window))
        created_visuals.extend(render_branch_timeseries(output_dir, aligned, algorithms, args.smoothing_window))
        created_visuals.extend(render_selected_four_timeseries(output_dir, aligned, algorithms, args.smoothing_window))
        created_visuals.extend(render_overall_charts(output_dir, overall, algorithms))
        if density_context is not None:
            steps = len(next(iter(aligned.values()))["qoe"])
            created_visuals.extend(render_active_users(output_dir, density_context, steps))
    else:
        created_visuals = existing_visualizations(output_dir)

    summary = {
        "profile": profile,
        "algorithms": list(algorithms),
        "runs": {algorithm: str(run_dirs[algorithm]) for algorithm in algorithms},
        "runs_relative": {algorithm: repo_relative_path(run_dirs[algorithm]) for algorithm in algorithms},
        "output_dir": str(output_dir),
        "output_dir_relative": repo_relative_path(output_dir),
        "result_root": str(RESULT_ROOT),
        "result_root_relative": repo_relative_path(RESULT_ROOT),
        "timeseries_csv": str(timeseries_csv),
        "timeseries_csv_relative": repo_relative_path(timeseries_csv),
        "overall_csv": str(output_dir / "comparison_overall.csv"),
        "overall_csv_relative": repo_relative_path(output_dir / "comparison_overall.csv"),
        "overall_json": str(output_dir / "comparison_overall.json"),
        "overall_json_relative": repo_relative_path(output_dir / "comparison_overall.json"),
        "section_means_csv": str(section_csv),
        "section_means_csv_relative": repo_relative_path(section_csv),
        "section_means_json": str(section_json),
        "section_means_json_relative": repo_relative_path(section_json),
        **regime_outputs,
        "visualizations": [str(path) for path in created_visuals],
        "visualizations_relative": repo_relative_paths(created_visuals),
    }
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Comparison runs:")
    for algorithm in algorithms:
        print(f"- {algorithm_label(algorithm)}: {run_dirs[algorithm]}")
    print(f"Comparison output: {output_dir}")
    print("Created:")
    print(f"- {timeseries_csv}")
    print(f"- {output_dir / 'comparison_overall.csv'}")
    print(f"- {output_dir / 'comparison_overall.json'}")
    print(f"- {section_csv}")
    print(f"- {section_json}")
    print(f"- {output_dir / 'comparison_summary.json'}")
    if args.no_visualize and created_visuals:
        print("Preserved visualizations:")
    for path in created_visuals:
        print(f"- {path}")
    return output_dir


def regenerate_from_timeseries_csv(args) -> Path:
    input_path = Path(args.timeseries_csv).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Timeseries CSV does not exist: {input_path}")
    aligned, algorithms = load_timeseries_csv(input_path)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_visualize:
        remove_obsolete_outputs(output_dir)

    timeseries_csv = write_timeseries_csv(output_dir, aligned, algorithms)
    overall = write_overall_comparison(output_dir, aligned, algorithms)
    section_csv, section_json = write_section_means(output_dir, aligned, algorithms)
    created_visuals = []
    if not args.no_visualize:
        created_visuals.extend(render_timeseries(output_dir, aligned, algorithms, args.smoothing_window))
        created_visuals.extend(render_branch_timeseries(output_dir, aligned, algorithms, args.smoothing_window))
        created_visuals.extend(render_selected_four_timeseries(output_dir, aligned, algorithms, args.smoothing_window))
        created_visuals.extend(render_overall_charts(output_dir, overall, algorithms))
    else:
        created_visuals = existing_visualizations(output_dir)

    summary_path = output_dir / "comparison_summary.json"
    existing_summary = {}
    if summary_path.is_file():
        with open(summary_path) as f:
            existing_summary = json.load(f)
    summary = {
        **existing_summary,
        "algorithms": list(algorithms),
        "regenerated_from_timeseries_csv": str(input_path),
        "regenerated_from_timeseries_csv_relative": repo_relative_path(input_path),
        "output_dir": str(output_dir),
        "output_dir_relative": repo_relative_path(output_dir),
        "timeseries_csv": str(timeseries_csv),
        "timeseries_csv_relative": repo_relative_path(timeseries_csv),
        "overall_csv": str(output_dir / "comparison_overall.csv"),
        "overall_csv_relative": repo_relative_path(output_dir / "comparison_overall.csv"),
        "overall_json": str(output_dir / "comparison_overall.json"),
        "overall_json_relative": repo_relative_path(output_dir / "comparison_overall.json"),
        "section_means_csv": str(section_csv),
        "section_means_csv_relative": repo_relative_path(section_csv),
        "section_means_json": str(section_json),
        "section_means_json_relative": repo_relative_path(section_json),
        "visualizations": [str(path) for path in created_visuals],
        "visualizations_relative": repo_relative_paths(created_visuals),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Regenerated comparison output from: {input_path}")
    print(f"Comparison output: {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare QECO-Adapt channel-aware experiment results.")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(DEFAULT_ALGORITHMS),
        help="Algorithms to compare. Defaults to tang_wong, qeco, and qeco_adapt.",
    )
    parser.add_argument("--tang-wong-run", help="Explicit Tang&Wong run directory.")
    parser.add_argument("--qeco-run", help="Explicit QECO run directory.")
    parser.add_argument("--qeco-adapt-run", help="Explicit QECO-ADAPT run directory.")
    parser.add_argument("--output-dir", help="Comparison output directory. Defaults to results/comparisons/<scenario>.")
    parser.add_argument(
        "--timeseries-csv",
        help="Regenerate CSV/JSON/PNGs from comparison_timeseries.csv without running experiments.",
    )
    parser.add_argument("--smoothing-window", type=int, default=25, help="Moving average window for smoothed chart.")
    parser.add_argument("--no-visualize", action="store_true", help="Only write CSV/JSON outputs; skip PNG visualization.")
    parser.add_argument(
        "--family",
        help="Experiment family folder (e.g. adapt2). Overrides QECO_EXPERIMENT_FAMILY; "
        "runs are read from results/<family>/<density>_density and comparisons written under results/<family>/comparisons/.",
    )
    parser.add_argument(
        "--density-mode",
        choices=("fixed", "random"),
        help="Density mode folder to read (fixed|random). Overrides QECO_DENSITY_MODE.",
    )
    parser.add_argument(
        "--density-choices",
        help="Comma-separated active-user counts for random-density run selection. Overrides QECO_DENSITY_CHOICES.",
    )
    args = parser.parse_args()
    if args.family is not None or args.density_mode is not None or args.density_choices is not None:
        global RESULT_ROOT, COMPARISON_ROOT
        if args.family is not None:
            SharedExperiment.EXPERIMENT_FAMILY = args.family.strip()
        if args.density_mode is not None:
            SharedExperiment.DENSITY_MODE = args.density_mode
        if args.density_choices is not None:
            SharedExperiment.DENSITY_CHOICES = tuple(normalize_density_choices(args.density_choices))
        RESULT_ROOT = SharedExperiment.result_base_dir()
        COMPARISON_ROOT = SharedExperiment.comparison_base_dir()
    try:
        if args.timeseries_csv:
            regenerate_from_timeseries_csv(args)
        else:
            run_comparison(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
