#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = ROOT / "env_channel" / "bin" / "python"
PYTHON_VERSION_MIN = (3, 10)
PYTHON_VERSION_MAX_EXCLUSIVE = (3, 11)
REQUIRED_RUNTIME_MODULES = ("numpy", "matplotlib", "tensorflow")

CORE_ALGORITHMS = ("tang-wong", "qeco", "qeco-adapt")
ABLATED_ALGORITHMS = (
    "qeco",
    "qeco-reward-weight-only",
    "qeco-gate-only",
    "qeco-fixed-weight-fixed-gate",
    "qeco-adaptive-weight-fixed-gate",
    "qeco-fixed-weight-adaptive-gate",
    "qeco-adapt-full",
)
REVIEW_ALGORITHMS = (
    "qeco",
    "qeco-gate-only",
    "qeco-adaptive-weight-fixed-gate",
    "qeco-adapt-full",
)
# qeco-adapt2 family: baseline, general (adaptive weight/fixed gate), adaptive gate, adapt2 selector.
ADAPT2_ALGORITHMS = (
    "qeco",
    "qeco-adapt-general",
    "qeco-adaptive-gate",
    "qeco-adapt2",
)
ADAPT2_SPARSE_ABLATION_ALGORITHMS = (
    "qeco",
    "qeco-gate-only",
    "qeco-adapt-general",
    "qeco-adaptive-gate",
)
ADAPT2_FAMILY = "adapt2"
DEFAULT_RANDOM_DENSITY_CHOICES = "100,300,500,800,1200,1500"


@dataclass(frozen=True)
class RuntimeCheck:
    path: Path
    source: str
    exists: bool
    executable: bool
    version: str | None
    supported: bool
    issues: tuple[str, ...]


def python_bin() -> Path:
    override = os.environ.get("QECO_PYTHON")
    if override:
        return Path(override)
    return DEFAULT_PYTHON


def python_source() -> str:
    return "QECO_PYTHON" if os.environ.get("QECO_PYTHON") else "env_channel"


def parse_python_version(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def is_supported_python_version(version_text: str | None) -> bool:
    parsed = parse_python_version(version_text or "")
    if parsed is None:
        return False
    return PYTHON_VERSION_MIN <= parsed[:2] < PYTHON_VERSION_MAX_EXCLUSIVE


def missing_runtime_modules(path: Path) -> tuple[str, ...]:
    module_names = repr(REQUIRED_RUNTIME_MODULES)
    completed = subprocess.run(
        [
            str(path),
            "-c",
            (
                "import importlib.util; "
                f"names={module_names}; "
                "print(','.join(name for name in names if importlib.util.find_spec(name) is None))"
            ),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return tuple(name for name in completed.stdout.strip().split(",") if name)


def check_python_runtime() -> RuntimeCheck:
    path = python_bin()
    issues: list[str] = []
    exists = path.exists()
    executable = os.access(path, os.X_OK) if exists else False
    version: str | None = None

    if not exists:
        issues.append(
            f"Python runtime not found at {path}. Create env_channel with Python 3.10 "
            "or set QECO_PYTHON to a Python 3.10 interpreter."
        )
    elif not executable:
        issues.append(f"Python runtime is not executable: {path}")
    else:
        try:
            completed = subprocess.run(
                [str(path), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            version = completed.stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            issues.append(f"Could not validate Python runtime {path}: {exc}")

    supported = is_supported_python_version(version)
    if version and not supported:
        issues.append(
            f"Unsupported Python version {version}; use Python 3.10 for the TensorFlow 2.8 runtime."
        )
    elif supported:
        try:
            missing_modules = missing_runtime_modules(path)
        except (OSError, subprocess.CalledProcessError) as exc:
            issues.append(f"Could not inspect required runtime modules in {path}: {exc}")
        else:
            if missing_modules:
                issues.append(
                    "Missing required runtime modules: "
                    + ", ".join(missing_modules)
                    + ". Install requirements.txt into env_channel."
                )

    return RuntimeCheck(
        path=path,
        source=python_source(),
        exists=exists,
        executable=executable,
        version=version,
        supported=supported,
        issues=tuple(issues),
    )


def ensure_python_runtime() -> Path:
    runtime = check_python_runtime()
    if runtime.issues:
        for issue in runtime.issues:
            print(f"[runtime error] {issue}", file=sys.stderr)
        raise SystemExit(1)
    return runtime.path


def warn_python_runtime_if_needed() -> None:
    runtime = check_python_runtime()
    if runtime.issues:
        for issue in runtime.issues:
            print(f"# Runtime warning: {issue}", file=sys.stderr)


def scenario_name(args: argparse.Namespace, algorithms: tuple[str, ...]) -> str:
    if getattr(args, "density_mode", "fixed") == "random":
        choices = [int(item) for item in density_choices_for(args).split(",") if item.strip()]
        user_part = f"user_{min(choices)}-{max(choices)}_random"
    else:
        user_part = f"user_{args.users}"
    if getattr(args, "family", None):
        prefix = "adapt2" if args.family == ADAPT2_FAMILY else args.family
    else:
        prefix = "ablation" if len(algorithms) > 3 or any("gate" in item or "weight" in item for item in algorithms) else "user"
    return f"{prefix}_{user_part}_seed_{args.seed}_ep_500_edge_3_ap_6_channel_4"


def density_choices_for(args: argparse.Namespace) -> str:
    choices = getattr(args, "density_choices", None)
    if choices:
        return choices
    return DEFAULT_RANDOM_DENSITY_CHOICES


def env_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Environment variables that select users, seed, family and density mode for child processes."""
    pairs = [("QECO_NUM_USERS", str(args.users)), ("QECO_RANDOM_SEED", str(args.seed))]
    family = getattr(args, "family", None)
    if family:
        pairs.append(("QECO_EXPERIMENT_FAMILY", family))
    density_mode = getattr(args, "density_mode", "fixed")
    if density_mode == "random":
        pairs.append(("QECO_DENSITY_MODE", "random"))
        pairs.append(("QECO_DENSITY_CHOICES", density_choices_for(args)))
    return pairs


def env_for(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in env_pairs(args):
        env[key] = value
    env["PYTHONUNBUFFERED"] = "1"
    return env


def env_prefix(args: argparse.Namespace) -> str:
    return " ".join(f"{key}={value}" for key, value in env_pairs(args))


def guard_users(args: argparse.Namespace) -> None:
    users = args.users
    if getattr(args, "density_mode", "fixed") == "random":
        choices = [int(item) for item in density_choices_for(args).split(",") if item.strip()]
        if max(choices) > users:
            raise SystemExit(
                f"--density-choices max {max(choices)} exceeds --users {users}; "
                "in random density mode --users is the agent pool size and must cover every choice."
            )
    if users >= 1200 and not args.allow_1200:
        raise SystemExit(
            "Refusing user>=1200 in this workflow session. "
            "Use --allow-1200 only when the deferred long-running experiment is intentionally resumed "
            "(adapt2 dense-regime and random-density runs need it as well)."
        )


def run_command(command: list[str], *, args: argparse.Namespace, dry: bool) -> None:
    print("$ " + env_prefix(args) + " " + " ".join(command))
    if dry:
        return
    subprocess.run(command, cwd=ROOT, env=env_for(args), check=True)


def common_required_files() -> list[Path]:
    return [
        ROOT / "channel_common_eval.py",
        ROOT / "compare_results.py",
        ROOT / "experiment_support.py",
        ROOT / "algorithm_adapters" / "registry.py",
        ROOT / "channel_env" / "channel_mec_env.py",
        ROOT / "qeco_runtime" / "D3QN.py",
    ]


def family_required_files(args: argparse.Namespace) -> list[Path]:
    if args.family == ADAPT2_FAMILY:
        return [
            ROOT / "docs" / "qeco_adapt2_design.md",
        ]
    return []


def required_files(args: argparse.Namespace) -> list[Path]:
    return common_required_files() + family_required_files(args)


def family_result_layout(args: argparse.Namespace) -> list[Path]:
    if args.family == ADAPT2_FAMILY:
        return [
            ROOT / "results" / "adapt2" / "fixed_density",
            ROOT / "results" / "adapt2" / "random_density",
            ROOT / "results" / "adapt2" / "comparisons" / "fixed_density",
            ROOT / "results" / "adapt2" / "comparisons" / "random_density",
        ]
    return [
        ROOT / "results" / "comparisons",
    ]


def scenario_user_max(path: Path) -> int | None:
    match = re.search(r"user_(\d+)(?:-(\d+))?", path.name)
    if not match:
        return None
    values = [int(value) for value in match.groups() if value is not None]
    return max(values) if values else None


def large_load_comparison_dirs() -> list[Path]:
    candidates: list[Path] = []
    roots = [
        ROOT / "results" / "comparisons",
        ROOT / "results" / "adapt2" / "comparisons" / "fixed_density",
        ROOT / "results" / "adapt2" / "comparisons" / "random_density",
    ]
    for base in roots:
        if not base.exists():
            continue
        for path in base.iterdir():
            if path.is_dir() and (scenario_user_max(path) or 0) >= 1200:
                candidates.append(path)
    return sorted(candidates)


def cmd_status(args: argparse.Namespace) -> None:
    print("[QECO workflow status]")
    print(f"- root: {ROOT}")
    runtime = check_python_runtime()
    print(f"- python: {runtime.path} ({runtime.source})")
    if runtime.version:
        print(f"- python_version: {runtime.version}")
    print(f"- users: {args.users}")
    print(f"- seed: {args.seed}")
    print(f"- family: {args.family or '(legacy layout)'}")
    print(f"- density_mode: {args.density_mode}")
    if args.density_mode == "random":
        print(f"- density_choices: {density_choices_for(args)}")
    print(f"- user>=1200 allowed: {args.allow_1200}")

    missing = [path for path in required_files(args) if not path.exists()]
    if missing:
        print("\n[missing]")
        for path in missing:
            print(f"- {path.relative_to(ROOT)}")
        raise SystemExit(1)

    print("\n[required files]")
    for path in required_files(args):
        print(f"- ok: {path.relative_to(ROOT)}")

    print("\n[result layout]")
    for path in family_result_layout(args):
        status = "ok" if path.exists() else "missing"
        print(f"- {status}: {path.relative_to(ROOT)}")

    retained = large_load_comparison_dirs()
    if retained:
        print("\n[retained large-load comparison evidence]")
        for path in retained:
            print(f"- {path.relative_to(ROOT)}")
    else:
        print("\n[retained large-load comparison evidence]")
        print("- ok: no user>=1200 comparison output detected")

    if runtime.issues:
        print("\n[runtime]")
        for issue in runtime.issues:
            print(f"- problem: {issue}")
        print("- note: status does not run experiments; run/compare fail fast until the runtime is fixed.")
        return

    print("\n[runtime]")
    print("- ok: Python runtime is present and compatible")


def comparison_output_dir(args: argparse.Namespace, algorithms: tuple[str, ...]) -> Path:
    if args.family:
        base = Path("results") / args.family / "comparisons" / f"{args.density_mode}_density"
    else:
        base = Path("results") / "comparisons"
    return base / scenario_name(args, algorithms)


def resolve_algorithms(args: argparse.Namespace) -> tuple[str, ...]:
    if args.algorithms:
        return tuple(args.algorithms)
    if args.family == ADAPT2_FAMILY:
        return ADAPT2_ALGORITHMS
    return REVIEW_ALGORITHMS


def cmd_commands(args: argparse.Namespace) -> None:
    guard_users(args)
    algorithms = resolve_algorithms(args)
    warn_python_runtime_if_needed()
    py = python_bin()
    prefix = env_prefix(args)
    print(f"cd {ROOT}")
    for algorithm in algorithms:
        print(f"{prefix} {py} channel_common_eval.py {algorithm}")
    output_dir = comparison_output_dir(args, algorithms)
    print(
        f"{prefix} {py} compare_results.py "
        f"--algorithms {' '.join(algorithms)} --output-dir {output_dir}"
    )


def cmd_run(args: argparse.Namespace) -> None:
    guard_users(args)
    py = str(ensure_python_runtime())
    for algorithm in resolve_algorithms(args):
        command = [py, "channel_common_eval.py", algorithm]
        if args.episodes is not None:
            command += ["--episodes", str(args.episodes)]
        run_command(command, args=args, dry=args.dry_run)


def cmd_compare(args: argparse.Namespace) -> None:
    guard_users(args)
    algorithms = resolve_algorithms(args)
    output_dir = args.output_dir or comparison_output_dir(args, algorithms)
    command = [
        str(ensure_python_runtime()),
        "compare_results.py",
        "--algorithms",
        *algorithms,
        "--output-dir",
        str(output_dir),
    ]
    run_command(command, args=args, dry=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QECO-ADAPT2 experiment workflow runner.")
    parser.add_argument("--users", type=int, default=800, help="Fixed user count, or the agent pool size in random density mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-1200", action="store_true", help="Allow deferred user>=1200 experiment commands.")
    parser.add_argument(
        "--family",
        default=None,
        help=f"Experiment family folder, e.g. '{ADAPT2_FAMILY}'. Routes runs to results/<family>/<density>_density/ "
        "and comparisons to results/<family>/comparisons/<density>_density/. Omit for the legacy layout.",
    )
    parser.add_argument(
        "--density-mode",
        choices=("fixed", "random"),
        default="fixed",
        help="fixed: every episode uses --users active users. random: per-episode active users sampled from --density-choices.",
    )
    parser.add_argument(
        "--density-choices",
        default=None,
        help=f"Comma-separated active-user counts for random density mode (default {DEFAULT_RANDOM_DENSITY_CHOICES}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Check review assets and deferred user=1200 evidence policy.")
    status.set_defaults(func=cmd_status)

    algorithm_help = (
        "Algorithms to use. Default: review set (qeco, gate-only, adaptive-weight/fixed-gate, full); "
        f"with --family {ADAPT2_FAMILY}: {', '.join(ADAPT2_ALGORITHMS)}."
    )

    commands = subparsers.add_parser("commands", help="Print reproducible terminal commands.")
    commands.add_argument("--algorithms", nargs="+", default=None, help=algorithm_help)
    commands.set_defaults(func=cmd_commands)

    run = subparsers.add_parser("run", help="Run selected algorithms in the common channel-aware environment.")
    run.add_argument("--algorithms", nargs="+", default=None, help=algorithm_help)
    run.add_argument("--episodes", type=int)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)

    compare = subparsers.add_parser("compare", help="Compare latest runs for selected algorithms.")
    compare.add_argument("--algorithms", nargs="+", default=None, help=algorithm_help)
    compare.add_argument("--output-dir", type=Path)
    compare.add_argument("--dry-run", action="store_true")
    compare.set_defaults(func=cmd_compare)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
