#!/usr/bin/env python3
"""Run dynamic-specialist experiment seeds as recoverable, bounded cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_experiment.provenance import SourceIdentity, source_identity

MANIFEST_NAME = "campaign_manifest.json"
STATUS_NAME = "status.json"
CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "QECO_AGENT_SESSION_MODE": "per-agent",
    "TF_NUM_INTRAOP_THREADS": "2",
    "TF_NUM_INTEROP_THREADS": "1",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_config_hash(path: Path) -> str:
    from dynamic_experiment.config import DynamicConfig

    return DynamicConfig.load(path).digest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def next_attempt(seed_dir: Path) -> Path:
    indices = []
    if seed_dir.exists():
        for path in seed_dir.iterdir():
            if path.is_dir() and path.name.startswith("attempt_"):
                try:
                    indices.append(int(path.name.removeprefix("attempt_")))
                except ValueError:
                    continue
    return seed_dir / f"attempt_{max(indices, default=0) + 1:03d}"


def checksum_valid(attempt: Path) -> bool:
    result = attempt / "result"
    checksum = result / "checksums.json"
    try:
        checksums = json.loads(checksum.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(checksums, dict) or "summary.json" not in checksums:
        return False
    artifacts = {
        path.name
        for path in result.iterdir()
        if path.is_file() and path.name != "checksums.json"
    }
    if set(checksums) != artifacts:
        return False
    for filename, expected in checksums.items():
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            return False
        artifact = result / filename
        if not artifact.is_file() or sha256_file(artifact) != expected:
            return False
    return True


def completed_attempt_valid(
    attempt: Path,
    *,
    seed: int,
    config_hash: str,
    revision: str,
    source_identity_hash: str,
) -> bool:
    status = read_json(attempt / STATUS_NAME)
    summary = read_json(attempt / "result" / "summary.json")
    return bool(
        status
        and status.get("state") == "complete"
        and summary
        and summary.get("status") == "complete"
        and summary.get("seed") == seed
        and summary.get("config_hash") == config_hash
        and summary.get("revision") == revision
        and summary.get("source_identity_hash") == source_identity_hash
        and isinstance(summary.get("metrics"), dict)
        and checksum_valid(attempt)
    )


def reusable_attempt(
    seed_dir: Path,
    *,
    seed: int,
    config_hash: str,
    revision: str,
    source_identity_hash: str,
) -> Path | None:
    if not seed_dir.is_dir():
        return None
    for attempt in sorted(seed_dir.glob("attempt_*"), reverse=True):
        if completed_attempt_valid(
            attempt,
            seed=seed,
            config_hash=config_hash,
            revision=revision,
            source_identity_hash=source_identity_hash,
        ):
            return attempt
    return None


def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


@dataclass(frozen=True)
class CellResult:
    state: str
    exit_code: int | None
    attempt: Path
    message: str


class StopRequested(RuntimeError):
    pass


def run_cell(
    *,
    python: Path,
    config: Path,
    seed: int,
    attempt: Path,
    config_hash: str,
    revision: str,
    source_identity_hash: str,
    timeout_seconds: float,
    min_free_gib: float,
    poll_seconds: float = 1.0,
) -> CellResult:
    attempt.mkdir(parents=True, exist_ok=False)
    log_path = attempt / "run.log"
    started = time.monotonic()
    command = [
        str(python),
        "-m",
        "dynamic_experiment.run",
        "--config",
        str(config),
        "--seed",
        str(seed),
        "--output-dir",
        str(attempt / "result"),
    ]
    status: dict[str, Any] = {
        "state": "running",
        "seed": seed,
        "config_hash": config_hash,
        "revision": revision,
        "source_identity_hash": source_identity_hash,
        "attempt": attempt.name,
        "started_at": utc_now(),
        "command": command,
        "cpu_environment": CPU_ENV,
    }
    atomic_json(attempt / STATUS_NAME, status)
    environment = os.environ.copy()
    environment.update(CPU_ENV)
    environment["QECO_CAMPAIGN_CONFIG_HASH"] = config_hash
    environment["QECO_CAMPAIGN_REVISION"] = revision
    environment["QECO_SOURCE_IDENTITY_HASH"] = source_identity_hash

    process: subprocess.Popen[Any] | None = None
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        if process is not None:
            terminate_process_group(process)
        raise StopRequested(f"campaign received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, handle_signal)

    state = "failed"
    message = "child process failed"
    exit_code: int | None = None
    try:
        with log_path.open("x", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    state = "timeout"
                    message = f"cell exceeded {timeout_seconds:.1f} seconds"
                    terminate_process_group(process)
                    break
                available = free_gib(attempt)
                if available < min_free_gib:
                    state = "disk_floor"
                    message = (
                        f"free disk {available:.2f} GiB fell below floor "
                        f"{min_free_gib:.2f} GiB"
                    )
                    terminate_process_group(process)
                    break
                time.sleep(min(poll_seconds, max(timeout_seconds - elapsed, 0.01)))
            exit_code = process.wait()
            if state not in {"timeout", "disk_floor"}:
                if exit_code == 0:
                    summary = read_json(attempt / "result" / "summary.json")
                    if (
                        summary
                        and summary.get("status") == "complete"
                        and summary.get("seed") == seed
                        and summary.get("config_hash") == config_hash
                        and summary.get("revision") == revision
                        and summary.get("source_identity_hash") == source_identity_hash
                        and isinstance(summary.get("metrics"), dict)
                        and checksum_valid(attempt)
                    ):
                        state = "complete"
                        message = "validated completed summary"
                    else:
                        state = "invalid_output"
                        message = "exit zero but summary identity or completion status is invalid"
                else:
                    state = "failed"
                    message = f"child exited with code {exit_code}"
    except StopRequested:
        state = "interrupted"
        message = "campaign interrupted; this attempt remains preserved"
        if process is not None:
            exit_code = process.poll()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        status.update(
            {
                "state": state,
                "exit_code": exit_code,
                "message": message,
                "finished_at": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        atomic_json(attempt / STATUS_NAME, status)
        if log_path.exists():
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[EXIT STATUS] {exit_code}\n")

    return CellResult(state=state, exit_code=exit_code, attempt=attempt, message=message)


def run_campaign(args: argparse.Namespace, *, poll_seconds: float = 1.0) -> int:
    root = ROOT
    config = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    python = Path(os.path.abspath(os.fspath(args.python.expanduser())))
    if not config.is_file():
        raise SystemExit(f"Config does not exist: {config}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise SystemExit(f"Python is not executable: {python}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if free_gib(output_dir) < args.min_free_gib:
        raise SystemExit(
            f"Free disk is below --min-free-gib {args.min_free_gib:.2f} at {output_dir}"
        )

    config_file_hash = sha256_file(config)
    config_hash = resolved_config_hash(config)
    source = source_identity(root)
    revision = source.revision
    deadline = time.monotonic() + args.budget_seconds
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (
            manifest is None
            or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("seeds"), dict)
        ):
            raise SystemExit(
                f"Existing campaign manifest is invalid; preserving it unchanged: {manifest_path}"
            )
    else:
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "seeds": {},
        }
    manifest.update(
        {
            "status": "running",
            "updated_at": utc_now(),
            "config": str(config),
            "config_hash": config_hash,
            "config_file_hash": config_file_hash,
            "revision": revision,
            "source": source.payload(),
            "source_identity_hash": source.identity_hash,
            "python": str(python),
            "budget_seconds": args.budget_seconds,
            "cell_timeout_seconds": args.cell_timeout_seconds,
            "min_free_gib": args.min_free_gib,
            "cpu_environment": CPU_ENV,
            "requested_seeds": args.seeds,
        }
    )
    atomic_json(manifest_path, manifest)

    campaign_ok = True
    try:
        for seed in args.seeds:
            seed_dir = output_dir / f"seed_{seed}"
            reusable = reusable_attempt(
                seed_dir,
                seed=seed,
                config_hash=config_hash,
                revision=revision,
                source_identity_hash=source.identity_hash,
            )
            if reusable is not None:
                manifest["seeds"][str(seed)] = {
                    "state": "complete",
                    "attempt": reusable.name,
                    "reused": True,
                }
                atomic_json(manifest_path, manifest)
                print(f"[campaign] seed={seed} reused {reusable}", flush=True)
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                campaign_ok = False
                manifest["seeds"][str(seed)] = {
                    "state": "budget_exhausted",
                    "message": "campaign deadline reached before cell start",
                }
                break
            if free_gib(output_dir) < args.min_free_gib:
                campaign_ok = False
                manifest["seeds"][str(seed)] = {
                    "state": "disk_floor",
                    "message": "disk floor reached before cell start",
                }
                break

            attempt = next_attempt(seed_dir)
            timeout = min(args.cell_timeout_seconds, remaining)
            print(f"[campaign] seed={seed} starting {attempt.name}", flush=True)
            result = run_cell(
                python=python,
                config=config,
                seed=seed,
                attempt=attempt,
                config_hash=config_hash,
                revision=revision,
                source_identity_hash=source.identity_hash,
                timeout_seconds=timeout,
                min_free_gib=args.min_free_gib,
                poll_seconds=poll_seconds,
            )
            manifest["seeds"][str(seed)] = {
                "state": result.state,
                "attempt": result.attempt.name,
                "exit_code": result.exit_code,
                "message": result.message,
                "reused": False,
            }
            atomic_json(manifest_path, manifest)
            print(
                f"[campaign] seed={seed} state={result.state} attempt={result.attempt.name}",
                flush=True,
            )
            if result.state != "complete":
                campaign_ok = False
    except StopRequested as exc:
        campaign_ok = False
        manifest["message"] = str(exc)
    finally:
        states = [manifest["seeds"].get(str(seed), {}).get("state") for seed in args.seeds]
        manifest["status"] = "complete" if all(state == "complete" for state in states) else "incomplete"
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
    return 0 if campaign_ok and manifest["status"] == "complete" else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--budget-seconds", type=float, default=21600)
    parser.add_argument("--cell-timeout-seconds", type=float, default=7200)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.budget_seconds)
        or not math.isfinite(args.cell_timeout_seconds)
        or args.budget_seconds <= 0
        or args.cell_timeout_seconds <= 0
    ):
        parser.error("time budgets must be positive")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        parser.error("--min-free-gib must be finite and non-negative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run_campaign(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
