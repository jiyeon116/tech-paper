"""Deterministic source identity for clean Git trees and deployed archives."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Iterable, Sequence


MANIFEST_NAME = "SOURCE_MANIFEST.json"
REVISION_NAME = "REVISION"
SCHEMA_VERSION = 1
EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".omx",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "artifact_manifests",
        "env_channel",
        "logs",
        "results",
    }
)
EXCLUDED_FILES = frozenset({".DS_Store", MANIFEST_NAME, REVISION_NAME})
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class ProvenanceError(RuntimeError):
    """The source tree cannot be assigned a trustworthy immutable identity."""


@dataclass(frozen=True)
class SourceIdentity:
    mode: str
    revision: str
    identity_hash: str
    file_count: int

    def payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "revision": self.revision,
            "identity_hash": self.identity_hash,
            "file_count": self.file_count,
        }


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_revision(revision: str) -> str:
    revision = revision.strip().lower()
    if not REVISION_PATTERN.fullmatch(revision):
        raise ProvenanceError("revision must be a full 40- or 64-character hexadecimal commit id")
    return revision


def _safe_relative(path: str) -> bool:
    pure = PurePosixPath(path)
    return bool(path) and not pure.is_absolute() and ".." not in pure.parts and str(pure) == path


def _excluded(relative: str) -> bool:
    pure = PurePosixPath(relative)
    return (
        pure.name in EXCLUDED_FILES
        or pure.suffix in {".pyc", ".pyo"}
        or any(part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info") for part in pure.parts)
    )


def _content(path: Path) -> bytes:
    if path.is_symlink():
        return os.readlink(path).encode("utf-8")
    return path.read_bytes()


def _digest(path: Path) -> str:
    return hashlib.sha256(_content(path)).hexdigest()


def _identity_digest(revision: str, files: dict[str, str]) -> str:
    canonical = json.dumps(
        {"schema_version": SCHEMA_VERSION, "revision": revision, "files": files},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _run_git(root: Path, arguments: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=check,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProvenanceError(f"Git provenance command failed: {' '.join(arguments)}") from exc


def _git_root(root: Path) -> Path | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return Path(completed.stdout.strip()).resolve()


def _tracked_paths(root: Path) -> list[str]:
    output = _run_git(root, ["ls-files", "-z"]).stdout
    paths = [item.decode("utf-8") for item in output.split(b"\0") if item]
    return sorted(path for path in paths if not _excluded(path))


def _git_identity(root: Path) -> SourceIdentity:
    repository = _git_root(root)
    if repository != root:
        raise ProvenanceError(f"source root is not the Git repository root: {root}")
    for arguments, label in (
        (["diff", "--quiet", "--ignore-submodules", "--"], "unstaged tracked changes"),
        (["diff", "--cached", "--quiet", "--ignore-submodules", "--"], "staged changes"),
    ):
        completed = _run_git(root, arguments, check=False)
        if completed.returncode == 1:
            raise ProvenanceError(f"Git source tree has {label}")
        if completed.returncode != 0:
            raise ProvenanceError(f"could not check Git source tree for {label}")
    untracked_output = _run_git(root, ["ls-files", "--others", "--exclude-standard", "-z"]).stdout
    untracked = sorted(
        item.decode("utf-8")
        for item in untracked_output.split(b"\0")
        if item and not _excluded(item.decode("utf-8"))
    )
    if untracked:
        preview = ", ".join(untracked[:5])
        raise ProvenanceError(f"Git source tree has nonignored untracked source files: {preview}")
    revision = _validate_revision(_run_git(root, ["rev-parse", "HEAD"]).stdout.decode("ascii"))
    files: dict[str, str] = {}
    for relative in _tracked_paths(root):
        path = root / relative
        if not path.is_file() and not path.is_symlink():
            raise ProvenanceError(f"tracked source path is missing or unsupported: {relative}")
        files[relative] = _digest(path)
    return SourceIdentity("git", revision, _identity_digest(revision, files), len(files))


def _archive_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for directory, names, filenames in os.walk(root):
        base = Path(directory)
        retained_directories = []
        for name in sorted(names):
            candidate = base / name
            relative = candidate.relative_to(root).as_posix()
            if _excluded(relative):
                continue
            if candidate.is_symlink():
                paths.append(relative)
            else:
                retained_directories.append(name)
        names[:] = retained_directories
        for filename in sorted(filenames):
            relative = (base / filename).relative_to(root).as_posix()
            if not _excluded(relative):
                paths.append(relative)
    return sorted(paths)


def _source_files(root: Path, paths: Iterable[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for relative in paths:
        if not _safe_relative(relative) or _excluded(relative):
            raise ProvenanceError(f"unsafe or excluded source manifest path: {relative!r}")
        path = root / relative
        if not path.is_file() and not path.is_symlink():
            raise ProvenanceError(f"source manifest path is missing: {relative}")
        files[relative] = _digest(path)
    return files


def create_archive_manifest(root: Path, revision: str) -> SourceIdentity:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ProvenanceError(f"archive root does not exist: {root}")
    revision = _validate_revision(revision)
    revision_path = root / REVISION_NAME
    if revision_path.exists():
        existing = _validate_revision(revision_path.read_text(encoding="ascii"))
        if existing != revision:
            raise ProvenanceError("existing REVISION differs from requested archive revision")
    else:
        revision_path.write_text(revision + "\n", encoding="ascii")
    paths = _archive_paths(root)
    files = _source_files(root, paths)
    identity_hash = _identity_digest(revision, files)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "revision": revision,
        "identity_hash": identity_hash,
        "files": files,
    }
    _atomic_json(root / MANIFEST_NAME, manifest)
    return SourceIdentity("archive", revision, identity_hash, len(files))


def _archive_identity(root: Path) -> SourceIdentity:
    revision_path = root / REVISION_NAME
    manifest_path = root / MANIFEST_NAME
    if not revision_path.is_file() or not manifest_path.is_file():
        raise ProvenanceError("archive source requires REVISION and SOURCE_MANIFEST.json")
    revision = _validate_revision(revision_path.read_text(encoding="ascii"))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError("archive source manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ProvenanceError("archive source manifest schema is invalid")
    if manifest.get("revision") != revision or not isinstance(manifest.get("files"), dict):
        raise ProvenanceError("archive revision and source manifest disagree")
    expected = manifest["files"]
    if not all(
        isinstance(path, str)
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest)
        for path, digest in expected.items()
    ):
        raise ProvenanceError("archive source manifest entries are invalid")
    actual_paths = _archive_paths(root)
    if set(actual_paths) != set(expected):
        missing = sorted(set(expected) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected))
        raise ProvenanceError(f"archive source file set differs: missing={missing[:5]}, extra={extra[:5]}")
    actual = _source_files(root, sorted(expected))
    if actual != expected:
        changed = sorted(path for path in expected if actual.get(path) != expected[path])
        raise ProvenanceError(f"archive source hashes differ: {changed[:5]}")
    identity_hash = _identity_digest(revision, actual)
    if manifest.get("identity_hash") != identity_hash:
        raise ProvenanceError("archive source identity hash differs")
    return SourceIdentity("archive", revision, identity_hash, len(actual))


def source_identity(root: Path) -> SourceIdentity:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ProvenanceError(f"source root does not exist: {root}")
    git_root = _git_root(root)
    if (root / ".git").exists() and git_root is None:
        raise ProvenanceError("Git metadata exists but the repository identity is unavailable")
    if git_root is not None:
        return _git_identity(root)
    return _archive_identity(root)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="Create metadata in an extracted source archive")
    create.add_argument("--root", required=True, type=Path)
    create.add_argument("--revision", required=True)
    verify = subparsers.add_parser("verify", help="Verify a clean Git tree or source archive")
    verify.add_argument("--root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    identity = (
        create_archive_manifest(args.root, args.revision)
        if args.command == "create"
        else source_identity(args.root)
    )
    print(json.dumps(identity.payload(), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
