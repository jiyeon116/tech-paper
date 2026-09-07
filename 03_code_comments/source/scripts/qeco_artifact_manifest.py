#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifact_manifests" / "qeco_artifacts.sha256"
MANIFEST_HEADER = "# QECO-Adapt artifact manifest v1"
NOISE_NAMES = {".DS_Store"}
NOISE_SUFFIXES = {".pyc", ".pyo"}
NOISE_PARTS = {"__pycache__"}


@dataclass(frozen=True)
class ArtifactEntry:
    sha256: str
    size: int
    path: str


def is_noise(path: Path) -> bool:
    if path.name in NOISE_NAMES:
        return True
    if path.suffix in NOISE_SUFFIXES:
        return True
    return any(part in NOISE_PARTS for part in path.parts)


def iter_artifact_files(root: Path) -> list[Path]:
    """Return result and log files covered by this manifest."""
    candidates: list[Path] = []
    for directory_name in ("results", "logs"):
        directory = root / directory_name
        if directory.exists():
            candidates.extend(path for path in directory.rglob("*") if path.is_file())
    filtered = [path for path in candidates if not is_noise(path.relative_to(root))]
    return sorted(filtered, key=lambda path: path.relative_to(root).as_posix())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_entries(root: Path) -> list[ArtifactEntry]:
    entries = []
    for path in iter_artifact_files(root):
        relative = path.relative_to(root).as_posix()
        entries.append(ArtifactEntry(sha256=sha256_file(path), size=path.stat().st_size, path=relative))
    return entries


def write_manifest(root: Path, manifest: Path) -> int:
    entries = build_entries(root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{MANIFEST_HEADER}\n")
        f.write("# Scope: results/**, logs/**\n")
        f.write("# Format: sha256<TAB>bytes<TAB>repo-relative-path\n")
        f.write("# This manifest detects artifact drift; it is not a backup copy.\n")
        for entry in entries:
            f.write(f"{entry.sha256}\t{entry.size}\t{entry.path}\n")
    print(f"Wrote {len(entries)} artifact entries to {manifest}")
    return 0


def read_manifest(manifest: Path) -> list[ArtifactEntry]:
    entries: list[ArtifactEntry] = []
    seen_paths: set[str] = set()
    with manifest.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid manifest line {line_number}: expected three tab-separated fields")
            digest, size_text, relative_path = parts
            parsed_path = PurePosixPath(relative_path)
            if (
                not relative_path
                or parsed_path.is_absolute()
                or parsed_path.as_posix() != relative_path
                or ".." in parsed_path.parts
            ):
                raise ValueError(f"Invalid manifest line {line_number}: unsafe repo-relative path: {relative_path}")
            if relative_path in seen_paths:
                raise ValueError(f"Invalid manifest line {line_number}: duplicate path: {relative_path}")
            seen_paths.add(relative_path)
            entries.append(ArtifactEntry(sha256=digest, size=int(size_text), path=relative_path))
    return entries


def verify_manifest(root: Path, manifest: Path) -> int:
    expected = {entry.path: entry for entry in read_manifest(manifest)}
    current = {entry.path: entry for entry in build_entries(root)}

    missing = sorted(set(expected) - set(current))
    extra = sorted(set(current) - set(expected))
    changed = sorted(
        path
        for path in set(expected) & set(current)
        if expected[path].sha256 != current[path].sha256 or expected[path].size != current[path].size
    )

    if not (missing or extra or changed):
        print(f"PASS: {len(expected)} artifact entries match {manifest}")
        return 0

    if missing:
        print("[missing]")
        for path in missing:
            print(f"- {path}")
    if changed:
        print("[changed]")
        for path in changed:
            print(f"- {path}")
    if extra:
        print("[extra]")
        for path in extra:
            print(f"- {path}")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write or verify QECO-ADAPT2 result and log checksums.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project root to scan. Defaults to the repository containing this script.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest file to write or verify. Defaults to <root>/artifact_manifests/qeco_artifacts.sha256.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("write", help="Write the artifact manifest.")
    subparsers.add_parser("verify", help="Verify files against the artifact manifest.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve() if args.manifest else root / "artifact_manifests" / DEFAULT_MANIFEST.name
    if args.command == "write":
        return write_manifest(root, manifest)
    if args.command == "verify":
        return verify_manifest(root, manifest)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
