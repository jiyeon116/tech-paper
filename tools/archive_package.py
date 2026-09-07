"""Create a new handoff ZIP and verify every member; never replace an existing ZIP."""
import hashlib
import json
from pathlib import Path
import zipfile

BUNDLE = Path(__file__).resolve().parents[1]
ARCHIVE = BUNDLE.with_suffix(".zip")


def digest(data):
    return hashlib.sha256(data).hexdigest()


if ARCHIVE.exists() or ARCHIVE.with_suffix(".zip.sha256").exists():
    raise SystemExit("Refusing to replace an existing delivery archive or checksum")

paths = sorted(p for p in BUNDLE.rglob("*") if p.is_file()
               and "__pycache__" not in p.parts and p.name != ".DS_Store"
               and p.name != "PACKAGE_MANIFEST.json")
records = []
for path in paths:
    relative = path.relative_to(BUNDLE)
    if path.is_symlink() or any(part.startswith(".") for part in relative.parts):
        raise ValueError(f"Unexpected hidden file or symlink: {relative}")
    if path.suffix.lower() in {".pyc", ".pyo", ".pem", ".key", ".hwpx", ".hwp"}:
        raise ValueError(f"Unexpected artifact: {relative}")
    records.append({"path": relative.as_posix(), "bytes": path.stat().st_size,
                    "sha256": digest(path.read_bytes())})
manifest = BUNDLE / "PACKAGE_MANIFEST.json"
manifest.write_text(json.dumps({"date": "2026-09-06", "self_excluded": True,
                               "files": records}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
paths.append(manifest)
with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in paths:
        archive.write(path, arcname=f"{BUNDLE.name}/{path.relative_to(BUNDLE).as_posix()}")
with zipfile.ZipFile(ARCHIVE) as archive:
    assert archive.testzip() is None
    expected = {f"{BUNDLE.name}/{p.relative_to(BUNDLE).as_posix()}": p for p in paths}
    assert set(archive.namelist()) == set(expected)
    assert len(archive.namelist()) == len(expected)
    for name, path in expected.items():
        assert digest(archive.read(name)) == digest(path.read_bytes()), name
    for record in records:
        assert digest(archive.read(f'{BUNDLE.name}/{record["path"]}')) == record["sha256"]
sha = digest(ARCHIVE.read_bytes())
ARCHIVE.with_suffix(".zip.sha256").write_text(f"{sha}  {ARCHIVE.name}\n", encoding="ascii")
print(json.dumps({"archive": str(ARCHIVE), "members": len(paths),
                  "bytes": ARCHIVE.stat().st_size, "sha256": sha,
                  "crc_and_all_members_verified": True}, indent=2))
