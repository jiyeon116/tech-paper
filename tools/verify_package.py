"""Read-only checks for the portable handoff. Optional first arg: Documents root."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from urllib.parse import unquote

from PIL import Image

BUNDLE = Path(__file__).resolve().parents[1]
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


manifest = json.loads((BUNDLE / "SOURCE_MANIFEST.json").read_text())
for item in manifest["source_records"]:
    path = BUNDLE / item["package_path"]
    assert not path.is_symlink(), path
    assert path.stat().st_size == item["bytes"], path
    assert digest(path) == item["sha256"], path
    if ROOT:
        assert digest(ROOT / item["source_relative_to_documents"]) == item["sha256"], path

refs = rows(BUNDLE / "02_references/reference_review.csv")
assert [int(r["reference_id"]) for r in refs] == list(range(1, 42))
pdfs = sorted((BUNDLE / "02_references/papers").glob("*.pdf"))
assert len(pdfs) == 33
assert all(r["coauthor_verdict"] == "PENDING" for r in refs)
for row in refs:
    if row["pdf_in_package"]:
        assert (BUNDLE / row["pdf_in_package"]).is_file()

qpdf = shutil.which("qpdf")
assert qpdf, "qpdf is required for PDF structural checks"
pdf_warnings = []
for path in pdfs:
    assert path.read_bytes().startswith(b"%PDF-"), path
    result = subprocess.run([qpdf, "--check", str(path)], capture_output=True, text=True)
    assert result.returncode in (0, 3), (path, result.stdout, result.stderr)
    if result.returncode == 3:
        pdf_warnings.append({"file": str(path.relative_to(BUNDLE)),
                             "output": result.stdout + result.stderr})

png_info = []
for path in sorted((BUNDLE / "01_figures").rglob("*.png")):
    with Image.open(path) as picture:
        png_info.append({"file": str(path.relative_to(BUNDLE)), "size": picture.size})
        picture.verify()

legacy = BUNDLE / "01_figures/data/legacy_random"
series = rows(legacy / "comparison_timeseries.csv")
assert len(series) == 500
assert [int(r["step"]) for r in series] == list(range(1, 501))
assert all(math.isfinite(float(v)) for r in series for v in r.values())
metrics = ("qoe", "delay", "energy", "drop", "completion_rate")
errors = []
for row in rows(legacy / "comparison_overall.csv"):
    for metric in metrics:
        computed = statistics.fmean(float(r[f'{row["algorithm"]}_{metric}']) for r in series)
        error = abs(computed - float(row[f"{metric}_mean"]))
        assert error < 1e-8, (row, metric, error)
        errors.append(error)
section_count = 0
for row in rows(legacy / "comparison_section_means.csv"):
    start, end = map(int, row["section"].split("-"))
    subset = series[start - 1:end]
    assert len(subset) == end - start + 1
    for metric in metrics:
        computed = statistics.fmean(float(r[f'{row["algorithm"]}_{metric}']) for r in subset)
        error = abs(computed - float(row[metric]))
        assert error < 1e-8, (row, metric, error)
        errors.append(error)
        section_count += 1
schedule = rows(legacy / "active_users_timeseries.csv")
assert [r["step"] for r in schedule] == [r["step"] for r in series]
assert {int(r["active_users"]) for r in schedule} == {100, 300, 500, 800, 1200, 1500}
regime_count = 0
for row in rows(legacy / "comparison_by_regime.csv"):
    subset = [r for r, activity in zip(series, schedule)
              if row["regime"] == "all" or activity["regime"] == row["regime"]]
    assert len(subset) == int(row["episodes"])
    for metric in metrics:
        computed = statistics.fmean(float(r[f'{row["algorithm"]}_{metric}']) for r in subset)
        error = abs(computed - float(row[f"{metric}_mean"]))
        assert error < 1e-8, (row, metric, error)
        errors.append(error)
        regime_count += 1

diagnostics = {}
for label, expected in (("dynamic_smoke", 16), ("dynamic_scale", 12)):
    result_dir = BUNDLE / "01_figures/data" / label / "seed_42/attempt_001/result"
    checksums = json.loads((result_dir / "checksums.json").read_text())
    for name, expected_digest in checksums.items():
        assert digest(result_dir / name) == expected_digest
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["completed_rollouts"] == summary["expected_rollouts"] == expected
    assert summary["evaluation_updates"] is False
    episodes = [json.loads(line) for line in (result_dir / "episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == expected
    evaluations = [r for r in episodes if r["phase"] == "evaluation"]
    for episode in {r["episode"] for r in evaluations}:
        group = [r for r in evaluations if r["episode"] == episode]
        assert {r["arm"] for r in group} == {"fixed", "adaptive", "contextual", "switching"}
        assert len({r["trace_hash"] for r in group}) == 1
    steps = [json.loads(line) for line in (result_dir / "steps.jsonl").read_text().splitlines()]
    switches = []
    for row in evaluations:
        selected = [r for r in steps if r["phase"] == "evaluation"
                    and r["arm"] == row["arm"] and r["episode"] == row["episode"]]
        assert [r["step"] for r in selected] == list(range(summary["config"]["total_steps"]))
        count = sum(a["policy"] != b["policy"] for a, b in zip(selected, selected[1:]))
        assert count == row["switch_count"]
        if row["arm"] == "switching":
            switches.append(count)
    diagnostics[label] = {"rollouts": expected, "checksums": len(checksums),
                          "evaluation_trace_pairs": len({r["episode"] for r in evaluations}),
                          "evaluation_switch_counts": switches}

# Historical/source Markdown is preserved byte-for-byte; its original workspace
# links are not promised to resolve inside this deliberately bounded snapshot.
historical = {"00_context/adapt2_extended_snapshot.md", "00_context/ultrasparse_partial_report.md",
              "00_context/dynamic_protocol.md"}
link_count = 0
for path in sorted(BUNDLE.rglob("*.md")):
    relative = path.relative_to(BUNDLE).as_posix()
    if relative.startswith("03_code_comments/source/") or relative in historical:
        continue
    for target in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", path.read_text()):
        target = target.strip("<>")
        if re.match(r"(?:https?://|mailto:|#)", target):
            continue
        target = unquote(target.split("#", 1)[0])
        assert not Path(target).is_absolute(), (relative, target, "non-portable link")
        resolved = (path.parent / target).resolve()
        assert resolved.is_relative_to(BUNDLE), (relative, target, "escapes package")
        assert resolved.exists(), (relative, target, "missing link")
        link_count += 1

python_files = list((BUNDLE / "03_code_comments/source").rglob("*.py")) + list((BUNDLE / "tools").glob("*.py"))
for path in python_files:
    ast.parse(path.read_text(), filename=str(path))
assert not list(BUNDLE.rglob("*.hwpx"))
assert not any(p.is_symlink() for p in BUNDLE.rglob("*"))

print(json.dumps({
    "source_copies": len(manifest["source_records"]), "source_rechecked": bool(ROOT),
    "references": len(refs), "pdfs": len(pdfs), "pdf_warnings": pdf_warnings,
    "missing_pdf_ids": [int(r["reference_id"]) for r in refs if not r["pdf_in_package"]],
    "pngs": png_info, "legacy_rows": len(series), "overall_means": 20,
    "section_means": section_count, "regime_means": regime_count, "max_mean_error": max(errors),
    "diagnostics": diagnostics, "authored_markdown_links": link_count,
    "syntax_parsed_python_files": len(python_files),
}, ensure_ascii=False, indent=2))
