"""Build this review snapshot from allowlisted local sources; never edit sources.

Usage: python3 tools/assemble.py /path/to/Documents
The resulting manifest records exact source and destination identities.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

BUNDLE = Path(__file__).resolve().parents[1]
ROOT = Path(sys.argv[1]).resolve()
PAPER = ROOT / "QECO-Adapt"
CODE = ROOT / "qeco-adapt2"
REVISION = "67c1a8c40c6b94de4add2f6d0a6d2dadbdddd3e7"
records = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, relative, purpose):
    target = BUNDLE / relative
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Missing or symlink source: {source}")
    before = digest(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and digest(target) != before:
        raise ValueError(f"Refusing to overwrite changed review file: {target}")
    if not target.exists():
        shutil.copyfile(source, target)
    assert digest(target) == digest(source) == before
    records.append({"source_relative_to_documents": str(source.relative_to(ROOT)),
                    "package_path": str(relative), "sha256": before,
                    "bytes": target.stat().st_size, "purpose": purpose})


def csv_output(relative, rows, fields):
    target = BUNDLE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


head = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
dirty = subprocess.check_output(["git", "-C", str(CODE), "status", "--porcelain"], text=True)
if head != REVISION or dirty:
    raise ValueError("Code source must be the exact clean experimental snapshot")
code_files = subprocess.check_output(["git", "-C", str(CODE), "ls-files"], text=True).splitlines()
for name in code_files:
    if not name.startswith("."):
        copy(CODE / name, Path("03_code_comments/source") / name, "exact committed code snapshot")

for src, dest in [
    (PAPER / "docs/adapt2/paper_draft_qeco_adapt2_kr_extended.md", "adapt2_extended_snapshot.md"),
    (PAPER / "docs/adapt2/qeco_adapt2_ultrasparse_partial_experiment_analysis.md", "ultrasparse_partial_report.md"),
    (CODE / "docs/dynamic_specialist_pilot.md", "dynamic_protocol.md"),
]:
    copy(src, Path("00_context") / dest, "read-only document snapshot; version boundaries in PROVENANCE_AND_LIMITS.md")

for name in ("topology_structure.png", "user_perspective_data_flow.png", "server_perspective_data_flow.png"):
    copy(PAPER / "docs" / name, Path("01_figures/assets") / name, "existing diagram reference, not a new specialist diagram")
copy(PAPER / "scripts/render_paper_figures.py", Path("01_figures/editable/render_paper_figures_legacy.py"), "legacy editable source; inspect portability caveat before running")

comparison = PAPER / "results/adapt2/comparisons/random_density/adapt2_user_100-1500_random_seed_42_ep_500_edge_3_ap_6_channel_4"
selected = ["comparison_timeseries.csv", "comparison_overall.csv", "comparison_by_regime.csv",
            "comparison_section_means.csv", "active_users_timeseries.csv", "comparison_summary.json",
            "ActiveUsers_Timeseries.png", "selected4_QoE_Timeseries.png", "selected4_CompletionRate_Timeseries.png"]
for name in selected:
    copy(comparison / name, Path("01_figures/data/legacy_random") / name, "historical episode-varying, single-seed, shared learner comparison")

for prefix, dest in [("smoke-67c1a8c", "dynamic_smoke"), ("scale-67c1a8c", "dynamic_scale")]:
    result = CODE / "results/dynamic_specialist_pilot" / prefix / "seed_42/attempt_001/result"
    for path in sorted(result.glob("*.json*")):
        copy(path, Path("01_figures/data") / dest / "seed_42/attempt_001/result" / path.name,
             "completed diagnostic/resource probe; not full-pilot performance evidence")

draft = (PAPER / "docs/adapt1/paper_draft_channel_aware_qeco_adapt_kr_extended.md").read_text()
refs = re.findall(r"^\[(\d+)\] (.+)$", draft, flags=re.M)
assert [int(n) for n, _ in refs] == list(range(1, 42))
registry = (PAPER / "journals/notion_reference_review_registry.md").read_text()
sections = {int(n): body for n, body in re.findall(r"^## \[(\d+)\] ([\s\S]*?)(?=^## \[|\Z)", registry, re.M)}
overrides = {
    8: "research_files/drl-task-offloading-mobile-edge-computing-systems/drl-task-offloading-mobile-edge-computing-systems_원문.pdf",
    9: "research_files/droo-online-computation-offloading-wireless-powered-mec/droo-online-computation-offloading-wireless-powered-mec_원문.pdf",
    10: "research_files/lydroo-stable-online-computation-offloading-mec/lydroo-stable-online-computation-offloading-mec_원문.pdf",
}
priority = {7, 8, 9, 10, 16, 17, 18, 23, 24, 25, 29, 30, 31, 40, 41}
web_checked = {
    7: ("https://arxiv.org/abs/2311.02525", "title/authors/journal/DOI and public QECO repository checked; Rahmaty/Rahmati edition spelling requires final comparison"),
    8: ("https://arxiv.org/abs/2005.02459", "preprint title/authors checked; journal volume/pages not rechecked"),
    17: ("https://proceedings.mlr.press/v48/wangf16.html", "official proceedings title/six authors/pages/year checked"),
    29: ("https://ojs.aaai.org/index.php/AAAI/article/view/11694", "official proceedings page checked"),
    30: ("https://arxiv.org/abs/1806.08295", "official preprint title/authors checked; do not treat as journal DOI"),
    31: ("https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html", "official proceedings page and author rliable repository checked"),
}
repo_checks = {
    7: ("https://github.com/ImanRHT/QECO", "author public repository verified 2026-09-06"),
    9: ("https://github.com/revenol/DROO", "public paper-associated repository verified 2026-09-06"),
    10: ("https://github.com/revenol/LyDROO", "public paper-associated repository verified 2026-09-06"),
    31: ("https://github.com/google-research/rliable", "author public evaluation repository verified 2026-09-06"),
}
rows = []
for n, citation in refs:
    number = int(n)
    section = sections[number]
    title = section.splitlines()[0]
    match = re.search(r"\*\*\[PDF 파일\]\*\*: `([^`]+)`", section)
    source = overrides.get(number, match.group(1) if match else "")
    path = ROOT / source if source else None
    dest = ""
    if path and path.is_file():
        dest = f"02_references/papers/R{number:02d}.pdf"
        copy(path, Path(dest), f"reference [{number}]: {title}; scholarly review only")
    doi = re.search(r"doi: ([^\s]+)", citation)
    doi = doi.group(1).rstrip(".") if doi else ""
    code_line = re.search(r"\*\*\[소스코드\]\*\*: (.+)", section)
    old_code = code_line.group(1) if code_line else "not investigated"
    checked_url, scope = web_checked.get(number, ("", "inherited bibliography; not externally rechecked this session"))
    repo_url, repo_scope = repo_checks.get(number, ("", "historical registry only; current availability not rechecked"))
    rows.append({"reference_id": number, "priority": "P1" if number in priority else "P2",
                 "title": title, "citation_as_in_current_draft": citation, "doi": doi,
                 "official_url_checked": checked_url, "metadata_check_scope": scope,
                 "pdf_in_package": dest, "canonical_or_original_source": source,
                 "pdf_status": "included; edition/content validation assigned to coauthor" if dest else "not found in mapped local sources; obtain from official publisher/library",
                 "code_url_checked": repo_url, "code_check_scope": repo_scope,
                 "historical_code_record": old_code, "coauthor_verdict": "PENDING",
                 "page_section_support": "", "proposed_correction": ""})
csv_output("02_references/reference_review.csv", rows, list(rows[0]))
(BUNDLE / "02_references/BIBLIOGRAPHY_SNAPSHOT.md").write_text(
    "# References — Existing Extended Draft Snapshot\n\n"
    "현재 adapt2 초안이 승계하는 adapt1 확장본의 [1]–[41]을 그대로 옮긴 검토용 사본이다. "
    "서지 수정 제안은 REVIEW_GUIDE.md와 reference_review.csv에 남기며 원고 번호는 변경하지 않았다.\n\n" +
    "\n\n".join(f"[{n}] {citation}" for n, citation in refs) + "\n", encoding="utf-8")
(BUNDLE / "SOURCE_MANIFEST.json").write_text(json.dumps({
    "snapshot_date": "2026-09-06", "code_revision": REVISION,
    "documents_root": "paths are relative to the local Documents folder",
    "source_records": records,
    "reference_count": len(refs), "included_reference_pdfs": sum(bool(r["pdf_in_package"]) for r in rows),
    "new_pdf_downloads": 0,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"copied_files": len(records), "references": len(refs),
                  "pdfs": sum(bool(r["pdf_in_package"]) for r in rows)}, ensure_ascii=False))
