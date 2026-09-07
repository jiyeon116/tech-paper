#!/usr/bin/env python3
"""Create coauthor-facing reference and diagnostic-data review exports.

This script reads only files already present in the handoff package.  It does
not alter source experiment outputs or reference records.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CSV = PACKAGE_ROOT / "02_references" / "reference_review.csv"
REFERENCE_MD = PACKAGE_ROOT / "02_references" / "REFERENCE_REVIEW.md"
DATA_DIR = PACKAGE_ROOT / "01_figures" / "data"
RESULT_DIR = DATA_DIR / "dynamic_scale" / "seed_42" / "attempt_001" / "result"
TIMELINE_CSV = DATA_DIR / "dynamic_scale_timeline.csv"
EVALUATION_CSV = DATA_DIR / "dynamic_scale_evaluation.csv"
DERIVED_MD = DATA_DIR / "DERIVED_DATA.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def markdown_value(value: str, fallback: str = "Not recorded") -> str:
    return value.strip() if value.strip() else fallback


def markdown_code(value: str, fallback: str = "Not recorded") -> str:
    value = value.strip()
    return f"`{value}`" if value else fallback


def write_reference_review() -> None:
    with REFERENCE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected_ids = [str(index) for index in range(1, 42)]
    actual_ids = [row["reference_id"] for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(f"Expected ordered reference IDs 1-41, found {actual_ids}")

    lines = [
        "# Notion용 참고문헌 검토 목록",
        "",
        "이 문서는 `reference_review.csv`의 41개 레코드를 검토하기 쉽게 펼친 자료이다. "
        "인용 문자열과 출처 기록은 원본 CSV에서 그대로 가져왔으며, `PENDING`은 공저자의 확인이 아직 필요하다는 뜻이다.",
        "",
        "동봉 PDF의 상대 경로는 이 패키지 안에서만 유효하다. `historical_code_record`는 과거 조사 기록이며, "
        "이번 패키지 작성 시점에 다시 확인한 사실로 해석하지 않는다.",
        "",
    ]

    for row in rows:
        ref_id = row["reference_id"]
        pdf_path = row["pdf_in_package"].strip()
        pdf_display = f"[{Path(pdf_path).name}]({Path(pdf_path).relative_to('02_references').as_posix()})" if pdf_path else "Not included"
        lines.extend(
            [
                f"## [{ref_id}] {row['title']}",
                "",
                f"- **Priority:** {markdown_value(row['priority'])}",
                f"- **Citation in current draft:** {row['citation_as_in_current_draft']}",
                f"- **DOI:** {markdown_value(row['doi'], 'None recorded')}",
                f"- **Official record checked:** {markdown_value(row['official_url_checked'], 'Not checked this session')}",
                f"- **Metadata check scope:** {markdown_value(row['metadata_check_scope'])}",
                f"- **Packaged PDF:** {pdf_display}",
                f"- **Canonical/original source record:** {markdown_code(row['canonical_or_original_source'])}",
                f"- **PDF status:** {markdown_value(row['pdf_status'])}",
                f"- **Source-code URL checked:** {markdown_value(row['code_url_checked'], 'Not checked this session')}",
                f"- **Code check scope:** {markdown_value(row['code_check_scope'])}",
                f"- **Historical code record (not reverified unless the scope above says otherwise):** {markdown_value(row['historical_code_record'])}",
                f"- **Coauthor verdict:** {markdown_value(row['coauthor_verdict'], 'PENDING')}",
                f"- **Supporting page/section:** {markdown_value(row['page_section_support'], 'PENDING')}",
                f"- **Proposed correction:** {markdown_value(row['proposed_correction'], 'PENDING')}",
                "",
            ]
        )

    REFERENCE_MD.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    fieldnames = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_dynamic_exports() -> None:
    steps = read_jsonl(RESULT_DIR / "steps.jsonl")
    episodes = read_jsonl(RESULT_DIR / "episodes.jsonl")

    selected_steps = [
        row
        for row in steps
        if row.get("phase") == "evaluation" and row.get("arm") == "switching" and row.get("episode") == 0
    ]
    if len(selected_steps) != 100:
        raise ValueError(f"Expected 100 evaluation/switching/episode-0 steps, found {len(selected_steps)}")
    if [row["step"] for row in selected_steps] != list(range(100)):
        raise ValueError("Selected timeline steps are not exactly 0-99 in source order")

    timeline_rows = []
    for row in selected_steps:
        context = row["context"]
        if not isinstance(context, list) or len(context) != 3:
            raise ValueError(f"Expected a three-value context vector at raw step {row['step']}")
        timeline_rows.append(
            {
                "display_step": row["step"] + 1,
                "raw_step": row["step"],
                "active_users": row["active_users"],
                "selected_policy": row["policy"],
                "normalized_active_user_fraction": context[0],
                "normalized_edge_backlog": context[1],
                "normalized_access_pressure": context[2],
                "context_bin": row["context_bin"],
                "decisions": row["decisions"],
                "gate_rejections": row["gate_rejections"],
                "completed_now": row["completed_now"],
                "action_seconds": row["action_seconds"],
                "context_selector_seconds": row["context_selector_seconds"],
                "controller_seconds": row["controller_seconds"],
            }
        )

    timeline_fields = [
        "display_step",
        "raw_step",
        "active_users",
        "selected_policy",
        "normalized_active_user_fraction",
        "normalized_edge_backlog",
        "normalized_access_pressure",
        "context_bin",
        "decisions",
        "gate_rejections",
        "completed_now",
        "action_seconds",
        "context_selector_seconds",
        "controller_seconds",
    ]
    write_csv(TIMELINE_CSV, timeline_fields, timeline_rows)

    selected_episodes = [
        row for row in episodes if row.get("phase") == "evaluation" and row.get("episode") == 0
    ]
    expected_arms = ["fixed", "adaptive", "contextual", "switching"]
    if [row["arm"] for row in selected_episodes] != expected_arms:
        raise ValueError(f"Expected evaluation arms {expected_arms}, found {[row['arm'] for row in selected_episodes]}")
    if len({row["trace_hash"] for row in selected_episodes}) != 1:
        raise ValueError("Evaluation arms do not share one held-out trace hash")

    evaluation_rows = []
    for row in selected_episodes:
        export_row = dict(row)
        export_row["completion_rate_percent"] = 100.0 * row["completion_rate"]
        evaluation_rows.append(export_row)

    evaluation_fields = [
        "phase",
        "episode",
        "arm",
        "trace_hash",
        "wall_seconds",
        "qoe",
        "delay",
        "energy",
        "drop",
        "completion_rate",
        "completion_rate_percent",
        "drop_rate",
        "completed_task",
        "arrived_task",
        "switch_count",
        "gate_rejections",
        "decisions",
        "action_seconds",
        "seconds_per_decision",
        "context_selector_seconds",
        "controller_seconds",
        "controller_seconds_per_step",
        "controller_seconds_per_decision",
        "delayed_across_switch",
        "context_saturated_steps",
    ]
    write_csv(EVALUATION_CSV, evaluation_fields, evaluation_rows)

    # Read the exports back and prove that every copied scalar is unchanged.
    with TIMELINE_CSV.open(encoding="utf-8", newline="") as handle:
        exported_timeline = list(csv.DictReader(handle))
    for source, exported in zip(selected_steps, exported_timeline, strict=True):
        context = source["context"]
        exact_pairs = {
            "display_step": source["step"] + 1,
            "raw_step": source["step"],
            "active_users": source["active_users"],
            "selected_policy": source["policy"],
            "normalized_active_user_fraction": context[0],
            "normalized_edge_backlog": context[1],
            "normalized_access_pressure": context[2],
            "context_bin": source["context_bin"],
            "decisions": source["decisions"],
            "gate_rejections": source["gate_rejections"],
            "completed_now": source["completed_now"],
            "action_seconds": source["action_seconds"],
            "context_selector_seconds": source["context_selector_seconds"],
            "controller_seconds": source["controller_seconds"],
        }
        for field, value in exact_pairs.items():
            if exported[field] != str(value):
                raise AssertionError(f"Timeline mismatch at raw step {source['step']}: {field}")

    with EVALUATION_CSV.open(encoding="utf-8", newline="") as handle:
        exported_evaluation = list(csv.DictReader(handle))
    for source, exported in zip(selected_episodes, exported_evaluation, strict=True):
        for field, value in source.items():
            expected = str(value)
            if exported[field] != expected:
                raise AssertionError(f"Evaluation mismatch for {source['arm']}: {field}")
        expected_percent = str(100.0 * source["completion_rate"])
        if exported["completion_rate_percent"] != expected_percent:
            raise AssertionError(f"Completion-rate percent mismatch for {source['arm']}")


def write_derived_data_guide() -> None:
    content = """# Derived Data Exports

These two CSV files are convenience views derived without changing the packaged raw JSONL records. They describe the completed `dynamic_scale` diagnostic/resource probe at code revision `67c1a8c40c6b94de4add2f6d0a6d2dadbdddd3e7`. They are **not** results from the pending full multi-seed pilot and must not be presented as confirmatory performance evidence.

## `dynamic_scale_timeline.csv`

Source: `dynamic_scale/seed_42/attempt_001/result/steps.jsonl`.

Filter: `phase == "evaluation"`, `arm == "switching"`, and `episode == 0`. The export contains exactly 100 source steps. `raw_step` preserves the zero-based source index; `display_step = raw_step + 1` is provided only for figure axes.

The three context fields map exactly to the implementation in [`dynamic_experiment/environment.py`](../../03_code_comments/source/dynamic_experiment/environment.py):

- `normalized_active_user_fraction = active_users / pool_users`, unitless and clipped by construction to the configured population range.
- `normalized_edge_backlog = clip(mean edge-computation backlog / 12, 0, 1)`, unitless. It is a normalized controller context, not raw queue bytes or task count.
- `normalized_access_pressure = clip(1 - mean best normalized AP-channel rate, 0, 1)`, unitless. It is not a measured packet-loss rate.

`action_seconds`, `context_selector_seconds`, and `controller_seconds` are seconds per recorded step. `controller_seconds` is the sum of the other two fields in the source record. `completion_now` is not exported; the exact source field is `completed_now`. The source has no per-step QoE, delay, energy, or drop value, so this timeline must not be used to invent per-step performance curves.

## `dynamic_scale_evaluation.csv`

Source: `dynamic_scale/seed_42/attempt_001/result/episodes.jsonl`.

Filter: `phase == "evaluation"` and `episode == 0`. The four rows are `fixed`, `adaptive`, `contextual`, and `switching`, all evaluated on the same held-out `trace_hash`. Source metrics are copied exactly. `completion_rate` remains a fraction in [0, 1]; `completion_rate_percent` is the sole derived metric and equals `100 * completion_rate`.

This is one seed and one evaluation episode from a scale/diagnostic run. It is suitable for checking the export and drafting figure structure, not for claiming statistical superiority or replacing the full paired multi-seed evaluation.

## Reproduce

From the handoff package root:

```sh
python3 tools/export_review.py
```

The script uses only the Python standard library, checks the expected phase/arm/episode filters and row counts, confirms the four evaluation arms share one trace, then reads both CSV files back and compares every exported source value with its JSONL record.
"""
    DERIVED_MD.write_text(content, encoding="utf-8")


def main() -> None:
    write_reference_review()
    write_dynamic_exports()
    write_derived_data_guide()
    print("Created REFERENCE_REVIEW.md with 41 entries")
    print("Created dynamic_scale_timeline.csv with 100 data rows")
    print("Created dynamic_scale_evaluation.csv with 4 data rows")
    print("Created DERIVED_DATA.md")


if __name__ == "__main__":
    main()
