#!/usr/bin/env python3
"""Generate Markdown summary from validated raw benchmark JSON artifacts.

Usage:
    python scripts/generate_evidence_markdown.py \
        --speed benchmarks/outputs/speed_matrix/report.json \
        --memory benchmarks/outputs/memory_matrix/report.json \
        --output evidence_summary.md
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "invalid"
        return f"{v:.3f}"
    return str(v)


def _speed_markdown(data: dict) -> str:
    records = data.get("records", [])
    lines = [
        "# Speed Evidence Summary",
        "",
        f"**Model:** `{data.get('model', 'unknown')}`  ",
        f"**MLX:** {data.get('mlx_version', 'unknown')}  ",
        f"**mlx_lm:** {data.get('mlx_lm_version', 'unknown')}  ",
        f"**Execution mode:** {data.get('execution_mode', 'unknown')}  ",
        f"**Seed:** {data.get('seed', 'unknown')}  ",
        f"**Trials per context:** {data.get('trials', 'unknown')}  ",
        "",
        "## Per-Context Results",
        "",
        "| Context | Valid trials | Dense tok/s | Turbo tok/s | Speedup |",
        "|---------|-------------:|------------:|------------:|--------:|",
    ]
    for r in records:
        lines.append(
            f"| {r.get('length', 0)} | {r.get('valid_trials', 0)} | "
            f"{_fmt(r.get('dense_mean_tok_per_sec'))} | "
            f"{_fmt(r.get('turbo_mean_tok_per_sec'))} | "
            f"{_fmt(r.get('speedup'))} |"
        )
    return "\n".join(lines) + "\n"


def _memory_markdown(data: dict) -> str:
    records = data.get("records", [])
    lines = [
        "# Memory Evidence Summary",
        "",
        f"**Model:** `{data.get('model', 'unknown')}`  ",
        "",
        (
            "Ratio convention: `dense_to_turbo_peak_ratio`. "
            "> 1 means Turbo uses less peak memory; "
            "< 1 means Turbo uses more peak memory."
        ),
        "",
        "## Per-Context Results",
        "",
        "| Context | Dense peak (B) | Turbo peak (B) | "
        "Logical ratio | Allocated ratio | Peak ratio |",
        "|---------|---------------:|---------------:|"
        "--------------:|----------------:|-----------:|",
    ]
    for r in records:
        lines.append(
            f"| {r.get('length', 0)} | "
            f"{r.get('dense_total_peak_bytes', 0)} | "
            f"{r.get('turbo_total_peak_bytes', 0)} | "
            f"{_fmt(r.get('logical_kv_ratio'))}× | "
            f"{_fmt(r.get('persistent_storage_ratio'))}× | "
            f"{_fmt(r.get('dense_to_turbo_peak_ratio'))}× |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown from raw benchmark JSON artifacts"
    )
    parser.add_argument(
        "--speed", type=Path, default=None,
        help="Path to speed matrix JSON artifact"
    )
    parser.add_argument(
        "--memory", type=Path, default=None,
        help="Path to memory matrix JSON artifact"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output Markdown file path"
    )
    args = parser.parse_args()

    sections: list[str] = []
    if args.speed and args.speed.exists():
        with open(args.speed, "r", encoding="utf-8") as f:
            sections.append(_speed_markdown(json.load(f)))
    if args.memory and args.memory.exists():
        with open(args.memory, "r", encoding="utf-8") as f:
            sections.append(_memory_markdown(json.load(f)))

    if not sections:
        print("No valid input artifacts found.")
        return

    text = "\n".join(sections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Markdown summary written to {args.output}")


if __name__ == "__main__":
    main()
