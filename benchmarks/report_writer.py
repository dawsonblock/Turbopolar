"""Write benchmark reports to JSON and Markdown.

Promotion decisions are owned by rfsn_v11.promotion.gate. This writer only
records measured metrics.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from benchmarks.report_schema import BenchmarkReport


def _convert(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def report_to_dict(report: BenchmarkReport) -> Dict[str, Any]:
    d = asdict(report)

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return _convert(o)

    return walk(d)


def write_json_report(report: BenchmarkReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.json"
    with open(path, "w") as f:
        json.dump(report_to_dict(report), f, indent=2)
    return path


def write_markdown_report(report: BenchmarkReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.md"

    lines = [
        "# TurboPolar Benchmark Report",
        "",
        f"**Model:** `{report.model}`  ",
        f"**MLX:** {report.mlx_version}  ",
        f"**mlx_lm:** {report.mlx_lm_version}  ",
        f"**dtype:** {report.dtype}  ",
        f"**seed:** {report.seed}  ",
        f"**Prompts:** {report.num_prompts} (metrics computed on prompts ≥ 64 tokens)  ",
        "",
        "> **Note:** This report records metrics only. Promotion decisions are made by "
        "`rfsn_v11.promotion.gate.PromotionGate.evaluate()` using the full evidence matrix.",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in report.aggregate.items():
        if v is None:
            lines.append(f"| {k} | skipped |")
        elif isinstance(v, float):
            lines.append(f"| {k} | {v:.6f} |")
        else:
            lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "## Per-Prompt Results",
        "",
        "| Prompt tokens | Cosine | Top-5 | Top-10 | PPL Δ | Compression |",
        "|---------------|--------|-------|--------|-------|-------------|",
    ])
    for p in report.prompts:
        lines.append(
            f"| {p.prompt_tokens} | {p.logit_cosine:.4f} | {p.top5_overlap:.4f} | "
            f"{p.top10_overlap:.4f} | {p.perplexity_delta:.4f} | {p.compression_ratio:.3f}× |"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
