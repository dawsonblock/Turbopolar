"""Write benchmark reports to JSON and Markdown."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from benchmarks.report_schema import BenchmarkReport, PROMOTION_GATES


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
        f"**Prompts:** {report.num_prompts}  ",
        "",
        "## Promotion Gates",
        "",
        "| Gate | Threshold | Actual | Passed |",
        "|------|-----------|--------|--------|",
    ]

    agg = report.aggregate
    gate_rows = [
        ("KV compression ratio", f"≥ {PROMOTION_GATES['kv_compression_ratio']:.2f}×", f"{agg.get('compression_ratio', 0.0):.3f}×"),
        ("Logit cosine", f"≥ {PROMOTION_GATES['logit_cosine']:.4f}", f"{agg.get('logit_cosine', 0.0):.4f}"),
        ("Top-5 overlap", f"≥ {PROMOTION_GATES['top5_overlap']:.2f}", f"{agg.get('top5_overlap', 0.0):.4f}"),
        ("Top-10 overlap", f"≥ {PROMOTION_GATES['top10_overlap']:.2f}", f"{agg.get('top10_overlap', 0.0):.4f}"),
        ("Perplexity delta", f"≤ {PROMOTION_GATES['perplexity_delta']:.3f}", f"{agg.get('perplexity_delta', 0.0):.4f}"),
        ("Decode speed ratio", f"≥ {PROMOTION_GATES['decode_tokens_per_sec']:.2f}×", f"{agg.get('decode_speed_ratio', 0.0):.3f}×"),
    ]
    for name, threshold, actual in gate_rows:
        passed = report.gate_passed.get(name.split()[0].lower().replace('-', '_'), False)
        # Map names to gate keys for lookup.
        key_map = {
            "KV compression ratio": "kv_compression_ratio",
            "Logit cosine": "logit_cosine",
            "Top-5 overlap": "top5_overlap",
            "Top-10 overlap": "top10_overlap",
            "Perplexity delta": "perplexity_delta",
            "Decode speed ratio": "decode_speed",
        }
        key = key_map[name]
        passed = report.gate_passed.get(key, False)
        lines.append(f"| {name} | {threshold} | {actual} | {'✅' if passed else '❌'} |")

    lines.extend([
        "",
        f"**Promotion allowed:** {'YES' if report.promotion_allowed else 'NO'}",
        f"**Visible drift:** {'YES' if report.visible_drift else 'NO'}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ])
    for k, v in agg.items():
        if isinstance(v, float):
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
