"""Write benchmark reports to JSON and Markdown.

Promotion decisions are owned by rfsn_v11.promotion.gate. This writer only
records measured metrics.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Union

from benchmarks.report_schema import BenchmarkReport, ForcedDecodeReport

ReportType = Union[BenchmarkReport, ForcedDecodeReport]


def _convert(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def report_to_dict(report: ReportType) -> Dict[str, Any]:
    d = asdict(report)

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return _convert(o)

    return walk(d)


def write_json_report(report: ReportType, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report_to_dict(report), f, indent=2)
    return path


def _markdown_for_benchmark_report(report: BenchmarkReport) -> str:
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

    lines.extend(
        [
            "",
            "## Per-Prompt Results",
            "",
            "| Prompt tokens | Cosine | Top-5 | Top-10 | PPL Δ | Compression |",
            "|---------------|--------|-------|--------|-------|-------------|",
        ]
    )
    for p in report.prompts:
        lines.append(
            f"| {p.prompt_tokens} | {p.logit_cosine:.4f} | {p.top5_overlap:.4f} | "
            f"{p.top10_overlap:.4f} | {p.perplexity_delta:.4f} | {p.compression_ratio:.3f}× |"
        )

    return "\n".join(lines) + "\n"


def _markdown_for_forced_decode_report(report: ForcedDecodeReport) -> str:
    agg = report.aggregate
    lines = [
        "# TurboPolar Fused Forced-Decode Report",
        "",
        f"**Model:** `{report.model}`  ",
        f"**MLX:** {report.mlx_version}  ",
        f"**mlx_lm:** {report.mlx_lm_version}  ",
        f"**dtype:** {report.dtype}  ",
        f"**seed:** {report.seed}  ",
        f"**Forced decode tokens:** {report.forced_decode_tokens}  ",
        f"**Fixtures:** {len(report.fixtures)}  ",
        "",
        "> **Note:** This report records metrics only. Promotion decisions are made by "
        "`rfsn_v11.promotion.gate.PromotionGate.evaluate()` using the full evidence matrix.",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| mean_logit_cosine | {agg.mean_logit_cosine:.6f} |",
        f"| median_logit_cosine | {agg.median_logit_cosine:.6f} |",
        f"| p05_logit_cosine | {agg.p05_logit_cosine:.6f} |",
        f"| p95_logit_cosine | {agg.p95_logit_cosine:.6f} |",
        f"| min_logit_cosine | {agg.min_logit_cosine:.6f} |",
        f"| max_logit_cosine | {agg.max_logit_cosine:.6f} |",
        f"| mean_top1_agreement | {agg.mean_top1_agreement:.6f} |",
        f"| mean_top5_overlap | {agg.mean_top5_overlap:.6f} |",
        f"| mean_top10_overlap | {agg.mean_top10_overlap:.6f} |",
        f"| mean_kl_divergence | {agg.mean_kl_divergence:.6f} |",
        f"| mean_js_divergence | {agg.mean_js_divergence:.6f} |",
        f"| mean_perplexity_delta | {agg.mean_perplexity_delta:.6f} |",
        f"| min_dense_argmax_rank | {agg.min_dense_argmax_rank} |",
        f"| max_dense_argmax_rank | {agg.max_dense_argmax_rank} |",
        f"| mean_dense_argmax_prob_delta | {agg.mean_dense_argmax_prob_delta:.6f} |",
        f"| worst_fixture_id | {agg.worst_fixture_id} |",
        f"| worst_position | {agg.worst_position} |",
        f"| first_argmax_divergence_position | {agg.first_argmax_divergence_position} |",
        f"| any_nans_or_infs | {agg.any_nans_or_infs} |",
        f"| online_attention_calls | {agg.online_attention_calls} |",
        f"| dense_tail_calls | {agg.dense_tail_calls} |",
        f"| fallback_calls | {agg.fallback_calls} |",
        "",
        "## Per-Fixture Summary",
        "",
        "| Fixture | Context | Continuation | Mean Cosine | Min Cosine | Top-1 Agree |",
        "|---------|---------|--------------|-------------|------------|-------------|",
    ]
    for f in report.fixtures:
        mean_cos = (
            sum(s.logit_cosine for s in f.steps) / len(f.steps) if f.steps else 0.0
        )
        min_cos = min((s.logit_cosine for s in f.steps), default=0.0)
        top1 = (
            sum(1 for s in f.steps if s.top1_agreement) / len(f.steps)
            if f.steps
            else 0.0
        )
        lines.append(
            f"| {f.fixture_id} | {f.context_length} | {f.continuation_length} | "
            f"{mean_cos:.4f} | {min_cos:.4f} | {top1:.4f} |"
        )

    return "\n".join(lines) + "\n"


def write_markdown_report(report: ReportType, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(report, ForcedDecodeReport):
        text = _markdown_for_forced_decode_report(report)
    else:
        text = _markdown_for_benchmark_report(report)
    with open(path, "w") as f:
        f.write(text)
    return path
