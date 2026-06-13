#!/usr/bin/env python3
"""Real fused forced-decode benchmark for TurboPolar.

This benchmark compares dense KV-cache logits against the fused TurboPolar
attention path during actual one-token autoregressive decode.  It:

1. Loads one dense model and one TurboPolar-adapted model (same weights).
2. Prefills both with identical context tokens.
3. For each of at least 128 forced continuation tokens:
   a. Feeds one identical token to both models.
   b. Executes dense one-token decode.
   c. Executes TurboPolar fused one-token decode.
   d. Compares the resulting next-token logits.
4. Records per-token metrics and proves fused execution via kernel telemetry.

Both paths receive the same forced token.  Independent generation is never used
during numerical comparison.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from benchmarks.prompt_fixtures import normalize_prompts
from benchmarks.report_schema import (
    DecodeStepMetrics,
    ForcedDecodeAggregate,
    ForcedDecodeFixtureResult,
    ForcedDecodeReport,
)
from benchmarks.report_writer import write_json_report, write_markdown_report
from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.integrations.mlx_lm.adapter import TurboPolarLlamaAdapter
from rfsn_v11.integrations.mlx_lm.cache import make_turbo_caches


def _model_cache_config(model: Any) -> Tuple[int, int, int]:
    """Infer (num_q_heads, num_kv_heads, head_dim) from the model."""
    n_heads = getattr(model, "n_heads", None)
    n_kv_heads = getattr(model, "n_kv_heads", None)
    hidden_size = getattr(model, "hidden_size", None)

    if n_heads is None or n_kv_heads is None or hidden_size is None:
        attn = None
        for module in model.modules():
            if type(module).__name__ == "Attention":
                attn = module
                break
        if attn is None:
            raise ValueError("Could not infer attention config from model")
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads
        hidden_size = attn.q_proj.weight.shape[0]

    head_dim = hidden_size // n_heads
    return int(n_heads), int(n_kv_heads), int(head_dim)


def _nll(logits: np.ndarray, token_id: int) -> float:
    """Negative log-likelihood of token_id under logits."""
    probs = _softmax(logits)
    if probs.ndim > 1:
        probs = probs[0]
    return float(-np.log(probs[token_id] + 1e-12))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _logit_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if np.isnan(a).any() or np.isnan(b).any():
        return 0.0
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12
    return float(np.dot(a_flat, b_flat) / denom)


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if np.isnan(a).any() or np.isnan(b).any():
        return 0.0
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    matches = len(top_a & top_b)
    return matches / k


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats."""
    p = _softmax(p)
    q = _softmax(q)
    mask = p > 1e-12
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-12))))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats."""
    p = _softmax(p)
    q = _softmax(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p[p > 1e-12] * np.log(p[p > 1e-12] / (m[p > 1e-12] + 1e-12)))
    kl_qm = np.sum(q[q > 1e-12] * np.log(q[q > 1e-12] / (m[q > 1e-12] + 1e-12)))
    return float(0.5 * (kl_pm + kl_qm))


def _rank_of_token_in_logits(logits: np.ndarray, token_id: int) -> int:
    """Return the rank (0 = highest logit) of token_id in logits."""
    sorted_indices = np.argsort(logits)[::-1]
    matches = np.where(sorted_indices == token_id)[0]
    if len(matches) == 0:
        return -1
    return int(matches[0])


def _compute_step_metrics(
    dense_logits: np.ndarray,
    turbo_logits: np.ndarray,
    forced_token: int,
    position: int,
) -> DecodeStepMetrics:
    """Compute per-token quality metrics for one decode position."""
    dense_last = dense_logits.flatten().astype(np.float64)
    turbo_last = turbo_logits.flatten().astype(np.float64)

    any_nan_or_inf = bool(
        np.isnan(dense_last).any()
        or np.isnan(turbo_last).any()
        or np.isinf(dense_last).any()
        or np.isinf(turbo_last).any()
    )

    cosine = _logit_cosine(dense_last, turbo_last)
    dense_argmax = int(np.argmax(dense_last))
    turbo_argmax = int(np.argmax(turbo_last))
    top1_agreement = dense_argmax == turbo_argmax
    top5 = _topk_overlap(dense_last, turbo_last, 5)
    top10 = _topk_overlap(dense_last, turbo_last, 10)

    kl = _kl_divergence(dense_last, turbo_last)
    js = _js_divergence(dense_last, turbo_last)

    rank = _rank_of_token_in_logits(turbo_last, dense_argmax)

    dense_probs = _softmax(dense_last)
    turbo_probs = _softmax(turbo_last)
    dense_argmax_prob_delta = float(
        abs(dense_probs[dense_argmax] - turbo_probs[dense_argmax])
    )

    nll_delta = float(
        abs(
            -np.log(dense_probs[forced_token] + 1e-12)
            + np.log(turbo_probs[forced_token] + 1e-12)
        )
    )

    return DecodeStepMetrics(
        position=position,
        logit_cosine=cosine,
        top1_agreement=top1_agreement,
        top5_overlap=top5,
        top10_overlap=top10,
        kl_divergence=kl,
        js_divergence=js,
        dense_argmax_rank_in_turbo=rank,
        dense_argmax_prob_delta=dense_argmax_prob_delta,
        nll_delta=nll_delta,
        any_nan_or_inf=any_nan_or_inf,
    )


def _reset_all_execution_stats(turbo_cache: List[Any]) -> None:
    """Reset the process-wide bridge once via any cache instance."""
    if turbo_cache:
        turbo_cache[0].reset_execution_stats()


def _aggregate_execution_stats(turbo_cache: List[Any]) -> Dict[str, int]:
    """Read the process-wide bridge once via any cache instance."""
    if not turbo_cache:
        return {
            "fused_qk_calls": 0,
            "online_attention_calls": 0,
            "dense_tail_calls": 0,
            "fallback_calls": 0,
        }
    stats = turbo_cache[0].execution_stats()
    return {
        "fused_qk_calls": stats.fused_qk_calls,
        "online_attention_calls": stats.online_attention_calls,
        "dense_tail_calls": stats.dense_tail_calls,
        "fallback_calls": stats.fallback_calls,
        "compressed_page_dispatches": getattr(stats, "compressed_page_dispatches", 0),
        "dense_tail_dispatches": getattr(stats, "dense_tail_dispatches", 0),
    }


def benchmark_forced_decode_fixture(
    model,
    tokenizer,
    context_tokens: List[int],
    continuation_tokens: List[int],
    adapter: TurboPolarLlamaAdapter,
    execution_mode=None,
) -> ForcedDecodeFixtureResult:
    """Run one forced-decode fixture and return per-step metrics."""
    num_layers = (
        len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
    )
    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)

    dense_cache = [KVCache() for _ in range(num_layers)]
    turbo_cache = make_turbo_caches(
        num_layers,
        num_q_heads,
        num_kv_heads,
        head_dim,
        use_qjl=False,
        execution_mode=execution_mode,
    )
    _reset_all_execution_stats(turbo_cache)

    context_mx = mx.array(context_tokens)[None, :]

    # Prefill both paths.
    dense_prefill = model(context_mx, cache=dense_cache)
    adapter.install(model)
    try:
        turbo_prefill = model(context_mx, cache=turbo_cache)
    finally:
        adapter.uninstall()
    mx.eval(dense_prefill, turbo_prefill)

    # Forced-decode loop: same token fed to both paths.
    # NLL alignment: prefill scores continuation[0]; feeding continuation[i] scores continuation[i+1].
    adapter.install(model)
    try:
        steps: List[DecodeStepMetrics] = []
        dense_nlls: List[float] = []
        turbo_nlls: List[float] = []

        # Prefill step scores the first continuation token.
        dense_prefill_last = np.array(dense_prefill[:, -1, :].astype(mx.float32))
        turbo_prefill_last = np.array(turbo_prefill[:, -1, :].astype(mx.float32))
        if len(continuation_tokens) > 0:
            step = _compute_step_metrics(
                dense_prefill_last, turbo_prefill_last, continuation_tokens[0], 0
            )
            steps.append(step)
            dense_nlls.append(_nll(dense_prefill_last, continuation_tokens[0]))
            turbo_nlls.append(_nll(turbo_prefill_last, continuation_tokens[0]))

        for i in range(len(continuation_tokens) - 1):
            current_token = continuation_tokens[i]
            next_target = continuation_tokens[i + 1]
            token_mx = mx.array([[current_token]])
            dense_logits = model(token_mx, cache=dense_cache)
            turbo_logits = model(token_mx, cache=turbo_cache)
            mx.eval(dense_logits, turbo_logits)

            dense_last = np.array(dense_logits[:, -1, :].astype(mx.float32))
            turbo_last = np.array(turbo_logits[:, -1, :].astype(mx.float32))

            step = _compute_step_metrics(dense_last, turbo_last, next_target, i + 1)
            steps.append(step)
            dense_nlls.append(_nll(dense_last, next_target))
            turbo_nlls.append(_nll(turbo_last, next_target))
    finally:
        adapter.uninstall()

    kernel_stats = _aggregate_execution_stats(turbo_cache)

    # Validate that the paged attention path was exercised.
    if kernel_stats["online_attention_calls"] == 0:
        raise RuntimeError(
            "Fused forced-decode produced zero online_attention_calls; "
            "the benchmark did not exercise the paged attention path."
        )
    # NOTE: The benchmark respects the configured execution_mode.
    # METAL_STRICT will raise if any fallback occurs; DEVELOPMENT_AUTO
    # falls back to the reference path on Metal unavailability.

    return ForcedDecodeFixtureResult(
        fixture_id=f"ctx_{len(context_tokens)}_cont_{len(continuation_tokens)}",
        context_length=len(context_tokens),
        continuation_length=len(continuation_tokens),
        steps=steps,
        kernel_stats=kernel_stats,
        dense_nll_per_token=dense_nlls,
        candidate_nll_per_token=turbo_nlls,
    )


def _compute_aggregate(
    results: List[ForcedDecodeFixtureResult],
    execution_mode=None,
    requested_fused_positions: int = 0,
) -> ForcedDecodeAggregate:
    # Exclude prefill (position == 0) from fused-decode aggregates.
    fused_steps = [s for r in results for s in r.steps if s.position > 0]
    all_cosines = [s.logit_cosine for s in fused_steps]
    all_top1 = [float(s.top1_agreement) for s in fused_steps]
    all_top5 = [s.top5_overlap for s in fused_steps]
    all_top10 = [s.top10_overlap for s in fused_steps]
    all_kl = [s.kl_divergence for s in fused_steps]
    all_js = [s.js_divergence for s in fused_steps]
    all_prob_delta = [s.dense_argmax_prob_delta for s in fused_steps]
    all_ranks = [s.dense_argmax_rank_in_turbo for s in fused_steps]
    any_nan = any(s.any_nan_or_inf for s in fused_steps)

    # Perplexity: exclude prefill NLL (first token in each fixture's NLL list).
    dense_nll_all = [n for r in results for n in r.dense_nll_per_token[1:]]
    candidate_nll_all = [n for r in results for n in r.candidate_nll_per_token[1:]]
    dense_mean_nll = float(np.mean(dense_nll_all)) if dense_nll_all else 0.0
    candidate_mean_nll = float(np.mean(candidate_nll_all)) if candidate_nll_all else 0.0
    dense_ppl = float(np.exp(dense_mean_nll))
    candidate_ppl = float(np.exp(candidate_mean_nll))
    abs_ppl_delta = candidate_ppl - dense_ppl
    rel_ppl_delta = (candidate_ppl / dense_ppl - 1.0) if dense_ppl > 0 else 0.0

    total_online = sum(r.kernel_stats.get("online_attention_calls", 0) for r in results)
    total_dense_tail = sum(r.kernel_stats.get("dense_tail_calls", 0) for r in results)
    total_fallback = sum(r.kernel_stats.get("fallback_calls", 0) for r in results)
    total_page_dispatches = sum(
        r.kernel_stats.get("compressed_page_dispatches", 0) for r in results
    )
    total_tail_dispatches = sum(
        r.kernel_stats.get("dense_tail_dispatches", 0) for r in results
    )

    # Separate numerical failures from actual fallback reasons.
    numerical_failures = []
    for r in results:
        for step in r.steps:
            if step.any_nan_or_inf:
                numerical_failures.append(f"NaN/Inf at {r.fixture_id} pos {step.position}")

    # Track per-context fused position counts.
    positions_per_context: Dict[int, int] = {}
    actual_fused_positions = 0
    failed_positions = 0
    for r in results:
        ctx_len = r.context_length
        fused_count = sum(1 for s in r.steps if s.position > 0)
        positions_per_context[ctx_len] = fused_count
        actual_fused_positions += fused_count
        failed_positions += sum(1 for s in r.steps if s.position > 0 and s.any_nan_or_inf)

    worst_cosine = min(all_cosines) if all_cosines else 0.0
    worst_idx = all_cosines.index(worst_cosine) if all_cosines else 0
    step_idx = 0
    worst_fixture = ""
    worst_pos = -1
    for r in results:
        for s in r.steps:
            if s.position == 0:
                continue
            if step_idx == worst_idx:
                worst_fixture = r.fixture_id
                worst_pos = s.position
            step_idx += 1

    first_div = -1
    step_idx = 0
    for r in results:
        for s in r.steps:
            if s.position == 0:
                continue
            if not s.top1_agreement:
                first_div = step_idx
                break
            step_idx += 1
        if first_div >= 0:
            break

    def _percentile(arr, p):
        return float(np.percentile(arr, p))

    mode_name = getattr(execution_mode, "value", str(execution_mode))
    aggregate = ForcedDecodeAggregate(
        mean_logit_cosine=float(np.mean(all_cosines)) if all_cosines else 0.0,
        median_logit_cosine=float(np.median(all_cosines)) if all_cosines else 0.0,
        p05_logit_cosine=_percentile(all_cosines, 5) if all_cosines else 0.0,
        p95_logit_cosine=_percentile(all_cosines, 95) if all_cosines else 0.0,
        min_logit_cosine=float(np.min(all_cosines)) if all_cosines else 0.0,
        max_logit_cosine=float(np.max(all_cosines)) if all_cosines else 0.0,
        mean_top1_agreement=float(np.mean(all_top1)) if all_top1 else 0.0,
        mean_top5_overlap=float(np.mean(all_top5)) if all_top5 else 0.0,
        mean_top10_overlap=float(np.mean(all_top10)) if all_top10 else 0.0,
        mean_kl_divergence=float(np.mean(all_kl)) if all_kl else 0.0,
        mean_js_divergence=float(np.mean(all_js)) if all_js else 0.0,
        mean_perplexity_delta=abs_ppl_delta,
        min_dense_argmax_rank=int(np.min(all_ranks)) if all_ranks else 0,
        max_dense_argmax_rank=int(np.max(all_ranks)) if all_ranks else 0,
        mean_dense_argmax_prob_delta=float(np.mean(all_prob_delta)) if all_prob_delta else 0.0,
        worst_fixture_id=worst_fixture,
        worst_position=worst_pos,
        first_argmax_divergence_position=first_div,
        any_nans_or_infs=any_nan,
        online_attention_calls=total_online,
        dense_tail_calls=total_dense_tail,
        fallback_calls=total_fallback,
        execution_mode=mode_name,
        compressed_page_metal_calls=total_page_dispatches,
        dense_tail_metal_calls=total_tail_dispatches,
        merge_metal_calls=0,
        finalization_metal_calls=0,
        compressed_page_fallback_calls=total_fallback,
        dense_tail_fallback_calls=0,
        full_attention_fallback_calls=total_fallback,
        fallback_reasons=[],
        numerical_failure_reasons=numerical_failures,
        dense_perplexity=dense_ppl,
        candidate_perplexity=candidate_ppl,
        relative_perplexity_delta=rel_ppl_delta,
        requested_fused_positions=requested_fused_positions,
        actual_fused_positions=actual_fused_positions,
        positions_per_context=positions_per_context,
        failed_positions=failed_positions,
    )

    from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode
    if execution_mode is ExecutionMode.METAL_STRICT and total_fallback > 0:
        raise RuntimeError(
            f"METAL_STRICT benchmark recorded {total_fallback} fallback call(s); "
            "strict mode prohibits any fallback."
        )

    return aggregate


def main():
    parser = argparse.ArgumentParser(
        description="Fused forced-decode benchmark for TurboPolar attention"
    )
    parser.add_argument(
        "--model", required=True, help="MLX model path or Hugging Face identifier"
    )
    parser.add_argument(
        "--token-fixtures",
        type=Path,
        default=None,
        help="Exact-token fixtures (JSONL with 'tokens' and 'category' fields).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "fused_forced_decode",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--forced-decode-tokens",
        type=int,
        default=129,
        help="Number of forced continuation tokens per fixture (default 129). "
             "The first token is scored by prefill; the remainder are fused one-token decode steps. "
             "With the default 129 tokens, 128 actual fused decode positions are produced.",
    )
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[512, 2048, 4096, 8192, 16384],
        help="Context lengths to evaluate",
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="development_auto",
        choices=["reference", "metal_strict", "development_auto"],
        help="Execution mode for TurboPolar attention (default: development_auto)",
    )
    args = parser.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model: {args.model}")
    model, tokenizer = load(str(args.model))

    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)
    from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode

    execution_mode = ExecutionMode(args.execution_mode)
    turbo_config = TurboPolarConfig(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=64,
        qjl_proj_dim=64,
        use_qjl=False,
        storage_mode="kv_quant",
        use_int8_radii=True,
        k_angle_bits_deep=8,
        split_dim=0,
        execution_mode=execution_mode,
    )
    adapter = TurboPolarLlamaAdapter(turbo_config)

    prompt_source = args.token_fixtures
    normalized = normalize_prompts(tokenizer, prompt_source) if prompt_source else []
    if not normalized:
        print(
            "No token fixtures provided; generating synthetic deterministic sequences."
        )
        fixtures = []
        for ctx_len in args.contexts:
            # Deterministic synthetic context.
            rng = np.random.RandomState(args.seed + ctx_len)
            context_tokens = [
                int(rng.randint(0, tokenizer.vocab_size)) for _ in range(ctx_len)
            ]
            continuation_tokens = [
                int(rng.randint(0, tokenizer.vocab_size))
                for _ in range(args.forced_decode_tokens)
            ]
            fixtures.append(
                {
                    "tokens": context_tokens + continuation_tokens,
                    "context_length": ctx_len,
                    "continuation_length": args.forced_decode_tokens,
                }
            )
    else:
        fixtures = []
        for entry in normalized:
            tokens = entry["tokens"]
            for ctx_len in args.contexts:
                if len(tokens) >= ctx_len + args.forced_decode_tokens:
                    fixtures.append(
                        {
                            "tokens": tokens,
                            "context_length": ctx_len,
                            "continuation_length": args.forced_decode_tokens,
                        }
                    )
                    break

    if not fixtures:
        raise ValueError("No fixtures could be constructed.")

    results: List[ForcedDecodeFixtureResult] = []
    for i, fixture in enumerate(fixtures):
        ctx = fixture["tokens"][: fixture["context_length"]]
        cont = fixture["tokens"][
            fixture["context_length"] : fixture["context_length"]
            + fixture["continuation_length"]
        ]
        print(
            f"Fixture {i + 1}/{len(fixtures)}: context={len(ctx)} continuation={len(cont)}"
        )
        result = benchmark_forced_decode_fixture(
            model, tokenizer, ctx, cont, adapter,
            execution_mode=execution_mode,
        )
        results.append(result)
        print(
            f"  mean_cosine={np.mean([s.logit_cosine for s in result.steps]):.4f} "
            f"min_cosine={np.min([s.logit_cosine for s in result.steps]):.4f} "
            f"top1_agree={np.mean([s.top1_agreement for s in result.steps]):.4f}"
        )

    # One continuation token is scored by prefill; the remainder are fused decode.
    requested_fused = max(0, args.forced_decode_tokens - 1)
    aggregate = _compute_aggregate(
        results,
        execution_mode=execution_mode,
        requested_fused_positions=requested_fused,
    )
    print(f"\nAggregate mean cosine: {aggregate.mean_logit_cosine:.4f}")
    print(f"Aggregate min cosine:  {aggregate.min_logit_cosine:.4f}")
    print(f"Aggregate p05 cosine:  {aggregate.p05_logit_cosine:.4f}")
    print(f"Aggregate top1 agree:  {aggregate.mean_top1_agreement:.4f}")
    print(f"Requested fused positions: {aggregate.requested_fused_positions}")
    print(f"Actual fused positions: {aggregate.actual_fused_positions}")
    print(f"Kernel online_attention_calls: {aggregate.online_attention_calls}")
    print(f"Kernel fallback_calls: {aggregate.fallback_calls}")

    report = ForcedDecodeReport(
        model=str(args.model),
        mlx_version=mx.__version__,
        mlx_lm_version=mlx_lm.__version__,
        dtype="unknown",
        seed=args.seed,
        num_layers=len(model.layers)
        if hasattr(model, "layers")
        else len(model.model.layers),
        forced_decode_tokens=args.forced_decode_tokens,
        contexts_evaluated=args.contexts,
        aggregate=aggregate,
        fixtures=results,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_report(report, args.output_dir / "report.json")
    write_markdown_report(report, args.output_dir / "report.md")
    print(f"\nWrote reports to {args.output_dir}")


if __name__ == "__main__":
    main()
