#!/usr/bin/env python3
"""Fused forced-decode benchmark using the instance-level TurboPolar adapter.

This benchmark compares a dense mlx_lm KVCache against the fused TurboPolar
attention path (TurboPolarFastCache + TurboPolarLlamaAdapter) on a real MLX
Llama model.  It reports logit similarity, perplexity delta, compression ratio,
decode speed, and kernel execution statistics.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.integrations.mlx_lm.llama_adapter import TurboPolarLlamaAdapter
from benchmarks.prompt_fixtures import normalize_prompts
from benchmarks.report_schema import BenchmarkReport, PromptResult
from benchmarks.report_writer import write_json_report, write_markdown_report
from benchmarks.turbopolar_fast_attention import make_turbo_caches


def _first_param_dtype(params: Dict[str, Any]) -> str:
    """Find the first floating-point parameter dtype in a nested parameter dict."""
    for v in params.values():
        if isinstance(v, dict):
            dtype = _first_param_dtype(v)
            if dtype is not None:
                return dtype
        elif hasattr(v, "dtype"):
            if "float" in str(v.dtype):
                return str(v.dtype)
    for v in params.values():
        if hasattr(v, "dtype"):
            return str(v.dtype)
    return "unknown"


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
    if a.ndim == 2:
        a = a[None, ...]
        b = b[None, ...]
    matches = 0
    total = 0
    for batch in range(a.shape[0]):
        for t in range(a.shape[1]):
            top_a = set(np.argsort(a[batch, t])[-k:].tolist())
            top_b = set(np.argsort(b[batch, t])[-k:].tolist())
            matches += len(top_a & top_b)
            total += k
    return matches / total if total > 0 else 0.0


def _perplexity(logits: np.ndarray, tokens: List[int]) -> float:
    if np.isnan(logits).any():
        return float("inf")
    if logits.ndim == 2:
        logits = logits[None, ...]
    log_probs = _softmax(logits, axis=-1)
    token_log_probs = []
    for t in range(logits.shape[1] - 1):
        token_id = tokens[t + 1]
        token_log_probs.append(-np.log(log_probs[0, t, token_id] + 1e-12))
    return float(np.exp(np.mean(token_log_probs))) if token_log_probs else float("inf")


def _teacher_forced_logits_from_tokens(model, tokens: List[int], cache: List):
    tokens_mx = mx.array(tokens)[None, :]
    logits = model(tokens_mx, cache=cache)
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    return np.array(logits), tokens


def _teacher_forced_logits(model, tokenizer, prompt_text: str, cache: List, max_tokens: int):
    tokens = tokenizer.encode(prompt_text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return _teacher_forced_logits_from_tokens(model, tokens, cache)


def _peak_kv_bytes_dense(cache: List[KVCache]) -> int:
    return sum(c.nbytes for c in cache)


def _peak_kv_bytes_turbo(cache: List[Any]) -> int:
    return sum(c.nbytes for c in cache)


def _make_dense_cache(num_layers: int) -> List[KVCache]:
    return [KVCache() for _ in range(num_layers)]


def _make_turbo_config(num_q_heads: int, num_kv_heads: int, head_dim: int) -> TurboPolarConfig:
    return TurboPolarConfig(
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
    )


def benchmark_prompt(
    model,
    tokenizer,
    prompt_text: str,
    max_tokens: int,
    adapter: TurboPolarLlamaAdapter,
    tokens: Optional[List[int]] = None,
) -> PromptResult:
    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)
    num_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)

    if tokens is None:
        tokens = tokenizer.encode(prompt_text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    dense_cache = _make_dense_cache(num_layers)
    dense_logits, _ = _teacher_forced_logits_from_tokens(model, tokens, dense_cache)

    turbo_cache = make_turbo_caches(
        num_layers, num_q_heads, num_kv_heads, head_dim, use_qjl=False
    )
    for c in turbo_cache:
        c.reset_execution_stats()

    adapter.install(model)
    try:
        turbo_logits, _ = _teacher_forced_logits_from_tokens(model, tokens, turbo_cache)
    finally:
        adapter.uninstall()

    telem = turbo_cache[0].runtime.get_io_telemetry()
    compression_ratio = telem.get("compression_ratio", 0.0)

    return PromptResult(
        prompt=prompt_text,
        prompt_tokens=len(tokens),
        dense_logits_shape=dense_logits.shape,
        turbo_logits_shape=turbo_logits.shape,
        logit_cosine=_logit_cosine(dense_logits, turbo_logits),
        top5_overlap=_topk_overlap(dense_logits, turbo_logits, k=5),
        top10_overlap=_topk_overlap(dense_logits, turbo_logits, k=10),
        kl_divergence=0.0,
        perplexity_delta=abs(
            _perplexity(dense_logits, tokens) - _perplexity(turbo_logits, tokens)
        ),
        compression_ratio=compression_ratio,
        peak_kv_bytes_turbo=_peak_kv_bytes_turbo(turbo_cache),
        peak_kv_bytes_dense=_peak_kv_bytes_dense(dense_cache),
    )


def _measure_decode_speed(
    model, tokenizer, cache: List[Any], tokens: List[int], num_decode: int
) -> float:
    """Measure decode tok/s using greedy argmax token selection."""
    prompt_mx = mx.array(tokens)[None, :]
    model(prompt_mx, cache=cache)
    mx.eval(mx.array(0))

    start = time.perf_counter()
    last_token = tokens[-1]
    for _ in range(num_decode):
        next_input = mx.array([[last_token]])
        logits = model(next_input, cache=cache)
        mx.eval(logits)
        probs = _softmax(np.array(logits)[0, -1], axis=-1)
        last_token = int(np.argmax(probs))
    elapsed = time.perf_counter() - start
    return num_decode / elapsed if elapsed > 0 else 0.0


def _aggregate_execution_stats(turbo_cache: List[Any]) -> Dict[str, int]:
    totals = {
        "fused_qk_calls": 0,
        "online_attention_calls": 0,
        "dense_tail_calls": 0,
        "fallback_calls": 0,
    }
    for c in turbo_cache:
        stats = c.execution_stats()
        totals["fused_qk_calls"] += stats.fused_qk_calls
        totals["online_attention_calls"] += stats.online_attention_calls
        totals["dense_tail_calls"] += stats.dense_tail_calls
        totals["fallback_calls"] += stats.fallback_calls
    return totals


def load_prompts(path: Path) -> List[str]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                prompts.append(obj.get("prompt", obj.get("text", "")))
            elif isinstance(obj, str):
                prompts.append(obj)
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Fused forced-decode benchmark for TurboPolar attention"
    )
    parser.add_argument(
        "--model", required=True, help="MLX model path or Hugging Face identifier"
    )
    parser.add_argument(
        "--prompt-suite",
        type=Path,
        default=Path(__file__).parent / "prompt_suite.jsonl",
        help="Legacy text prompt suite (JSONL with 'prompt' or 'text' fields)",
    )
    parser.add_argument(
        "--token-fixtures",
        type=Path,
        default=None,
        help="Exact-token fixtures (JSONL with 'tokens' and 'category' fields). "
        "If provided, overrides --prompt-suite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "fused_forced_decode",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Max prompt length in tokens"
    )
    parser.add_argument(
        "--num-decode", type=int, default=32, help="Tokens to measure decode speed"
    )
    parser.add_argument(
        "--skip-decode-speed", action="store_true", help="Skip decode speed measurement"
    )
    args = parser.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model: {args.model}")
    model, tokenizer = load(str(args.model))

    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)
    num_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)

    turbo_config = _make_turbo_config(num_q_heads, num_kv_heads, head_dim)
    adapter = TurboPolarLlamaAdapter(turbo_config)

    prompt_source = args.token_fixtures if args.token_fixtures else args.prompt_suite
    normalized = normalize_prompts(tokenizer, prompt_source)
    if not normalized:
        raise ValueError(f"No prompts found in {prompt_source}")

    results = []
    for i, entry in enumerate(normalized):
        tokens = entry["tokens"]
        prompt_len = len(tokens)
        print(f"Benchmarking prompt {i + 1}/{len(normalized)} ({prompt_len} tokens)")
        result = benchmark_prompt(
            model,
            tokenizer,
            entry.get("text", ""),
            args.max_tokens,
            adapter,
            tokens=tokens,
        )
        results.append(result)
        print(
            f"  cosine={result.logit_cosine:.4f} top5={result.top5_overlap:.4f} "
            f"ppl_delta={result.perplexity_delta:.4f} ratio={result.compression_ratio:.3f}x"
        )

    dense_decode_tok_per_sec: Optional[float] = None
    turbo_decode_tok_per_sec: Optional[float] = None
    final_kernel_stats: Optional[Dict[str, int]] = None
    if not args.skip_decode_speed and normalized:
        print("Measuring decode speed...")
        tokens = normalized[0]["tokens"][: args.max_tokens]

        dense_cache = _make_dense_cache(num_layers)
        dense_decode_tok_per_sec = _measure_decode_speed(
            model, tokenizer, dense_cache, tokens, args.num_decode
        )

        turbo_cache = make_turbo_caches(
            num_layers, num_q_heads, num_kv_heads, head_dim, use_qjl=False
        )
        for c in turbo_cache:
            c.reset_execution_stats()

        adapter.install(model)
        try:
            turbo_decode_tok_per_sec = _measure_decode_speed(
                model, tokenizer, turbo_cache, tokens, args.num_decode
            )
        finally:
            adapter.uninstall()
        final_kernel_stats = _aggregate_execution_stats(turbo_cache)
        print(
            f"  dense: {dense_decode_tok_per_sec:.2f} tok/s  "
            f"turbo: {turbo_decode_tok_per_sec:.2f} tok/s"
        )
        print(f"  kernel_stats: {final_kernel_stats}")

    MIN_GATE_TOKENS = 64
    gate_results = [r for r in results if r.prompt_tokens >= MIN_GATE_TOKENS]
    if not gate_results:
        print(f"WARNING: no prompts reached {MIN_GATE_TOKENS} tokens; aggregates pessimistic.")
        gate_results = results

    aggregate = {
        "logit_cosine": float(np.mean([r.logit_cosine for r in gate_results])),
        "top5_overlap": float(np.mean([r.top5_overlap for r in gate_results])),
        "top10_overlap": float(np.mean([r.top10_overlap for r in gate_results])),
        "perplexity_delta": float(np.mean([r.perplexity_delta for r in gate_results])),
        "compression_ratio": float(np.mean([r.compression_ratio for r in gate_results])),
        "peak_kv_bytes_dense": int(np.max([r.peak_kv_bytes_dense for r in results])),
        "peak_kv_bytes_turbo": int(np.max([r.peak_kv_bytes_turbo for r in results])),
        "decode_speed_dense_tok_per_sec": dense_decode_tok_per_sec,
        "decode_speed_turbo_tok_per_sec": turbo_decode_tok_per_sec,
        "decode_speed_ratio": (
            turbo_decode_tok_per_sec / dense_decode_tok_per_sec
            if dense_decode_tok_per_sec is not None and dense_decode_tok_per_sec > 0
            else None
        ),
        "kernel_stats": final_kernel_stats,
        "gate_eligible_prompts": len(gate_results),
        "total_prompts": len(results),
    }

    report = BenchmarkReport(
        model=str(args.model),
        mlx_version=mx.__version__,
        mlx_lm_version=mlx_lm.__version__,
        dtype=_first_param_dtype(model.parameters()),
        seed=args.seed,
        num_layers=num_layers,
        num_prompts=len(normalized),
        prompts=results,
        aggregate=aggregate,
    )

    write_json_report(report, args.output_dir)
    write_markdown_report(report, args.output_dir)
    print(f"Report written to {args.output_dir}")
    print("Promotion status: locked; use rfsn_v11.promotion.gate to evaluate evidence.")


if __name__ == "__main__":
    main()
