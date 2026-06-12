#!/usr/bin/env python3
"""Dense-vs-TurboPolar teacher-forced benchmark on a real MLX model."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.models.cache import KVCache
from mlx_lm import load

# Ensure project root is on path.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from benchmarks.turbopolar_mlxlm_cache import TurboPolarMLXLMCache
from benchmarks.report_schema import BenchmarkReport, PromptResult
from benchmarks.report_writer import write_json_report, write_markdown_report


def _first_param_dtype(params):
    for v in params.values():
        if hasattr(v, "dtype"):
            return v.dtype
        if isinstance(v, dict):
            return _first_param_dtype(v)
    return "unknown"


def _model_cache_config(model) -> Tuple[int, int, int]:
    """Infer (num_q_heads, num_kv_heads, head_dim) from the model."""
    n_heads = getattr(model, "n_heads", None)
    n_kv_heads = getattr(model, "n_kv_heads", None)
    hidden_size = getattr(model, "hidden_size", None)

    if n_heads is None or n_kv_heads is None or hidden_size is None:
        # Fallback: inspect the first attention layer.
        attn = None
        for module in model.modules():
            name = type(module).__name__
            if name == "Attention":
                attn = module
                break
        if attn is None:
            raise ValueError("Could not infer attention config from model")
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads
        hidden_size = attn.q_proj.weight.shape[0]

    head_dim = hidden_size // n_heads
    return int(n_heads), int(n_kv_heads), int(head_dim)


def _make_dense_cache(num_layers: int):
    return [KVCache() for _ in range(num_layers)]


def _make_turbo_cache(num_layers: int, num_q_heads: int, num_kv_heads: int, head_dim: int):
    if head_dim not in (64, 128):
        raise ValueError(f"TurboPolar only supports head_dim 64 or 128, got {head_dim}")
    config = TurboPolarConfig(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=64,
        qjl_proj_dim=32 if head_dim == 64 else 64,
        use_qjl=False,
        storage_mode="kv_quant",
    )
    return [TurboPolarMLXLMCache(config) for _ in range(num_layers)]


def _run_forward(model, tokens: mx.array, cache: List):
    return model(tokens, cache=cache)


def _teacher_forced_logits(model, tokenizer, prompt_text: str, cache: List, max_tokens: int):
    tokens = tokenizer.encode(prompt_text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    tokens_mx = mx.array(tokens)[None, :]  # (1, L)
    logits = _run_forward(model, tokens_mx, cache)
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    return np.array(logits), tokens, cache


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
    # a, b shape: (B, T, V) or (T, V)
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


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if np.isnan(p).any() or np.isnan(q).any():
        return float("inf")
    # p, q shape: (B, T, V)
    p = _softmax(p, axis=-1)
    q = _softmax(q, axis=-1)
    eps = 1e-12
    kl = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)
    return float(np.mean(kl))


def _perplexity(logits: np.ndarray, tokens: List[int]) -> float:
    if np.isnan(logits).any():
        return float("inf")
    # logits shape: (1, T, V) or (T, V)
    if logits.ndim == 2:
        logits = logits[None, ...]
    log_probs = _softmax(logits, axis=-1)
    token_log_probs = []
    for t in range(logits.shape[1] - 1):
        token_id = tokens[t + 1]
        token_log_probs.append(-np.log(log_probs[0, t, token_id] + 1e-12))
    return float(np.exp(np.mean(token_log_probs))) if token_log_probs else float("inf")


def _peak_kv_bytes_dense(cache: List[KVCache]) -> int:
    return sum(c.nbytes for c in cache)


def _peak_kv_bytes_turbo(cache: List[TurboPolarMLXLMCache]) -> int:
    return sum(c.nbytes for c in cache)


def _measure_decode_speed(model, tokenizer, cache: List, tokens: List[int], num_decode: int = 64) -> float:
    """Rough decode tokens/sec for the last layer only is not trivial; measure full-model decode."""
    # Prime the cache with prompt tokens.
    prompt_mx = mx.array(tokens)[None, :]
    _run_forward(model, prompt_mx, cache)
    mx.eval(mx.array(0))

    start = time.perf_counter()
    last_token = tokens[-1]
    for _ in range(num_decode):
        next_input = mx.array([[last_token]])
        logits = _run_forward(model, next_input, cache)
        mx.eval(logits)
        probs = _softmax(np.array(logits)[0, -1], axis=-1)
        last_token = int(np.argmax(probs))
    elapsed = time.perf_counter() - start
    return num_decode / elapsed if elapsed > 0 else 0.0


def benchmark_prompt(model, tokenizer, prompt_text: str, max_tokens: int, num_decode: int) -> PromptResult:
    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)
    num_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)

    dense_cache = _make_dense_cache(num_layers)
    dense_logits, tokens, dense_cache = _teacher_forced_logits(model, tokenizer, prompt_text, dense_cache, max_tokens)

    turbo_cache = _make_turbo_cache(num_layers, num_q_heads, num_kv_heads, head_dim)
    turbo_logits, _tokens, turbo_cache = _teacher_forced_logits(model, tokenizer, prompt_text, turbo_cache, max_tokens)

    if _tokens != tokens:
        raise RuntimeError("Token mismatch between dense and TurboPolar runs")

    telem = turbo_cache[0].runtime.get_io_telemetry()
    compression_ratio = telem.get("compression_ratio", 0.0)
    peak_dense = _peak_kv_bytes_dense(dense_cache)
    peak_turbo = _peak_kv_bytes_turbo(turbo_cache)

    return PromptResult(
        prompt=prompt_text,
        prompt_tokens=len(tokens),
        dense_logits_shape=dense_logits.shape,
        turbo_logits_shape=turbo_logits.shape,
        logit_cosine=_logit_cosine(dense_logits, turbo_logits),
        top5_overlap=_topk_overlap(dense_logits, turbo_logits, k=5),
        top10_overlap=_topk_overlap(dense_logits, turbo_logits, k=10),
        kl_divergence=_kl_divergence(dense_logits, turbo_logits),
        perplexity_delta=abs(_perplexity(dense_logits, tokens) - _perplexity(turbo_logits, tokens)),
        compression_ratio=compression_ratio,
        peak_kv_bytes_turbo=peak_turbo,
        peak_kv_bytes_dense=peak_dense,
    )


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
    parser = argparse.ArgumentParser(description="Dense vs TurboPolar benchmark on an MLX model")
    parser.add_argument("--model", required=True, help="MLX model path or Hugging Face identifier")
    parser.add_argument("--prompt-suite", type=Path, default=Path(__file__).parent / "prompt_suite.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=128, help="Max prompt length in tokens")
    parser.add_argument("--num-decode", type=int, default=32, help="Tokens to measure decode speed")
    parser.add_argument("--skip-decode-speed", action="store_true", help="Skip decode speed measurement")
    args = parser.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    model_path = Path(args.model)
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)

    prompts = load_prompts(args.prompt_suite)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_suite}")

    num_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)

    results = []
    for i, prompt in enumerate(prompts):
        print(f"Benchmarking prompt {i + 1}/{len(prompts)} ({len(tokenizer.encode(prompt))} tokens)")
        result = benchmark_prompt(model, tokenizer, prompt, args.max_tokens, args.num_decode)
        results.append(result)
        print(f"  cosine={result.logit_cosine:.4f} top5={result.top5_overlap:.4f} ppl_delta={result.perplexity_delta:.4f} ratio={result.compression_ratio:.3f}x")

    # Decode speed measured separately on the first prompt so it does not corrupt per-prompt caches.
    dense_decode_tok_per_sec = 0.0
    turbo_decode_tok_per_sec = 0.0
    if not args.skip_decode_speed and prompts:
        print("Measuring decode speed...")
        tokens = tokenizer.encode(prompts[0])[: args.max_tokens]
        num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)
        dense_cache = _make_dense_cache(num_layers)
        dense_decode_tok_per_sec = _measure_decode_speed(model, tokenizer, dense_cache, tokens, args.num_decode)
        turbo_cache = _make_turbo_cache(num_layers, num_q_heads, num_kv_heads, head_dim)
        turbo_decode_tok_per_sec = _measure_decode_speed(model, tokenizer, turbo_cache, tokens, args.num_decode)
        print(f"  dense: {dense_decode_tok_per_sec:.2f} tok/s  turbo: {turbo_decode_tok_per_sec:.2f} tok/s")

    aggregate = {
        "logit_cosine": float(np.mean([r.logit_cosine for r in results])),
        "top5_overlap": float(np.mean([r.top5_overlap for r in results])),
        "top10_overlap": float(np.mean([r.top10_overlap for r in results])),
        "kl_divergence": float(np.mean([r.kl_divergence for r in results])),
        "perplexity_delta": float(np.mean([r.perplexity_delta for r in results])),
        "compression_ratio": float(np.mean([r.compression_ratio for r in results])),
        "peak_kv_bytes_dense": int(np.max([r.peak_kv_bytes_dense for r in results])),
        "peak_kv_bytes_turbo": int(np.max([r.peak_kv_bytes_turbo for r in results])),
        "decode_speed_dense_tok_per_sec": dense_decode_tok_per_sec,
        "decode_speed_turbo_tok_per_sec": turbo_decode_tok_per_sec,
        "decode_speed_ratio": turbo_decode_tok_per_sec / dense_decode_tok_per_sec if dense_decode_tok_per_sec > 0 else 0.0,
    }

    report = BenchmarkReport(
        model=str(args.model),
        mlx_version=mx.__version__,
        mlx_lm_version=mlx_lm.__version__,
        dtype=str(_first_param_dtype(model.parameters())),
        seed=args.seed,
        num_layers=num_layers,
        num_prompts=len(prompts),
        prompts=results,
        aggregate=aggregate,
    )
    report.evaluate_gates()

    write_json_report(report, args.output_dir)
    write_markdown_report(report, args.output_dir)
    print(f"Report written to {args.output_dir}")
    print(f"Promotion allowed: {report.promotion_allowed}")


if __name__ == "__main__":
    main()
