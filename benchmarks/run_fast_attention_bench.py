#!/usr/bin/env python3
"""Benchmark dense KV cache vs fused TurboPolar attention on an MLX Llama model."""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from benchmarks.turbopolar_fast_attention import (
    TurboPolarFastCache,
    make_turbo_caches,
    patch_llama_attention,
    unpatch_llama_attention,
)


def _model_cache_config(model) -> Tuple[int, int, int]:
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


def _teacher_forced_logits(model, tokenizer, prompt_text: str, cache: List, max_tokens: int):
    tokens = tokenizer.encode(prompt_text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    tokens_mx = mx.array(tokens)[None, :]
    logits = model(tokens_mx, cache=cache)
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    return np.array(logits), tokens


def _measure_decode_speed(
    model, tokenizer, cache: List, tokens: List[int], num_decode: int
) -> float:
    """Measure decode tok/s using greedy argmax sampling."""
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


def _peak_kv_bytes_dense(cache: List[KVCache]) -> int:
    return sum(c.nbytes for c in cache)


def _peak_kv_bytes_turbo(cache: List[TurboPolarFastCache]) -> int:
    return sum(c.nbytes for c in cache)


def main():
    parser = argparse.ArgumentParser(
        description="Dense vs fused TurboPolar attention benchmark"
    )
    parser.add_argument(
        "--model", required=True, help="MLX model path or Hugging Face identifier"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Max prompt length in tokens"
    )
    parser.add_argument(
        "--num-decode", type=int, default=32, help="Tokens to measure decode speed"
    )
    parser.add_argument(
        "--use-qjl", action="store_true", help="Enable QJL residual sketches"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)
    num_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)

    prompt_text = (
        "The turbo-polar cache compresses key-value tensors for transformer decoding. "
        "In this benchmark we compare dense attention against a fused Metal kernel "
        "that operates directly on compressed polar keys and quantized values. "
        "Large language models rely on key-value caches to avoid recomputing hidden states "
        "for every previously generated token. As context windows grow to thousands of tokens, "
        "these caches become the dominant memory consumer during inference. "
        "Compression methods aim to reduce this footprint while preserving output quality. "
        "Polar quantization represents each key vector in radial coordinates, separating magnitude "
        "from angle. Grouped int8 quantization stores values with per-block scales. "
        "Custom Metal kernels fuse dequantization with attention so the cache never needs to be "
        "expanded back to dense floating-point form on the device."
    )

    print("Running dense baseline...")
    dense_cache = [KVCache() for _ in range(num_layers)]
    dense_logits, tokens = _teacher_forced_logits(
        model, tokenizer, prompt_text, dense_cache, args.max_tokens
    )

    print("Running fused TurboPolar attention...")
    turbo_cache = make_turbo_caches(
        num_layers, num_q_heads, num_kv_heads, head_dim, use_qjl=args.use_qjl
    )
    patch_llama_attention(model)
    try:
        turbo_logits, _ = _teacher_forced_logits(
            model, tokenizer, prompt_text, turbo_cache, args.max_tokens
        )
    finally:
        unpatch_llama_attention(model)

    cosine = _logit_cosine(dense_logits, turbo_logits)
    top1 = _topk_overlap(dense_logits, turbo_logits, k=1)
    top5 = _topk_overlap(dense_logits, turbo_logits, k=5)
    ppl_dense = _perplexity(dense_logits, tokens)
    ppl_turbo = _perplexity(turbo_logits, tokens)
    ppl_delta = abs(ppl_dense - ppl_turbo)

    telem = turbo_cache[0].runtime.get_io_telemetry()
    compression_ratio = telem.get("compression_ratio", 0.0)

    print("Measuring decode speed...")
    dense_cache = [KVCache() for _ in range(num_layers)]
    dense_decode_tok_s = _measure_decode_speed(
        model, tokenizer, dense_cache, tokens, args.num_decode
    )

    turbo_cache = make_turbo_caches(
        num_layers, num_q_heads, num_kv_heads, head_dim, use_qjl=args.use_qjl
    )
    patch_llama_attention(model)
    try:
        turbo_decode_tok_s = _measure_decode_speed(
            model, tokenizer, turbo_cache, tokens, args.num_decode
        )
    finally:
        unpatch_llama_attention(model)

    peak_dense = _peak_kv_bytes_dense(dense_cache)
    peak_turbo = _peak_kv_bytes_turbo(turbo_cache)

    print("\n=== Results ===")
    print(f"Model: {args.model}")
    print(f"MLX: {mx.__version__}, mlx_lm: {mlx_lm.__version__}")
    print(f"use_qjl: {args.use_qjl}")
    print(f"Prompt tokens: {len(tokens)}")
    print(f"Layers / heads / kv_heads / head_dim: {num_layers} / {num_q_heads} / {num_kv_heads} / {head_dim}")
    print(f"Logit cosine similarity: {cosine:.6f}")
    print(f"Top-1 overlap: {top1:.4f}")
    print(f"Top-5 overlap: {top5:.4f}")
    print(f"Dense perplexity: {ppl_dense:.4f}")
    print(f"TurboPolar perplexity: {ppl_turbo:.4f}")
    print(f"Perplexity delta: {ppl_delta:.4f}")
    print(f"KV compression ratio: {compression_ratio:.3f}x")
    print(f"Peak KV bytes dense: {peak_dense:,}")
    print(f"Peak KV bytes turbo: {peak_turbo:,}")
    print(f"Dense decode speed: {dense_decode_tok_s:.2f} tok/s")
    print(f"TurboPolar decode speed: {turbo_decode_tok_s:.2f} tok/s")
    if dense_decode_tok_s > 0:
        print(f"Speed ratio (turbo/dense): {turbo_decode_tok_s / dense_decode_tok_s:.3f}")


if __name__ == "__main__":
    main()
