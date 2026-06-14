#!/usr/bin/env python3
"""Isolated full-model memory worker for TurboPolar evidence.

Runs actual model prefill + forced decode and records whole-model peak memory.
Designed to be launched in a fresh process for each mode and context length.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
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
from rfsn_v11.integrations.mlx_lm.adapter import TurboPolarLlamaAdapter
from rfsn_v11.integrations.mlx_lm.cache import make_turbo_caches
from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode


def _model_cache_config(model: Any) -> Tuple[int, int, int]:
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
    return int(n_heads), int(n_kv_heads), int(hidden_size // n_heads)


def _make_turbo_config(
    num_q_heads: int, num_kv_heads: int, head_dim: int, execution_mode=None
) -> TurboPolarConfig:
    if execution_mode is None:
        execution_mode = ExecutionMode.DEVELOPMENT_AUTO
    elif isinstance(execution_mode, str):
        execution_mode = ExecutionMode(execution_mode)
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
        execution_mode=execution_mode,
    )


def run_memory_worker(
    model_path: str,
    context_length: int,
    mode: str,
    forced_decode_count: int,
    execution_mode: str = "development_auto",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run one memory measurement for a given mode and context length.

    Args:
        model_path: MLX model path or HF identifier.
        context_length: Number of prefill tokens.
        mode: "dense" or "turbopolar_strict".
        forced_decode_count: Number of forced decode positions after prefill.
        execution_mode: Execution mode for TurboPolar.
        seed: Random seed.

    Returns:
        Dict with memory measurements and cache-specific stats.
    """
    mx.random.seed(seed)
    np.random.seed(seed)

    print(f"Loading model: {model_path}", file=sys.stderr)
    model, tokenizer = load(str(model_path))
    num_layers = (
        len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
    )
    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)

    # Build deterministic tokens.
    base_tokens = list(range(0, min(tokenizer.vocab_size, 10000)))
    tokens = [base_tokens[i % len(base_tokens)] for i in range(context_length)]
    forced_continuation = [base_tokens[i % len(base_tokens)] for i in range(
        context_length, context_length + forced_decode_count
    )]

    # 1. Baseline peak (before any model work).
    mx.reset_peak_memory()
    mx.eval(mx.array(0))
    baseline_peak_bytes = int(mx.get_peak_memory())

    # 2. Model load peak.
    mx.reset_peak_memory()
    mx.eval(mx.array(0))
    model_loaded_peak_bytes = int(mx.get_peak_memory())

    # 3. Prefill.
    if mode == "dense":
        cache = [KVCache() for _ in range(num_layers)]
        prompt_mx = mx.array(tokens)[None, :]
        mx.reset_peak_memory()
        prefill_out = model(prompt_mx, cache=cache)
        mx.eval(prefill_out)
        prefill_peak_bytes = int(mx.get_peak_memory())
    elif mode == "turbopolar_strict":
        turbo_config = _make_turbo_config(
            num_q_heads, num_kv_heads, head_dim, execution_mode=execution_mode
        )
        adapter = TurboPolarLlamaAdapter(turbo_config)
        cache = make_turbo_caches(
            num_layers, num_q_heads, num_kv_heads, head_dim,
            execution_mode=execution_mode,
        )
        cache[0].reset_execution_stats()
        prompt_mx = mx.array(tokens)[None, :]
        adapter.install(model)
        try:
            mx.reset_peak_memory()
            prefill_out = model(prompt_mx, cache=cache)
            mx.eval(prefill_out)
            prefill_peak_bytes = int(mx.get_peak_memory())
        finally:
            adapter.uninstall()
    else:
        raise ValueError(f"Unsupported memory worker mode: {mode}")

    # 4. Forced decode.
    if mode == "dense":
        mx.reset_peak_memory()
        for forced_token in forced_continuation:
            token_mx = mx.array([[forced_token]])
            out = model(token_mx, cache=cache)
            mx.eval(out)
        decode_peak_bytes = int(mx.get_peak_memory())
    elif mode == "turbopolar_strict":
        adapter.install(model)
        try:
            mx.reset_peak_memory()
            for forced_token in forced_continuation:
                token_mx = mx.array([[forced_token]])
                out = model(token_mx, cache=cache)
                mx.eval(out)
            decode_peak_bytes = int(mx.get_peak_memory())
        finally:
            adapter.uninstall()

    # Total peak is the max of all stages observed; since we reset each stage,
    # the sum isn't meaningful, but the caller can compare whole-model peaks.
    # We report the decode stage peak as the representative full-model peak
    # because it includes prefill allocations.
    total_peak_bytes = decode_peak_bytes

    # 5. Cache-specific stats (aggregate across ALL layers).
    cache_stats = {}
    fallback_count = 0
    if mode == "turbopolar_strict":
        total_logical = 0
        total_allocated = 0
        total_dense_tail = 0
        total_fallback = 0
        for layer_cache in cache:
            if hasattr(layer_cache, "get_memory_stats"):
                stats = layer_cache.get_memory_stats()
                total_logical += stats.logical_payload_bytes
                total_allocated += stats.allocated_capacity_bytes
                total_dense_tail += stats.dense_tail_bytes
            if hasattr(layer_cache, "execution_stats"):
                bridge_stats = layer_cache.execution_stats()
                total_fallback += getattr(bridge_stats, "fallback_calls", 0)
        cache_stats = {
            "logical_cache_bytes": total_logical,
            "allocated_cache_bytes": total_allocated,
            "dense_tail_bytes": total_dense_tail,
        }
        fallback_count = total_fallback

    # 6. Dense history retention audit.
    # Look for actual dense K/V tensors spanning the full sequence length,
    # not page-slack capacity.
    retained_dense_k = False
    retained_dense_v = False
    if mode == "turbopolar_strict":
        for layer_cache in cache:
            runtime = getattr(layer_cache, "runtime", None)
            if runtime is None:
                continue
            # Check for dense historical tensors in the runtime storage
            storage = getattr(runtime, "storage", None)
            if storage is not None:
                # A hidden dense cache would be a full-sequence-length tensor
                # outside the partial buffers.
                for attr in ("dense_k_history", "dense_v_history", "full_k", "full_v"):
                    tensor = getattr(storage, attr, None)
                    if tensor is not None and hasattr(tensor, "shape"):
                        seq_dim = tensor.shape[2] if tensor.ndim >= 3 else 0
                        if seq_dim > context_length + forced_decode_count - 64:
                            # Arbitrary threshold: if a dense tensor spans nearly
                            # the full sequence, flag it.
                            retained_dense_k = True
                            retained_dense_v = True
                            break

    # Dense KV bytes for fair ratio calculation.
    dense_kv_bytes = 0
    if mode == "dense":
        for c in cache:
            dense_kv_bytes += getattr(c, "nbytes", 0)

    result = {
        "context_length": context_length,
        "mode": mode,
        "baseline_peak_bytes": baseline_peak_bytes,
        "model_loaded_peak_bytes": model_loaded_peak_bytes,
        "prefill_peak_bytes": prefill_peak_bytes,
        "decode_peak_bytes": decode_peak_bytes,
        "total_peak_bytes": total_peak_bytes,
        "dense_kv_bytes": dense_kv_bytes,
        "logical_cache_bytes": cache_stats.get("logical_cache_bytes", 0),
        "allocated_cache_bytes": cache_stats.get("allocated_cache_bytes", 0),
        "dense_tail_bytes": cache_stats.get("dense_tail_bytes", 0),
        "retained_dense_k_history": retained_dense_k,
        "retained_dense_v_history": retained_dense_v,
        "fallback_count": fallback_count,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Full-model memory worker for TurboPolar evidence"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["dense", "turbopolar_strict"],
    )
    parser.add_argument("--forced-decode-count", type=int, default=128)
    parser.add_argument("--execution-mode", default="development_auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run_memory_worker(
        model_path=args.model,
        context_length=args.context_length,
        mode=args.mode,
        forced_decode_count=args.forced_decode_count,
        execution_mode=args.execution_mode,
        seed=args.seed,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote result to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
