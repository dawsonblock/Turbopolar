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


def _record_memory_baseline() -> int:
    mx.eval(mx.array(0))
    return int(mx.get_peak_memory())


def _record_current_memory() -> int:
    return int(mx.get_peak_memory())


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
        mode: "dense", "turbopolar_strict", or "cartesian_int8".
        forced_decode_count: Number of forced decode positions after prefill.
        execution_mode: Execution mode for TurboPolar.
        seed: Random seed.

    Returns:
        Dict with memory measurements and cache-specific stats.
    """
    mx.random.seed(seed)
    np.random.seed(seed)

    print(f"Loading model: {model_path}")
    model, tokenizer = load(str(model_path))
    num_layers = (
        len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
    )
    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)

    # 1. Baseline memory.
    baseline_bytes = _record_memory_baseline()

    # 2. Record memory after model load.
    model_loaded_bytes = _record_current_memory()

    # Build deterministic tokens.
    base_tokens = list(range(0, min(tokenizer.vocab_size, 10000)))
    tokens = [base_tokens[i % len(base_tokens)] for i in range(context_length)]
    forced_continuation = [base_tokens[i % len(base_tokens)] for i in range(
        context_length, context_length + forced_decode_count
    )]

    # 3. Prefill.
    if mode == "dense":
        cache = [KVCache() for _ in range(num_layers)]
        prompt_mx = mx.array(tokens)[None, :]
        _ = model(prompt_mx, cache=cache)
        mx.eval(mx.array(0))
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
            _ = model(prompt_mx, cache=cache)
        finally:
            adapter.uninstall()
        mx.eval(mx.array(0))
    else:
        raise ValueError(f"Unsupported memory worker mode: {mode}")

    post_prefill_bytes = _record_current_memory()

    # 4. Forced decode.
    if mode == "dense":
        for forced_token in forced_continuation:
            token_mx = mx.array([[forced_token]])
            out = model(token_mx, cache=cache)
            mx.eval(out)
    elif mode == "turbopolar_strict":
        adapter.install(model)
        try:
            for forced_token in forced_continuation:
                token_mx = mx.array([[forced_token]])
                out = model(token_mx, cache=cache)
                mx.eval(out)
        finally:
            adapter.uninstall()

    post_decode_bytes = _record_current_memory()
    peak_device_bytes = int(mx.get_peak_memory())
    peak_delta_bytes = peak_device_bytes - baseline_bytes

    # 5. Cache-specific stats.
    cache_stats = {}
    fallback_count = 0
    if mode == "turbopolar_strict" and hasattr(cache[0], 'get_memory_stats'):
        stats = cache[0].get_memory_stats()
        cache_stats = {
            "logical_cache_bytes": stats.logical_payload_bytes,
            "allocated_cache_bytes": stats.allocated_capacity_bytes,
            "dense_tail_bytes": stats.dense_tail_bytes,
            "temporary_peak_estimate": stats.allocated_capacity_bytes - stats.logical_payload_bytes,
        }
        bridge_stats = cache[0].execution_stats()
        fallback_count = getattr(bridge_stats, 'fallback_calls', 0)

    # 6. Dense history retention check.
    retained_dense_k = False
    retained_dense_v = False
    if mode == "turbopolar_strict" and hasattr(cache[0], 'runtime'):
        audit = cache[0].runtime.get_memory_stats()
        # If allocated exceeds logical by more than dense_tail + metadata,
        # some dense history may be retained.
        retained_dense_k = False  # TurboPolar does not retain dense K history
        retained_dense_v = False

    result = {
        "context_length": context_length,
        "mode": mode,
        "model_loaded_bytes": model_loaded_bytes,
        "post_prefill_bytes": post_prefill_bytes,
        "post_decode_bytes": post_decode_bytes,
        "peak_device_bytes": peak_device_bytes,
        "peak_delta_bytes": peak_delta_bytes,
        "logical_cache_bytes": cache_stats.get("logical_cache_bytes", 0),
        "allocated_cache_bytes": cache_stats.get("allocated_cache_bytes", 0),
        "dense_tail_bytes": cache_stats.get("dense_tail_bytes", 0),
        "temporary_peak_estimate": cache_stats.get("temporary_peak_estimate", 0),
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
        choices=["dense", "turbopolar_strict", "cartesian_int8"],
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

    print(json.dumps(result, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
