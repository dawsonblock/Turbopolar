#!/usr/bin/env python3
"""Isolated full-model memory worker for TurboPolar evidence.

Runs actual model prefill + forced decode and records whole-model peak memory.
Designed to be launched in a fresh process for each mode and context length.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def _get_tokens_from_fixtures(
    tokenizer,
    token_fixtures_path: Optional[Path],
    fixture_category: Optional[str],
    context_length: int,
) -> tuple:
    """Load tokens from canonical fixtures if available.
    
    Returns:
        Tuple of (tokens_list, fixture_id, fixture_hash) or (None, None, None) if not available.
    """
    if token_fixtures_path is None or not token_fixtures_path.exists():
        return None, None, None
    
    try:
        from benchmarks.prompt_fixtures import load_token_fixtures_canonical
        fixtures = load_token_fixtures_canonical(token_fixtures_path)
        
        # Find fixture matching criteria
        for fixture in fixtures:
            if fixture_category and fixture.category == fixture_category:
                if fixture.length == context_length:
                    return list(fixture.tokens), fixture.fixture_id, fixture.content_hash
            elif not fixture_category and fixture.length == context_length:
                return list(fixture.tokens), fixture.fixture_id, fixture.content_hash
        
        return None, None, None
    except Exception:
        return None, None, None


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
    token_fixtures_path: Optional[Path] = None,
    fixture_category: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one memory measurement for a given mode and context length.

    Args:
        model_path: MLX model path or HF identifier.
        context_length: Number of prefill tokens.
        mode: "dense" or "turbopolar_strict".
        forced_decode_count: Number of forced decode positions after prefill.
        execution_mode: Execution mode for TurboPolar.
        seed: Random seed.
        token_fixtures_path: Optional path to canonical token fixtures JSONL.
        fixture_category: Optional category to select from fixtures (e.g., "short", "medium").
            If None and token_fixtures_path is provided, selects fixture matching context_length.

    Returns:
        Dict with memory measurements and cache-specific stats.
    """
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache

    mx.random.seed(seed)
    np.random.seed(seed)

    # Whole-run peak measurement: reset once at start, never again.
    # This captures the true peak across all stages (model load, prefill, decode).
    mx.reset_peak_memory()
    mx.eval(mx.array(0))

    # 1. Model load.
    print(f"Loading model: {model_path}", file=sys.stderr)
    model, tokenizer = load(str(model_path))
    # Evaluate the model to ensure parameters are materialized on device.
    mx.eval(model)

    num_layers = (
        len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
    )
    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)

    # Build deterministic tokens from canonical fixtures if available,
    # otherwise fall back to deterministic sequence.
    tokens, fixture_id, fixture_hash = _get_tokens_from_fixtures(
        tokenizer, token_fixtures_path, fixture_category, context_length
    )
    if tokens is None:
        # Fall back to deterministic sequence
        base_tokens = list(range(0, min(tokenizer.vocab_size, 10000)))
        tokens = [base_tokens[i % len(base_tokens)] for i in range(context_length)]
        fixture_id = None
        fixture_hash = None
    
    # Build continuation tokens (not from fixtures, as these are generated)
    base_tokens = list(range(0, min(tokenizer.vocab_size, 10000)))
    forced_continuation = [base_tokens[i % len(base_tokens)] for i in range(
        context_length, context_length + forced_decode_count
    )]

    # 2. Prefill.
    if mode == "dense":
        cache = [KVCache() for _ in range(num_layers)]
        prompt_mx = mx.array(tokens)[None, :]
        prefill_out = model(prompt_mx, cache=cache)
        mx.eval(prefill_out)
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
            prefill_out = model(prompt_mx, cache=cache)
            mx.eval(prefill_out)
        finally:
            adapter.uninstall()
    else:
        raise ValueError(f"Unsupported memory worker mode: {mode}")

    # 3. Forced decode.
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

    # Capture whole-run peak (never reset between stages).
    whole_run_peak_bytes = int(mx.get_peak_memory())

    # 5. Cache-specific stats (aggregate across ALL layers).
    cache_stats = {}
    fallback_count = 0
    if mode == "turbopolar_strict":
        total_logical = 0
        total_allocated = 0
        total_dense_tail = 0
        for layer_cache in cache:
            if hasattr(layer_cache, "get_memory_stats"):
                stats = layer_cache.get_memory_stats()
                total_logical += stats.logical_payload_bytes
                total_allocated += stats.allocated_capacity_bytes
                total_dense_tail += stats.dense_tail_bytes
        cache_stats = {
            "logical_cache_bytes": total_logical,
            "allocated_cache_bytes": total_allocated,
            "dense_tail_bytes": total_dense_tail,
        }
        # Read singleton bridge statistics once, not per layer.
        fallback_count = getattr(
            cache[0].execution_stats() if hasattr(cache[0], "execution_stats") else {},
            "fallback_calls",
            0,
        )

    # 6. Dense history retention audit.
    # Check runtime for dense arrays that could indicate improper full-sequence retention.
    # TurboPolar should only keep block_size (64) tokens in dense partial buffers.
    retained_dense_k = False
    retained_dense_v = False
    if mode == "turbopolar_strict":
        for layer_cache in cache:
            runtime = getattr(layer_cache, "runtime", None)
            if runtime is None:
                continue
            # Check partial buffers - they should only hold up to block_size (64) tokens
            partial_k = getattr(runtime, "partial_k_buffer", None)
            partial_v = getattr(runtime, "partial_v_buffer", None)
            
            # Check K partial buffer
            if partial_k is not None and hasattr(partial_k, "shape"):
                if partial_k.ndim >= 3:
                    seq_dim = partial_k.shape[2] if partial_k.ndim >= 3 else 0
                    # Partial buffer should not exceed block_size (64)
                    if seq_dim > 64:
                        retained_dense_k = True
            
            # Check V partial buffer  
            if partial_v is not None and hasattr(partial_v, "shape"):
                if partial_v.ndim >= 3:
                    seq_dim = partial_v.shape[2] if partial_v.ndim >= 3 else 0
                    # Partial buffer should not exceed block_size (64)
                    if seq_dim > 64:
                        retained_dense_v = True

    # Dense KV bytes for fair ratio calculation.
    dense_kv_bytes = 0
    if mode == "dense":
        for c in cache:
            dense_kv_bytes += getattr(c, "nbytes", 0)

    result = {
        "context_length": context_length,
        "mode": mode,
        # Whole-run peak: measured once across all stages (model load, prefill, decode)
        # This captures true peak memory usage across the entire benchmark run.
        "whole_run_peak_bytes": whole_run_peak_bytes,
        # Per-stage peaks are now measured from the same continuous run for accuracy.
        # They represent the peak at each stage boundary, not independent measurements.
        "dense_kv_bytes": dense_kv_bytes,
        "logical_cache_bytes": cache_stats.get("logical_cache_bytes", 0),
        "allocated_cache_bytes": cache_stats.get("allocated_cache_bytes", 0),
        "dense_tail_bytes": cache_stats.get("dense_tail_bytes", 0),
        "retained_dense_k_history": retained_dense_k,
        "retained_dense_v_history": retained_dense_v,
        "fallback_count": fallback_count,
        # Backward compatibility: total_peak_bytes now equals whole_run_peak_bytes
        "total_peak_bytes": whole_run_peak_bytes,
        # Fixture provenance for reproducibility
        "fixture_id": fixture_id,
        "fixture_hash": fixture_hash,
        "token_fixtures_path": str(token_fixtures_path) if token_fixtures_path else None,
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
    parser.add_argument(
        "--token-fixtures",
        type=Path,
        default=None,
        help="Path to canonical token fixtures JSONL file"
    )
    parser.add_argument(
        "--fixture-category",
        type=str,
        default=None,
        help="Category of fixture to use (e.g., short, medium, long)"
    )
    args = parser.parse_args()

    result = run_memory_worker(
        model_path=args.model,
        context_length=args.context_length,
        mode=args.mode,
        forced_decode_count=args.forced_decode_count,
        execution_mode=args.execution_mode,
        seed=args.seed,
        token_fixtures_path=args.token_fixtures,
        fixture_category=args.fixture_category,
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
