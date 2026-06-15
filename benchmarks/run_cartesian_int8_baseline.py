#!/usr/bin/env python3
"""Fair dense-vs-Cartesian-vs-TurboPolar baseline comparison.

Runs all three caches through the same forced-decode fixtures on a real model and
reports quality, memory, and speed deltas.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.cartesian_int8_paged_cache import PagedCartesianInt8KVCache
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


def _compare_fixture(
    model,
    tokenizer,
    length: int,
    num_decode: int,
    execution_mode: str,
    seed: int,
) -> Dict[str, Any]:
    """Compare dense, Cartesian-int8, and TurboPolar for one context length."""
    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)
    num_layers = (
        len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
    )

    # Deterministic tokens.
    np.random.seed(seed + length)
    base_tokens = list(range(0, min(tokenizer.vocab_size, 10000)))
    tokens = [base_tokens[i % len(base_tokens)] for i in range(length)]
    forced_continuation = [base_tokens[i % len(base_tokens)] for i in range(
        length, length + num_decode
    )]

    prompt_mx = mx.array(tokens)[None, :]

    # Dense path.
    dense_cache = [KVCache() for _ in range(num_layers)]
    dense_prefill = model(prompt_mx, cache=dense_cache)
    mx.eval(dense_prefill)

    dense_logits = []
    for forced_token in forced_continuation:
        token_mx = mx.array([[forced_token]])
        logits = model(token_mx, cache=dense_cache)
        mx.eval(logits)
        dense_logits.append(np.array(logits[:, -1, :].astype(mx.float32)).flatten())

    # Cartesian-int8 path.
    cartesian_cache = [PagedCartesianInt8KVCache(block_size=64) for _ in range(num_layers)]
    cartesian_prefill = model(prompt_mx, cache=cartesian_cache)
    mx.eval(cartesian_prefill)

    cartesian_logits = []
    for forced_token in forced_continuation:
        token_mx = mx.array([[forced_token]])
        logits = model(token_mx, cache=cartesian_cache)
        mx.eval(logits)
        cartesian_logits.append(np.array(logits[:, -1, :].astype(mx.float32)).flatten())

    # TurboPolar path.
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
        execution_mode=ExecutionMode(execution_mode),
    )
    adapter = TurboPolarLlamaAdapter(turbo_config)
    turbo_cache = make_turbo_caches(
        num_layers, num_q_heads, num_kv_heads, head_dim,
        execution_mode=ExecutionMode(execution_mode),
    )
    turbo_cache[0].reset_execution_stats()

    adapter.install(model)
    try:
        turbo_prefill = model(prompt_mx, cache=turbo_cache)
        mx.eval(turbo_prefill)

        turbo_logits = []
        for forced_token in forced_continuation:
            token_mx = mx.array([[forced_token]])
            logits = model(token_mx, cache=turbo_cache)
            mx.eval(logits)
            turbo_logits.append(np.array(logits[:, -1, :].astype(mx.float32)).flatten())
    finally:
        adapter.uninstall()

    # Quality metrics: Compare each candidate against dense.
    # TurboPolar vs dense.
    cosines_turbo_vs_dense = []
    argmax_agreements_turbo_vs_dense = []
    top5s_turbo_vs_dense = []
    for d, t in zip(dense_logits, turbo_logits):
        cosines_turbo_vs_dense.append(_logit_cosine(d, t))
        argmax_agreements_turbo_vs_dense.append(float(np.argmax(d) == np.argmax(t)))
        top5s_turbo_vs_dense.append(_topk_overlap(d, t, 5))

    # Cartesian vs dense.
    cosines_cartesian_vs_dense = []
    argmax_agreements_cartesian_vs_dense = []
    top5s_cartesian_vs_dense = []
    for d, c in zip(dense_logits, cartesian_logits):
        cosines_cartesian_vs_dense.append(_logit_cosine(d, c))
        argmax_agreements_cartesian_vs_dense.append(float(np.argmax(d) == np.argmax(c)))
        top5s_cartesian_vs_dense.append(_topk_overlap(d, c, 5))

    # Memory: report logical and allocated separately.
    dense_bytes = 0
    for c in dense_cache:
        dense_bytes += getattr(c, "nbytes", 0)

    cartesian_logical = 0
    cartesian_allocated = 0
    for c in cartesian_cache:
        cartesian_logical += getattr(c, "nbytes", 0)
        cartesian_allocated += getattr(c, "allocated_bytes", 0)

    turbo_logical = 0
    turbo_allocated = 0
    for c in turbo_cache:
        if hasattr(c, "get_memory_stats"):
            stats = c.get_memory_stats()
            turbo_logical += stats.logical_payload_bytes
            turbo_allocated += stats.allocated_capacity_bytes

    # Speed: quick 16-token measurement for all three.
    import time as time_mod
    mx.random.seed(seed)
    np.random.seed(seed)
    warm_tokens = [base_tokens[i % len(base_tokens)] for i in range(length, length + 16)]

    # Dense speed.
    dense_cache_s = [KVCache() for _ in range(num_layers)]
    _ = model(prompt_mx, cache=dense_cache_s)
    mx.eval(_)
    t0 = time_mod.perf_counter()
    for tok in warm_tokens:
        out = model(mx.array([[tok]]), cache=dense_cache_s)
        mx.eval(out)
    dense_time = time_mod.perf_counter() - t0

    # Cartesian speed.
    cartesian_cache_s = [PagedCartesianInt8KVCache(block_size=64) for _ in range(num_layers)]
    _ = model(prompt_mx, cache=cartesian_cache_s)
    mx.eval(_)
    t0 = time_mod.perf_counter()
    for tok in warm_tokens:
        out = model(mx.array([[tok]]), cache=cartesian_cache_s)
        mx.eval(out)
    cartesian_time = time_mod.perf_counter() - t0

    # Turbo speed.
    turbo_cache_s = make_turbo_caches(
        num_layers, num_q_heads, num_kv_heads, head_dim,
        execution_mode=ExecutionMode(execution_mode),
    )
    adapter.install(model)
    try:
        _ = model(prompt_mx, cache=turbo_cache_s)
        mx.eval(_)
        t0 = time_mod.perf_counter()
        for tok in warm_tokens:
            out = model(mx.array([[tok]]), cache=turbo_cache_s)
            mx.eval(out)
        turbo_time = time_mod.perf_counter() - t0
    finally:
        adapter.uninstall()

    speedup_vs_dense = dense_time / turbo_time if turbo_time > 0 else 0.0
    speedup_vs_cartesian = cartesian_time / turbo_time if turbo_time > 0 else 0.0

    return {
        "length": length,
        "num_decode": num_decode,
        "turbo_cosine_vs_dense": float(np.mean(cosines_turbo_vs_dense)),
        "turbo_argmax_agreement_vs_dense": float(np.mean(argmax_agreements_turbo_vs_dense)),
        "turbo_top5_vs_dense": float(np.mean(top5s_turbo_vs_dense)),
        "cartesian_cosine_vs_dense": float(np.mean(cosines_cartesian_vs_dense)),
        "cartesian_argmax_agreement_vs_dense": float(np.mean(argmax_agreements_cartesian_vs_dense)),
        "cartesian_top5_vs_dense": float(np.mean(top5s_cartesian_vs_dense)),
        "dense_bytes": dense_bytes,
        "cartesian_logical_bytes": cartesian_logical,
        "cartesian_allocated_bytes": cartesian_allocated,
        "turbo_logical_bytes": turbo_logical,
        "turbo_allocated_bytes": turbo_allocated,
        "dense_time_16tok": dense_time,
        "cartesian_time_16tok": cartesian_time,
        "turbo_time_16tok": turbo_time,
        "speedup_vs_dense_16tok": speedup_vs_dense,
        "speedup_vs_cartesian_16tok": speedup_vs_cartesian,
    }


def main():
    parser = argparse.ArgumentParser(description="Dense vs Cartesian vs TurboPolar baseline comparison")
    parser.add_argument("--model", required=True, help="MLX model path or HF identifier")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    )
    parser.add_argument("--num-decode", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "cartesian_baseline",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="metal_strict",
        choices=["reference", "metal_strict", "development_auto"],
    )
    args = parser.parse_args()

    mx.random.seed(args.seed)
    print(f"Loading model: {args.model}")
    model, tokenizer = load(str(args.model))

    records = []
    cartesian_implemented = True
    for length in args.lengths:
        try:
            record = _compare_fixture(
                model, tokenizer, length, args.num_decode, args.execution_mode, args.seed
            )
            records.append(record)
            print(
                f"length={length:5d} "
                f"turbo_cos(dense)={record['turbo_cosine_vs_dense']:.4f} "
                f"cart_cos(dense)={record['cartesian_cosine_vs_dense']:.4f} "
                f"turbo_argmax(dense)={record['turbo_argmax_agreement_vs_dense']:.4f} "
                f"cart_argmax(dense)={record['cartesian_argmax_agreement_vs_dense']:.4f} "
                f"mem_dense={record['dense_bytes']} "
                f"mem_cart_alloc={record['cartesian_allocated_bytes']} "
                f"mem_turbo_alloc={record['turbo_allocated_bytes']} "
                f"speedup(dense)={record['speedup_vs_dense_16tok']:.2f}x "
                f"speedup(cart)={record['speedup_vs_cartesian_16tok']:.2f}x"
            )
        except Exception as exc:
            print(f"length={length:5d} FAILED: {exc}")
            cartesian_implemented = False

    # Winner decisions: TurboPolar wins quality if it is closer to dense than Cartesian is.
    # Compare both candidates against dense.
    if cartesian_implemented and records:
        wins_quality = all(
            r["turbo_cosine_vs_dense"] >= r["cartesian_cosine_vs_dense"] for r in records
        )
        wins_memory = all(
            r["turbo_allocated_bytes"] < r["cartesian_allocated_bytes"] for r in records
        )
        wins_speed = all(
            r["speedup_vs_cartesian_16tok"] > 1.0 for r in records
        )
        recommendation = (
            "TurboPolar shows competitive quality and memory savings vs Cartesian int8."
            if wins_quality and wins_memory else
            "Further tuning required."
        )
    else:
        wins_quality = None
        wins_memory = None
        wins_speed = None
        recommendation = "Cartesian int8 baseline did not complete successfully."

    # Capture architecture info for equivalence verification
    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)
    num_layers = (
        len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
    )
    
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "records": records,
        "contexts_evaluated": [r["length"] for r in records],
        # P1-34: Architecture equivalence metadata for fair comparison
        "architecture_equivalence": {
            "num_layers": num_layers,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "block_size": 64,  # All implementations use 64-token blocks
            "execution_mode": args.execution_mode,
            "comparison_basis": "all_implementations_same_model_same_config",
            "dense_implementation": "mlx_lm KVCache (baseline)",
            "cartesian_implementation": "PagedCartesianInt8KVCache",
            "turbopolar_implementation": "TurboPolar with config matching model",
            "equivalence_verification": {
                "same_model_weights": True,
                "same_token_sequences": True,  # All use same forced tokens
                "same_attention_pattern": "causal full attention",
                "same_quantization_target": "int8 for Cartesian, configurable for TurboPolar",
            },
        },
        "baseline_comparison_report": {
            "cartesian_int8_baseline_implemented": cartesian_implemented,
            "turbo_polar_wins_on_quality": wins_quality,
            "turbo_polar_wins_on_memory": wins_memory,
            "turbo_polar_wins_on_speed": wins_speed,
            "recommendation": recommendation,
            "notes": [
                f"Execution mode: {args.execution_mode}",
                f"Contexts: {[r['length'] for r in records]}",
                f"Architecture: {num_layers} layers, {num_q_heads} Q heads, {num_kv_heads} KV heads, {head_dim} head dim",
            ],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {json_path}")


if __name__ == "__main__":
    main()
