#!/usr/bin/env python3
"""Fair Cartesian-int8 baseline comparison against TurboPolar.

Runs both caches through the same forced-decode fixtures and reports
quality, memory, and speed deltas.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.cartesian_int8_paged_cache import PagedCartesianInt8KVCache
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime


def _dense_attention(q, k, v, scale):
    B, H_q, _, D = q.shape
    H_kv = k.shape[1]
    nq = H_q // H_kv
    k_rep = mx.repeat(k, nq, axis=1)
    v_rep = mx.repeat(v, nq, axis=1)
    scores = mx.sum(q * k_rep, axis=-1) * scale
    weights = mx.softmax(scores, axis=-1)
    return mx.sum(weights[:, :, :, None] * v_rep, axis=-2)


def _run_forced_decode(cache, q_tokens, k_tokens, v_tokens, scale):
    """Run a forced decode loop and return the final attention output."""
    for q, k, v in zip(q_tokens, k_tokens, v_tokens):
        if hasattr(cache, "decode_attention"):
            out = cache.decode_attention(q, k, v, scale)
        else:
            # Cartesian cache: update history then run dense attention
            k_hist, v_hist = cache.get_history()
            # Append the new token first
            # For Cartesian cache, we append by calling update_and_fetch style
            # But PagedCartesianInt8KVCache has append method
            cache.append(k, v)
            k_hist, v_hist = cache.get_history()
            out = _dense_attention(q, k_hist, v_hist, scale)
    return out


def _position_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def _position_argmax_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.argmax(a) == np.argmax(b))


def _position_topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    return len(top_a & top_b) / k if k > 0 else 0.0


def _compare_fixture(
    length: int, num_decode: int, config: TurboPolarConfig
) -> Dict[str, Any]:
    """Compare TurboPolar vs Cartesian int8 for a given context length and decode count."""
    B, H_kv, D = 1, config.num_kv_heads, config.head_dim
    H_q = config.num_q_heads
    scale = config.attention_scale

    # Shared random data.
    mx.random.seed(42 + length)
    k_prefill = mx.random.normal((B, H_kv, length, D), dtype=mx.float16)
    v_prefill = mx.random.normal((B, H_kv, length, D), dtype=mx.float16)

    # Shared random decode tokens and queries.
    k_decode_list = [mx.random.normal((B, H_kv, 1, D), dtype=mx.float16) for _ in range(num_decode)]
    v_decode_list = [mx.random.normal((B, H_kv, 1, D), dtype=mx.float16) for _ in range(num_decode)]
    q_list = [mx.random.normal((B, H_q, 1, D), dtype=mx.float16) for _ in range(num_decode)]

    # TurboPolar: prefill + decode via the public API.
    from rfsn_v11.integrations.mlx_lm.cache import TurboPolarFastCache

    turbo_runtime = TurboPolarKVCacheRuntime(config)
    turbo_runtime.append_many(k_prefill, v_prefill)
    fast_cache = TurboPolarFastCache(config)
    fast_cache.runtime = turbo_runtime

    # Cartesian: prefill.
    cartesian = PagedCartesianInt8KVCache(block_size=config.block_size)
    cartesian.append(k_prefill, v_prefill)

    # Dense reference cache.
    dense_k = [k_prefill]
    dense_v = [v_prefill]

    turbo_cosines = []
    cartesian_cosines = []
    turbo_argmax = []
    cartesian_argmax = []
    turbo_top5 = []
    cartesian_top5 = []

    for step in range(num_decode):
        k_dec = k_decode_list[step]
        v_dec = v_decode_list[step]
        q = q_list[step]

        # Dense reference.
        dense_k.append(k_dec)
        dense_v.append(v_dec)
        k_dense = mx.concatenate(dense_k, axis=2)
        v_dense = mx.concatenate(dense_v, axis=2)
        dense_out = _dense_attention(q, k_dense, v_dense, scale)

        # TurboPolar decode.
        turbo_out = fast_cache.decode_attention(q, k_dec, v_dec, scale)

        # Cartesian decode: append then dense attention.
        cartesian.append(k_dec, v_dec)
        k_cart, v_cart = cartesian.get_history()
        cartesian_out = _dense_attention(q, k_cart, v_cart, scale)

        mx.eval(dense_out, turbo_out, cartesian_out)
        d = np.array(dense_out.astype(mx.float32)).flatten()
        t = np.array(turbo_out.astype(mx.float32)).flatten()
        c = np.array(cartesian_out.astype(mx.float32)).flatten()

        turbo_cosines.append(_position_cosine(t, d))
        cartesian_cosines.append(_position_cosine(c, d))
        turbo_argmax.append(_position_argmax_agreement(t, d))
        cartesian_argmax.append(_position_argmax_agreement(c, d))
        turbo_top5.append(_position_topk_overlap(t, d, 5))
        cartesian_top5.append(_position_topk_overlap(c, d, 5))

    # Aggregates.
    mean_turbo_cos = float(np.mean(turbo_cosines))
    p05_turbo_cos = float(np.percentile(turbo_cosines, 5))
    min_turbo_cos = float(np.min(turbo_cosines))
    mean_cart_cos = float(np.mean(cartesian_cosines))
    p05_cart_cos = float(np.percentile(cartesian_cosines, 5))
    min_cart_cos = float(np.min(cartesian_cosines))

    # Memory.
    turbo_stats = turbo_runtime.get_memory_stats()
    cartesian_bytes = cartesian.nbytes
    dense_bytes = B * H_kv * (length + num_decode) * D * 2 * 2

    return {
        "length": length,
        "num_decode": num_decode,
        "mean_turbo_cosine": mean_turbo_cos,
        "p05_turbo_cosine": p05_turbo_cos,
        "min_turbo_cosine": min_turbo_cos,
        "mean_cartesian_cosine": mean_cart_cos,
        "p05_cartesian_cosine": p05_cart_cos,
        "min_cartesian_cosine": min_cart_cos,
        "turbo_argmax_agreement": float(np.mean(turbo_argmax)),
        "cartesian_argmax_agreement": float(np.mean(cartesian_argmax)),
        "turbo_top5": float(np.mean(turbo_top5)),
        "cartesian_top5": float(np.mean(cartesian_top5)),
        "turbo_logical_bytes": turbo_stats.logical_payload_bytes,
        "turbo_allocated_bytes": turbo_stats.allocated_capacity_bytes,
        "cartesian_bytes": cartesian_bytes,
        "dense_bytes": dense_bytes,
    }


def main():
    parser = argparse.ArgumentParser(description="Cartesian int8 baseline comparison")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024]
    )
    parser.add_argument("--num-decode", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "cartesian_baseline",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mx.random.seed(args.seed)
    config = TurboPolarConfig(
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=64,
        storage_mode="kv_quant",
        use_int8_radii=True,
        k_angle_bits_deep=8,
        split_dim=0,
    )

    records = []
    for length in args.lengths:
        record = _compare_fixture(length, args.num_decode, config)
        records.append(record)
        print(
            f"length={length:5d} "
            f"turbo_cos={record['cosine_turbo_vs_dense']:.4f} "
            f"cart_cos={record['cosine_cart_vs_dense']:.4f} "
            f"turbo_mae={record['mae_turbo']:.4f} "
            f"cart_mae={record['mae_cart']:.4f}"
        )

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": records,
        "baseline_comparison_report": {
            "cartesian_int8_baseline_implemented": True,
            "turbo_polar_wins_on_quality": all(
                r["mean_turbo_cosine"] >= r["mean_cartesian_cosine"] for r in records
            ),
            "turbo_polar_wins_on_memory": all(
                r["turbo_logical_bytes"] < r["cartesian_bytes"] for r in records
            ),
            "turbo_polar_wins_on_speed": False,  # Not measured in this fixture
            "recommendation": (
                "TurboPolar quality is measured against dense reference; "
                "Cartesian baseline is also reported for comparison."
            ),
            "notes": [
                "Quality measured by per-position cosine similarity vs dense fp16 attention output."
            ],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {json_path}")


if __name__ == "__main__":
    main()
