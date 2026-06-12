#!/usr/bin/env python3
"""Memory benchmark for TurboPolar vs dense KV cache.

Measures logical, allocated, and peak-device-memory savings at a range of
sequence lengths using the truthful accounting exposed by
``TurboPolarKVCacheRuntime``.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime


def _dense_peak_bytes(B: int, H: int, T: int, D: int) -> int:
    """Allocate dense fp16 K+V and return the peak MLX bytes observed."""
    mx.reset_peak_memory()
    k = mx.zeros((B, H, T, D), dtype=mx.float16)
    v = mx.zeros((B, H, T, D), dtype=mx.float16)
    mx.eval(k, v)
    return int(mx.get_peak_memory())


def _measure_length(length: int, config: TurboPolarConfig) -> Dict[str, Any]:
    B, H, D = 1, config.num_kv_heads, config.head_dim
    runtime = TurboPolarKVCacheRuntime(config)

    k = mx.random.normal((B, H, length, D), dtype=mx.float16)
    v = mx.random.normal((B, H, length, D), dtype=mx.float16)

    turbo_peak = runtime.measure_append_peak_memory(k, v)
    stats = runtime.get_memory_stats()
    dense_equivalent = stats.dense_equivalent_bytes

    dense_peak = _dense_peak_bytes(B, H, length, D)

    return {
        "length": length,
        "logical_kv_ratio": (
            dense_equivalent / stats.logical_payload_bytes
            if stats.logical_payload_bytes > 0
            else 0.0
        ),
        "persistent_storage_ratio": (
            dense_equivalent / stats.allocated_capacity_bytes
            if stats.allocated_capacity_bytes > 0
            else 0.0
        ),
        "peak_device_memory_ratio": (
            dense_peak / turbo_peak if turbo_peak > 0 else 0.0
        ),
        "dense_equivalent_bytes": dense_equivalent,
        "logical_payload_bytes": stats.logical_payload_bytes,
        "allocated_capacity_bytes": stats.allocated_capacity_bytes,
        "dense_peak_bytes": dense_peak,
        "turbo_peak_bytes": turbo_peak,
    }


def main():
    parser = argparse.ArgumentParser(description="TurboPolar memory benchmark")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        help="Sequence lengths to measure",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "memory_bench",
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
    print(f"Benchmarking lengths: {args.lengths}")
    for length in sorted(args.lengths):
        t0 = time.perf_counter()
        record = _measure_length(length, config)
        elapsed = time.perf_counter() - t0
        records.append(record)
        print(
            f"  length={length:5d} logical_ratio={record['logical_kv_ratio']:.3f}x "
            f"allocated_ratio={record['persistent_storage_ratio']:.3f}x "
            f"peak_ratio={record['peak_device_memory_ratio']:.3f}x "
            f"({elapsed:.2f}s)"
        )

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": asdict(config),
        "records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {json_path}")


if __name__ == "__main__":
    main()
