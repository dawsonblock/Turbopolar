#!/usr/bin/env python3
"""Run isolated-subprocess memory benchmarks across sequence lengths.

Each length is measured in a fresh Python process to avoid allocator
fragmentation and ensure peak-memory readings are not polluted by prior
allocations.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig


def _dense_peak_bytes(B: int, H: int, T: int, D: int) -> int:
    import mlx.core as mx

    mx.reset_peak_memory()
    k = mx.zeros((B, H, T, D), dtype=mx.float16)
    v = mx.zeros((B, H, T, D), dtype=mx.float16)
    mx.eval(k, v)
    return int(mx.get_peak_memory())


def _measure_length(
    length: int, config: TurboPolarConfig, seed: int, worker: Path
) -> Dict[str, Any]:
    payload = json.dumps(
        {
            "length": length,
            "seed": seed,
            "config": {
                "num_q_heads": config.num_q_heads,
                "num_kv_heads": config.num_kv_heads,
                "head_dim": config.head_dim,
                "block_size": config.block_size,
                "storage_mode": config.storage_mode,
                "use_int8_radii": config.use_int8_radii,
                "k_angle_bits_level1": config.k_angle_bits_level1,
                "k_angle_bits_deep": config.k_angle_bits_deep,
                "split_dim": config.split_dim,
            },
        }
    )
    result = subprocess.run(
        [sys.executable, str(worker)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"memory_worker failed for length={length}: {result.stderr}")
    turbo = json.loads(result.stdout)

    B, H, D = 1, config.num_kv_heads, config.head_dim
    dense_peak = _dense_peak_bytes(B, H, length, D)

    turbo["dense_peak_bytes"] = dense_peak
    turbo["peak_device_memory_ratio"] = (
        dense_peak / turbo["peak_device_memory_bytes"]
        if turbo["peak_device_memory_bytes"] > 0
        else 0.0
    )
    return turbo


def main():
    parser = argparse.ArgumentParser(description="TurboPolar memory matrix benchmark")
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
        default=Path(__file__).parent / "outputs" / "memory_matrix",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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

    worker = Path(__file__).parent / "memory_worker.py"
    records: List[Dict[str, Any]] = []

    print(f"Benchmarking lengths: {args.lengths}")
    for length in sorted(args.lengths):
        t0 = time.perf_counter()
        record = _measure_length(length, config, args.seed, worker)
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
        "config": {
            "num_q_heads": config.num_q_heads,
            "num_kv_heads": config.num_kv_heads,
            "head_dim": config.head_dim,
            "block_size": config.block_size,
        },
        "records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "memory_matrix.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {json_path}")


if __name__ == "__main__":
    main()
