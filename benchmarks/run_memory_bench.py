#!/usr/bin/env python3
"""Memory benchmark for TurboPolar vs dense KV cache.

Measures logical, allocated, and peak-device-memory savings at a range of
sequence lengths using isolated subprocess workers to avoid allocator
cross-contamination between dense and Turbo measurements.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig


def _run_worker(length: int, seed: int, config_dict: dict, mode: str) -> Dict[str, Any]:
    """Run memory_worker.py in a fresh subprocess for one mode."""
    worker = Path(__file__).parent / "memory_worker.py"
    payload = json.dumps({"length": length, "seed": seed, "config": config_dict, "mode": mode})
    result = subprocess.run(
        [sys.executable, str(worker)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"memory_worker {mode} failed: {result.stderr}")
    return json.loads(result.stdout)


def _measure_length(length: int, config: TurboPolarConfig, seed: int) -> Dict[str, Any]:
    config_dict = {
        k: v for k, v in asdict(config).items()
        if not k.startswith("_")
    }

    turbo_result = _run_worker(length, seed, config_dict, "turbo")
    dense_result = _run_worker(length, seed, config_dict, "dense")

    turbo_peak = turbo_result["peak_device_memory_bytes"]
    dense_peak = dense_result["peak_device_memory_bytes"]
    dense_equivalent = turbo_result["dense_equivalent_bytes"]

    return {
        "length": length,
        "logical_kv_ratio": (
            dense_equivalent / turbo_result["logical_payload_bytes"]
            if turbo_result.get("logical_payload_bytes", 0) > 0
            else 0.0
        ),
        "persistent_storage_ratio": (
            dense_equivalent / turbo_result["allocated_capacity_bytes"]
            if turbo_result.get("allocated_capacity_bytes", 0) > 0
            else 0.0
        ),
        "dense_to_turbo_peak_ratio": (
            dense_peak / turbo_peak if turbo_peak > 0 else 0.0
        ),
        "dense_equivalent_bytes": dense_equivalent,
        "logical_payload_bytes": turbo_result.get("logical_payload_bytes", 0),
        "allocated_capacity_bytes": turbo_result.get("allocated_capacity_bytes", 0),
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
        record = _measure_length(length, config, args.seed)
        elapsed = time.perf_counter() - t0
        records.append(record)
        print(
            f"  length={length:5d} logical_ratio={record['logical_kv_ratio']:.3f}x "
            f"allocated_ratio={record['persistent_storage_ratio']:.3f}x "
            f"dense_to_turbo_peak_ratio={record['dense_to_turbo_peak_ratio']:.3f}x "
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
