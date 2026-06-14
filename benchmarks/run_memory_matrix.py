#!/usr/bin/env python3
"""Run full-model memory benchmarks across sequence lengths.

Each length is measured in a fresh Python process for each mode to isolate
peak-memory readings from allocator fragmentation.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def _measure_mode(
    model: str,
    length: int,
    mode: str,
    execution_mode: str,
    seed: int,
    worker: Path,
    forced_decode_count: int = 128,
) -> Dict[str, Any]:
    """Run full_model_memory_worker.py for one mode and length."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        output_path = tmp.name

    cmd = [
        sys.executable,
        str(worker),
        "--model", model,
        "--context-length", str(length),
        "--mode", mode,
        "--forced-decode-count", str(forced_decode_count),
        "--execution-mode", execution_mode,
        "--seed", str(seed),
        "--output", output_path,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"memory_worker failed for length={length} mode={mode}: {result.stderr}"
        )
    with open(output_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="TurboPolar memory matrix benchmark")
    parser.add_argument("--model", required=True, help="MLX model path or HF identifier")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "memory_matrix",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="metal_strict",
        help="Execution mode for TurboPolar (default: metal_strict)",
    )
    parser.add_argument(
        "--forced-decode-count",
        type=int,
        default=128,
        help="Forced decode positions per measurement",
    )
    args = parser.parse_args()

    worker = Path(__file__).parent / "full_model_memory_worker.py"
    records: List[Dict[str, Any]] = []

    print(f"Benchmarking lengths: {args.lengths}")
    for length in sorted(args.lengths):
        t0 = time.perf_counter()
        dense = _measure_mode(
            args.model, length, "dense", args.execution_mode, args.seed, worker,
            forced_decode_count=args.forced_decode_count,
        )
        turbo = _measure_mode(
            args.model, length, "turbopolar_strict", args.execution_mode, args.seed, worker,
            forced_decode_count=args.forced_decode_count,
        )

        # Use separate numerators and denominators so each ratio compares
        # compatible quantities.
        dense_kv_bytes = dense.get("dense_kv_bytes", 0)
        dense_total_peak = dense.get("total_peak_bytes", 0)

        turbo_logical = turbo.get("logical_cache_bytes", 0)
        turbo_allocated = turbo.get("allocated_cache_bytes", 0)
        turbo_total_peak = turbo.get("total_peak_bytes", 0)

        logical_kv_ratio = (
            dense_kv_bytes / turbo_logical if turbo_logical > 0 else 0.0
        )
        persistent_storage_ratio = (
            dense_kv_bytes / turbo_allocated if turbo_allocated > 0 else 0.0
        )
        peak_device_memory_ratio = (
            dense_total_peak / turbo_total_peak if turbo_total_peak > 0 else 0.0
        )

        record = {
            "length": length,
            "dense_kv_bytes": dense_kv_bytes,
            "dense_total_peak_bytes": dense_total_peak,
            "turbo_logical_bytes": turbo_logical,
            "turbo_allocated_bytes": turbo_allocated,
            "turbo_total_peak_bytes": turbo_total_peak,
            "logical_kv_ratio": logical_kv_ratio,
            "persistent_storage_ratio": persistent_storage_ratio,
            "peak_device_memory_ratio": peak_device_memory_ratio,
            "hidden_dense_cache_detected": turbo.get("retained_dense_k_history", False),
            "fallback_count": turbo.get("fallback_count", 0),
        }
        records.append(record)
        elapsed = time.perf_counter() - t0
        print(
            f"  length={length:5d} logical_ratio={record['logical_kv_ratio']:.3f}x "
            f"allocated_ratio={record['persistent_storage_ratio']:.3f}x "
            f"peak_ratio={record['peak_device_memory_ratio']:.3f}x "
            f"({elapsed:.2f}s)"
        )

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "memory_matrix.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {json_path}")


if __name__ == "__main__":
    main()
