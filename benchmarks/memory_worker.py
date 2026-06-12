#!/usr/bin/env python3
"""Isolated subprocess worker for TurboPolar memory measurement.

Reads a JSON config from stdin, runs one append operation, and prints
a JSON result to stdout.  Designed to be invoked by ``run_memory_matrix.py``.

Exit codes:
  0  success
  1  runtime error
  2  invalid input
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime


def main():
    raw = sys.stdin.read()
    if not raw:
        print("No input JSON provided.", file=sys.stderr)
        sys.exit(2)
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        sys.exit(2)

    length = int(cfg.get("length", 0))
    seed = int(cfg.get("seed", 42))
    config_dict = cfg.get("config", {})

    if length <= 0:
        print("length must be positive", file=sys.stderr)
        sys.exit(2)

    mx.random.seed(seed)
    config = TurboPolarConfig(**config_dict)
    runtime = TurboPolarKVCacheRuntime(config)

    B, H, D = 1, config.num_kv_heads, config.head_dim
    k = mx.random.normal((B, H, length, D), dtype=mx.float16)
    v = mx.random.normal((B, H, length, D), dtype=mx.float16)

    mx.reset_peak_memory()
    runtime.append_many(k, v)
    runtime._eval_state()
    peak = int(mx.get_peak_memory())

    stats = runtime.get_memory_stats()
    audit = runtime.audit_cache_residency()

    result = {
        "length": length,
        "peak_device_memory_bytes": peak,
        "logical_payload_bytes": stats.logical_payload_bytes,
        "allocated_capacity_bytes": stats.allocated_capacity_bytes,
        "dense_equivalent_bytes": stats.dense_equivalent_bytes,
        "dense_tail_bytes": stats.dense_tail_bytes,
        "metadata_bytes": stats.metadata_bytes,
        "logical_kv_ratio": stats.logical_compression_ratio,
        "persistent_storage_ratio": stats.allocated_compression_ratio,
        "dense_tail_tokens": audit.dense_tail_tokens,
        "materialized_compressed_history_present": audit.materialized_compressed_history_present,
        "hidden_dense_cache_detected": (
            audit.dense_full_k_history_present or audit.dense_full_v_history_present
        ),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
