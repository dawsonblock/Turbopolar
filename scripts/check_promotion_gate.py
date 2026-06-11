#!/usr/bin/env python3
"""Check whether TurboPolar promotion gates are satisfied (synthetic smoke only)."""

import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime


def check():
    cfg = TurboPolarConfig(num_q_heads=8, num_kv_heads=4, head_dim=128, block_size=64)
    cache = TurboPolarKVCacheRuntime(cfg)
    for _ in range(256):
        k = mx.random.normal((1, cfg.num_kv_heads, 1, cfg.head_dim)).astype(mx.float16)
        v = mx.random.normal((1, cfg.num_kv_heads, 1, cfg.head_dim)).astype(mx.float16)
        cache.append(k, v)

    telem = cache.get_io_telemetry()
    seq_len = telem["total_blocks"] * cfg.block_size + telem["partial_tokens"]
    gates = {
        "kv_reduction_ge_1_7x": telem["compression_ratio"] >= 1.7,
        "no_tail_crash": seq_len == 256,
    }

    print("Promotion gate status (synthetic only):")
    for name, ok in gates.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    if all(gates.values()):
        print("All synthetic gates passed. Real model validation still required before promotion.")
    else:
        print("Some synthetic gates failed.")


if __name__ == "__main__":
    check()
