#!/usr/bin/env python3
"""Synthetic smoke check for the TurboPolar cache. Does not declare promotion."""

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

    print("Synthetic smoke check (not a promotion decision):")
    print(f"  sequence length: {seq_len}")
    print(f"  compression ratio: {telem['compression_ratio']:.3f}x")
    print(f"  total blocks: {telem['total_blocks']}")
    print(f"  partial tokens: {telem['partial_tokens']}")

    if seq_len == 256 and telem["compression_ratio"] >= 1.7:
        print("Smoke check passed. Real-model evidence is required for promotion.")
    else:
        print("Smoke check failed.")


if __name__ == "__main__":
    check()
