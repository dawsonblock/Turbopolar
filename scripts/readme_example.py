#!/usr/bin/env python3
"""Verify the README quickstart example runs and produces sane telemetry."""

import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime


def main():
    cfg = TurboPolarConfig(
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=128,
        block_size=64,
    )
    cache = TurboPolarKVCacheRuntime(cfg)

    for _ in range(100):
        k = mx.random.normal((1, cfg.num_kv_heads, 1, cfg.head_dim)).astype(mx.float16)
        v = mx.random.normal((1, cfg.num_kv_heads, 1, cfg.head_dim)).astype(mx.float16)
        cache.append(k, v)

    telem = cache.get_io_telemetry()
    assert telem["compression_ratio"] > 1.0, telem
    print("README example OK")


if __name__ == "__main__":
    main()
