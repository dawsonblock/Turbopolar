#!/usr/bin/env python3
"""Quick smoke test: append tokens to a TurboPolar cache and print telemetry."""

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

    steps = 200
    for _ in range(steps):
        k = mx.random.normal((1, cfg.num_kv_heads, 1, cfg.head_dim)).astype(mx.float16)
        v = mx.random.normal((1, cfg.num_kv_heads, 1, cfg.head_dim)).astype(mx.float16)
        cache.append(k, v)

    telem = cache.get_io_telemetry()
    print("TurboPolar smoke test passed.")
    print(f"  Total blocks:    {telem['total_blocks']}")
    print(f"  Partial tokens:  {telem['partial_tokens']}")
    print(f"  KV bytes:        {telem['actual_cache_bytes']}")
    print(f"  Dense bytes:     {telem['dense_kv_bytes']}")
    print(f"  Compression:     {telem['compression_ratio']:.2f}x")


if __name__ == "__main__":
    main()
