#!/usr/bin/env python3
"""Micro-benchmark: cache compression ratio across sequence lengths."""

import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime


def bench(seq_lens=[64, 128, 256, 512, 1024, 2048]):
    cfg = TurboPolarConfig(
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=128,
        block_size=64,
    )

    print(f"{'seq_len':>8} {'dense_mb':>10} {'cache_mb':>10} {'ratio':>8}")
    for seq_len in seq_lens:
        cache = TurboPolarKVCacheRuntime(cfg)
        k = mx.random.normal((1, cfg.num_kv_heads, seq_len, cfg.head_dim)).astype(
            mx.float16
        )
        v = mx.random.normal((1, cfg.num_kv_heads, seq_len, cfg.head_dim)).astype(
            mx.float16
        )
        cache.append(k, v)

        telem = cache.get_io_telemetry()
        dense_mb = telem["dense_kv_bytes"] / (1024 * 1024)
        cache_mb = telem["actual_cache_bytes"] / (1024 * 1024)
        ratio = telem["compression_ratio"]
        print(f"{seq_len:>8} {dense_mb:>10.3f} {cache_mb:>10.3f} {ratio:>8.3f}")


if __name__ == "__main__":
    bench()
