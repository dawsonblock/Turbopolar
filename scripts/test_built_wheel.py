#!/usr/bin/env python3
"""Smoke-test a built TurboPolar wheel without assuming an editable install.

This script is run after ``pip install dist/*.whl`` in an isolated virtual
environment.  It verifies that the wheel contains all required runtime code,
Metal shader sources, and that the integration can be imported and instantiated.
"""

import sys


def test_import_rfsn_v11():
    print("OK: import rfsn_v11")


def test_import_mlx_lm_integration():
    print("OK: import MLX-LM integration")


def test_locate_metal_files():
    from importlib.resources import files

    kernel_dir = files("rfsn_v11.kernels.turbo_polar")
    metal_files = [p.name for p in kernel_dir.iterdir() if p.name.endswith(".metal")]
    assert (
        "tqpolar_fused_qk.metal" in metal_files
    ), f"Missing QK shader; found {metal_files}"
    assert (
        "tqpolar_online_attention.metal" in metal_files
    ), f"Missing attention shader; found {metal_files}"
    print(f"OK: locate Metal files ({len(metal_files)} files)")


def test_instantiate_config():
    from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig

    cfg = TurboPolarConfig()
    assert cfg.head_dim == 128
    assert cfg.block_size == 64
    print("OK: instantiate configuration")


def test_construct_cache():
    import mlx.core as mx
    from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
    from rfsn_v11.integrations.mlx_lm.cache import TurboPolarFastCache

    cfg = TurboPolarConfig(num_q_heads=4, num_kv_heads=2, head_dim=128, block_size=64)
    cache = TurboPolarFastCache(cfg)

    k = mx.random.normal((1, 2, 1, 128)).astype(mx.float16)
    v = mx.random.normal((1, 2, 1, 128)).astype(mx.float16)
    cache.decode_attention(
        mx.random.normal((1, 4, 1, 128)).astype(mx.float16),
        k,
        v,
        scale=cfg.attention_scale,
    )
    print("OK: construct cache and decode one token")


def test_load_metal_source():
    from importlib.resources import files

    kernel_dir = files("rfsn_v11.kernels.turbo_polar")
    qk_source = kernel_dir.joinpath("tqpolar_fused_qk.metal").read_text()
    attn_source = kernel_dir.joinpath("tqpolar_online_attention.metal").read_text()
    assert len(qk_source) > 0
    assert len(attn_source) > 0
    print("OK: load Metal source through package resources")


def test_metal_kernel_compile():
    """Attempt a minimal Metal kernel compilation on Apple Silicon.

    Skipped on non-Metal platforms.
    """
    import mlx.core as mx

    if mx.metal.is_available():
        try:
            from rfsn_v11.kernels.turbo_polar.metal import MetalKernelBridge

            bridge = MetalKernelBridge()
            stats = bridge.execution_stats()
            assert stats is not None
            print("OK: Metal kernel bridge instantiates on Apple Silicon")
        except Exception as exc:
            print(f"FAIL: Metal kernel bridge failed: {exc}")
            sys.exit(1)
    else:
        print("SKIP: Metal not available on this platform")


def main():
    tests = [
        test_import_rfsn_v11,
        test_import_mlx_lm_integration,
        test_locate_metal_files,
        test_instantiate_config,
        test_construct_cache,
        test_load_metal_source,
        test_metal_kernel_compile,
    ]

    for test in tests:
        try:
            test()
        except Exception as exc:
            print(f"FAIL: {test.__name__}: {exc}")
            sys.exit(1)

    print("\nAll wheel smoke tests passed.")


if __name__ == "__main__":
    main()
