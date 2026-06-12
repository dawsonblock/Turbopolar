"""Smoke tests that the installed wheel contains all required runtime assets.

These tests pass in both editable and wheel installs.  CI runs them after
``pip install dist/*.whl`` in a fresh venv.
"""

import sys

import mlx.core as mx
import pytest

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.integrations.mlx_lm.cache import TurboPolarFastCache


class TestInstalledWheel:
    def test_import_rfsn_v11(self):
        import rfsn_v11

        assert hasattr(rfsn_v11, "__version__") or hasattr(rfsn_v11, "__path__")

    def test_import_mlx_lm_integration(self):
        from rfsn_v11.integrations.mlx_lm import (
            TurboPolarFastCache,
            TurboPolarLlamaAdapter,
            TurboPolarLlamaAttention,
            make_turbo_caches,
        )

        assert TurboPolarFastCache is not None
        assert TurboPolarLlamaAdapter is not None
        assert TurboPolarLlamaAttention is not None
        assert make_turbo_caches is not None

    def test_locate_metal_files(self):
        from importlib.resources import files

        kernel_dir = files("rfsn_v11.kernels.turbo_polar")
        metal_files = [p.name for p in kernel_dir.iterdir() if p.name.endswith(".metal")]
        assert "tqpolar_fused_qk.metal" in metal_files
        assert "tqpolar_online_attention.metal" in metal_files

    def test_instantiate_config(self):
        cfg = TurboPolarConfig()
        assert cfg.head_dim == 128
        assert cfg.block_size == 64

    def test_construct_cache(self):
        cfg = TurboPolarConfig(num_q_heads=4, num_kv_heads=2, head_dim=128, block_size=64)
        cache = TurboPolarFastCache(cfg)
        k = mx.random.normal((1, 2, 1, 128)).astype(mx.float16)
        v = mx.random.normal((1, 2, 1, 128)).astype(mx.float16)
        out = cache.decode_attention(
            mx.random.normal((1, 4, 1, 128)).astype(mx.float16),
            k,
            v,
            scale=cfg.attention_scale,
        )
        assert out is not None

    def test_load_metal_source(self):
        from importlib.resources import files

        kernel_dir = files("rfsn_v11.kernels.turbo_polar")
        qk_source = kernel_dir.joinpath("tqpolar_fused_qk.metal").read_text()
        attn_source = kernel_dir.joinpath("tqpolar_online_attention.metal").read_text()
        assert len(qk_source) > 0
        assert len(attn_source) > 0

    @pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
    def test_metal_kernel_compile(self):
        from rfsn_v11.kernels.turbo_polar.metal import MetalKernelBridge

        bridge = MetalKernelBridge()
        stats = bridge.execution_stats()
        assert stats is not None
