"""Tests that the online attention kernels scale the QJL correction by attention_scale."""

import unittest

import mlx.core as mx
import numpy as np

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.qjl.encoder import QJLResidualEncoder
from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer
from rfsn_v11.kernels.turbo_polar.metal import MetalKernelBridge


class TestQJLScaledOnlineAttention(unittest.TestCase):
    """CPU and Metal online-attention paths must apply the same QJL scaling."""

    def setUp(self):
        if not mx.metal.is_available():
            self.skipTest("Metal not available.")
        self.config = TurboPolarConfig(
            head_dim=128,
            qjl_proj_dim=64,
            block_size=64,
            split_dim=0,
            num_q_heads=4,
            num_kv_heads=4,
            seed=42,
        )
        self.encoder = PolarQuantEncoder(self.config)
        self.decoder = PolarQuantDecoder()
        self.qjl_encoder = QJLResidualEncoder(self.config)
        self.v_quantizer = GroupedVQuantizer(group_size=32)
        self.bridge = MetalKernelBridge()

    def _encode_unified(self, k_original):
        B, H, T, D = k_original.shape
        S = T // self.config.block_size
        k_blocked = k_original.reshape(B, H, S, self.config.block_size, D)
        blocks = [self.encoder.encode_block(k_blocked[:, :, s, :, :]) for s in range(S)]
        return blocks[0].__class__(
            radii=mx.stack([b.radii for b in blocks], axis=2),
            angle_codes_l1=mx.stack([b.angle_codes_l1 for b in blocks], axis=2),
            angle_codes_deep=mx.stack([b.angle_codes_deep for b in blocks], axis=2),
            shape=(B, H, T, D),
            block_size=self.config.block_size,
            head_dim=D,
            metadata=blocks[0].metadata,
        )

    def _pack_q_signs(self, q_proj):
        B, H, P = q_proj.shape
        signs = q_proj >= 0
        reshaped = signs.reshape(B, H, P // 8, 8)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        return mx.sum(reshaped.astype(mx.uint8) * powers, axis=-1).astype(mx.uint8)

    def test_qjl_score_contribution_scales_via_qk_kernel(self):
        """Per-token QJL score contribution must scale linearly with attention_scale."""
        B, H, S, L, D = 1, 4, 2, 64, 128
        mx.random.seed(self.config.seed)
        k_original = mx.random.normal((B, H, S * L, D))
        q = mx.random.normal((B, H, D))
        unified = self._encode_unified(k_original)
        k_recon = self.decoder.decode_block(unified)
        residual = k_original - k_recon
        qjl_payload = self.qjl_encoder.compute_residual_sketch(
            residual.reshape(B, H, S, L, D)
        )
        q_proj = mx.matmul(q, self.qjl_encoder.W)
        q_packed = self._pack_q_signs(q_proj)

        contributions = []
        for scale in (1.0, 0.5, 0.25):
            cfg = TurboPolarConfig(
                head_dim=128,
                qjl_proj_dim=64,
                block_size=64,
                split_dim=0,
                num_q_heads=4,
                num_kv_heads=4,
                seed=42,
                attention_scale=scale,
            )
            scores_no_qjl = self.bridge.execute_fused_qk(q, unified, cfg)
            scores_qjl = self.bridge.execute_fused_qk_qjl(
                q, unified, qjl_payload, q_packed, cfg
            )
            mx.eval(scores_no_qjl, scores_qjl)
            contrib = float(mx.mean(mx.abs(scores_qjl - scores_no_qjl)).item())
            contributions.append(contrib / scale)

        # Normalized contribution should be constant across scales.
        self.assertAlmostEqual(contributions[0], contributions[1], places=3)
        self.assertAlmostEqual(contributions[1], contributions[2], places=3)

    def test_cpu_metal_online_attention_qjl_scale_match(self):
        """CPU fallback and Metal online attention must agree with scaled QJL."""
        B, H, S, L, D = 1, 4, 2, 64, 128
        mx.random.seed(self.config.seed)
        k_original = mx.random.normal((B, H, S * L, D))
        v_original = mx.random.normal((B, H, S, L, D))
        q = mx.random.normal((B, H, D))
        quant_v = self.v_quantizer.quantize_block(v_original)
        unified = self._encode_unified(k_original)
        k_recon = self.decoder.decode_block(unified)
        residual = k_original - k_recon
        qjl_payload = self.qjl_encoder.compute_residual_sketch(
            residual.reshape(B, H, S, L, D)
        )
        q_proj = mx.matmul(q, self.qjl_encoder.W)
        q_packed = self._pack_q_signs(q_proj)

        cfg = TurboPolarConfig(
            head_dim=128,
            qjl_proj_dim=64,
            block_size=64,
            split_dim=0,
            num_q_heads=4,
            num_kv_heads=4,
            seed=42,
            attention_scale=0.25,
        )

        metal_out, _ = self.bridge.execute_online_attention_quant_v(
            q,
            unified,
            quant_v,
            qjl_payload,
            q_packed,
            cfg,
            actual_seq_len=S * L,
            use_qjl=True,
        )
        cpu_out, _ = self.bridge._cpu_online_attention(
            q,
            unified,
            self.v_quantizer.dequantize_block(quant_v).reshape(B, H, S * L, D),
            qjl_payload,
            q_packed,
            cfg,
            actual_seq_len=S * L,
            use_qjl=True,
            quant_v_used=True,
        )
        mx.eval(metal_out, cpu_out)

        np.testing.assert_allclose(
            np.array(metal_out), np.array(cpu_out), rtol=1e-2, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
