import unittest
import mlx.core as mx
import numpy as np

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.qjl.encoder import QJLResidualEncoder
from rfsn_v11.kernels.turbo_polar.metal import MetalKernelBridge


class TestTurboPolarKernelQK(unittest.TestCase):
    def setUp(self):
        self.config = TurboPolarConfig(
            head_dim=128,
            qjl_proj_dim=64,
            block_size=64,
            split_dim=64,
            num_q_heads=4,
            num_kv_heads=4,
            seed=42,
        )
        self.encoder = PolarQuantEncoder(self.config)
        self.decoder = PolarQuantDecoder()
        self.qjl_encoder = QJLResidualEncoder(self.config)
        if not mx.metal.is_available():
            self.skipTest("Metal not available.")
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

    def test_phase_6_fused_qk_precision(self):
        B, H, S, L, D = 1, 4, 2, 64, 128
        mx.random.seed(self.config.seed)
        k_original = mx.random.normal(shape=[B, H, S * L, D])
        q = mx.random.normal(shape=[B, H, D])
        unified = self._encode_unified(k_original)
        k_recon = self.decoder.decode_block(unified)
        ref_scores = (
            mx.sum(q[:, :, None, :] * k_recon, axis=-1) * self.config.attention_scale
        )
        gpu_scores = self.bridge.execute_fused_qk(q, unified, self.config)
        mx.eval(ref_scores, gpu_scores)
        ref_np = np.array(ref_scores)
        gpu_np = np.array(gpu_scores)
        max_error = np.max(np.abs(ref_np - gpu_np))
        cosine = np.dot(ref_np.flatten(), gpu_np.flatten()) / (
            np.linalg.norm(ref_np) * np.linalg.norm(gpu_np) + 1e-12
        )
        # The Metal kernel accumulates in fp16; allow a slightly larger absolute
        # tolerance than the dense fp32 reference while still requiring high
        # cosine similarity.
        self.assertLessEqual(max_error, 1e-2)
        self.assertGreaterEqual(cosine, 0.999)

    def test_phase_7_qjl_fused_qk(self):
        B, H, S, L, D = 1, 2, 1, 64, 128
        mx.random.seed(self.config.seed)
        k_original = mx.random.normal(shape=[B, H, S * L, D])
        q = mx.random.normal(shape=[B, H, D])
        unified = self._encode_unified(k_original)
        k_recon = self.decoder.decode_block(unified)
        residual_E = k_original - k_recon
        qjl_payload = self.qjl_encoder.compute_residual_sketch(
            residual_E.reshape(B, H, S, L, D)
        )
        q_proj = mx.matmul(q, self.qjl_encoder.W)
        q_signs = q_proj >= 0
        reshaped = q_signs.reshape(B, H, self.config.qjl_proj_dim // 8, 8)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        q_packed = mx.sum(reshaped.astype(mx.uint8) * powers, axis=-1).astype(mx.uint8)
        gpu_scores = self.bridge.execute_fused_qk_qjl(
            q, unified, qjl_payload, q_packed, self.config
        )
        mx.eval(gpu_scores)
        self.assertFalse(np.isnan(np.array(gpu_scores)).any())

    def test_high_quality_config_fused_qk(self):
        """Metal path supports int8 log-radii + 8-bit deep angles + split_dim=0."""
        config = TurboPolarConfig(
            head_dim=128,
            qjl_proj_dim=64,
            block_size=64,
            split_dim=0,
            num_q_heads=4,
            num_kv_heads=4,
            seed=42,
            use_int8_radii=True,
            k_angle_bits_deep=8,
        )
        encoder = PolarQuantEncoder(config)
        decoder = PolarQuantDecoder()
        bridge = MetalKernelBridge()
        B, H, S, L, D = 1, 4, 2, 64, 128
        mx.random.seed(config.seed)
        k_original = mx.random.normal(shape=[B, H, S * L, D])
        q = mx.random.normal(shape=[B, H, D])
        k_blocked = k_original.reshape(B, H, S, L, D)
        blocks = [encoder.encode_block(k_blocked[:, :, s, :, :]) for s in range(S)]
        unified = blocks[0].__class__(
            radii=mx.stack([b.radii for b in blocks], axis=2),
            angle_codes_l1=mx.stack([b.angle_codes_l1 for b in blocks], axis=2),
            angle_codes_deep=mx.stack([b.angle_codes_deep for b in blocks], axis=2),
            radii_scales=mx.stack([b.radii_scales for b in blocks], axis=2),
            shape=(B, H, S * L, D),
            block_size=L,
            head_dim=D,
            metadata=blocks[0].metadata,
        )
        k_recon = decoder.decode_block(unified)
        ref_scores = (
            mx.sum(q[:, :, None, :] * k_recon, axis=-1) * config.attention_scale
        )
        gpu_scores = bridge.execute_fused_qk(q, unified, config)
        mx.eval(ref_scores, gpu_scores)
        ref_np = np.array(ref_scores)
        gpu_np = np.array(gpu_scores)
        max_error = np.max(np.abs(ref_np - gpu_np))
        cosine = np.dot(ref_np.flatten(), gpu_np.flatten()) / (
            np.linalg.norm(ref_np) * np.linalg.norm(gpu_np) + 1e-12
        )
        self.assertFalse(np.isnan(gpu_np).any())
        self.assertLessEqual(max_error, 1e-2)
        self.assertGreaterEqual(cosine, 0.999)

    def test_gqa_fused_qk(self):
        """Test GQA: 4 query heads, 2 KV heads."""
        config = TurboPolarConfig(
            head_dim=128,
            qjl_proj_dim=64,
            block_size=64,
            split_dim=64,
            num_q_heads=4,
            num_kv_heads=2,
            seed=42,
        )
        encoder = PolarQuantEncoder(config)
        B, H_q, H_kv, S, L, D = 1, 4, 2, 2, 64, 128
        mx.random.seed(config.seed)
        k_original = mx.random.normal(shape=[B, H_kv, S * L, D])
        q = mx.random.normal(shape=[B, H_q, D])
        # Encode at KV head resolution
        k_blocked = k_original.reshape(B, H_kv, S, L, D)
        blocks = [encoder.encode_block(k_blocked[:, :, s, :, :]) for s in range(S)]
        unified = blocks[0].__class__(
            radii=mx.stack([b.radii for b in blocks], axis=2),
            angle_codes_l1=mx.stack([b.angle_codes_l1 for b in blocks], axis=2),
            angle_codes_deep=mx.stack([b.angle_codes_deep for b in blocks], axis=2),
            shape=(B, H_kv, S * L, D),
            block_size=L,
            head_dim=D,
            metadata=blocks[0].metadata,
        )
        # Bridge should handle GQA via num_queries_per_kv
        bridge = MetalKernelBridge()
        gpu_scores = bridge.execute_fused_qk(q, unified, config)
        mx.eval(gpu_scores)
        self.assertEqual(gpu_scores.shape, (B, H_q, S * L))
        self.assertFalse(np.isnan(np.array(gpu_scores)).any())


if __name__ == "__main__":
    unittest.main()
