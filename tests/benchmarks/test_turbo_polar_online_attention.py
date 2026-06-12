import unittest
import mlx.core as mx
import numpy as np

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.qjl.encoder import QJLResidualEncoder
from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer
from rfsn_v11.kernels.turbo_polar.metal import MetalKernelBridge


class TestTurboPolarOnlineAttention(unittest.TestCase):
    def setUp(self):
        self.config = TurboPolarConfig(
            head_dim=128, qjl_proj_dim=64, block_size=64, split_dim=64,
            num_q_heads=2, num_kv_heads=2, seed=2026,
        )
        self.polar_encoder = PolarQuantEncoder(self.config)
        self.polar_decoder = PolarQuantDecoder()
        self.qjl_encoder = QJLResidualEncoder(self.config)
        self.v_quantizer = GroupedVQuantizer(group_size=32)
        if not mx.metal.is_available():
            self.skipTest("Metal not available.")
        self.bridge = MetalKernelBridge()

    def _encode_unified(self, k_original):
        B, H, T, D = k_original.shape
        S = T // self.config.block_size
        k_blocked = k_original.reshape(B, H, S, self.config.block_size, D)
        blocks = [self.polar_encoder.encode_block(k_blocked[:, :, s, :, :]) for s in range(S)]
        return blocks[0].__class__(
            radii=mx.stack([b.radii for b in blocks], axis=2),
            angle_codes_l1=mx.stack([b.angle_codes_l1 for b in blocks], axis=2),
            angle_codes_deep=mx.stack([b.angle_codes_deep for b in blocks], axis=2),
            shape=(B, H, T, D), block_size=self.config.block_size, head_dim=D,
            metadata=blocks[0].metadata,
        )

    def test_phase_8_dense_v_gate(self):
        B, H, S, L, D = 1, 2, 2, 64, 128
        mx.random.seed(self.config.seed)
        q = mx.random.normal(shape=[B, H, D])
        k_original = mx.random.normal(shape=[B, H, S * L, D])
        v_dense = mx.random.normal(shape=[B, H, S, L, D])
        unified = self._encode_unified(k_original)
        k_recon = self.polar_decoder.decode_block(unified)
        scores = mx.sum(q[:, :, None, :] * k_recon, axis=-1) * self.config.attention_scale
        weights = mx.softmax(scores, axis=-1)
        ref_output = mx.sum(weights[:, :, :, None] * v_dense.reshape(B, H, S * L, D), axis=-2)
        gpu_output, _ = self.bridge.execute_online_attention_dense_v(
            q, unified, v_dense, None, None, self.config, actual_seq_len=S*L, use_qjl=False
        )
        mx.eval(ref_output, gpu_output)
        ref_np = np.array(ref_output)
        gpu_np = np.array(gpu_output)
        cosine = np.dot(ref_np.flatten(), gpu_np.flatten()) / (
            np.linalg.norm(ref_np) * np.linalg.norm(gpu_np) + 1e-12
        )
        max_err = np.max(np.abs(ref_np - gpu_np))
        self.assertGreaterEqual(cosine, 0.999)
        self.assertLessEqual(max_err, 1e-3)

    def test_phase_8_dense_v_with_qjl(self):
        """Test QJL correction in online attention."""
        B, H, S, L, D = 1, 2, 2, 64, 128
        mx.random.seed(self.config.seed)
        q = mx.random.normal(shape=[B, H, D])
        k_original = mx.random.normal(shape=[B, H, S * L, D])
        v_dense = mx.random.normal(shape=[B, H, S, L, D])
        unified = self._encode_unified(k_original)
        k_recon = self.polar_decoder.decode_block(unified)
        residual = k_original - k_recon
        qjl_payload = self.qjl_encoder.compute_residual_sketch(residual.reshape(B, H, S, L, D))
        q_proj = mx.matmul(q, self.qjl_encoder.W)
        q_signs = q_proj >= 0
        reshaped = q_signs.reshape(B, H, self.config.qjl_proj_dim // 8, 8)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        q_packed = mx.sum(reshaped.astype(mx.uint8) * powers, axis=-1).astype(mx.uint8)
        gpu_output, trace = self.bridge.execute_online_attention_dense_v(
            q, unified, v_dense, qjl_payload, q_packed, self.config, actual_seq_len=S*L, use_qjl=True
        )
        mx.eval(gpu_output)
        self.assertTrue(trace["qjl_used"])
        self.assertFalse(np.isnan(np.array(gpu_output)).any())

    def test_phase_9_quant_v_gate(self):
        B, H, S, L, D = 1, 2, 2, 64, 128
        mx.random.seed(self.config.seed)
        q = mx.random.normal(shape=[B, H, D])
        k_original = mx.random.normal(shape=[B, H, S * L, D])
        v_original = mx.random.normal(shape=[B, H, S, L, D])
        quant_v = self.v_quantizer.quantize_block(v_original)
        v_dequant = self.v_quantizer.dequantize_block(quant_v)
        unified = self._encode_unified(k_original)
        k_recon = self.polar_decoder.decode_block(unified)
        scores = mx.sum(q[:, :, None, :] * k_recon, axis=-1) * self.config.attention_scale
        weights = mx.softmax(scores, axis=-1)
        ref_output = mx.sum(weights[:, :, :, None] * v_dequant.reshape(B, H, S * L, D), axis=-2)
        gpu_output, _ = self.bridge.execute_online_attention_quant_v(
            q, unified, quant_v, None, None, self.config, actual_seq_len=S*L, use_qjl=False
        )
        mx.eval(ref_output, gpu_output)
        ref_np = np.array(ref_output)
        gpu_np = np.array(gpu_output)
        cosine = np.dot(ref_np.flatten(), gpu_np.flatten()) / (
            np.linalg.norm(ref_np) * np.linalg.norm(gpu_np) + 1e-12
        )
        max_err = np.max(np.abs(ref_np - gpu_np))
        self.assertGreaterEqual(cosine, 0.999)
        self.assertLessEqual(max_err, 1e-3)

    def test_long_context_float_accumulator_stability(self):
        """Dense-V attention at 4k and 8k should stay close to CPU reference."""
        for T in (4096, 8192):
            with self.subTest(T=T):
                B, H, S, L, D = 1, 2, T // self.config.block_size, self.config.block_size, self.config.head_dim
                mx.random.seed(self.config.seed)
                q = mx.random.normal(shape=[B, H, D])
                k_original = mx.random.normal(shape=[B, H, S * L, D])
                v_dense = mx.random.normal(shape=[B, H, S, L, D])
                unified = self._encode_unified(k_original)
                k_recon = self.polar_decoder.decode_block(unified)
                scores = mx.sum(q[:, :, None, :] * k_recon, axis=-1) * self.config.attention_scale
                weights = mx.softmax(scores, axis=-1)
                ref_output = mx.sum(weights[:, :, :, None] * v_dense.reshape(B, H, S * L, D), axis=-2)
                gpu_output, _ = self.bridge.execute_online_attention_dense_v(
                    q, unified, v_dense, None, None, self.config, actual_seq_len=S * L, use_qjl=False
                )
                mx.eval(ref_output, gpu_output)
                ref_np = np.array(ref_output)
                gpu_np = np.array(gpu_output)
                cosine = np.dot(ref_np.flatten(), gpu_np.flatten()) / (
                    np.linalg.norm(ref_np) * np.linalg.norm(gpu_np) + 1e-12
                )
                max_err = np.max(np.abs(ref_np - gpu_np))
                # Tolerance grows with sqrt(seq_len) because online softmax rescaling
                # accumulates rounding differences.
                allowed_err = 2e-3 * np.sqrt(T / 64.0)
                self.assertGreaterEqual(cosine, 0.999, f"cosine={cosine} at T={T}")
                self.assertLessEqual(max_err, allowed_err, f"max_err={max_err} at T={T}")

    def test_high_quality_config_online_attention(self):
        """Metal online attention supports int8 log-radii + 8-bit deep angles."""
        config = TurboPolarConfig(
            head_dim=128, qjl_proj_dim=64, block_size=64, split_dim=0,
            num_q_heads=4, num_kv_heads=4, seed=2026,
            use_int8_radii=True, k_angle_bits_deep=8,
        )
        polar_encoder = PolarQuantEncoder(config)
        v_quantizer = GroupedVQuantizer(group_size=32)
        decoder = PolarQuantDecoder()
        bridge = MetalKernelBridge()
        B, H, S, L, D = 1, 4, 2, 64, 128
        mx.random.seed(config.seed)
        q = mx.random.normal(shape=[B, H, D])
        k_original = mx.random.normal(shape=[B, H, S * L, D])
        v_original = mx.random.normal(shape=[B, H, S, L, D])
        quant_v = v_quantizer.quantize_block(v_original)
        v_dequant = v_quantizer.dequantize_block(quant_v)
        k_blocked = k_original.reshape(B, H, S, L, D)
        blocks = [polar_encoder.encode_block(k_blocked[:, :, s, :, :]) for s in range(S)]
        unified = blocks[0].__class__(
            radii=mx.stack([b.radii for b in blocks], axis=2),
            angle_codes_l1=mx.stack([b.angle_codes_l1 for b in blocks], axis=2),
            angle_codes_deep=mx.stack([b.angle_codes_deep for b in blocks], axis=2),
            radii_scales=mx.stack([b.radii_scales for b in blocks], axis=2),
            shape=(B, H, S * L, D), block_size=L, head_dim=D,
            metadata=blocks[0].metadata,
        )
        k_recon = decoder.decode_block(unified)
        scores = mx.sum(q[:, :, None, :] * k_recon, axis=-1) * config.attention_scale
        weights = mx.softmax(scores, axis=-1)
        ref_output = mx.sum(weights[:, :, :, None] * v_dequant.reshape(B, H, S * L, D), axis=-2)
        gpu_output, trace = bridge.execute_online_attention_quant_v(
            q, unified, quant_v, None, None, config, actual_seq_len=S*L, use_qjl=False
        )
        mx.eval(ref_output, gpu_output)
        ref_np = np.array(ref_output)
        gpu_np = np.array(gpu_output)
        cosine = np.dot(ref_np.flatten(), gpu_np.flatten()) / (
            np.linalg.norm(ref_np) * np.linalg.norm(gpu_np) + 1e-12
        )
        max_err = np.max(np.abs(ref_np - gpu_np))
        self.assertFalse(np.isnan(gpu_np).any())
        self.assertGreaterEqual(cosine, 0.999)
        self.assertLessEqual(max_err, 1e-3)

    def test_gqa_quant_v_attention(self):
        """Test GQA with quantized V attention."""
        config = TurboPolarConfig(
            head_dim=128, qjl_proj_dim=64, block_size=64, split_dim=64,
            num_q_heads=4, num_kv_heads=2, seed=2026,
        )
        polar_encoder = PolarQuantEncoder(config)
        v_quantizer = GroupedVQuantizer(group_size=32)
        bridge = MetalKernelBridge()
        B, H_q, H_kv, S, L, D = 1, 4, 2, 2, 64, 128
        mx.random.seed(config.seed)
        q = mx.random.normal(shape=[B, H_q, D])
        k_original = mx.random.normal(shape=[B, H_kv, S * L, D])
        v_original = mx.random.normal(shape=[B, H_kv, S, L, D])
        quant_v = v_quantizer.quantize_block(v_original)
        k_blocked = k_original.reshape(B, H_kv, S, L, D)
        blocks = [polar_encoder.encode_block(k_blocked[:, :, s, :, :]) for s in range(S)]
        unified = blocks[0].__class__(
            radii=mx.stack([b.radii for b in blocks], axis=2),
            angle_codes_l1=mx.stack([b.angle_codes_l1 for b in blocks], axis=2),
            angle_codes_deep=mx.stack([b.angle_codes_deep for b in blocks], axis=2),
            shape=(B, H_kv, S * L, D), block_size=L, head_dim=D,
            metadata=blocks[0].metadata,
        )
        gpu_output, trace = bridge.execute_online_attention_quant_v(
            q, unified, quant_v, None, None, config, actual_seq_len=S*L, use_qjl=False
        )
        mx.eval(gpu_output)
        self.assertEqual(gpu_output.shape, (B, H_q, D))
        self.assertEqual(trace["num_queries_per_kv"], 2)
        self.assertFalse(np.isnan(np.array(gpu_output)).any())


if __name__ == "__main__":
    unittest.main()
