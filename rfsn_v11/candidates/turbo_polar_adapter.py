import mlx.core as mx
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.candidates.turbo_polar_metrics import (
    mean_token_kl,
    topk_set_overlap_np,
    calculate_logit_deltas,
)
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.kernels.turbo_polar.metal import MetalKernelBridge
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime


class TurboPolarAdapter:
    """DEPRECATED. Raises RuntimeError on instantiation."""

    def __init__(self, config: TurboPolarConfig):
        raise RuntimeError(
            "TurboPolarAdapter is deprecated and structurally broken. "
            "Use TurboPolarOfflineEvaluator or TurboPolarKVCacheRuntime."
        )


class TurboPolarOfflineEvaluator:
    """
    Offline validation harness with GQA and bit-packed PolarQuant support.
    """

    def __init__(self, config: TurboPolarConfig):
        self.config = config
        self.metal_bridge = MetalKernelBridge()
        self.decoder = PolarQuantDecoder()

    def run_qjl_ablation(
        self, q: mx.array, k_original: mx.array
    ) -> Tuple[bool, float, float]:
        B, H, T, D = k_original.shape
        cache = TurboPolarKVCacheRuntime(self.config)
        cache.append(k_original, mx.zeros_like(k_original))

        if cache.total_blocks == 0:
            return False, 0.0, 0.0

        block, _, _, qjl_payload, actual_len = cache.get_blocks_for_attention()
        if block is None:
            return False, 0.0, 0.0

        k_recon = self.decoder.decode_block(block)[:, :, :actual_len, :]

        scores_ref = (
            mx.sum(q[:, :, None, :] * k_original, axis=-1) * self.config.attention_scale
        )
        scores_polar = (
            mx.sum(q[:, :, None, :] * k_recon, axis=-1) * self.config.attention_scale
        )
        error_without = float(mx.max(mx.abs(scores_ref - scores_polar)))
        topk_overlap_without = topk_set_overlap_np(scores_ref, scores_polar, k=10)

        q_proj = mx.matmul(q, cache.qjl_encoder.W)
        q_signs = q_proj >= 0
        reshaped = q_signs.reshape(B, H, self.config.qjl_proj_dim // 8, 8)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        q_packed = mx.sum(reshaped.astype(mx.uint8) * powers, axis=-1).astype(mx.uint8)

        scores_corrected_full = self.metal_bridge.execute_fused_qk_qjl(
            q, block, qjl_payload, q_packed, self.config
        )
        scores_corrected = scores_corrected_full[:, :, :actual_len]
        mx.eval(scores_ref, scores_corrected)

        error_with = float(mx.max(mx.abs(scores_ref - scores_corrected)))
        topk_overlap_with = topk_set_overlap_np(scores_ref, scores_corrected, k=10)

        use_qjl = (
            error_with <= error_without * 0.90
            and topk_overlap_with >= topk_overlap_without
        )
        return use_qjl, error_without, error_with

    def run_teacher_forced_validation(
        self,
        baseline_logits: mx.array,
        candidate_logits: mx.array,
        token_sequence: list[int],
        output_dir: Path,
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)

        kl_div = mean_token_kl(baseline_logits, candidate_logits)
        top5_overlap = topk_set_overlap_np(baseline_logits, candidate_logits, k=5)
        top10_overlap = topk_set_overlap_np(baseline_logits, candidate_logits, k=10)
        deltas = calculate_logit_deltas(baseline_logits, candidate_logits)

        flat_base = baseline_logits.flatten()
        flat_cand = candidate_logits.flatten()
        norm_b = mx.sqrt(mx.sum(flat_base**2))
        norm_c = mx.sqrt(mx.sum(flat_cand**2))
        logit_cosine = float(mx.sum(flat_base * flat_cand) / (norm_b * norm_c + 1e-12))

        passed = (
            logit_cosine >= 0.999
            and kl_div <= 0.05
            and top5_overlap >= 0.90
            and top10_overlap >= 0.95
            and deltas["mean_abs_logit_delta"] <= 0.02
            and deltas["p99_abs_logit_delta"] <= 0.2
            and deltas["max_logit_delta"] <= 1.0
        )

        seq_bytes = json.dumps(token_sequence).encode("utf-8")
        token_hash = hashlib.sha256(seq_bytes).hexdigest()

        artifact = {
            "benchmark_methodology": "teacher_forced_logit_v1_validated",
            "token_sequence_hash": token_hash,
            "methodology_status": "TEACHER_FORCED_RERUN_COMPLETE_NO_PROMOTION",
            "promotion_allowed": False,
            "gate_passed": passed,
            "metrics": {
                "logit_cosine": logit_cosine,
                "kl_divergence": kl_div,
                "top5_overlap": top5_overlap,
                "top10_overlap": top10_overlap,
                **deltas,
            },
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(artifact, f, indent=2)

        return artifact
