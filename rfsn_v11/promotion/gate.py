"""Single promotion authority for TurboPolar.

No other component may declare promotion. All required evidence must be present
and passing; missing evidence results in INCOMPLETE/FAILED.
"""

from typing import List

from rfsn_v11.promotion.schema import (
    GitTreeState,
    PromotionDecision,
    PromotionEvidence,
    PromotionState,
)


class PromotionGate:
    """Evaluate PromotionEvidence and render a single PromotionDecision."""

    # Temporarily locked: maximum state is REVIEW_REQUIRED until the full
    # evidence suite (fused decode, isolated memory, fair baseline, wheel test)
    # has been independently validated.
    PROMOTION_LOCKED = True

    # Correctness thresholds
    MEAN_COSINE = 0.995
    P05_COSINE = 0.990
    MIN_COSINE = 0.975
    MEAN_TOP5 = 0.95
    MEAN_TOP10 = 0.97
    ARGMAX_AGREE = 0.97
    MAX_PPL_DELTA = 0.02

    # Memory thresholds
    LOGICAL_KV_RATIO = 1.85
    PERSISTENT_STORAGE_RATIO = 1.75
    PEAK_MEMORY_RATIO_8192 = 1.20

    # Speed thresholds
    MAX_REGRESSION_AT_4096_PLUS = 0.97  # no more than 3% regression
    MIN_IMPROVEMENT_AT_ANY_LONG_CONTEXT = 1.05
    MIN_MEDIAN_RATIO_AT_8192_PLUS = 1.03

    # Experiment completeness requirements
    REQUIRED_CONTEXTS = {512, 2048, 4096, 8192, 16384}
    REQUIRED_FORCED_DECODE_TOKENS = 128
    MIN_TRIALS_PER_CONTEXT = 5

    # Required native Metal test modules (prefix match).
    REQUIRED_NATIVE_METAL_TESTS = {
        "tests.kernels.test_paged_online_attention",
        "tests.kernels.test_qjl_scaled_fused_qk",
        "tests.kernels.test_qjl_scaled_online_attention",
        "tests.benchmarks.test_turbopolar_fast_attention",
        "tests.benchmarks.test_turbo_polar_online_attention",
    }

    def evaluate(self, evidence: PromotionEvidence) -> PromotionDecision:
        reasons: List[str] = []

        if evidence is None:
            return PromotionDecision(
                state=PromotionState.INCOMPLETE,
                reasons=["No promotion evidence provided."],
            )

        # Missing reports are failures, not successes.
        required_reports = [
            ("kernel_report", evidence.kernel_report),
            ("teacher_forced_report", evidence.teacher_forced_report),
            ("fused_decode_report", evidence.fused_decode_report),
            ("speed_report", evidence.speed_report),
            ("memory_report", evidence.memory_report),
            ("baseline_comparison_report", evidence.baseline_comparison_report),
            ("provenance", evidence.provenance),
        ]
        for name, report in required_reports:
            if report is None:
                reasons.append(f"Missing required report: {name}")

        if reasons:
            return PromotionDecision(
                state=PromotionState.INCOMPLETE,
                reasons=reasons,
                evidence=evidence,
            )

        # Kernel correctness
        kr = evidence.kernel_report
        if not kr.all_unit_tests_passed:
            reasons.append("Unit tests did not all pass.")
        if not kr.all_kernel_tests_passed:
            reasons.append("Kernel tests did not all pass.")
        if not kr.all_integration_tests_passed:
            reasons.append("Integration tests did not all pass.")

        # Required native Metal tests
        present = set(kr.metal_tests_present)
        passed = set(kr.metal_tests_passed)
        missing = self.REQUIRED_NATIVE_METAL_TESTS - present
        if missing:
            reasons.append(
                f"Required Metal tests missing from collection: {sorted(missing)}"
            )
        failed = self.REQUIRED_NATIVE_METAL_TESTS - passed
        if failed:
            reasons.append(f"Required Metal tests did not pass: {sorted(failed)}")

        # Teacher-forced quality
        tf = evidence.teacher_forced_report
        if tf.mean_logit_cosine is None or tf.mean_logit_cosine < self.MEAN_COSINE:
            reasons.append(
                f"Teacher-forced mean cosine {tf.mean_logit_cosine} < {self.MEAN_COSINE}"
            )
        if tf.p05_logit_cosine is None or tf.p05_logit_cosine < self.P05_COSINE:
            reasons.append(
                f"Teacher-forced p05 cosine {tf.p05_logit_cosine} < {self.P05_COSINE}"
            )
        if tf.min_logit_cosine is None or tf.min_logit_cosine < self.MIN_COSINE:
            reasons.append(
                f"Teacher-forced min cosine {tf.min_logit_cosine} < {self.MIN_COSINE}"
            )
        if tf.mean_top5_overlap is None or tf.mean_top5_overlap < self.MEAN_TOP5:
            reasons.append(
                f"Teacher-forced top-5 overlap {tf.mean_top5_overlap} < {self.MEAN_TOP5}"
            )
        if tf.mean_top10_overlap is None or tf.mean_top10_overlap < self.MEAN_TOP10:
            reasons.append(
                f"Teacher-forced top-10 overlap {tf.mean_top10_overlap} < {self.MEAN_TOP10}"
            )
        if tf.argmax_agreement is None or tf.argmax_agreement < self.ARGMAX_AGREE:
            reasons.append(
                f"Teacher-forced argmax agreement {tf.argmax_agreement} < {self.ARGMAX_AGREE}"
            )
        if (
            tf.mean_perplexity_delta is None
            or tf.mean_perplexity_delta > self.MAX_PPL_DELTA
        ):
            reasons.append(
                f"Teacher-forced perplexity delta {tf.mean_perplexity_delta} > {self.MAX_PPL_DELTA}"
            )
        if tf.any_nans_or_infs:
            reasons.append("Teacher-forced run contained NaNs or infinities.")

        # Fused decode quality
        fd = evidence.fused_decode_report
        if fd.mean_logit_cosine is None or fd.mean_logit_cosine < self.MEAN_COSINE:
            reasons.append(
                f"Fused decode mean cosine {fd.mean_logit_cosine} < {self.MEAN_COSINE}"
            )
        if fd.p05_logit_cosine is None or fd.p05_logit_cosine < self.P05_COSINE:
            reasons.append(
                f"Fused decode p05 cosine {fd.p05_logit_cosine} < {self.P05_COSINE}"
            )
        if fd.min_logit_cosine is None or fd.min_logit_cosine < self.MIN_COSINE:
            reasons.append(
                f"Fused decode min cosine {fd.min_logit_cosine} < {self.MIN_COSINE}"
            )
        if fd.mean_top5_overlap is None or fd.mean_top5_overlap < self.MEAN_TOP5:
            reasons.append(
                f"Fused decode top-5 overlap {fd.mean_top5_overlap} < {self.MEAN_TOP5}"
            )
        if fd.mean_top10_overlap is None or fd.mean_top10_overlap < self.MEAN_TOP10:
            reasons.append(
                f"Fused decode top-10 overlap {fd.mean_top10_overlap} < {self.MEAN_TOP10}"
            )
        if fd.argmax_agreement is None or fd.argmax_agreement < self.ARGMAX_AGREE:
            reasons.append(
                f"Fused decode argmax agreement {fd.argmax_agreement} < {self.ARGMAX_AGREE}"
            )
        if (
            fd.mean_perplexity_delta is None
            or fd.mean_perplexity_delta > self.MAX_PPL_DELTA
        ):
            reasons.append(
                f"Fused decode perplexity delta {fd.mean_perplexity_delta} > {self.MAX_PPL_DELTA}"
            )
        if fd.any_nans_or_infs:
            reasons.append("Fused decode run contained NaNs or infinities.")

        # Speed
        sr = evidence.speed_report
        if (
            sr.min_ratio_at_4096_plus is None
            or sr.min_ratio_at_4096_plus < self.MAX_REGRESSION_AT_4096_PLUS
        ):
            reasons.append(
                f"Speed ratio at 4096+ minimum {sr.min_ratio_at_4096_plus} < {self.MAX_REGRESSION_AT_4096_PLUS}"
            )
        if (
            sr.max_ratio_at_4096_plus is None
            or sr.max_ratio_at_4096_plus < self.MIN_IMPROVEMENT_AT_ANY_LONG_CONTEXT
        ):
            reasons.append(
                f"No long-context tier improved by >= {self.MIN_IMPROVEMENT_AT_ANY_LONG_CONTEXT}: "
                f"max ratio {sr.max_ratio_at_4096_plus}"
            )
        if (
            sr.median_ratio_at_8192_plus is None
            or sr.median_ratio_at_8192_plus < self.MIN_MEDIAN_RATIO_AT_8192_PLUS
        ):
            reasons.append(
                f"Median 8192+ speed ratio {sr.median_ratio_at_8192_plus} < {self.MIN_MEDIAN_RATIO_AT_8192_PLUS}"
            )

        # Memory
        mr = evidence.memory_report
        if mr.logical_kv_ratio is None or mr.logical_kv_ratio < self.LOGICAL_KV_RATIO:
            reasons.append(
                f"Logical KV ratio {mr.logical_kv_ratio} < {self.LOGICAL_KV_RATIO}"
            )
        if (
            mr.persistent_storage_ratio is None
            or mr.persistent_storage_ratio < self.PERSISTENT_STORAGE_RATIO
        ):
            reasons.append(
                f"Persistent storage ratio {mr.persistent_storage_ratio} < {self.PERSISTENT_STORAGE_RATIO}"
            )
        if (
            mr.peak_device_memory_ratio_at_8192_plus is None
            or mr.peak_device_memory_ratio_at_8192_plus < self.PEAK_MEMORY_RATIO_8192
        ):
            reasons.append(
                f"Peak memory ratio at 8192+ {mr.peak_device_memory_ratio_at_8192_plus} < {self.PEAK_MEMORY_RATIO_8192}"
            )
        if mr.hidden_dense_cache_detected:
            reasons.append("Hidden dense full-history cache detected.")

        # Baseline comparison
        br = evidence.baseline_comparison_report
        if not br.cartesian_int8_baseline_implemented:
            reasons.append("Cartesian int8 baseline not implemented.")
        elif not (
            br.turbo_polar_wins_on_quality
            or br.turbo_polar_wins_on_memory
            or br.turbo_polar_wins_on_speed
        ):
            reasons.append(
                "TurboPolar does not differentiate from Cartesian int8 on quality, memory, or speed."
            )

        # Experiment completeness
        fd = evidence.fused_decode_report
        contexts_set = (
            set(fd.contexts_evaluated) if fd and fd.contexts_evaluated else set()
        )
        if contexts_set != self.REQUIRED_CONTEXTS:
            reasons.append(
                f"Fused decode contexts incomplete: expected {self.REQUIRED_CONTEXTS}, got {contexts_set}"
            )
        sr = evidence.speed_report
        if sr and sr.trials_per_context < self.MIN_TRIALS_PER_CONTEXT:
            reasons.append(
                f"Speed trials per context {sr.trials_per_context} < {self.MIN_TRIALS_PER_CONTEXT}"
            )

        # Provenance
        pv = evidence.provenance
        if pv.git_tree_state == GitTreeState.UNKNOWN:
            reasons.append("Git tree state unknown; cannot verify reproducibility.")
            return PromotionDecision(
                state=PromotionState.REVIEW_REQUIRED,
                reasons=reasons,
                evidence=evidence,
            )
        if pv.git_tree_state == GitTreeState.DIRTY:
            reasons.append(
                f"Source tree was dirty (diff hash {pv.git_diff_hash}); promotion requires a clean tree."
            )
        if not pv.model_repo_id or not pv.model_revision:
            reasons.append("Model provenance incomplete.")
        if not pv.turbopolar_config_hash:
            reasons.append("TurboPolar config hash missing.")

        if reasons:
            return PromotionDecision(
                state=PromotionState.FAILED,
                reasons=reasons,
                evidence=evidence,
            )

        if self.PROMOTION_LOCKED:
            return PromotionDecision(
                state=PromotionState.REVIEW_REQUIRED,
                reasons=[
                    "All quantitative thresholds pass, but promotion is capped at REVIEW_REQUIRED "
                    "until the full evidence suite (fused decode, isolated memory, fair baseline, "
                    "installed-wheel test) has been independently validated on native Apple Silicon."
                ],
                evidence=evidence,
            )

        return PromotionDecision(
            state=PromotionState.PROMOTED_EXPERIMENTAL,
            reasons=["All required evidence present and passing."],
            evidence=evidence,
        )
