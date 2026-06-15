"""Single promotion authority for TurboPolar.

No other component may declare promotion. All required evidence must be present
and passing; missing evidence results in INCOMPLETE/FAILED.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional

from rfsn_v11.evidence.platform_validation import (
    validate_apple_silicon_platform,
    validate_metal_execution_mode,
)
from rfsn_v11.evidence.provenance import validate_provenance_immutable_fields
from rfsn_v11.evidence.speed import RawSpeedArtifact, validate_speed_trials
from rfsn_v11.evidence.trace_validation import (
    EvidenceValidationError,
    TraceArtifactError,
    TraceTopologyError,
    parse_trace_artifact,
    validate_artifact_file,
    validate_trace_topology,
)
from rfsn_v11.promotion.schema import (
    FusedDecodeReport,
    GitTreeState,
    PromotionDecision,
    PromotionEvidence,
    PromotionState,
)


def _recompute_teacher_forced_summary(
    raw_metrics: Dict[str, Any]
) -> Optional[Dict[str, float]]:
    """Recompute teacher-forced summary from position-level metrics.

    Handles three formats:
    1. BenchmarkReport format:
       {"prompts": [{"position_metrics": [...], ...}],
        "aggregate": {...}}
    2. Legacy format: {"positions": [...]}
    3. TeacherForcedEvidence format:
       {"context_results": {context: {"positions": [...]}}}
    """
    try:
        # Try BenchmarkReport format first
        # (actual run_dense_vs_turbopolar.py output)
        if "prompts" in raw_metrics:
            prompts = raw_metrics.get("prompts", [])
            if not prompts:
                return None

            positions = []
            total_positions = 0
            prompt_perplexity_deltas = []

            for prompt in prompts:
                if not isinstance(prompt, dict):
                    continue
                prompt_positions = prompt.get("position_metrics", [])
                if isinstance(prompt_positions, list):
                    positions.extend(prompt_positions)
                    total_positions += len(prompt_positions)

                # Collect prompt-level perplexity delta
                ppl_delta = prompt.get("perplexity_delta")
                if ppl_delta is not None:
                    prompt_perplexity_deltas.append(ppl_delta)

            if not positions:
                return None

            # Require total_positions > 0
            if total_positions == 0:
                return None

            # Store total position count for validation
            raw_metrics["_computed_total_positions"] = total_positions
        # Try TeacherForcedEvidence format
        elif "context_results" in raw_metrics:
            context_results = raw_metrics.get("context_results", {})
            if not context_results:
                return None

            positions = []
            total_positions = 0
            for context, result in context_results.items():
                if not isinstance(result, dict):
                    continue
                context_positions = result.get("positions", [])
                if isinstance(context_positions, list):
                    positions.extend(context_positions)
                    total_positions += len(context_positions)

            if not positions or total_positions == 0:
                return None

            raw_metrics["_computed_total_positions"] = total_positions
        else:
            # Try legacy format
            positions = raw_metrics.get("positions", [])
            if not positions:
                return None
            total_positions = len(positions)
            if total_positions == 0:
                return None
            raw_metrics["_computed_total_positions"] = total_positions

        cosines = []
        top5_overlaps = []
        top10_overlaps = []
        argmax_agreements = []
        any_nan_or_inf = False

        for pos in positions:
            cos = pos.get("logit_cosine")
            if cos is not None:
                cosines.append(cos)

            top5 = pos.get("top5_overlap")
            if top5 is not None:
                top5_overlaps.append(top5)

            top10 = pos.get("top10_overlap")
            if top10 is not None:
                top10_overlaps.append(top10)

            argmax = pos.get("argmax_agreement")
            if argmax is not None:
                argmax_agreements.append(1 if argmax else 0)

            # Note: PositionMetrics does not contain perplexity_delta,
            # only kl_divergence. Perplexity delta is only available at
            # the prompt level in PromptResult. We do not use KL
            # divergence as a substitute for perplexity delta.

            if pos.get("any_nan_or_inf", False):
                any_nan_or_inf = True

        if not cosines:
            return None

        cosines.sort()
        mean_cosine = sum(cosines) / len(cosines)
        # Use numpy percentile with default interpolation to match
        # benchmark implementation
        p05_cosine = float(np.percentile(cosines, 5)) if cosines else 0.0
        min_cosine = cosines[0] if cosines else 0

        mean_top5 = (sum(top5_overlaps) / len(top5_overlaps)
                     if top5_overlaps else 0)
        mean_top10 = (sum(top10_overlaps) / len(top10_overlaps)
                      if top10_overlaps else 0)
        argmax_agreement = (sum(argmax_agreements) / len(argmax_agreements)
                            if argmax_agreements else 0)

        # Use prompt-level perplexity delta with normalized formula
        # relative_delta = abs(candidate_perplexity / dense_perplexity - 1.0)
        mean_ppl_delta = None
        if prompt_perplexity_deltas:
            mean_ppl_delta = (sum(prompt_perplexity_deltas)
                              / len(prompt_perplexity_deltas))

        return {
            "mean_cosine": mean_cosine,
            "p05_cosine": p05_cosine,
            "min_cosine": min_cosine,
            "mean_top5": mean_top5,
            "mean_top10": mean_top10,
            "argmax_agreement": argmax_agreement,
            "mean_ppl_delta": mean_ppl_delta,
            "any_nan_or_inf": any_nan_or_inf,
            "total_positions": total_positions,
        }
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def _recompute_speed_ratios(
    raw_timing: Dict[str, Any]
) -> Optional[Dict[str, float]]:
    """Recompute speed ratios from raw timing trials.

    Handles three formats:
    1. Unified schema format: {"schema_version": 1, "speed_evidence": {...}}
    2. Legacy format: {"trials": {"context": [{"latencies": [],
        "turbo_time_ms": ..., "baseline_time_ms": ...}]}}
    3. Benchmark format: {"trial_results": [{"context_length": ...,
        "mode": "dense/turbo", "per_token_ms": [...], ...}]}
    """
    try:
        # Try unified schema format first (new SpeedEvidence format)
        if "schema_version" in raw_timing and "speed_evidence" in raw_timing:
            speed_evidence = raw_timing.get("speed_evidence", {})
            trial_results = speed_evidence.get("trial_results", [])
            if not trial_results:
                return None

            context_ratios_4096_plus = []
            context_ratios_8192_plus = []

            # Group trials by context and mode
            context_trials: Dict[int, Dict[str, List[Dict]]] = {}
            for trial in trial_results:
                if not isinstance(trial, dict):
                    continue
                context = trial.get("context_length")
                mode = trial.get("mode")
                if context is None or mode is None:
                    continue

                if context not in context_trials:
                    context_trials[context] = {"dense": [], "turbo": []}
                context_trials[context][mode].append(trial)

            # Compute ratios for each context
            for context, modes in context_trials.items():
                dense_trials = modes.get("dense", [])
                turbo_trials = modes.get("turbo", [])

                if not dense_trials or not turbo_trials:
                    continue

                # Use throughput (tokens per second) to match benchmark
                # calculation. Benchmark uses:
                # mean(turbo throughput) / mean(dense throughput)
                dense_throughputs = [
                    t.get("throughput_tps", 0.0)
                    for t in dense_trials if t.get("throughput_tps", 0.0) > 0
                ]
                turbo_throughputs = [
                    t.get("throughput_tps", 0.0)
                    for t in turbo_trials if t.get("throughput_tps", 0.0) > 0
                ]

                if not dense_throughputs or not turbo_throughputs:
                    continue

                dense_avg_tps = sum(dense_throughputs) / len(dense_throughputs)
                turbo_avg_tps = sum(turbo_throughputs) / len(turbo_throughputs)

                if dense_avg_tps > 0:
                    ratio = turbo_avg_tps / dense_avg_tps

                    if context >= 4096:
                        context_ratios_4096_plus.append(ratio)
                    if context >= 8192:
                        context_ratios_8192_plus.append(ratio)

            if not context_ratios_4096_plus:
                return None

            min_ratio_4096_plus = min(context_ratios_4096_plus)
            max_ratio_4096_plus = max(context_ratios_4096_plus)

            if context_ratios_8192_plus:
                sorted_ratios = sorted(context_ratios_8192_plus)
                median_ratio_8192_plus = sorted_ratios[len(sorted_ratios) // 2]
            else:
                median_ratio_8192_plus = 0

            return {
                "min_ratio_4096_plus": min_ratio_4096_plus,
                "max_ratio_4096_plus": max_ratio_4096_plus,
                "median_ratio_8192_plus": median_ratio_8192_plus,
            }

        # Try benchmark format (run_speed_matrix.py output)
        if "trial_results" in raw_timing:
            trial_results = raw_timing.get("trial_results", [])
            if not trial_results:
                return None

            context_ratios_4096_plus = []
            context_ratios_8192_plus = []

            # Group trials by context and mode
            context_trials: Dict[int, Dict[str, List[Dict]]] = {}
            for trial in trial_results:
                if not isinstance(trial, dict):
                    continue
                context = trial.get("context_length")
                mode = trial.get("mode")
                if context is None or mode is None:
                    continue

                if context not in context_trials:
                    context_trials[context] = {"dense": [], "turbo": []}
                context_trials[context][mode].append(trial)

            # Compute ratios for each context
            for context, modes in context_trials.items():
                dense_trials = modes.get("dense", [])
                turbo_trials = modes.get("turbo", [])

                if not dense_trials or not turbo_trials:
                    continue

                # Use throughput (tokens per second) to match benchmark
                # calculation. Benchmark uses:
                # mean(turbo throughput) / mean(dense throughput)
                dense_throughputs = [
                    t.get("throughput_tps", 0.0)
                    for t in dense_trials if t.get("throughput_tps", 0.0) > 0
                ]
                turbo_throughputs = [
                    t.get("throughput_tps", 0.0)
                    for t in turbo_trials if t.get("throughput_tps", 0.0) > 0
                ]

                if not dense_throughputs or not turbo_throughputs:
                    continue

                dense_avg_tps = sum(dense_throughputs) / len(dense_throughputs)
                turbo_avg_tps = sum(turbo_throughputs) / len(turbo_throughputs)

                if dense_avg_tps > 0:
                    ratio = turbo_avg_tps / dense_avg_tps

                    if context >= 4096:
                        context_ratios_4096_plus.append(ratio)
                    if context >= 8192:
                        context_ratios_8192_plus.append(ratio)

            if not context_ratios_4096_plus:
                return None

            min_ratio_4096_plus = min(context_ratios_4096_plus)
            max_ratio_4096_plus = max(context_ratios_4096_plus)

            if context_ratios_8192_plus:
                sorted_ratios = sorted(context_ratios_8192_plus)
                median_ratio_8192_plus = sorted_ratios[len(sorted_ratios) // 2]
            else:
                median_ratio_8192_plus = 0

            return {
                "min_ratio_4096_plus": min_ratio_4096_plus,
                "max_ratio_4096_plus": max_ratio_4096_plus,
                "median_ratio_8192_plus": median_ratio_8192_plus,
            }

        # Try legacy format
        trials = raw_timing.get("trials", {})
        if not trials:
            return None

        context_ratios_4096_plus = []
        context_ratios_8192_plus = []

        for context, context_trials in trials.items():
            if not isinstance(context_trials, list):
                continue

            for trial in context_trials:
                if not isinstance(trial, dict):
                    continue

                turbo_time = trial.get("turbo_time_ms")
                baseline_time = trial.get("baseline_time_ms")

                if (turbo_time is not None and baseline_time is not None
                        and baseline_time > 0):
                    ratio = baseline_time / turbo_time

                    if context >= 4096:
                        context_ratios_4096_plus.append(ratio)
                    if context >= 8192:
                        context_ratios_8192_plus.append(ratio)

        if not context_ratios_4096_plus:
            return None

        min_ratio_4096_plus = min(context_ratios_4096_plus)
        max_ratio_4096_plus = max(context_ratios_4096_plus)

        if context_ratios_8192_plus:
            sorted_ratios = sorted(context_ratios_8192_plus)
            median_ratio_8192_plus = sorted_ratios[len(sorted_ratios) // 2]
        else:
            median_ratio_8192_plus = 0

        return {
            "min_ratio_4096_plus": min_ratio_4096_plus,
            "max_ratio_4096_plus": max_ratio_4096_plus,
            "median_ratio_8192_plus": median_ratio_8192_plus,
        }
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def _recompute_memory_ratios(
    raw_memory: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Recompute memory ratios from raw memory matrix records.

    Returns a dict with:
      - logical_kv_ratio (worst across 8192+ contexts)
      - persistent_storage_ratio (worst across 8192+ contexts)
      - peak_device_memory_ratio (worst across 8192+ contexts)
      - contexts_evaluated
      - fallback_calls (total across all contexts)
      - fixture_identities (list of per-context fixture info)
      - hidden_dense_detected (any record indicates hidden dense cache)
    """
    try:
        records = raw_memory.get("records", [])
        if not records:
            return None

        contexts = []
        fallback_calls = 0
        hidden_dense_detected = False
        fixture_identities = []
        long_ratios: Dict[str, List[float]] = {
            "logical": [], "persistent": [], "peak": []
        }

        for r in records:
            length = r.get("length")
            if length is None:
                continue
            contexts.append(length)

            # Require raw byte fields for independent recomputation
            dense_kv = r.get("dense_kv_bytes")
            turbo_logical = r.get("turbo_logical_bytes")
            turbo_allocated = r.get("turbo_allocated_bytes")
            dense_peak = r.get("dense_total_peak_bytes")
            turbo_peak = r.get("turbo_total_peak_bytes")

            if any(
                v is None for v in (
                    dense_kv, turbo_logical, turbo_allocated,
                    dense_peak, turbo_peak,
                )
            ):
                return None

            if any(
                v < 0 for v in (
                    dense_kv, turbo_logical, turbo_allocated,
                    dense_peak, turbo_peak,
                )
            ):
                return None

            fallback_calls += r.get("fallback_count", 0)
            if r.get("hidden_dense_cache_detected", False):
                hidden_dense_detected = True

            # Recompute ratios from byte counts
            logical_ratio = (
                dense_kv / turbo_logical if turbo_logical > 0 else 0.0
            )
            persistent_ratio = (
                dense_kv / turbo_allocated if turbo_allocated > 0 else 0.0
            )
            peak_ratio = (
                dense_peak / turbo_peak if turbo_peak > 0 else 0.0
            )

            if length >= 8192:
                long_ratios["logical"].append(logical_ratio)
                long_ratios["persistent"].append(persistent_ratio)
                long_ratios["peak"].append(peak_ratio)

            # Validate fixture identity
            fixture_id = r.get("fixture_id")
            fixture_hash = r.get("fixture_hash")
            if fixture_id:
                if not fixture_hash or len(fixture_hash) != 64:
                    return None
                if not all(
                    c in "0123456789abcdef" for c in fixture_hash.lower()
                ):
                    return None
                fixture_identities.append({
                    "context": length,
                    "fixture_id": fixture_id,
                    "fixture_hash": fixture_hash,
                })

        if not contexts:
            return None

        if long_ratios["logical"]:
            logical_kv_ratio = min(long_ratios["logical"])
            persistent_storage_ratio = min(long_ratios["persistent"])
            peak_device_memory_ratio = min(long_ratios["peak"])
        else:
            # Fallback to last record if no long contexts
            last = records[-1]
            dense_kv = last.get("dense_kv_bytes", 0)
            turbo_logical = last.get("turbo_logical_bytes", 1)
            turbo_allocated = last.get("turbo_allocated_bytes", 1)
            dense_peak = last.get("dense_total_peak_bytes", 1)
            turbo_peak = last.get("turbo_total_peak_bytes", 1)
            logical_kv_ratio = (
                dense_kv / turbo_logical if turbo_logical > 0 else 0.0
            )
            persistent_storage_ratio = (
                dense_kv / turbo_allocated if turbo_allocated > 0 else 0.0
            )
            peak_device_memory_ratio = (
                dense_peak / turbo_peak if turbo_peak > 0 else 0.0
            )

        return {
            "logical_kv_ratio": logical_kv_ratio,
            "persistent_storage_ratio": persistent_storage_ratio,
            "peak_device_memory_ratio": peak_device_memory_ratio,
            "contexts_evaluated": contexts,
            "fallback_calls": fallback_calls,
            "fixture_identities": fixture_identities,
            "hidden_dense_detected": hidden_dense_detected,
        }
    except (KeyError, TypeError, ValueError):
        return None


def _fused_fallback_total(report: FusedDecodeReport) -> int:
    """Calculate total fallback calls from fused decode report."""
    values = (
        report.compressed_page_fallback_calls,
        report.dense_tail_fallback_calls,
        report.full_attention_fallback_calls,
    )
    return sum(int(value or 0) for value in values)


class PromotionGate:
    """Evaluate PromotionEvidence and render a single PromotionDecision."""

    # Locked until the strict no-fallback Metal suite passes end-to-end.
    # A correct fallback result does not prove the Metal implementation
    # works. This must remain True until all evidence systems are
    # scientifically trustworthy.
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
        "tests.kernels.test_metal_strict",
        "tests.kernels.test_fallback_injection",
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
            ("baseline_comparison_report",
             evidence.baseline_comparison_report),
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
                f"Required Metal tests missing from collection: "
                f"{sorted(missing)}"
            )
        failed = self.REQUIRED_NATIVE_METAL_TESTS - passed
        if failed:
            reasons.append(
                f"Required Metal tests did not pass: {sorted(failed)}"
            )

        # Validate skipped status
        if kr.metal_tests_skipped:
            reasons.append(
                f"Native Metal tests were skipped: "
                f"{sorted(kr.metal_tests_skipped)}"
            )

        # Teacher-forced quality
        tf = evidence.teacher_forced_report
        if (tf.mean_logit_cosine is None
                or tf.mean_logit_cosine < self.MEAN_COSINE):
            reasons.append(
                f"Teacher-forced mean cosine {tf.mean_logit_cosine} < "
                f"{self.MEAN_COSINE}"
            )
        if (tf.p05_logit_cosine is None
                or tf.p05_logit_cosine < self.P05_COSINE):
            reasons.append(
                f"Teacher-forced p05 cosine {tf.p05_logit_cosine} < "
                f"{self.P05_COSINE}"
            )
        if (tf.min_logit_cosine is None
                or tf.min_logit_cosine < self.MIN_COSINE):
            reasons.append(
                f"Teacher-forced min cosine {tf.min_logit_cosine} < "
                f"{self.MIN_COSINE}"
            )
        if (tf.mean_top5_overlap is None
                or tf.mean_top5_overlap < self.MEAN_TOP5):
            reasons.append(
                f"Teacher-forced top-5 overlap {tf.mean_top5_overlap} < "
                f"{self.MEAN_TOP5}"
            )
        if (tf.mean_top10_overlap is None
                or tf.mean_top10_overlap < self.MEAN_TOP10):
            reasons.append(
                f"Teacher-forced top-10 overlap {tf.mean_top10_overlap} < "
                f"{self.MEAN_TOP10}"
            )
        if (tf.argmax_agreement is None
                or tf.argmax_agreement < self.ARGMAX_AGREE):
            reasons.append(
                f"Teacher-forced argmax agreement {tf.argmax_agreement} < "
                f"{self.ARGMAX_AGREE}"
            )
        if (
            tf.mean_perplexity_delta is None
            or tf.mean_perplexity_delta > self.MAX_PPL_DELTA
        ):
            reasons.append(
                f"Teacher-forced absolute perplexity delta "
                f"{tf.mean_perplexity_delta} > {self.MAX_PPL_DELTA}"
            )
        if tf.any_nans_or_infs:
            reasons.append("Teacher-forced run contained NaNs or infinities.")

        # Validate teacher-forced raw metrics artifact
        recomputed: dict[str, Any] | None = None
        if not tf.raw_metrics_path:
            reasons.append("Teacher-forced raw_metrics_path is missing.")
        elif not tf.raw_metrics_hash:
            reasons.append("Teacher-forced raw_metrics_hash is missing.")
        else:
            try:
                content = validate_artifact_file(
                    tf.raw_metrics_path,
                    tf.raw_metrics_hash,
                    artifact_name="Teacher-forced raw metrics artifact",
                )
                # P1-24: Parse position records and recompute summaries
                raw_metrics = json.loads(content)
                if not isinstance(raw_metrics, dict):
                    reasons.append(
                        "Teacher-forced raw metrics must be a JSON object")
                else:
                    # Recompute summary from position records
                    recomputed = _recompute_teacher_forced_summary(raw_metrics)
                    if recomputed is None:
                        reasons.append(
                            "Teacher-forced raw metrics could not be "
                            "recomputed; expected positions array or "
                            "context_results with nested positions"
                        )
                    else:
                        # Compare with report values
                        mean_cos = recomputed["mean_cosine"]
                        tf_mean_cos = tf.mean_logit_cosine or 0
                        if abs(mean_cos - tf_mean_cos) > 0.001:
                            reasons.append(
                                f"Teacher-forced mean cosine mismatch: "
                                f"report {tf.mean_logit_cosine} != "
                                f"recomputed {recomputed['mean_cosine']}"
                            )
                        p05_cos = recomputed["p05_cosine"]
                        tf_p05_cos = tf.p05_logit_cosine or 0
                        if abs(p05_cos - tf_p05_cos) > 0.001:
                            reasons.append(
                                f"Teacher-forced p05 cosine mismatch: "
                                f"report {tf.p05_logit_cosine} != "
                                f"recomputed {recomputed['p05_cosine']}"
                            )
                        min_cos = recomputed["min_cosine"]
                        tf_min_cos = tf.min_logit_cosine or 0
                        if abs(min_cos - tf_min_cos) > 0.001:
                            reasons.append(
                                f"Teacher-forced min cosine mismatch: "
                                f"report {tf.min_logit_cosine} != "
                                f"recomputed {recomputed['min_cosine']}"
                            )
                        argmax_agr = recomputed["argmax_agreement"]
                        tf_argmax = tf.argmax_agreement or 0
                        if abs(argmax_agr - tf_argmax) > 0.001:
                            reasons.append(
                                f"Teacher-forced argmax agreement mismatch: "
                                f"report {tf.argmax_agreement} != "
                                f"recomputed {recomputed['argmax_agreement']}"
                            )
                        mean_top5 = recomputed["mean_top5"]
                        tf_top5 = tf.mean_top5_overlap or 0
                        if abs(mean_top5 - tf_top5) > 0.001:
                            reasons.append(
                                f"Teacher-forced top-5 overlap mismatch: "
                                f"report {tf.mean_top5_overlap} != "
                                f"recomputed {recomputed['mean_top5']}"
                            )
                        mean_top10 = recomputed["mean_top10"]
                        tf_top10 = tf.mean_top10_overlap or 0
                        if abs(mean_top10 - tf_top10) > 0.001:
                            reasons.append(
                                f"Teacher-forced top-10 overlap mismatch: "
                                f"report {tf.mean_top10_overlap} != "
                                f"recomputed {recomputed['mean_top10']}"
                            )
                        # Skip perplexity delta recomputation check since it's
                        # not available at position level. The gate trusts
                        # the prompt-level perplexity_delta from the benchmark
                        # report
                        if recomputed["any_nan_or_inf"] != tf.any_nans_or_infs:
                            reasons.append(
                                f"Teacher-forced NaN/Inf mismatch: "
                                f"report {tf.any_nans_or_infs} != "
                                f"recomputed {recomputed['any_nan_or_inf']}"
                            )
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
                EvidenceValidationError,
                TraceArtifactError,
                TraceTopologyError,
            ) as exc:
                reasons.append(
                    f"Invalid teacher-forced raw metrics artifact: {exc}"
                )

            # Fail if recomputation is impossible
            if recomputed is None:
                reasons.append(
                    "Teacher-forced raw metrics recomputation failed - "
                    "cannot verify summary fields."
                )
            else:
                # Validate recomputed total positions
                recomputed_total = recomputed.get("total_positions", 0)
                if recomputed_total == 0:
                    reasons.append(
                        "Teacher-forced recomputed total_positions is "
                        "zero - no positions were evaluated."
                    )

                # Validate that report total matches recomputed total
                if tf.total_positions != recomputed_total:
                    total_pos = tf.total_positions
                    reasons.append(
                        f"Teacher-forced report total_positions "
                        f"({total_pos}) != recomputed total_positions "
                        f"({recomputed_total})"
                    )

        # Fused decode quality
        fd = evidence.fused_decode_report
        if fd.requested_fused_positions_per_context != 128:
            reasons.append(
                f"Fused decode requested_fused_positions_per_context "
                f"must be 128, got {fd.requested_fused_positions_per_context}"
            )
        if (fd.mean_logit_cosine is None
                or fd.mean_logit_cosine < self.MEAN_COSINE):
            reasons.append(
                f"Fused decode mean cosine {fd.mean_logit_cosine} < "
                f"{self.MEAN_COSINE}"
            )
        if (fd.p05_logit_cosine is None
                or fd.p05_logit_cosine < self.P05_COSINE):
            reasons.append(
                f"Fused decode p05 cosine {fd.p05_logit_cosine} < "
                f"{self.P05_COSINE}"
            )
        if (fd.min_logit_cosine is None
                or fd.min_logit_cosine < self.MIN_COSINE):
            reasons.append(
                f"Fused decode min cosine {fd.min_logit_cosine} < "
                f"{self.MIN_COSINE}"
            )
        if (fd.mean_top5_overlap is None
                or fd.mean_top5_overlap < self.MEAN_TOP5):
            reasons.append(
                f"Fused decode top-5 overlap {fd.mean_top5_overlap} < "
                f"{self.MEAN_TOP5}"
            )
        if (fd.mean_top10_overlap is None
                or fd.mean_top10_overlap < self.MEAN_TOP10):
            reasons.append(
                f"Fused decode top-10 overlap {fd.mean_top10_overlap} < "
                f"{self.MEAN_TOP10}"
            )
        if (fd.argmax_agreement is None
                or fd.argmax_agreement < self.ARGMAX_AGREE):
            reasons.append(
                f"Fused decode argmax agreement {fd.argmax_agreement} < "
                f"{self.ARGMAX_AGREE}"
            )
        if (
            fd.mean_perplexity_delta is None
            or fd.mean_perplexity_delta > self.MAX_PPL_DELTA
        ):
            reasons.append(
                f"Fused decode absolute perplexity delta "
                f"{fd.mean_perplexity_delta} > {self.MAX_PPL_DELTA}"
            )
        if fd.any_nans_or_infs:
            reasons.append("Fused decode run contained NaNs or infinities.")

        # Trace artifact validation
        # Skip artifact validation for synthetic dry-run evidence
        if evidence.provenance.evidence_kind == "synthetic_dry_run":
            if fd.trace_artifact_path or fd.trace_artifact_hash:
                reasons.append(
                    "Synthetic dry-run evidence should not have trace "
                    "artifacts."
                )
        elif evidence.provenance.evidence_kind == "experimental":
            # Experimental evidence requires trace artifacts
            if not fd.trace_artifact_path:
                reasons.append("Fused decode trace artifact path is missing.")
            elif not fd.trace_artifact_hash:
                reasons.append("Fused decode trace artifact hash is missing.")
            else:
                # Use fail-safe artifact validation
                try:
                    content = validate_artifact_file(
                        fd.trace_artifact_path,
                        fd.trace_artifact_hash,
                        artifact_name="Fused decode trace artifact",
                    )
                    raw_traces = json.loads(content)
                    parsed_traces = parse_trace_artifact(raw_traces)

                    # Validate trace topology - require model_layer_count to be
                    # set
                    if fd.model_layer_count <= 0:
                        reasons.append(
                            f"Fused decode model_layer_count must be > 0, "
                            f"got {fd.model_layer_count}"
                        )
                    else:
                        topology_result = validate_trace_topology(
                            parsed_traces,
                            model_layer_count=fd.model_layer_count,
                            required_contexts=self.REQUIRED_CONTEXTS,
                            requested_positions_per_context=(
                                fd.requested_fused_positions_per_context
                            ),
                        )

                        # Reconcile trace totals with report totals (P0-16-21)
                        for context in self.REQUIRED_CONTEXTS:
                            if context not in (
                                fd.compressed_page_dispatches_per_context
                            ):
                                reasons.append(
                                    f"Fused decode report missing "
                                    f"compressed_page_dispatches_per_context "
                                    f"for context {context}"
                                )
                                continue

                            dispatches = (
                                fd.compressed_page_dispatches_per_context
                            )
                            report_page_ops = dispatches[context]
                            trace_page_ops = topology_result.get(
                                "trace_computed_page_ops", {}
                            ).get(context, 0)
                            if report_page_ops != trace_page_ops:
                                reasons.append(
                                    f"Context {context}: report "
                                    f"compressed_page_dispatches "
                                    f"{report_page_ops} != trace-computed "
                                    f"{trace_page_ops}"
                                )

                            report_tail_ops = (
                                fd.dense_tail_dispatches_per_context.get(
                                    context, 0)
                            )
                            trace_tail_ops = topology_result.get(
                                "trace_computed_tail_ops", {}
                            ).get(context, 0)
                            if report_tail_ops != trace_tail_ops:
                                reasons.append(
                                    f"Context {context}: report "
                                    f"dense_tail_dispatches {report_tail_ops} "
                                    f"!= trace-computed {trace_tail_ops}"
                                )

                            report_fallback_ops = (
                                fd.fallback_calls_per_context.get(context, 0)
                            )
                            trace_fallback_ops = topology_result.get(
                                "trace_computed_fallback_ops", {}
                            ).get(context, 0)
                            if report_fallback_ops != trace_fallback_ops:
                                reasons.append(
                                    f"Context {context}: report "
                                    f"fallback_calls {report_fallback_ops} "
                                    f"!= trace-computed {trace_fallback_ops}"
                                )

                        # Reconcile trace totals with global bridge totals
                        # (P1-27)
                        total_trace_page_ops = sum(
                            topology_result.get(
                                "trace_computed_page_ops", {}).values()
                        )
                        total_trace_tail_ops = sum(
                            topology_result.get(
                                "trace_computed_tail_ops", {}).values()
                        )
                        total_trace_fallback_ops = sum(
                            topology_result.get(
                                "trace_computed_fallback_ops", {}
                            ).values()
                        )

                        if fd.compressed_page_metal_calls is not None:
                            page_calls = fd.compressed_page_metal_calls
                            if page_calls != total_trace_page_ops:
                                reasons.append(
                                    f"Global compressed_page_metal_calls "
                                    f"{page_calls} != trace-computed "
                                    f"total {total_trace_page_ops}"
                                )

                        if fd.dense_tail_metal_calls is not None:
                            tail_calls = fd.dense_tail_metal_calls
                            if tail_calls != total_trace_tail_ops:
                                reasons.append(
                                    f"Global dense_tail_metal_calls "
                                    f"{tail_calls} != trace-computed "
                                    f"total {total_trace_tail_ops}"
                                )

                        # Sum of fallback types should match total fallbacks
                        total_type_fallbacks = _fused_fallback_total(fd)

                        if total_type_fallbacks != total_trace_fallback_ops:
                            tfb = total_type_fallbacks
                            ttfo = total_trace_fallback_ops
                            reasons.append(
                                f"Sum of typed fallbacks {tfb} != "
                                f"trace-computed total {ttfo}"
                            )

                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                    EvidenceValidationError,
                    TraceArtifactError,
                    TraceTopologyError,
                ) as exc:
                    reasons.append(
                        f"Invalid trace artifact: {exc}"
                    )
        else:
            # Unknown evidence kind - treat as incomplete
            reasons.append(
                f"Unknown evidence kind: {evidence.provenance.evidence_kind}"
            )

        # Strict Metal execution verification.
        if fd.execution_mode is None:
            reasons.append("Fused decode execution_mode is missing.")
        elif fd.execution_mode != "metal_strict":
            reasons.append(
                f"Fused decode execution_mode='{fd.execution_mode}' "
                f"!= 'metal_strict'; only strict Metal runs are "
                f"eligible for promotion."
            )

        # Required Metal dispatch counts must be present.
        # merge_metal_calls and finalization_metal_calls are optional because
        # merge/finalization currently use ordinary MLX operations, not custom
        # Metal kernels. This is a known hybrid execution policy.
        required_metal_fields = [
            "compressed_page_metal_calls",
            "dense_tail_metal_calls",
        ]
        for field in required_metal_fields:
            val = getattr(fd, field, None)
            if val is None:
                reasons.append(f"Fused decode missing required field: {field}")
            elif val == 0:
                reasons.append(
                    f"Fused decode {field}=0; no Metal dispatches recorded."
                )
        if fd.merge_metal_calls is not None and fd.merge_metal_calls > 0:
            reasons.append(
                "Fused decode reported merge_metal_calls>0, but merge uses "
                "MLX operations."
            )
        if (fd.finalization_metal_calls is not None
                and fd.finalization_metal_calls > 0):
            reasons.append(
                "Fused decode reported finalization_metal_calls>0, but "
                "finalization uses MLX operations."
            )

        # Fallback counts must be present and zero.
        required_fallback_fields = [
            "compressed_page_fallback_calls",
            "dense_tail_fallback_calls",
            "full_attention_fallback_calls",
        ]
        for field in required_fallback_fields:
            val = getattr(fd, field, None)
            if val is None:
                reasons.append(f"Fused decode missing required field: {field}")
            elif val != 0:
                reasons.append(
                    f"Fused decode {field}={val}; fallback occurred in "
                    f"strict mode."
                )

        if fd.fallback_reasons is None:
            reasons.append("Fused decode fallback_reasons missing.")
        elif fd.fallback_reasons:
            reasons.append(
                f"Fused decode had fallback reasons: {fd.fallback_reasons}"
            )

        # Speed
        sr = evidence.speed_report
        if (
            sr.min_ratio_at_4096_plus is None
            or sr.min_ratio_at_4096_plus < self.MAX_REGRESSION_AT_4096_PLUS
        ):
            reasons.append(
                f"Speed ratio at 4096+ minimum {sr.min_ratio_at_4096_plus} < "
                f"{self.MAX_REGRESSION_AT_4096_PLUS}"
            )

        # Require strict execution and zero fallbacks in speed evidence
        if sr.execution_mode is None:
            reasons.append("Speed evidence execution_mode is missing.")
        elif sr.execution_mode != "metal_strict":
            reasons.append(
                f"Speed evidence execution_mode='{sr.execution_mode}' "
                f"!= 'metal_strict'; only strict Metal runs are "
                f"eligible for promotion."
            )
        if sr.fallback_calls != 0:
            reasons.append(
                f"Speed evidence fallback_calls={sr.fallback_calls}; "
                f"fallback occurred in strict mode."
            )
        if not sr.raw_timing_path:
            reasons.append("Speed evidence raw_timing_path is missing.")
        elif not sr.raw_timing_hash:
            reasons.append("Speed evidence raw_timing_hash is missing.")
        else:
            # Validate raw timing artifact
            recomputed_speed: dict[str, float] | None = None
            try:
                content = validate_artifact_file(
                    sr.raw_timing_path,
                    sr.raw_timing_hash,
                    artifact_name="Speed raw timing artifact",
                )
                # P1-25: Parse speed trials using canonical schema validator
                raw_timing = json.loads(content)
                if not isinstance(raw_timing, dict):
                    reasons.append("Speed raw timing must be a JSON object")
                else:
                    # Use canonical RawSpeedArtifact validator for unified
                    # validation
                    try:
                        artifact = RawSpeedArtifact.from_dict(raw_timing)
                        validation_errors = validate_speed_trials(artifact)
                        for err in validation_errors:
                            reasons.append(f"Speed validation: {err}")
                    except (ValueError, TypeError, KeyError) as exc:
                        reasons.append(
                            f"Speed canonical validation failed: {exc}"
                        )

                    # P1-25: Recompute speed ratios from raw timing
                    recomputed_speed = _recompute_speed_ratios(raw_timing)
                    if recomputed_speed is not None:
                        min_r4096 = recomputed_speed["min_ratio_4096_plus"]
                        sr_min = sr.min_ratio_at_4096_plus or 0
                        if abs(min_r4096 - sr_min) > 0.01:
                            report = sr.min_ratio_at_4096_plus
                            rec = recomputed_speed['min_ratio_4096_plus']
                            reasons.append(
                                f"Speed min ratio 4096+ mismatch: "
                                f"report {report} != recomputed {rec}"
                            )
                        max_r4096 = recomputed_speed["max_ratio_4096_plus"]
                        sr_max = sr.max_ratio_at_4096_plus or 0
                        if abs(max_r4096 - sr_max) > 0.01:
                            report = sr.max_ratio_at_4096_plus
                            rec = recomputed_speed['max_ratio_4096_plus']
                            reasons.append(
                                f"Speed max ratio 4096+ mismatch: "
                                f"report {report} != recomputed {rec}"
                            )
                        med_r8192 = recomputed_speed["median_ratio_8192_plus"]
                        sr_med = sr.median_ratio_at_8192_plus or 0
                        if abs(med_r8192 - sr_med) > 0.01:
                            report = sr.median_ratio_at_8192_plus
                            rec = recomputed_speed['median_ratio_8192_plus']
                            reasons.append(
                                f"Speed median ratio 8192+ mismatch: "
                                f"report {report} != recomputed {rec}"
                            )
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
                EvidenceValidationError,
            ) as exc:
                reasons.append(
                    f"Invalid speed raw timing artifact: {exc}"
                )

        # Require baseline contexts through 16K
        if 16384 not in sr.contexts_evaluated:
            reasons.append(
                "Speed evidence missing required 16K context length.")

        # Baseline comparison must include 16K
        bc = evidence.baseline_comparison_report
        if bc and 16384 not in bc.contexts_evaluated:
            reasons.append(
                "Baseline comparison missing required 16K context length."
            )
        if (sr.max_ratio_at_4096_plus is None
                or sr.max_ratio_at_4096_plus
                < self.MIN_IMPROVEMENT_AT_ANY_LONG_CONTEXT):
            reasons.append(
                f"No long-context tier improved by >= "
                f"{self.MIN_IMPROVEMENT_AT_ANY_LONG_CONTEXT}: "
                f"max ratio {sr.max_ratio_at_4096_plus}"
            )
        if (sr.median_ratio_at_8192_plus is None
                or sr.median_ratio_at_8192_plus
                < self.MIN_MEDIAN_RATIO_AT_8192_PLUS):
            reasons.append(
                f"Median 8192+ speed ratio {sr.median_ratio_at_8192_plus} < "
                f"{self.MIN_MEDIAN_RATIO_AT_8192_PLUS}"
            )

        # Memory
        mr = evidence.memory_report
        if (mr.logical_kv_ratio is None
                or mr.logical_kv_ratio < self.LOGICAL_KV_RATIO):
            reasons.append(
                f"Logical KV ratio {mr.logical_kv_ratio} < "
                f"{self.LOGICAL_KV_RATIO}"
            )
        if (
            mr.persistent_storage_ratio is None
            or mr.persistent_storage_ratio < self.PERSISTENT_STORAGE_RATIO
        ):
            reasons.append(
                f"Persistent storage ratio {mr.persistent_storage_ratio} < "
                f"{self.PERSISTENT_STORAGE_RATIO}"
            )
        if (mr.peak_device_memory_ratio_at_8192_plus is None
                or mr.peak_device_memory_ratio_at_8192_plus
                < self.PEAK_MEMORY_RATIO_8192):
            reasons.append(
                f"Peak memory ratio at 8192+ "
                f"{mr.peak_device_memory_ratio_at_8192_plus} < "
                f"{self.PEAK_MEMORY_RATIO_8192}"
            )
        if mr.hidden_dense_cache_detected:
            reasons.append("Hidden dense full-history cache detected.")
        if mr.fallback_calls > 0:
            reasons.append(
                f"Memory fallback_calls={mr.fallback_calls}; "
                f"fallback occurred in strict mode."
            )

        # P0: Validate raw memory artifact independently
        if not mr.raw_memory_path:
            reasons.append("Memory evidence raw_memory_path is missing.")
        elif not mr.raw_memory_hash:
            reasons.append("Memory evidence raw_memory_hash is missing.")
        else:
            try:
                content = validate_artifact_file(
                    mr.raw_memory_path,
                    mr.raw_memory_hash,
                    artifact_name="Memory raw matrix artifact",
                )
                raw_memory = json.loads(content)
                if not isinstance(raw_memory, dict):
                    reasons.append(
                        "Memory raw matrix must be a JSON object"
                    )
                else:
                    recomputed = _recompute_memory_ratios(raw_memory)
                    if recomputed is None:
                        reasons.append(
                            "Memory raw matrix could not be recomputed"
                        )
                    else:
                        # Validate required contexts
                        raw_contexts = set(recomputed["contexts_evaluated"])
                        if not self.REQUIRED_CONTEXTS.issubset(raw_contexts):
                            missing = self.REQUIRED_CONTEXTS - raw_contexts
                            reasons.append(
                                f"Memory raw contexts incomplete: "
                                f"missing {missing}"
                            )
                        # Validate zero fallback
                        if recomputed["fallback_calls"] > 0:
                            reasons.append(
                                f"Memory fallback_calls="
                                f"{recomputed['fallback_calls']}; "
                                f"fallback occurred in strict mode."
                            )
                        # Validate fixture identity per context
                        fixture_contexts = {
                            f["context"] for f in recomputed["fixture_identities"]
                        }
                        if not self.REQUIRED_CONTEXTS.issubset(fixture_contexts):
                            missing = self.REQUIRED_CONTEXTS - fixture_contexts
                            reasons.append(
                                "Memory raw matrix lacks fixture identities "
                                f"for required contexts: {missing}"
                            )
                        # Validate hidden dense cache detection
                        if (
                            recomputed["hidden_dense_detected"]
                            != mr.hidden_dense_cache_detected
                        ):
                            reasons.append(
                                "Memory hidden_dense_cache_detected summary "
                                "does not match raw matrix records."
                            )
                        # Cross-check summary values
                        tolerance = 1e-6
                        if (mr.logical_kv_ratio is not None
                                and recomputed["logical_kv_ratio"] is not None
                                and abs(mr.logical_kv_ratio
                                        - recomputed["logical_kv_ratio"])
                                > tolerance):
                            reasons.append(
                                "Memory logical_kv_ratio does not match "
                                "raw matrix recomputation."
                            )
                        if (mr.persistent_storage_ratio is not None
                                and recomputed["persistent_storage_ratio"]
                                is not None
                                and abs(
                                    mr.persistent_storage_ratio
                                    - recomputed["persistent_storage_ratio"]
                                ) > tolerance):
                            reasons.append(
                                "Memory persistent_storage_ratio does not "
                                "match raw matrix recomputation."
                            )
                        if (mr.peak_device_memory_ratio_at_8192_plus
                                is not None
                                and recomputed["peak_device_memory_ratio"]
                                is not None
                                and abs(
                                    mr.peak_device_memory_ratio_at_8192_plus
                                    - recomputed["peak_device_memory_ratio"]
                                ) > tolerance):
                            reasons.append(
                                "Memory peak_device_memory_ratio_at_8192+ "
                                "does not match raw matrix recomputation."
                            )
            except Exception as e:
                reasons.append(
                    f"Memory raw matrix artifact validation failed: {e}"
                )

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
                "TurboPolar does not differentiate from Cartesian int8 "
                "on quality, memory, or speed."
            )

        # Experiment completeness: per-context fused decode evidence.
        # Use required-set inclusion so reports with diagnostic contexts still
        # pass.
        fd = evidence.fused_decode_report
        contexts_set = (
            set(fd.contexts_evaluated)
            if fd and fd.contexts_evaluated else set()
        )
        if not self.REQUIRED_CONTEXTS.issubset(contexts_set):
            reasons.append(
                f"Fused decode contexts incomplete: required "
                f"{self.REQUIRED_CONTEXTS} not in {contexts_set}"
            )

        # Per-context completeness gates.
        for context in self.REQUIRED_CONTEXTS:
            if context not in fd.positions_per_context:
                reasons.append(
                    f"Missing fused decode positions for context {context}"
                )
            elif (fd.positions_per_context.get(context, 0)
                  < self.REQUIRED_FORCED_DECODE_TOKENS):
                positions = fd.positions_per_context.get(context, 0)
                reasons.append(
                    f"Context {context}: fused positions {positions} "
                    f"< required {self.REQUIRED_FORCED_DECODE_TOKENS}"
                )
            if fd.failed_positions_per_context.get(context, 0) != 0:
                reasons.append(
                    f"Context {context}: has failed fused positions"
                )
            if fd.fallback_calls_per_context.get(context, 0) != 0:
                reasons.append(
                    f"Context {context}: has fallback calls"
                )

        # Legacy fallback check (global).
        if fd and fd.actual_fused_positions is not None:
            if fd.actual_fused_positions < self.REQUIRED_FORCED_DECODE_TOKENS:
                actual_pos = fd.actual_fused_positions
                required_pos = self.REQUIRED_FORCED_DECODE_TOKENS
                reasons.append(
                    f"Fused decode actual positions {actual_pos} < "
                    f"required {required_pos}"
                )
        else:
            reasons.append("Fused decode actual_fused_positions missing.")

        sr = evidence.speed_report
        if sr and sr.trials_per_context < self.MIN_TRIALS_PER_CONTEXT:
            reasons.append(
                f"Speed trials per context {sr.trials_per_context} < "
                f"{self.MIN_TRIALS_PER_CONTEXT}"
            )
        speed_contexts = set(
            sr.contexts_evaluated) if sr and sr.contexts_evaluated else set()
        if not self.REQUIRED_CONTEXTS.issubset(speed_contexts):
            reasons.append(
                f"Speed contexts incomplete: required "
                f"{self.REQUIRED_CONTEXTS} not in {speed_contexts}"
            )
        mr = evidence.memory_report
        memory_contexts = set(
            mr.contexts_evaluated) if mr and mr.contexts_evaluated else set()
        if not self.REQUIRED_CONTEXTS.issubset(memory_contexts):
            reasons.append(
                f"Memory contexts incomplete: required "
                f"{self.REQUIRED_CONTEXTS} not in {memory_contexts}"
            )

        # Provenance
        pv = evidence.provenance

        # P1-29: Require token fixtures hash for reproducibility
        if not pv.token_fixtures_hash:
            reasons.append(
                "Token fixtures hash is missing; exact fixtures required "
                "for reproducibility."
            )
        elif len(pv.token_fixtures_hash) < 32:  # At least half of SHA-256 hex
            reasons.append(
                f"Token fixtures hash '{pv.token_fixtures_hash}' is too "
                f"short; full SHA-256 hash required for reproducibility."
            )

        # P1-30: Require immutable model and tokenizer revisions
        if not pv.model_revision:
            reasons.append(
                "Model revision is empty; immutable git commit hash required."
            )
        elif pv.model_revision == "unknown":
            reasons.append(
                "Model revision is 'unknown'; immutable git commit hash "
                "required."
            )
        elif len(pv.model_revision) < 7:  # At least short git hash
            reasons.append(
                f"Model revision '{pv.model_revision}' is too short; "
                "immutable git commit hash required (at least 7 characters)."
            )

        if not pv.tokenizer_revision:
            reasons.append(
                "Tokenizer revision is empty; immutable git commit hash "
                "required."
            )
        elif pv.tokenizer_revision == "unknown":
            reasons.append(
                "Tokenizer revision is 'unknown'; immutable git commit "
                "hash required."
            )
        elif len(pv.tokenizer_revision) < 7:
            reasons.append(
                f"Tokenizer revision '{pv.tokenizer_revision}' is too short; "
                "immutable git commit hash required (at least 7 characters)."
            )

        # P0: Validate all five workload hashes are present and well-formed
        _workload_hashes = {
            "speed": pv.speed_workload_hash,
            "memory": pv.memory_workload_hash,
            "fused_decode": pv.fused_decode_workload_hash,
            "cartesian": pv.cartesian_workload_hash,
            "teacher_forced": pv.teacher_forced_workload_hash,
        }
        for name, h in _workload_hashes.items():
            if not h:
                reasons.append(
                    f"{name}_workload_hash is missing; workload identity "
                    f"required for reproducibility."
                )
            elif len(h) != 64:
                reasons.append(
                    f"{name}_workload_hash must be a full 64-character "
                    f"SHA-256 hex string (got {len(h)} chars)."
                )
            elif not all(c in "0123456789abcdef" for c in h.lower()):
                reasons.append(
                    f"{name}_workload_hash must be a valid SHA-256 "
                    f"hexadecimal string."
                )
        # Cross-family uniqueness: no two workload hashes may be identical
        _non_empty = [h for h in _workload_hashes.values() if h]
        if len(_non_empty) != len(set(_non_empty)):
            reasons.append(
                "Duplicate workload hash detected across benchmark families; "
                "each workload must have a unique identity."
            )

        if pv.git_tree_state == GitTreeState.UNKNOWN:
            reasons.append(
                "Git tree state unknown; cannot verify reproducibility."
            )
        if pv.git_tree_state == GitTreeState.DIRTY:
            reasons.append(
                f"Source tree was dirty (diff hash {pv.git_diff_hash}); "
                "promotion requires a clean tree."
            )
        if not pv.model_repo_id or not pv.model_revision:
            reasons.append("Model provenance incomplete.")
        if not pv.turbopolar_config_hash:
            reasons.append("TurboPolar config hash missing.")

        # Integrate canonical provenance validator
        provenance_errors = validate_provenance_immutable_fields(pv)
        for err in provenance_errors:
            reasons.append(f"Provenance: {err}")

        # Integrate platform validator for Apple Silicon benchmarks
        platform_errors = validate_apple_silicon_platform(pv)
        for err in platform_errors:
            reasons.append(f"Platform: {err}")

        # Integrate Metal execution mode validator
        metal_errors = validate_metal_execution_mode(pv)
        for err in metal_errors:
            reasons.append(f"Execution: {err}")

        # Explicit promotion decision ordering
        # 1. Hard quantitative/artifact failures -> FAILED
        if reasons:
            return PromotionDecision(
                state=PromotionState.FAILED,
                reasons=reasons,
                evidence=evidence,
            )

        # 2. Valid synthetic evidence -> REVIEW_REQUIRED
        if pv.evidence_kind != "experimental":
            return PromotionDecision(
                state=PromotionState.REVIEW_REQUIRED,
                reasons=[
                    "Synthetic or non-experimental evidence is never "
                    "promotable."
                ],
                evidence=evidence,
            )

        # 3. Evidence missing or provenance unknown -> INCOMPLETE
        if pv.git_tree_state == GitTreeState.UNKNOWN:
            return PromotionDecision(
                state=PromotionState.INCOMPLETE,
                reasons=[
                    "Git tree state unknown; cannot verify reproducibility."
                ],
                evidence=evidence,
            )

        # 4. Promotion locked -> REVIEW_REQUIRED
        if self.PROMOTION_LOCKED:
            return PromotionDecision(
                state=PromotionState.REVIEW_REQUIRED,
                reasons=[
                    "Promotion is locked pending independent native "
                    "evidence review."
                ],
                evidence=evidence,
            )

        # 4. All checks pass -> PROMOTED_EXPERIMENTAL
        return PromotionDecision(
            state=PromotionState.PROMOTED_EXPERIMENTAL,
            reasons=["All required evidence present and passing."],
            evidence=evidence,
        )
