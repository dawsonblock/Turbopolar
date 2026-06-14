"""Single promotion authority for TurboPolar.

No other component may declare promotion. All required evidence must be present
and passing; missing evidence results in INCOMPLETE/FAILED.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional

from rfsn_v11.evidence.trace_validation import (
    EvidenceValidationError,
    TraceArtifactError,
    TraceTopologyError,
    parse_trace_artifact,
    validate_artifact_file,
    validate_trace_topology,
)
from rfsn_v11.promotion.schema import (
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
            for prompt in prompts:
                if not isinstance(prompt, dict):
                    continue
                prompt_positions = prompt.get("position_metrics", [])
                if isinstance(prompt_positions, list):
                    positions.extend(prompt_positions)
                    total_positions += len(prompt_positions)

            if not positions:
                return None

            # Store total position count for validation
            raw_metrics["_computed_total_positions"] = total_positions
        # Try TeacherForcedEvidence format
        elif "context_results" in raw_metrics:
            context_results = raw_metrics.get("context_results", {})
            if not context_results:
                return None

            positions = []
            for context, result in context_results.items():
                if not isinstance(result, dict):
                    continue
                context_positions = result.get("positions", [])
                if isinstance(context_positions, list):
                    positions.extend(context_positions)

            if not positions:
                return None
        else:
            # Try legacy format
            positions = raw_metrics.get("positions", [])
            if not positions:
                return None

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
        
        mean_top5 = sum(top5_overlaps) / len(top5_overlaps) if top5_overlaps else 0
        mean_top10 = sum(top10_overlaps) / len(top10_overlaps) if top10_overlaps else 0
        argmax_agreement = sum(argmax_agreements) / len(argmax_agreements) if argmax_agreements else 0
        
        # Perplexity delta is not available at position level, only at prompt level
        # Return None for position-level perplexity delta recomputation
        # The gate should use the prompt-level perplexity_delta from the report instead
        mean_ppl_delta = None
        
        return {
            "mean_cosine": mean_cosine,
            "p05_cosine": p05_cosine,
            "min_cosine": min_cosine,
            "mean_top5": mean_top5,
            "mean_top10": mean_top10,
            "argmax_agreement": argmax_agreement,
            "mean_ppl_delta": mean_ppl_delta,
            "any_nan_or_inf": any_nan_or_inf,
        }
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def _recompute_speed_ratios(raw_timing: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Recompute speed ratios from raw timing trials.

    Handles three formats:
    1. Unified schema format: {"schema_version": 1, "speed_evidence": {...}}
    2. Legacy format: {"trials": {"context": [{"latencies": [], "turbo_time_ms": ..., "baseline_time_ms": ...}]}}
    3. Benchmark format: {"trial_results": [{"context_length": ..., "mode": "dense/turbo", "per_token_ms": [...], ...}]}
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

                # Use throughput (tokens per second) to match benchmark calculation
                # Benchmark uses: mean(turbo throughput) / mean(dense throughput)
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
                median_ratio_8192_plus = sorted(context_ratios_8192_plus)[len(context_ratios_8192_plus) // 2]
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
                
                # Use throughput (tokens per second) to match benchmark calculation
                # Benchmark uses: mean(turbo throughput) / mean(dense throughput)
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
                median_ratio_8192_plus = sorted(context_ratios_8192_plus)[len(context_ratios_8192_plus) // 2]
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
                
                if turbo_time is not None and baseline_time is not None and baseline_time > 0:
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
            median_ratio_8192_plus = sorted(context_ratios_8192_plus)[len(context_ratios_8192_plus) // 2]
        else:
            median_ratio_8192_plus = 0
        
        return {
            "min_ratio_4096_plus": min_ratio_4096_plus,
            "max_ratio_4096_plus": max_ratio_4096_plus,
            "median_ratio_8192_plus": median_ratio_8192_plus,
        }
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def _fused_fallback_total(report: "FusedDecodeReport") -> int:
    """Calculate total fallback calls from fused decode report."""
    from rfsn_v11.promotion.schema import FusedDecodeReport

    values = (
        report.compressed_page_fallback_calls,
        report.dense_tail_fallback_calls,
        report.full_attention_fallback_calls,
    )
    return sum(int(value or 0) for value in values)


class PromotionGate:
    """Evaluate PromotionEvidence and render a single PromotionDecision."""

    # Locked until the strict no-fallback Metal suite passes end-to-end.
    # A correct fallback result does not prove the Metal implementation works.
    # This must remain True until all evidence systems are scientifically trustworthy.
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

        # Validate skipped status
        if kr.metal_tests_skipped:
            reasons.append(f"Native Metal tests were skipped: {sorted(kr.metal_tests_skipped)}")

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
                f"Teacher-forced absolute perplexity delta {tf.mean_perplexity_delta} > {self.MAX_PPL_DELTA}"
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
                    reasons.append("Teacher-forced raw metrics must be a JSON object")
                else:
                    # Recompute summary from position records
                    recomputed = _recompute_teacher_forced_summary(raw_metrics)
                    if recomputed is None:
                        reasons.append(
                            "Teacher-forced raw metrics could not be recomputed; "
                            "expected positions array or context_results with nested positions"
                        )
                    else:
                        # Compare with report values
                        if abs(recomputed["mean_cosine"] - (tf.mean_logit_cosine or 0)) > 0.001:
                            reasons.append(
                                f"Teacher-forced mean cosine mismatch: report {tf.mean_logit_cosine} "
                                f"!= recomputed {recomputed['mean_cosine']}"
                            )
                        if abs(recomputed["p05_cosine"] - (tf.p05_logit_cosine or 0)) > 0.001:
                            reasons.append(
                                f"Teacher-forced p05 cosine mismatch: report {tf.p05_logit_cosine} "
                                f"!= recomputed {recomputed['p05_cosine']}"
                            )
                        if abs(recomputed["min_cosine"] - (tf.min_logit_cosine or 0)) > 0.001:
                            reasons.append(
                                f"Teacher-forced min cosine mismatch: report {tf.min_logit_cosine} "
                                f"!= recomputed {recomputed['min_cosine']}"
                            )
                        if abs(recomputed["argmax_agreement"] - (tf.argmax_agreement or 0)) > 0.001:
                            reasons.append(
                                f"Teacher-forced argmax agreement mismatch: report {tf.argmax_agreement} "
                                f"!= recomputed {recomputed['argmax_agreement']}"
                            )
                        if abs(recomputed["mean_top5"] - (tf.mean_top5_overlap or 0)) > 0.001:
                            reasons.append(
                                f"Teacher-forced top-5 overlap mismatch: report {tf.mean_top5_overlap} "
                                f"!= recomputed {recomputed['mean_top5']}"
                            )
                        if abs(recomputed["mean_top10"] - (tf.mean_top10_overlap or 0)) > 0.001:
                            reasons.append(
                                f"Teacher-forced top-10 overlap mismatch: report {tf.mean_top10_overlap} "
                                f"!= recomputed {recomputed['mean_top10']}"
                            )
                        # Skip perplexity delta recomputation check since it's not available at position level
                        # The gate trusts the prompt-level perplexity_delta from the benchmark report
                        if recomputed["any_nan_or_inf"] != tf.any_nans_or_infs:
                            reasons.append(
                                f"Teacher-forced NaN/Inf mismatch: report {tf.any_nans_or_infs} "
                                f"!= recomputed {recomputed['any_nan_or_inf']}"
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
                reasons.append(f"Invalid teacher-forced raw metrics artifact: {exc}")

            # Fail if recomputation is impossible
            if recomputed is None:
                reasons.append("Teacher-forced raw metrics recomputation failed - cannot verify summary fields.")

            # Require nonzero total position count
            if tf.total_positions == 0:
                reasons.append("Teacher-forced total_positions is zero - no positions were evaluated.")

        # Fused decode quality
        fd = evidence.fused_decode_report
        if fd.requested_fused_positions_per_context != 128:
            reasons.append(
                f"Fused decode requested_fused_positions_per_context must be 128, got {fd.requested_fused_positions_per_context}"
            )
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
                f"Fused decode absolute perplexity delta {fd.mean_perplexity_delta} > {self.MAX_PPL_DELTA}"
            )
        if fd.any_nans_or_infs:
            reasons.append("Fused decode run contained NaNs or infinities.")

        # Trace artifact validation
        # Skip artifact validation for synthetic dry-run evidence
        if evidence.provenance.evidence_kind == "synthetic_dry_run":
            if fd.trace_artifact_path or fd.trace_artifact_hash:
                reasons.append("Synthetic dry-run evidence should not have trace artifacts.")
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
                    
                    # Validate trace topology - require model_layer_count to be set
                    if fd.model_layer_count <= 0:
                        reasons.append(
                            f"Fused decode model_layer_count must be > 0, got {fd.model_layer_count}"
                        )
                    else:
                        topology_result = validate_trace_topology(
                            parsed_traces,
                            model_layer_count=fd.model_layer_count,
                            required_contexts=self.REQUIRED_CONTEXTS,
                            requested_positions_per_context=fd.requested_fused_positions_per_context,
                        )
                        
                        # Reconcile trace totals with report totals (P0-16-21)
                        for context in self.REQUIRED_CONTEXTS:
                            if context not in fd.compressed_page_dispatches_per_context:
                                reasons.append(
                                    f"Fused decode report missing compressed_page_dispatches_per_context for context {context}"
                                )
                                continue
                            
                            report_page_ops = fd.compressed_page_dispatches_per_context[context]
                            trace_page_ops = topology_result.get("trace_computed_page_ops", {}).get(context, 0)
                            if report_page_ops != trace_page_ops:
                                reasons.append(
                                    f"Context {context}: report compressed_page_dispatches {report_page_ops} "
                                    f"!= trace-computed {trace_page_ops}"
                                )
                            
                            report_tail_ops = fd.dense_tail_dispatches_per_context.get(context, 0)
                            trace_tail_ops = topology_result.get("trace_computed_tail_ops", {}).get(context, 0)
                            if report_tail_ops != trace_tail_ops:
                                reasons.append(
                                    f"Context {context}: report dense_tail_dispatches {report_tail_ops} "
                                    f"!= trace-computed {trace_tail_ops}"
                                )
                            
                            report_fallback_ops = fd.fallback_calls_per_context.get(context, 0)
                            trace_fallback_ops = topology_result.get("trace_computed_fallback_ops", {}).get(context, 0)
                            if report_fallback_ops != trace_fallback_ops:
                                reasons.append(
                                    f"Context {context}: report fallback_calls {report_fallback_ops} "
                                    f"!= trace-computed {trace_fallback_ops}"
                                )
                        
                        # Reconcile trace totals with global bridge totals (P1-27)
                        total_trace_page_ops = sum(topology_result.get("trace_computed_page_ops", {}).values())
                        total_trace_tail_ops = sum(topology_result.get("trace_computed_tail_ops", {}).values())
                        total_trace_fallback_ops = sum(topology_result.get("trace_computed_fallback_ops", {}).values())
                        
                        if fd.compressed_page_metal_calls is not None:
                            if fd.compressed_page_metal_calls != total_trace_page_ops:
                                reasons.append(
                                    f"Global compressed_page_metal_calls {fd.compressed_page_metal_calls} "
                                    f"!= trace-computed total {total_trace_page_ops}"
                                )
                        
                        if fd.dense_tail_metal_calls is not None:
                            if fd.dense_tail_metal_calls != total_trace_tail_ops:
                                reasons.append(
                                    f"Global dense_tail_metal_calls {fd.dense_tail_metal_calls} "
                                    f"!= trace-computed total {total_trace_tail_ops}"
                                )
                        
                        # Sum of fallback types should match total fallbacks
                        total_type_fallbacks = _fused_fallback_total(fd)

                        if total_type_fallbacks != total_trace_fallback_ops:
                            reasons.append(
                                f"Sum of typed fallbacks {total_type_fallbacks} "
                                f"!= trace-computed total {total_trace_fallback_ops}"
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
                    reasons.append(f"Invalid trace artifact: {exc}")
        else:
            # Unknown evidence kind - treat as incomplete
            reasons.append(f"Unknown evidence kind: {evidence.provenance.evidence_kind}")

        # Strict Metal execution verification.
        if fd.execution_mode is None:
            reasons.append("Fused decode execution_mode is missing.")
        elif fd.execution_mode != "metal_strict":
            reasons.append(
                f"Fused decode execution_mode='{fd.execution_mode}' != 'metal_strict'; "
                "only strict Metal runs are eligible for promotion."
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
                reasons.append(f"Fused decode {field}=0; no Metal dispatches recorded.")
        if fd.merge_metal_calls is not None and fd.merge_metal_calls > 0:
            reasons.append("Fused decode reported merge_metal_calls>0, but merge uses MLX operations.")
        if fd.finalization_metal_calls is not None and fd.finalization_metal_calls > 0:
            reasons.append("Fused decode reported finalization_metal_calls>0, but finalization uses MLX operations.")

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
                    f"Fused decode {field}={val}; fallback occurred in strict mode."
                )

        if fd.fallback_reasons is None:
            reasons.append("Fused decode fallback_reasons missing.")
        elif fd.fallback_reasons:
            reasons.append(f"Fused decode had fallback reasons: {fd.fallback_reasons}")

        # Speed
        sr = evidence.speed_report
        if (
            sr.min_ratio_at_4096_plus is None
            or sr.min_ratio_at_4096_plus < self.MAX_REGRESSION_AT_4096_PLUS
        ):
            reasons.append(
                f"Speed ratio at 4096+ minimum {sr.min_ratio_at_4096_plus} < {self.MAX_REGRESSION_AT_4096_PLUS}"
            )
        
        # Require strict execution and zero fallbacks in speed evidence
        if sr.execution_mode is None:
            reasons.append("Speed evidence execution_mode is missing.")
        elif sr.execution_mode != "metal_strict":
            reasons.append(
                f"Speed evidence execution_mode='{sr.execution_mode}' != 'metal_strict'; "
                "only strict Metal runs are eligible for promotion."
            )
        if sr.fallback_calls != 0:
            reasons.append(
                f"Speed evidence fallback_calls={sr.fallback_calls}; fallback occurred in strict mode."
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
                # P1-25: Parse speed trials and recompute ratios
                raw_timing = json.loads(content)
                if not isinstance(raw_timing, dict):
                    reasons.append("Speed raw timing must be a JSON object")
                else:
                    # P1-26: Require five trials and 128 latency values per context
                    # Handle unified schema (speed_evidence.trial_results), benchmark format (trial_results), and legacy format (trials)
                    trial_results = None
                    if "speed_evidence" in raw_timing:
                        # Unified schema format
                        speed_evidence = raw_timing.get("speed_evidence", {})
                        trial_results = speed_evidence.get("trial_results", [])
                    elif "trial_results" in raw_timing:
                        # Benchmark format from run_speed_matrix.py
                        trial_results = raw_timing.get("trial_results", [])
                    
                    if trial_results is not None:
                        if not isinstance(trial_results, list):
                            reasons.append("Speed trial_results must be a list")
                        else:
                            # Group by context and mode
                            context_mode_counts: Dict[int, Dict[str, int]] = {}
                            for trial in trial_results:
                                if not isinstance(trial, dict):
                                    continue
                                context = trial.get("context_length")
                                mode = trial.get("mode")
                                if context is None or mode is None:
                                    continue
                                
                                if context not in context_mode_counts:
                                    context_mode_counts[context] = {"dense": 0, "turbo": 0}
                                context_mode_counts[context][mode] += 1
                            
                            # Validate trial counts
                            for context in self.REQUIRED_CONTEXTS:
                                if context not in context_mode_counts:
                                    reasons.append(f"Speed raw timing missing context {context}")
                                    continue
                                
                                dense_count = context_mode_counts[context].get("dense", 0)
                                turbo_count = context_mode_counts[context].get("turbo", 0)
                                
                                if dense_count < self.MIN_TRIALS_PER_CONTEXT:
                                    reasons.append(
                                        f"Context {context}: has {dense_count} dense trials "
                                        f"< required {self.MIN_TRIALS_PER_CONTEXT}"
                                    )
                                if turbo_count < self.MIN_TRIALS_PER_CONTEXT:
                                    reasons.append(
                                        f"Context {context}: has {turbo_count} turbo trials "
                                        f"< required {self.MIN_TRIALS_PER_CONTEXT}"
                                    )
                                
                                # Validate latency counts
                                for trial in trial_results:
                                    if trial.get("context_length") == context:
                                        latencies = trial.get("per_token_ms", [])
                                        if not isinstance(latencies, list) or len(latencies) < self.REQUIRED_FORCED_DECODE_TOKENS:
                                            reasons.append(
                                                f"Context {context} trial: has {len(latencies) if isinstance(latencies, list) else 0} latencies "
                                                f"< required {self.REQUIRED_FORCED_DECODE_TOKENS}"
                                            )
                    else:
                        # Legacy format
                        trials = raw_timing.get("trials", {})
                        for context in self.REQUIRED_CONTEXTS:
                            if context not in trials:
                                reasons.append(f"Speed raw timing missing context {context}")
                                continue
                            context_trials = trials[context]
                            if not isinstance(context_trials, list) or len(context_trials) < self.MIN_TRIALS_PER_CONTEXT:
                                reasons.append(
                                    f"Context {context}: has {len(context_trials) if isinstance(context_trials, list) else 0} trials "
                                    f"< required {self.MIN_TRIALS_PER_CONTEXT}"
                                )
                            for trial_idx, trial in enumerate(context_trials):
                                if not isinstance(trial, dict):
                                    reasons.append(f"Context {context} trial {trial_idx}: not a dict")
                                    continue
                                latencies = trial.get("latencies", [])
                                if not isinstance(latencies, list) or len(latencies) < self.REQUIRED_FORCED_DECODE_TOKENS:
                                    reasons.append(
                                        f"Context {context} trial {trial_idx}: has {len(latencies) if isinstance(latencies, list) else 0} latencies "
                                        f"< required {self.REQUIRED_FORCED_DECODE_TOKENS}"
                                    )
                    
                    # P1-25: Recompute speed ratios from raw timing
                    recomputed_speed = _recompute_speed_ratios(raw_timing)
                    if recomputed_speed is not None:
                        if abs(recomputed_speed["min_ratio_4096_plus"] - (sr.min_ratio_at_4096_plus or 0)) > 0.01:
                            reasons.append(
                                f"Speed min ratio 4096+ mismatch: report {sr.min_ratio_at_4096_plus} "
                                f"!= recomputed {recomputed_speed['min_ratio_4096_plus']}"
                            )
                        if abs(recomputed_speed["max_ratio_4096_plus"] - (sr.max_ratio_at_4096_plus or 0)) > 0.01:
                            reasons.append(
                                f"Speed max ratio 4096+ mismatch: report {sr.max_ratio_at_4096_plus} "
                                f"!= recomputed {recomputed_speed['max_ratio_4096_plus']}"
                            )
                        if abs(recomputed_speed["median_ratio_8192_plus"] - (sr.median_ratio_at_8192_plus or 0)) > 0.01:
                            reasons.append(
                                f"Speed median ratio 8192+ mismatch: report {sr.median_ratio_at_8192_plus} "
                                f"!= recomputed {recomputed_speed['median_ratio_8192_plus']}"
                            )
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
                EvidenceValidationError,
            ) as exc:
                reasons.append(f"Invalid speed raw timing artifact: {exc}")
        
        # Require baseline contexts through 16K
        if 16384 not in sr.contexts_evaluated:
            reasons.append("Speed evidence missing required 16K context length.")
        
        # Baseline comparison must include 16K
        bc = evidence.baseline_comparison_report
        if bc and 16384 not in bc.contexts_evaluated:
            reasons.append("Baseline comparison missing required 16K context length.")
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

        # Experiment completeness: per-context fused decode evidence.
        # Use required-set inclusion so reports with diagnostic contexts still pass.
        fd = evidence.fused_decode_report
        contexts_set = (
            set(fd.contexts_evaluated) if fd and fd.contexts_evaluated else set()
        )
        if not self.REQUIRED_CONTEXTS.issubset(contexts_set):
            reasons.append(
                f"Fused decode contexts incomplete: required {self.REQUIRED_CONTEXTS} not in {contexts_set}"
            )

        # Per-context completeness gates.
        for context in self.REQUIRED_CONTEXTS:
            if context not in fd.positions_per_context:
                reasons.append(f"Missing fused decode positions for context {context}")
            elif fd.positions_per_context.get(context, 0) < self.REQUIRED_FORCED_DECODE_TOKENS:
                reasons.append(
                    f"Context {context}: fused positions {fd.positions_per_context.get(context, 0)} "
                    f"< required {self.REQUIRED_FORCED_DECODE_TOKENS}"
                )
            if fd.failed_positions_per_context.get(context, 0) != 0:
                reasons.append(f"Context {context}: has failed fused positions")
            if fd.fallback_calls_per_context.get(context, 0) != 0:
                reasons.append(f"Context {context}: has fallback calls")

        # Legacy fallback check (global).
        if fd and fd.actual_fused_positions is not None:
            if fd.actual_fused_positions < self.REQUIRED_FORCED_DECODE_TOKENS:
                reasons.append(
                    f"Fused decode actual positions {fd.actual_fused_positions} < "
                    f"required {self.REQUIRED_FORCED_DECODE_TOKENS}"
                )
        else:
            reasons.append("Fused decode actual_fused_positions missing.")

        sr = evidence.speed_report
        if sr and sr.trials_per_context < self.MIN_TRIALS_PER_CONTEXT:
            reasons.append(
                f"Speed trials per context {sr.trials_per_context} < {self.MIN_TRIALS_PER_CONTEXT}"
            )
        speed_contexts = set(sr.contexts_evaluated) if sr and sr.contexts_evaluated else set()
        if not self.REQUIRED_CONTEXTS.issubset(speed_contexts):
            reasons.append(
                f"Speed contexts incomplete: required {self.REQUIRED_CONTEXTS} not in {speed_contexts}"
            )
        mr = evidence.memory_report
        memory_contexts = set(mr.contexts_evaluated) if mr and mr.contexts_evaluated else set()
        if not self.REQUIRED_CONTEXTS.issubset(memory_contexts):
            reasons.append(
                f"Memory contexts incomplete: required {self.REQUIRED_CONTEXTS} not in {memory_contexts}"
            )

        # Provenance
        pv = evidence.provenance

        # P1-29: Require token fixtures hash for reproducibility
        if not pv.token_fixtures_hash:
            reasons.append("Token fixtures hash is missing; exact fixtures required for reproducibility.")
        elif len(pv.token_fixtures_hash) < 32:  # At least half of SHA-256 hex
            reasons.append(
                f"Token fixtures hash '{pv.token_fixtures_hash}' is too short; "
                "full SHA-256 hash required for reproducibility."
            )

        # P1-30: Require immutable model and tokenizer revisions
        if not pv.model_revision:
            reasons.append("Model revision is empty; immutable git commit hash required.")
        elif pv.model_revision == "unknown":
            reasons.append("Model revision is 'unknown'; immutable git commit hash required.")
        elif len(pv.model_revision) < 7:  # At least short git hash
            reasons.append(
                f"Model revision '{pv.model_revision}' is too short; "
                "immutable git commit hash required (at least 7 characters)."
            )
        
        if not pv.tokenizer_revision:
            reasons.append("Tokenizer revision is empty; immutable git commit hash required.")
        elif pv.tokenizer_revision == "unknown":
            reasons.append("Tokenizer revision is 'unknown'; immutable git commit hash required.")
        elif len(pv.tokenizer_revision) < 7:
            reasons.append(
                f"Tokenizer revision '{pv.tokenizer_revision}' is too short; "
                "immutable git commit hash required (at least 7 characters)."
            )
        
        # Check evidence kind - synthetic evidence is never promotable
        if pv.evidence_kind != "experimental":
            return PromotionDecision(
                state=PromotionState.REVIEW_REQUIRED,
                reasons=[
                    "Synthetic or non-experimental evidence is never promotable."
                ],
                evidence=evidence,
            )
        
        if pv.git_tree_state == GitTreeState.UNKNOWN:
            reasons.append("Git tree state unknown; cannot verify reproducibility.")
        if pv.git_tree_state == GitTreeState.DIRTY:
            reasons.append(
                f"Source tree was dirty (diff hash {pv.git_diff_hash}); promotion requires a clean tree."
            )
        if not pv.model_repo_id or not pv.model_revision:
            reasons.append("Model provenance incomplete.")
        if not pv.turbopolar_config_hash:
            reasons.append("TurboPolar config hash missing.")

        # Explicit promotion decision ordering
        # 1. Hard quantitative/artifact failures -> FAILED
        if reasons:
            return PromotionDecision(
                state=PromotionState.FAILED,
                reasons=reasons,
                evidence=evidence,
            )

        # 2. Evidence missing or provenance unknown -> INCOMPLETE
        if pv.git_tree_state == GitTreeState.UNKNOWN:
            return PromotionDecision(
                state=PromotionState.INCOMPLETE,
                reasons=["Git tree state unknown; cannot verify reproducibility."],
                evidence=evidence,
            )

        # 3. Promotion locked -> REVIEW_REQUIRED
        if self.PROMOTION_LOCKED:
            return PromotionDecision(
                state=PromotionState.REVIEW_REQUIRED,
                reasons=[
                    "Promotion is locked pending independent native evidence review."
                ],
                evidence=evidence,
            )

        # 4. All checks pass -> PROMOTED_EXPERIMENTAL
        return PromotionDecision(
            state=PromotionState.PROMOTED_EXPERIMENTAL,
            reasons=["All required evidence present and passing."],
            evidence=evidence,
        )
