"""Operation-level execution trace collection tests.

Verify that decode_attention preserves detailed traces with layer/step/page
identity, and that strict validation catches missing pages and fallbacks.
"""

import pytest
import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.integrations.mlx_lm.cache import TurboPolarFastCache
from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode
from rfsn_v11.evidence.execution_trace import (
    AttentionExecutionTrace,
    KernelOperationTrace,
    ExecutionTraceCollector,
)


def _make_config():
    return TurboPolarConfig(
        head_dim=128,
        block_size=64,
        num_q_heads=4,
        num_kv_heads=4,
        use_int8_radii=True,
        k_angle_bits_level1=8,
        k_angle_bits_deep=8,
        storage_mode="kv_quant",
    )


@pytest.mark.native_metal_required
class TestExecutionTraceCollection:
    def test_one_page_trace(self):
        """64 tokens = 1 page, no tail."""
        mx.random.seed(7000)
        config = _make_config()
        cache = TurboPolarFastCache(config)
        cache.reset_execution_stats()
        cache.clear_execution_traces()

        k = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
        cache.runtime.append(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        _ = cache.decode_attention(
            q, k[:, :, -1:, :], v[:, :, -1:, :],
            config.attention_scale,
            layer_index=0, decode_step=0, experiment_id="exp_1page",
        )

        traces = cache.execution_traces()
        assert len(traces) == 1
        t = traces[0]
        assert t.layer_index == 0
        assert t.decode_step == 0
        assert t.expected_page_count == 1
        assert len(t.page_traces) == 1
        assert t.page_traces[0].operation == "compressed_page"
        assert t.page_traces[0].page_index == 0
        assert t.page_traces[0].metal_executed is True
        assert t.fallback_count == 0

    def test_two_pages_trace(self):
        """1088 tokens = 17 blocks = 2 pages (16 + 1)."""
        mx.random.seed(7001)
        config = _make_config()
        cache = TurboPolarFastCache(config)
        cache.reset_execution_stats()
        cache.clear_execution_traces()

        k = mx.random.normal((1, 4, 1088, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 1088, 128)).astype(mx.float16)
        cache.runtime.append(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        _ = cache.decode_attention(
            q, k[:, :, -1:, :], v[:, :, -1:, :],
            config.attention_scale,
            layer_index=1, decode_step=5, experiment_id="exp_2page",
        )

        traces = cache.execution_traces()
        assert len(traces) == 1
        t = traces[0]
        assert t.layer_index == 1
        assert t.decode_step == 5
        assert t.expected_page_count == 2
        assert len(t.page_traces) == 2
        assert t.page_traces[0].page_index == 0
        assert t.page_traces[1].page_index == 1
        assert t.fallback_count == 0

    def test_pages_plus_dense_tail_trace(self):
        """65 tokens = 1 page + 1 tail token."""
        mx.random.seed(7002)
        config = _make_config()
        cache = TurboPolarFastCache(config)
        cache.reset_execution_stats()
        cache.clear_execution_traces()

        k = mx.random.normal((1, 4, 65, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 65, 128)).astype(mx.float16)
        cache.runtime.append(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        _ = cache.decode_attention(
            q, k[:, :, -1:, :], v[:, :, -1:, :],
            config.attention_scale,
            layer_index=2, decode_step=10, experiment_id="exp_tail",
        )

        traces = cache.execution_traces()
        assert len(traces) == 1
        t = traces[0]
        assert t.layer_index == 2
        assert t.decode_step == 10
        assert len(t.page_traces) == 1
        assert t.dense_tail_trace is not None
        assert t.dense_tail_trace.operation == "dense_tail"
        assert t.fallback_count == 0

    def test_multiple_layers_and_steps(self):
        """Simulate multiple layers and decode steps with unique identities."""
        mx.random.seed(7003)
        config = _make_config()
        cache = TurboPolarFastCache(config)
        cache.reset_execution_stats()
        cache.clear_execution_traces()

        k = mx.random.normal((1, 4, 128, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 128, 128)).astype(mx.float16)
        cache.runtime.append(k, v)

        q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
        for layer in range(3):
            for step in range(4):
                _ = cache.decode_attention(
                    q, k[:, :, -1:, :], v[:, :, -1:, :],
                    config.attention_scale,
                    layer_index=layer, decode_step=step,
                    experiment_id="exp_multi",
                )

        traces = cache.execution_traces()
        assert len(traces) == 12
        seen = set()
        for t in traces:
            key = (t.layer_index, t.decode_step)
            assert key not in seen
            seen.add(key)
        assert len(seen) == 12

    def test_strict_missing_page_raises(self):
        """Inject a missing page trace and verify strict validation raises."""
        mx.random.seed(7004)
        config = _make_config()
        import dataclasses
        config = dataclasses.replace(config, execution_mode=ExecutionMode.METAL_STRICT)
        cache = TurboPolarFastCache(config)
        cache.reset_execution_stats()
        cache.clear_execution_traces()

        k = mx.random.normal((1, 4, 128, 128)).astype(mx.float16)
        v = mx.random.normal((1, 4, 128, 128)).astype(mx.float16)
        cache.runtime.append(k, v)

        # Manually inject a trace with wrong page count to test validation.
        from rfsn_v11.evidence.execution_trace import AttentionExecutionTrace, KernelOperationTrace
        from rfsn_v11.evidence.trace_validation import validate_trace_topology, TraceTopologyError

        cache._trace_collector.record(
            AttentionExecutionTrace(
                experiment_id="exp",
                context_length=512,
                fixture_id="fixture-001",
                layer_index=0,
                decode_step=512,
                decode_ordinal=0,
                cache_offset_before=512,
                cache_tokens_before=512,
                cache_tokens_after=513,
                partial_tail_length=1,
                expected_page_count=2,
                page_traces=[
                    KernelOperationTrace(
                        experiment_id="exp",
                        context_length=512,
                        fixture_id="fixture-001",
                        layer_index=0,
                        decode_step=512,
                        decode_ordinal=0,
                        cache_offset_before=512,
                        operation="compressed_page",
                        page_index=0,
                        kernel_name="test",
                        execution_mode="metal_strict",
                        metal_requested=True,
                        metal_executed=True,
                        fallback_used=False,
                        fallback_reason=None,
                        expected_tokens=64,
                        processed_tokens=64,
                    )
                ],
            )
        )

        # Retrieve traces and run topology validation
        traces = cache.execution_traces()
        assert len(traces) == 1

        # Parse the raw trace for validation
        from rfsn_v11.evidence.trace_validation import parse_attention_trace
        parsed = [parse_attention_trace(traces[0].model_dump(), entry_index=0)]

        # Validate topology - should fail due to missing page (expected 2, got 1)
        with self.assertRaises(TraceTopologyError) as cm:
            validate_trace_topology(
                traces=parsed,
                model_layer_count=1,
                required_contexts={512},
                requested_positions_per_context=1,
            )
        self.assertIn("page count", str(cm.exception).lower())

    def test_commit_provisional_preserves_all_fields(self):
        """Test that commit_provisional preserves all trace identity and topology fields."""
        collector = ExecutionTraceCollector(experiment_id="test_exp")
        
        # Create a provisional trace with all new fields
        provisional_trace = AttentionExecutionTrace(
            experiment_id="test_exp",
            context_length=512,
            fixture_id="fixture-001",
            layer_index=0,
            decode_step=512,
            decode_ordinal=0,
            cache_offset_before=512,
            cache_tokens_before=512,
            cache_tokens_after=513,
            partial_tail_length=1,
            expected_page_count=1,
            page_traces=[
                KernelOperationTrace(
                    experiment_id="test_exp",
                    context_length=512,
                    fixture_id="fixture-001",
                    layer_index=0,
                    decode_step=512,
                    decode_ordinal=0,
                    cache_offset_before=512,
                    operation="compressed_page",
                    page_index=0,
                    kernel_name="test_kernel",
                    execution_mode="metal_strict",
                    metal_requested=True,
                    metal_executed=True,
                    fallback_used=False,
                    fallback_reason=None,
                    expected_tokens=64,
                    processed_tokens=64,
                    output_evaluated=False,  # Will be updated on commit
                )
            ],
            dense_tail_trace=None,
        )
        
        collector.record_provisional(provisional_trace)
        assert len(collector._provisional) == 1
        assert len(collector._traces) == 0
        
        # Commit with output_evaluated=True
        collector.commit_provisional(output_evaluated=True)
        
        assert len(collector._provisional) == 0
        assert len(collector._traces) == 1
        
        committed = collector._traces[0]
        
        # Verify all identity fields are preserved
        assert committed.experiment_id == "test_exp"
        assert committed.context_length == 512
        assert committed.fixture_id == "fixture-001"
        assert committed.layer_index == 0
        assert committed.decode_step == 512
        assert committed.decode_ordinal == 0
        assert committed.cache_offset_before == 512
        assert committed.cache_tokens_before == 512
        assert committed.cache_tokens_after == 513
        assert committed.partial_tail_length == 1
        assert committed.expected_page_count == 1
        
        # Verify output_evaluated was updated
        assert committed.page_traces[0].output_evaluated == True
