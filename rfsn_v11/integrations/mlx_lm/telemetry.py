"""Telemetry dataclasses for TurboPolar MLX-LM integration."""

from dataclasses import dataclass


@dataclass
class KernelExecutionStats:
    """Process-level Metal kernel execution counters for TurboPolar attention."""

    fused_qk_calls: int = 0
    online_attention_calls: int = 0
    dense_tail_calls: int = 0
    fallback_calls: int = 0
    compressed_page_dispatches: int = 0
    dense_tail_dispatches: int = 0
