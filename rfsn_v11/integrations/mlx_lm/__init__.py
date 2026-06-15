"""Instance-level mlx_lm Llama adapter for TurboPolar.

This package provides a non-global, reversible adapter that replaces Llama
attention modules with wrappers routing decode steps through the fused
TurboPolar Metal kernels.
"""

from rfsn_v11.integrations.mlx_lm.telemetry import KernelExecutionStats

try:
    from rfsn_v11.integrations.mlx_lm.adapter import TurboPolarLlamaAdapter
    from rfsn_v11.integrations.mlx_lm.attention import TurboPolarLlamaAttention
    from rfsn_v11.integrations.mlx_lm.cache import TurboPolarFastCache, make_turbo_caches
except ImportError:
    # MLX-dependent modules are unavailable in non-MLX environments.
    # Telemetry dataclasses remain importable for testing and evidence
    # processing.
    TurboPolarLlamaAdapter = None  # type: ignore[misc,assignment]
    TurboPolarLlamaAttention = None  # type: ignore[misc,assignment]
    TurboPolarFastCache = None  # type: ignore[misc,assignment]
    make_turbo_caches = None  # type: ignore[misc,assignment]

__all__ = [
    "TurboPolarLlamaAdapter",
    "TurboPolarLlamaAttention",
    "TurboPolarFastCache",
    "make_turbo_caches",
    "KernelExecutionStats",
]
