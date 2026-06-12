"""Instance-level mlx_lm Llama adapter for TurboPolar.

This package provides a non-global, reversible adapter that replaces Llama
attention modules with wrappers routing decode steps through the fused
TurboPolar Metal kernels.
"""

from rfsn_v11.integrations.mlx_lm.llama_adapter import TurboPolarLlamaAdapter
from rfsn_v11.integrations.mlx_lm.llama_attention import TurboPolarLlamaAttention

__all__ = ["TurboPolarLlamaAdapter", "TurboPolarLlamaAttention"]
