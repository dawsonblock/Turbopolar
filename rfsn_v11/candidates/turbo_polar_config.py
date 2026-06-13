import math
from dataclasses import dataclass, field
from typing import Dict, Any

from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode


def validate_supported_configuration(config: "TurboPolarConfig") -> None:
    """Reject unsupported configurations immediately.

    Supported contract:
      - Platform: Apple Silicon
      - Runtime: MLX + mlx-lm
      - Model: one verified mlx_lm Llama implementation
      - Batch size: 1
      - Attention: full-history causal GQA
      - Decode query length: 1
      - Mask: None
      - Head dimension: 128
      - Block size: 64
      - K format: log-int8 radius + 8-bit angle
      - V format: grouped int8
      - QJL: disabled
      - Sliding window: unsupported
      - Speculative decoding: unsupported
    """
    if config.head_dim != 128:
        raise ValueError("TurboPolar requires head_dim=128")
    if config.block_size != 64:
        raise ValueError("TurboPolar requires block_size=64")
    if config.num_q_heads <= 0 or config.num_kv_heads <= 0:
        raise ValueError("Head counts must be positive")
    if config.num_q_heads % config.num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if config.use_qjl:
        raise NotImplementedError("QJL remains disabled")
    if config.storage_mode != "kv_quant":
        raise NotImplementedError("Only kv_quant storage is supported")
    if config.attention_scale <= 0:
        raise ValueError("attention_scale must be positive")


@dataclass(frozen=True)
class TurboPolarConfig:
    """
    Immutable configuration structure governing the TurboPolar Alpha 9 runtime.

    Supported initial target (Revision 4):
      - Runtime: MLX + mlx-lm
      - Model: one exact mlx_lm Llama class
      - Batch size: 1
      - Attention: full-history causal GQA
      - Decode query length: 1
      - Head dimension: 128
      - KV block size: 64
      - K format: log-int8 radius + 8-bit angle
      - V format: grouped int8
      - QJL: disabled
      - Mask: None only
      - Sliding window: unsupported
      - Speculative decoding: unsupported

    Anything outside this narrow scope is unsupported and raises an error.
    """

    k_angle_bits_level1: int = 8
    k_angle_bits_deep: int = 8
    use_int8_radii: bool = True
    v_bits: int = 8
    block_size: int = 64
    head_dim: int = 128
    qjl_proj_dim: int = 64
    use_qjl: bool = False
    storage_mode: str = "kv_quant"
    seed: int = 42
    split_dim: int = 0
    attention_scale: float = 0.0  # 0 = auto-compute as 1/sqrt(head_dim)
    num_q_heads: int = 32
    num_kv_heads: int = 8
    validate_finite_inputs: bool = False
    finite_audit_interval: int = 0
    execution_mode: ExecutionMode = ExecutionMode.DEVELOPMENT_AUTO
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.num_q_heads <= 0:
            raise ValueError("num_q_heads must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        if self.head_dim != 128:
            raise ValueError(
                "TurboPolar fused MLX path currently requires head_dim=128"
            )
        if self.block_size != 64:
            raise ValueError(
                "TurboPolar fused MLX path currently requires block_size=64"
            )
        if self.use_qjl:
            raise NotImplementedError(
                "QJL is disabled until fused real-model validation passes"
            )
        if self.storage_mode != "kv_quant":
            raise ValueError(
                "TurboPolar only supports storage_mode='kv_quant' in this release"
            )
        if self.v_bits != 8:
            raise ValueError(
                "v_bits must be 8 (4-bit V quantization is not yet implemented)"
            )

        if self.k_angle_bits_level1 not in (4, 8):
            raise ValueError("k_angle_bits_level1 must be 4 or 8")
        if self.k_angle_bits_deep not in (2, 4, 8):
            raise ValueError("k_angle_bits_deep must be 2, 4, or 8")

        if self.qjl_proj_dim <= 0:
            raise ValueError("qjl_proj_dim must be positive")
        if self.qjl_proj_dim % 8 != 0:
            raise ValueError("qjl_proj_dim must be divisible by 8 for bit packing")

        if (
            self.split_dim < 0
            or self.split_dim > self.head_dim
            or self.split_dim % 2 != 0
        ):
            raise ValueError("split_dim must be even and within [0, head_dim]")

        if self.finite_audit_interval < 0:
            raise ValueError("finite_audit_interval must be non-negative")

        # Normalize execution_mode to enum for type safety.
        mode = self.execution_mode
        if isinstance(mode, str):
            mode = ExecutionMode(mode)
        if not isinstance(mode, ExecutionMode):
            raise TypeError(
                f"execution_mode must be an ExecutionMode, got {type(mode).__name__}"
            )
        object.__setattr__(self, "execution_mode", mode)

        attention_scale = self.attention_scale
        if attention_scale == 0.0:
            attention_scale = 1.0 / math.sqrt(self.head_dim)
        if attention_scale <= 0:
            raise ValueError("attention_scale must be positive")
        object.__setattr__(self, "attention_scale", float(attention_scale))
