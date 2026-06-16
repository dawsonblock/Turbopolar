import math
from dataclasses import dataclass, field
from typing import Dict, Any

from rfsn_v11.kernels.turbo_polar.execution import (
    ExecutionMode,
    TraceValidationMode,
)


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
      - Page capacity: 16 blocks per page
      - K format: log-int8 radius + 8-bit angle (both level1 and deep)
      - V format: grouped int8
      - QJL: disabled
      - Sliding window: unsupported
      - Speculative decoding: unsupported
    """
    if config.head_dim not in (64, 128):
        raise ValueError("TurboPolar requires head_dim=64 or 128")
    if config.block_size != 64:
        raise ValueError("TurboPolar requires block_size=64")
    if config.page_capacity_blocks != 16:
        raise ValueError("TurboPolar requires page_capacity_blocks=16")
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
    if config.k_angle_bits_level1 != 8:
        raise ValueError(
            "Supported configuration requires 8-bit level-1 angles"
        )
    if config.k_angle_bits_deep != 8:
        raise ValueError("Supported configuration requires 8-bit deep angles")


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
    page_capacity_blocks: int = 16
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
    trace_validation_mode: TraceValidationMode = (
        TraceValidationMode.SYNCHRONOUS_EVIDENCE
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Dense-tail capacity: how many recent tokens to keep in dense fp16 before
    # flushing to compressed storage.  Larger values reduce per-decode
    # overhead (the primary speed bottleneck) at the cost of more memory.
    # Must be >= block_size and a multiple of block_size so flushes are clean.
    dense_tail_capacity: int = 4096
    # Flush batch size: how many tokens to compress and flush at once when
    # the dense tail reaches capacity.  Must be a multiple of block_size,
    # >= block_size, and <= dense_tail_capacity.
    # Default is half the dense tail so the most recent tokens always stay
    # in fast dense memory.
    flush_batch_size: int = 2048
    # Page pool preallocation: how many empty pages to pre-allocate after the
    # first compressed block is stored.  Eliminates per-decode Metal allocator
    # fragmentation at long contexts.  0 = disabled (allocate on demand).
    page_pool_prealloc: int = 0
    # Warm-cache capacity in blocks.  Uncompressed dense blocks evicted from
    # the hot window are kept here before being compressed to cold storage.
    # 0 = disabled (backward compatible).  Must be a multiple of
    # ``flush_batch_size // block_size`` for clean batch transitions.
    warm_cache_capacity_blocks: int = 0

    def __post_init__(self):
        if self.num_q_heads <= 0:
            raise ValueError("num_q_heads must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        if self.head_dim not in (64, 128):
            raise ValueError(
                "TurboPolar fused MLX path currently requires head_dim=64 or 128"
            )
        if self.block_size != 64:
            raise ValueError(
                "TurboPolar fused MLX path currently requires block_size=64"
            )
        if self.page_capacity_blocks != 16:
            raise ValueError(
                "TurboPolar requires page_capacity_blocks=16"
            )
        if self.use_qjl:
            raise NotImplementedError(
                "QJL is disabled until fused real-model validation passes"
            )
        if self.storage_mode != "kv_quant":
            raise ValueError(
                "TurboPolar only supports storage_mode='kv_quant'"
            )
        if self.v_bits != 8:
            raise ValueError(
                "v_bits must be 8 (4-bit V quantization not yet implemented)"
            )

        if self.k_angle_bits_level1 not in (4, 8):
            raise ValueError("k_angle_bits_level1 must be 4 or 8")
        if self.k_angle_bits_deep not in (2, 4, 8):
            raise ValueError("k_angle_bits_deep must be 2, 4, or 8")

        if self.qjl_proj_dim <= 0:
            raise ValueError("qjl_proj_dim must be positive")
        if self.qjl_proj_dim % 8 != 0:
            raise ValueError(
                "qjl_proj_dim must be divisible by 8 for bit packing"
            )

        if (
            self.split_dim < 0
            or self.split_dim > self.head_dim
            or self.split_dim % 2 != 0
        ):
            raise ValueError("split_dim must be even and within [0, head_dim]")

        if self.finite_audit_interval < 0:
            raise ValueError("finite_audit_interval must be non-negative")

        if self.dense_tail_capacity < self.block_size:
            raise ValueError(
                f"dense_tail_capacity ({self.dense_tail_capacity}) must be >= "
                f"block_size ({self.block_size})"
            )
        if self.dense_tail_capacity % self.block_size != 0:
            raise ValueError(
                f"dense_tail_capacity ({self.dense_tail_capacity}) must be a "
                f"multiple of block_size ({self.block_size})"
            )
        if self.flush_batch_size < self.block_size:
            raise ValueError(
                f"flush_batch_size ({self.flush_batch_size}) must be >= "
                f"block_size ({self.block_size})"
            )
        if self.flush_batch_size % self.block_size != 0:
            raise ValueError(
                f"flush_batch_size ({self.flush_batch_size}) must be a "
                f"multiple of block_size ({self.block_size})"
            )
        if self.flush_batch_size > self.dense_tail_capacity:
            raise ValueError(
                f"flush_batch_size ({self.flush_batch_size}) must be <= "
                f"dense_tail_capacity ({self.dense_tail_capacity})"
            )
        if self.page_pool_prealloc < 0:
            raise ValueError("page_pool_prealloc must be non-negative")
        if self.warm_cache_capacity_blocks < 0:
            raise ValueError(
                "warm_cache_capacity_blocks must be non-negative"
            )

        # Normalize execution_mode to enum for type safety.
        mode = self.execution_mode
        if isinstance(mode, str):
            mode = ExecutionMode(mode)
        if not isinstance(mode, ExecutionMode):
            raise TypeError(
                "execution_mode must be an ExecutionMode, got "
                f"{type(mode).__name__}"
            )
        object.__setattr__(self, "execution_mode", mode)

        # Normalize trace_validation_mode to enum for type safety.
        trace_mode = self.trace_validation_mode
        if isinstance(trace_mode, str):
            trace_mode = TraceValidationMode(trace_mode)
        if not isinstance(trace_mode, TraceValidationMode):
            raise TypeError(
                "trace_validation_mode must be a TraceValidationMode, got "
                f"{type(trace_mode).__name__}"
            )
        object.__setattr__(self, "trace_validation_mode", trace_mode)

        attention_scale = self.attention_scale
        if attention_scale == 0.0:
            attention_scale = 1.0 / math.sqrt(self.head_dim)
        if attention_scale <= 0:
            raise ValueError("attention_scale must be positive")
        object.__setattr__(self, "attention_scale", float(attention_scale))
