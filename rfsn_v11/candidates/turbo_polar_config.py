import math
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class TurboPolarConfig:
    """
    Immutable configuration structure governing the TurboPolar Alpha 9 runtime.
    """
    k_angle_bits_level1: int = 4
    k_angle_bits_deep: int = 2
    v_bits: int = 8
    block_size: int = 64
    head_dim: int = 128
    qjl_proj_dim: int = 64
    use_qjl: bool = False
    use_metal: bool = True
    storage_mode: str = "kv_quant"
    seed: int = 42
    split_dim: int = 64
    attention_scale: float = 0.0  # 0 = auto-compute as 1/sqrt(head_dim)
    num_q_heads: int = 32
    num_kv_heads: int = 8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be divisible by 2 for polar pair coordinate mapping")
        if self.qjl_proj_dim <= 0:
            raise ValueError("qjl_proj_dim must be positive")
        if self.qjl_proj_dim % 8 != 0:
            raise ValueError("qjl_proj_dim must be divisible by 8 for bit packing")
        if self.v_bits != 8:
            raise ValueError("v_bits must be 8 (4-bit V quantization is not yet implemented)")
        if self.storage_mode not in {"k_only_first", "kv_quant", "dense_v_debug"}:
            raise ValueError("unknown storage_mode")
        if self.block_size != 64:
            raise ValueError("current Metal kernels require block_size == 64")
        if self.head_dim % 32 != 0:
            raise ValueError("current Metal kernels require head_dim divisible by 32")
        if self.head_dim > 128:
            raise ValueError("current Metal kernels only support head_dim <= 128")
        if self.split_dim < 0 or self.split_dim > self.head_dim or self.split_dim % 2 != 0:
            raise ValueError("split_dim must be even and within [0, head_dim]")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads for GQA")
        if self.attention_scale == 0.0:
            object.__setattr__(self, "attention_scale", 1.0 / math.sqrt(self.head_dim))
