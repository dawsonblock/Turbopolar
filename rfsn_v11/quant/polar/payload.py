from dataclasses import dataclass
from typing import Tuple, Dict, Any
import mlx.core as mx


@dataclass
class PolarKeyBlock:
    radii: mx.array
    angle_codes_l1: mx.array
    angle_codes_deep: mx.array
    shape: Tuple[int, ...]
    block_size: int
    head_dim: int
    metadata: Dict[str, Any]
