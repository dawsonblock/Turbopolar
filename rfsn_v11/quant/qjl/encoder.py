import mlx.core as mx
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig


@dataclass
class QJLPayload:
    packed_signs: mx.array
    norms: mx.array
    proj_dim: int
    seed: int
    shape: tuple


class QJLResidualEncoder:
    def __init__(self, config: "TurboPolarConfig"):
        self.config = config
        self.proj_dim = config.qjl_proj_dim
        self.seed = getattr(config, "seed", 42)
        self.head_dim = config.head_dim
        rng = np.random.default_rng(self.seed)
        w_np = rng.standard_normal((self.head_dim, self.proj_dim), dtype=np.float32)
        w_np = w_np / (np.linalg.norm(w_np, axis=0, keepdims=True) + 1e-12)
        self.W = mx.array(w_np)

    def compute_residual_sketch(self, residual_E: mx.array) -> QJLPayload:
        B, H, S, L, D = residual_E.shape
        assert D == self.head_dim
        flat_E = residual_E.reshape(-1, D)
        proj = mx.matmul(flat_E, self.W)
        proj = proj.reshape(B, H, S, L, self.proj_dim)
        signs = proj >= 0
        reshaped = signs.reshape(B, H, S, L, self.proj_dim // 8, 8)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        packed = mx.sum(reshaped.astype(mx.uint8) * powers, axis=-1).astype(mx.uint8)
        norms = mx.sqrt(mx.sum(residual_E * residual_E, axis=-1)).astype(mx.float16)
        return QJLPayload(
            packed_signs=packed,
            norms=norms,
            proj_dim=self.proj_dim,
            seed=self.seed,
            shape=(B, H, S, L, D),
        )
