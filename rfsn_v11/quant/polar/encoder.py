import mlx.core as mx
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig

from rfsn_v11.quant.polar.payload import PolarKeyBlock


class PolarQuantEncoder:
    def __init__(self, config: "TurboPolarConfig"):
        self.config = config
        self.head_dim = config.head_dim
        self.block_size = config.block_size
        self.split_dim = getattr(config, "split_dim", config.head_dim // 2)
        self.k_angle_bits_level1 = config.k_angle_bits_level1
        self.k_angle_bits_deep = config.k_angle_bits_deep
        self.l1_levels = 2 ** self.k_angle_bits_level1
        self.deep_levels = 2 ** self.k_angle_bits_deep
        self.l1_scale = float(self.l1_levels - 1)
        self.deep_scale = float(self.deep_levels - 1)

    def encode_block(self, k_block: mx.array) -> PolarKeyBlock:
        B, H, L, D = k_block.shape
        assert D == self.head_dim
        assert D % 2 == 0
        half_d = D // 2
        split_half = self.split_dim // 2
        k_pairs = k_block.reshape(B, H, L, half_d, 2)
        x = k_pairs[..., 0]
        y = k_pairs[..., 1]
        radii = mx.sqrt(x * x + y * y).astype(mx.float16)
        angles = mx.arctan2(y, x)
        shifted = angles + np.pi
        norm_angles = shifted / (2.0 * np.pi)
        norm_angles = mx.clip(norm_angles, 0.0, 1.0)
        epsilon = 1e-6
        norm_angles = mx.where(norm_angles > (1.0 - epsilon), mx.array(0.0), norm_angles)
        norm_l1 = norm_angles[..., :split_half]
        norm_deep = norm_angles[..., split_half:]
        codes_l1 = mx.clip(mx.round(norm_l1 * self.l1_scale), 0, self.l1_levels - 1).astype(mx.uint8)
        codes_deep = mx.clip(mx.round(norm_deep * self.deep_scale), 0, self.deep_levels - 1).astype(mx.uint8)
        # BIT-PACK: 4-bit codes -> 2 per byte
        codes_l1_packed = self._pack_4bit(codes_l1)
        # BIT-PACK: deep codes using configured bit width
        if self.k_angle_bits_deep == 4:
            codes_deep_packed = self._pack_4bit(codes_deep)
        elif self.k_angle_bits_deep == 2:
            codes_deep_packed = self._pack_2bit(codes_deep)
        else:
            raise ValueError(f"unsupported k_angle_bits_deep: {self.k_angle_bits_deep}")
        return PolarKeyBlock(
            radii=radii,
            angle_codes_l1=codes_l1_packed,
            angle_codes_deep=codes_deep_packed,
            shape=(B, H, L, D),
            block_size=L,
            head_dim=D,
            metadata={
                "l1_scale": self.l1_scale,
                "deep_scale": self.deep_scale,
                "split_dim": self.split_dim,
                "l1_packed": True,
                "deep_packed": True,
                "deep_bits": self.k_angle_bits_deep,
                "l1_original_len": split_half,
                "deep_original_len": half_d - split_half,
            },
        )

    @staticmethod
    def _pack_4bit(codes: mx.array) -> mx.array:
        # codes: [..., N] uint8, values 0-15
        # packed: [..., ceil(N/2)] uint8
        N = codes.shape[-1]
        pad = (2 - N % 2) % 2
        if pad > 0:
            padded = mx.pad(codes, [(0, 0)] * (codes.ndim - 1) + [(0, pad)])
        else:
            padded = codes
        even = padded[..., 0::2]
        odd = padded[..., 1::2]
        packed = (even & 0x0F) | ((odd & 0x0F) << 4)
        return packed.astype(mx.uint8)

    @staticmethod
    def _pack_2bit(codes: mx.array) -> mx.array:
        # codes: [..., N] uint8, values 0-3
        # packed: [..., ceil(N/4)] uint8
        N = codes.shape[-1]
        pad = (4 - N % 4) % 4
        if pad > 0:
            padded = mx.pad(codes, [(0, 0)] * (codes.ndim - 1) + [(0, pad)])
        else:
            padded = codes
        b0 = padded[..., 0::4] & 0x03
        b1 = (padded[..., 1::4] & 0x03) << 2
        b2 = (padded[..., 2::4] & 0x03) << 4
        b3 = (padded[..., 3::4] & 0x03) << 6
        packed = b0 | b1 | b2 | b3
        return packed.astype(mx.uint8)
