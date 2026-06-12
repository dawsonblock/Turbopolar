import mlx.core as mx
import numpy as np
from rfsn_v11.quant.polar.payload import PolarKeyBlock


class PolarQuantDecoder:
    def decode_block(self, block: PolarKeyBlock) -> mx.array:
        # Accept both single-block 4D payloads [B,H,L,D/2] and unified 5D payloads [B,H,S,L,D/2]
        if block.radii.ndim == 4:
            radii = mx.expand_dims(block.radii, axis=2)
            angle_l1 = mx.expand_dims(block.angle_codes_l1, axis=2)
            angle_deep = mx.expand_dims(block.angle_codes_deep, axis=2)
            radii_scales = (
                mx.expand_dims(block.radii_scales, axis=2)
                if block.radii_scales is not None
                else None
            )
            block = PolarKeyBlock(
                radii=radii,
                angle_codes_l1=angle_l1,
                angle_codes_deep=angle_deep,
                radii_scales=radii_scales,
                shape=block.shape[:2] + (1,) + block.shape[2:],
                block_size=block.block_size,
                head_dim=block.head_dim,
                metadata=block.metadata,
            )
        B, H, S, L, _ = block.radii.shape
        D = block.head_dim
        split_half = block.metadata.get("split_dim", D // 2) // 2
        l1_scale = float(block.metadata["l1_scale"])
        deep_scale = float(block.metadata["deep_scale"])
        l1_orig = block.metadata.get("l1_original_len", split_half)
        deep_orig = block.metadata.get("deep_original_len", D // 2 - split_half)

        # Unpack level1 (4-bit or 8-bit)
        l1_bits = block.metadata.get("l1_bits", 4)
        if block.metadata.get("l1_packed", False):
            norm_l1 = self._unpack_4bit(block.angle_codes_l1, l1_orig).astype(mx.float32) / l1_scale
        else:
            norm_l1 = block.angle_codes_l1[..., :l1_orig].astype(mx.float32) / l1_scale

        # Unpack deep (2-bit, 4-bit, or 8-bit depending on metadata)
        deep_bits = block.metadata.get("deep_bits", 2)
        if block.metadata.get("deep_packed", False):
            if deep_bits == 4:
                norm_deep = self._unpack_4bit(block.angle_codes_deep, deep_orig).astype(mx.float32) / deep_scale
            elif deep_bits == 2:
                norm_deep = self._unpack_2bit(block.angle_codes_deep, deep_orig).astype(mx.float32) / deep_scale
            else:
                raise ValueError(f"unsupported deep_bits: {deep_bits}")
        else:
            norm_deep = block.angle_codes_deep[..., :deep_orig].astype(mx.float32) / deep_scale

        norm_angles = mx.concatenate([norm_l1, norm_deep], axis=-1)
        angles = norm_angles * (2.0 * np.pi) - np.pi
        if block.radii_scales is not None:
            if block.metadata.get("log_radii", False):
                radii_fp = mx.exp(block.radii.astype(mx.float32) * block.radii_scales.astype(mx.float32))
            else:
                radii_fp = block.radii.astype(mx.float32) * block.radii_scales.astype(mx.float32)
        else:
            radii_fp = block.radii.astype(mx.float32)
        k_x = radii_fp * mx.cos(angles)
        k_y = radii_fp * mx.sin(angles)
        k_pairs = mx.stack([k_x, k_y], axis=-1)
        return k_pairs.reshape(B, H, S * L, D).astype(mx.float16)

    @staticmethod
    def _unpack_4bit(packed: mx.array, original_len: int) -> mx.array:
        # packed: [..., M] uint8
        # unpacked: [..., original_len] uint8
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        interleaved = mx.stack([low, high], axis=-1).reshape(packed.shape[:-1] + (-1,))
        return interleaved[..., :original_len]

    @staticmethod
    def _unpack_2bit(packed: mx.array, original_len: int) -> mx.array:
        b0 = packed & 0x03
        b1 = (packed >> 2) & 0x03
        b2 = (packed >> 4) & 0x03
        b3 = (packed >> 6) & 0x03
        interleaved = mx.stack([b0, b1, b2, b3], axis=-1).reshape(packed.shape[:-1] + (-1,))
        return interleaved[..., :original_len]
