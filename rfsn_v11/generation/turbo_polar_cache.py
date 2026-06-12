import mlx.core as mx
from typing import Dict, Any, Tuple, Optional

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.storage import PolarKBlockStorage, QuantVBlockStorage
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.polar.payload import PolarKeyBlock
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer, QuantizedVBlock
from rfsn_v11.quant.qjl.encoder import QJLResidualEncoder, QJLPayload


def _nbytes(x: mx.array) -> int:
    """Return the number of bytes occupied by an MLX array's data."""
    return int(x.size * x.itemsize)


class TurboPolarKVCacheRuntime:
    """
    Stateful incremental Key-Value Cache with bit-packed PolarQuant.
    Handles GQA: KV heads stored at native resolution, broadcasted at attention time.

    Storage:
      - Completed full blocks are compressed once into persistent, capacity-growing
        storage objects (PolarKBlockStorage, QuantVBlockStorage).
      - The active partial block (length < block_size) remains dense and mutable.
      - This eliminates per-token re-encoding of the partial tail and avoids
        rebuilding the full history via mx.stack() on every decode step.
    """
    def __init__(self, config: TurboPolarConfig):
        self.config = config
        self.polar_encoder = PolarQuantEncoder(config)
        self.v_quantizer = GroupedVQuantizer(group_size=32)
        self.qjl_encoder = QJLResidualEncoder(config)
        self.decoder = PolarQuantDecoder()

        self.partial_k: Optional[mx.array] = None
        self.partial_v: Optional[mx.array] = None
        self.k_storage = PolarKBlockStorage()
        self.v_storage = QuantVBlockStorage()
        self.qjl_blocks: list[QJLPayload] = []
        self.actual_seq_len = 0
        self.total_blocks = 0
        self.bytes_written = 0
        self.bytes_read = 0
        self.compression_time_ns = 0

        # Persistent invariants validated across all appends, even after full flushes.
        self._initialized = False
        self._batch_size: Optional[int] = None
        self._num_kv_heads: Optional[int] = None
        self._head_dim: Optional[int] = None
        self._input_dtype = None

    def _validate_append_inputs(self, k_new: mx.array, v_new: mx.array):
        if not isinstance(k_new, mx.array) or not isinstance(v_new, mx.array):
            raise TypeError("append() expects mlx.core.array inputs")
        if k_new.ndim != 4 or v_new.ndim != 4:
            raise ValueError(f"append() expects 4-D inputs (B, H_kv, T, D), got {k_new.shape} and {v_new.shape}")
        if k_new.shape != v_new.shape:
            raise ValueError(f"k_new shape {k_new.shape} must match v_new shape {v_new.shape}")
        B, H_kv, T_new, D = k_new.shape
        if T_new < 1:
            raise ValueError(f"input must have at least one token, got T={T_new}")
        if k_new.dtype != v_new.dtype:
            raise ValueError(f"k_new dtype {k_new.dtype} must match v_new dtype {v_new.dtype}")
        if k_new.dtype not in (mx.float16, mx.float32):
            raise ValueError(f"only float16 and float32 inputs are supported, got {k_new.dtype}")
        if not mx.isfinite(k_new).all().item() or not mx.isfinite(v_new).all().item():
            raise ValueError("append() inputs must contain finite values")

        # Persistent invariants: validated regardless of partial-block state.
        if self._initialized:
            if B != self._batch_size:
                raise ValueError(
                    f"batch size changed from {self._batch_size} to {B}; "
                    "TurboPolar cache does not support varying batch size."
                )
            if H_kv != self._num_kv_heads:
                raise ValueError(
                    f"KV head count changed from {self._num_kv_heads} to {H_kv}; "
                    "TurboPolar cache does not support varying head counts."
                )
            if D != self._head_dim:
                raise ValueError(
                    f"head_dim changed from {self._head_dim} to {D}; "
                    "TurboPolar cache does not support varying head dimension."
                )
            if k_new.dtype != self._input_dtype:
                raise ValueError(
                    f"input dtype changed from {self._input_dtype} to {k_new.dtype}; "
                    "TurboPolar cache does not support varying dtype."
                )
        else:
            # First append establishes invariants.
            if H_kv != self.config.num_kv_heads:
                raise ValueError(f"input has {H_kv} KV heads but config expects {self.config.num_kv_heads}")
            if D != self.config.head_dim:
                raise ValueError(f"input head_dim {D} does not match config {self.config.head_dim}")
            self._batch_size = B
            self._num_kv_heads = H_kv
            self._head_dim = D
            self._input_dtype = k_new.dtype
            self._initialized = True

    def append(self, k_new: mx.array, v_new: mx.array):
        self._validate_append_inputs(k_new, v_new)
        B, H, T_new, D = k_new.shape
        self.actual_seq_len += T_new
        if self.partial_k is None:
            self.partial_k = k_new
            self.partial_v = v_new
        else:
            self.partial_k = mx.concatenate([self.partial_k, k_new], axis=2)
            self.partial_v = mx.concatenate([self.partial_v, v_new], axis=2)

        L = self.config.block_size
        while self.partial_k is not None and self.partial_k.shape[2] >= L:
            k_block = self.partial_k[:, :, :L, :]
            v_block = self.partial_v[:, :, :L, :]
            self._flush_block(k_block, v_block)
            self.partial_k = self.partial_k[:, :, L:, :]
            self.partial_v = self.partial_v[:, :, L:, :]
            if self.partial_k.shape[2] == 0:
                self.partial_k = None
                self.partial_v = None

    def _flush_block(self, k_block: mx.array, v_block: mx.array):
        B, H, L, D = k_block.shape
        polar_block = self.polar_encoder.encode_block(k_block)
        quant_v = self.v_quantizer.quantize_block(v_block.reshape(B, H, 1, L, D))

        self.k_storage.append(polar_block)
        self.v_storage.append(quant_v)
        self.total_blocks += 1

        if self.config.use_qjl:
            unified_for_decode = PolarKeyBlock(
                radii=mx.expand_dims(polar_block.radii, axis=2),
                angle_codes_l1=mx.expand_dims(polar_block.angle_codes_l1, axis=2),
                angle_codes_deep=mx.expand_dims(polar_block.angle_codes_deep, axis=2),
                radii_scales=mx.expand_dims(polar_block.radii_scales, axis=2) if polar_block.radii_scales is not None else None,
                shape=(B, H, 1, L, D),
                block_size=L,
                head_dim=D,
                metadata=polar_block.metadata,
            )
            k_recon = self.decoder.decode_block(unified_for_decode).reshape(B, H, L, D)
            residual = k_block - k_recon
            qjl_payload = self.qjl_encoder.compute_residual_sketch(residual.reshape(B, H, 1, L, D))
            self.qjl_blocks.append(qjl_payload)
            self.bytes_written += _nbytes(qjl_payload.packed_signs) + _nbytes(qjl_payload.norms)

        self.bytes_written += (
            _nbytes(polar_block.radii) +
            _nbytes(polar_block.angle_codes_l1) +
            _nbytes(polar_block.angle_codes_deep)
        )
        if polar_block.radii_scales is not None:
            self.bytes_written += _nbytes(polar_block.radii_scales)
        self.bytes_written += _nbytes(quant_v.codes) + _nbytes(quant_v.scales)

    def get_fused_attention_inputs(
        self
    ) -> Tuple[Optional[PolarKeyBlock], Optional[QuantizedVBlock], Optional[mx.array], Optional[mx.array], Optional[QJLPayload], int]:
        """
        Return inputs for fused attention without re-encoding the partial tail.

        Returns:
            compressed_k: PolarKeyBlock for completed full blocks, or None.
            quant_v: QuantizedVBlock for completed full blocks, or None.
            partial_k: dense [B, H_kv, T_part, D] tail, or None.
            partial_v: dense [B, H_kv, T_part, D] tail, or None.
            qjl_payload: unified QJL payload for completed blocks, or None.
            actual_seq_len: total number of valid tokens.
        """
        compressed_k = None
        quant_v = None
        if self.k_storage.block_count > 0:
            B = self._batch_size
            H = self._num_kv_heads
            D = self._head_dim
            L = self.config.block_size
            compressed_k = self.k_storage.to_unified_block((B, H, self.k_storage.block_count * L, D))
            quant_v = self.v_storage.to_quantized_block()

        unified_qjl: Optional[QJLPayload] = None
        if self.qjl_blocks:
            qjl_signs = mx.squeeze(mx.stack([b.packed_signs for b in self.qjl_blocks], axis=2), axis=3)
            qjl_norms = mx.squeeze(mx.stack([b.norms for b in self.qjl_blocks], axis=2), axis=3)
            unified_qjl = QJLPayload(
                packed_signs=qjl_signs, norms=qjl_norms,
                proj_dim=self.qjl_blocks[0].proj_dim,
                seed=self.qjl_blocks[0].seed,
                shape=(qjl_signs.shape[0], qjl_signs.shape[1], qjl_signs.shape[2], qjl_signs.shape[3], self.config.head_dim),
            )

        return compressed_k, quant_v, self.partial_k, self.partial_v, unified_qjl, self.actual_seq_len

    def get_blocks_for_attention(self) -> Tuple[Optional[PolarKeyBlock], Optional[QuantizedVBlock], Optional[mx.array], Optional[QJLPayload], int]:
        """
        Return unified attention blocks for the decompress-on-read path.
        The partial tail, if any, is padded into a temporary final block so that
        every token in [0, actual_seq_len) is attendable.
        """
        partial = self._maybe_encode_partial()

        if self.k_storage.block_count == 0:
            if partial is None:
                return None, None, None, None, self.actual_seq_len
            unified_k = PolarKeyBlock(
                radii=partial["radii"],
                angle_codes_l1=partial["angle_l1"],
                angle_codes_deep=partial["angle_deep"],
                radii_scales=partial.get("radii_scales"),
                shape=(partial["radii"].shape[0], partial["radii"].shape[1],
                       partial["radii"].shape[2] * partial["radii"].shape[3],
                       partial["radii"].shape[4] * 2),
                block_size=self.config.block_size,
                head_dim=self.config.head_dim,
                metadata=partial["metadata"],
            )
            return unified_k, partial["quant_v"], None, partial.get("qjl"), self.actual_seq_len

        compressed_k = self.k_storage.to_unified_block(
            (self._batch_size, self._num_kv_heads, self.k_storage.block_count * self.config.block_size, self._head_dim)
        )
        quant_v = self.v_storage.to_quantized_block()

        partial_quant_v: Optional[QuantizedVBlock] = None
        partial_qjl: Optional[QJLPayload] = None

        if partial is not None:
            radii = mx.concatenate([compressed_k.radii, partial["radii"]], axis=2)
            angle_l1 = mx.concatenate([compressed_k.angle_codes_l1, partial["angle_l1"]], axis=2)
            angle_deep = mx.concatenate([compressed_k.angle_codes_deep, partial["angle_deep"]], axis=2)
            radii_scales = None
            if compressed_k.radii_scales is not None and partial.get("radii_scales") is not None:
                radii_scales = mx.concatenate([compressed_k.radii_scales, partial["radii_scales"]], axis=2)
            compressed_k = PolarKeyBlock(
                radii=radii,
                angle_codes_l1=angle_l1,
                angle_codes_deep=angle_deep,
                radii_scales=radii_scales,
                shape=(radii.shape[0], radii.shape[1], radii.shape[2] * radii.shape[3], radii.shape[4] * 2),
                block_size=self.config.block_size,
                head_dim=self.config.head_dim,
                metadata=compressed_k.metadata,
            )
            partial_quant_v = partial["quant_v"]
            partial_qjl = partial.get("qjl")

        if partial_quant_v is not None:
            quant_v = QuantizedVBlock(
                codes=mx.concatenate([quant_v.codes, partial_quant_v.codes], axis=2),
                scales=mx.concatenate([quant_v.scales, partial_quant_v.scales], axis=2),
                group_size=quant_v.group_size,
            )

        unified_qjl: Optional[QJLPayload] = None
        if self.qjl_blocks:
            qjl_signs = mx.squeeze(mx.stack([b.packed_signs for b in self.qjl_blocks], axis=2), axis=3)
            qjl_norms = mx.squeeze(mx.stack([b.norms for b in self.qjl_blocks], axis=2), axis=3)
            if partial_qjl is not None:
                qjl_signs = mx.concatenate([qjl_signs, partial_qjl.packed_signs], axis=2)
                qjl_norms = mx.concatenate([qjl_norms, partial_qjl.norms], axis=2)
            unified_qjl = QJLPayload(
                packed_signs=qjl_signs, norms=qjl_norms,
                proj_dim=self.qjl_blocks[0].proj_dim,
                seed=self.qjl_blocks[0].seed,
                shape=(qjl_signs.shape[0], qjl_signs.shape[1], qjl_signs.shape[2], qjl_signs.shape[3], self.config.head_dim),
            )

        return compressed_k, quant_v, None, unified_qjl, self.actual_seq_len

    def _maybe_encode_partial(self) -> Optional[Dict[str, Any]]:
        """Encode the partial tail padded to a full block, returning unified-shape tensors."""
        if self.partial_k is None:
            return None
        B, H, T_part, D = self.partial_k.shape
        L = self.config.block_size
        pad = L - T_part
        pad_width = [(0, 0), (0, 0), (0, pad), (0, 0)]
        k_padded = mx.pad(self.partial_k, pad_width)
        v_padded = mx.pad(self.partial_v, pad_width)

        polar_block = self.polar_encoder.encode_block(k_padded)
        radii = mx.expand_dims(polar_block.radii, axis=2)
        radii_scales = mx.expand_dims(polar_block.radii_scales, axis=2) if polar_block.radii_scales is not None else None
        angle_l1 = mx.expand_dims(polar_block.angle_codes_l1, axis=2)
        angle_deep = mx.expand_dims(polar_block.angle_codes_deep, axis=2)

        quant_v = self.v_quantizer.quantize_block(v_padded.reshape(B, H, 1, L, D))

        qjl_payload: Optional[QJLPayload] = None
        if self.config.use_qjl:
            unified_tmp = PolarKeyBlock(
                radii=radii,
                angle_codes_l1=angle_l1,
                angle_codes_deep=angle_deep,
                radii_scales=radii_scales,
                shape=(B, H, 1, L, D),
                block_size=L,
                head_dim=D,
                metadata=polar_block.metadata,
            )
            k_recon = self.decoder.decode_block(unified_tmp).reshape(B, H, L, D)
            residual = k_padded - k_recon
            qjl_payload = self.qjl_encoder.compute_residual_sketch(residual.reshape(B, H, 1, L, D))

        return {
            "radii": radii,
            "radii_scales": radii_scales,
            "angle_l1": angle_l1,
            "angle_deep": angle_deep,
            "quant_v": quant_v,
            "qjl": qjl_payload,
            "metadata": polar_block.metadata,
        }

    def fetch_blocks(self, indices: Optional[list[int]] = None) -> mx.array:
        if self.k_storage.block_count == 0:
            return mx.array([])
        if indices is None:
            indices = list(range(self.k_storage.block_count))
        fetched = []
        for idx in indices:
            block = PolarKeyBlock(
                radii=self.k_storage.radii[:, :, idx:idx + 1, :, :],
                angle_codes_l1=self.k_storage.angle_codes_l1[:, :, idx:idx + 1, :, :],
                angle_codes_deep=self.k_storage.angle_codes_deep[:, :, idx:idx + 1, :, :],
                radii_scales=self.k_storage.radii_scales[:, :, idx:idx + 1, :, :] if self.k_storage.radii_scales is not None else None,
                shape=(self._batch_size, self._num_kv_heads, self.config.block_size, self._head_dim),
                block_size=self.config.block_size,
                head_dim=self._head_dim,
                metadata=self.k_storage.metadata,
            )
            self.bytes_read += (
                _nbytes(block.radii) +
                _nbytes(block.angle_codes_l1) +
                _nbytes(block.angle_codes_deep)
            )
            if block.radii_scales is not None:
                self.bytes_read += _nbytes(block.radii_scales)
            fetched.append(self.decoder.decode_block(block))
        return mx.concatenate(fetched, axis=2)

    def get_io_telemetry(self) -> Dict[str, Any]:
        if self.k_storage.block_count == 0 and self.partial_k is None:
            return {}
        D = self.config.head_dim
        if self.k_storage.block_count > 0:
            B = int(self.k_storage.radii.shape[0])
            H = int(self.k_storage.radii.shape[1])
        else:
            B = int(self.partial_k.shape[0])
            H = int(self.partial_k.shape[1])

        # Dense baseline: full fp16 K + full fp16 V for every token in the sequence.
        dense_kv_bytes = B * H * self.actual_seq_len * D * 2 * 2

        # Compressed cache: flushed payloads plus raw partial K/V tails.
        compressed_bytes = self.bytes_written
        if self.partial_k is not None:
            compressed_bytes += _nbytes(self.partial_k) + _nbytes(self.partial_v)

        return {
            "dense_kv_bytes": dense_kv_bytes,
            "actual_cache_bytes": compressed_bytes,
            "compression_ratio": float(dense_kv_bytes / compressed_bytes) if compressed_bytes > 0 else 0.0,
            "total_blocks": self.total_blocks,
            "partial_tokens": self.partial_k.shape[2] if self.partial_k is not None else 0,
            "k_storage_capacity": self.k_storage.capacity,
            "v_storage_capacity": self.v_storage.capacity,
            "k_storage_reallocs": self.k_storage.reallocation_count,
            "v_storage_reallocs": self.v_storage.reallocation_count,
        }

    def reset(self):
        """Clear all cache state and persistent invariants."""
        self.partial_k = None
        self.partial_v = None
        self.k_storage = PolarKBlockStorage()
        self.v_storage = QuantVBlockStorage()
        self.qjl_blocks = []
        self.actual_seq_len = 0
        self.total_blocks = 0
        self.bytes_written = 0
        self.bytes_read = 0
        self.compression_time_ns = 0
        self._initialized = False
        self._batch_size = None
        self._num_kv_heads = None
        self._head_dim = None
        self._input_dtype = None
