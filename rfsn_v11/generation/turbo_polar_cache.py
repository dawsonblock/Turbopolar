import mlx.core as mx
from typing import List, Dict, Any, Tuple, Optional

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
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

    Storage modes:
      - "kv_quant": default; polar-compressed K + grouped int8 V + optional QJL.
      - "dense_v_debug": polar-compressed K + dense fp16 V + optional QJL.
      - "k_only_first": polar-compressed K only (no V storage, no QJL).
    """
    def __init__(self, config: TurboPolarConfig):
        self.config = config
        self.polar_encoder = PolarQuantEncoder(config)
        self.v_quantizer = GroupedVQuantizer(group_size=32)
        self.qjl_encoder = QJLResidualEncoder(config)
        self.decoder = PolarQuantDecoder()

        self.partial_k: Optional[mx.array] = None
        self.partial_v: Optional[mx.array] = None
        self.compressed_k_blocks: List[PolarKeyBlock] = []
        self.compressed_v_blocks: List[QuantizedVBlock] = []
        self.dense_v_blocks: List[mx.array] = []
        self.qjl_blocks: List[QJLPayload] = []
        self.actual_seq_len = 0
        self.total_blocks = 0
        self.bytes_written = 0
        self.bytes_read = 0

    def _validate_append_inputs(self, k_new: mx.array, v_new: mx.array):
        if not isinstance(k_new, mx.array) or not isinstance(v_new, mx.array):
            raise TypeError("append() expects mlx.core.array inputs")
        if k_new.ndim != 4 or v_new.ndim != 4:
            raise ValueError(f"append() expects 4-D inputs (B, H_kv, T, D), got {k_new.shape} and {v_new.shape}")
        if k_new.shape != v_new.shape:
            raise ValueError(f"k_new shape {k_new.shape} must match v_new shape {v_new.shape}")
        B, H_kv, T_new, D = k_new.shape
        if H_kv != self.config.num_kv_heads:
            raise ValueError(f"input has {H_kv} KV heads but config expects {self.config.num_kv_heads}")
        if D != self.config.head_dim:
            raise ValueError(f"input head_dim {D} does not match config {self.config.head_dim}")
        if T_new < 1:
            raise ValueError(f"input must have at least one token, got T={T_new}")
        if self.partial_k is not None and B != self.partial_k.shape[0]:
            raise ValueError(f"batch size changed from {self.partial_k.shape[0]} to {B}")
        if k_new.dtype != v_new.dtype:
            raise ValueError(f"k_new dtype {k_new.dtype} must match v_new dtype {v_new.dtype}")
        if k_new.dtype not in (mx.float16, mx.float32):
            raise ValueError(f"only float16 and float32 inputs are supported, got {k_new.dtype}")

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

        # Decode the freshly-encoded block so we can compute a residual sketch for QJL.
        # The decoder accepts unified 5D payloads, so expand the single-block dims.
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

        quant_v: Optional[QuantizedVBlock] = None
        dense_v: Optional[mx.array] = None
        qjl_payload: Optional[QJLPayload] = None

        if self.config.storage_mode == "kv_quant":
            quant_v = self.v_quantizer.quantize_block(v_block.reshape(B, H, 1, L, D))
            self.compressed_v_blocks.append(quant_v)
        elif self.config.storage_mode == "dense_v_debug":
            dense_v = v_block.reshape(B, H, 1, L, D)
            self.dense_v_blocks.append(dense_v)
        # "k_only_first" stores no V payload.

        if self.config.use_qjl:
            residual = k_block - k_recon
            qjl_payload = self.qjl_encoder.compute_residual_sketch(residual.reshape(B, H, 1, L, D))
            self.qjl_blocks.append(qjl_payload)

        self.compressed_k_blocks.append(polar_block)
        self.total_blocks += 1

        # Honest byte accounting: count every payload that is actually stored.
        self.bytes_written += (
            _nbytes(polar_block.radii) +
            _nbytes(polar_block.angle_codes_l1) +
            _nbytes(polar_block.angle_codes_deep)
        )
        if polar_block.radii_scales is not None:
            self.bytes_written += _nbytes(polar_block.radii_scales)
        if quant_v is not None:
            self.bytes_written += _nbytes(quant_v.codes) + _nbytes(quant_v.scales)
        if dense_v is not None:
            self.bytes_written += _nbytes(dense_v)
        if qjl_payload is not None:
            self.bytes_written += _nbytes(qjl_payload.packed_signs) + _nbytes(qjl_payload.norms)

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

        quant_v: Optional[QuantizedVBlock] = None
        dense_v: Optional[mx.array] = None
        if self.config.storage_mode == "kv_quant":
            quant_v = self.v_quantizer.quantize_block(v_padded.reshape(B, H, 1, L, D))
        elif self.config.storage_mode == "dense_v_debug":
            dense_v = v_padded.reshape(B, H, 1, L, D)

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
            "dense_v": dense_v,
            "qjl": qjl_payload,
            "metadata": polar_block.metadata,
        }

    def get_blocks_for_attention(self) -> Tuple[Optional[PolarKeyBlock], Optional[QuantizedVBlock], Optional[mx.array], Optional[QJLPayload], int]:
        """
        Return unified attention blocks. The partial tail, if any, is padded into a
        temporary final block so that every token in [0, actual_seq_len) is attendable.
        """
        partial = self._maybe_encode_partial()

        if not self.compressed_k_blocks:
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
            return unified_k, partial["quant_v"], partial["dense_v"], partial["qjl"], self.actual_seq_len

        radii = mx.stack([b.radii for b in self.compressed_k_blocks], axis=2)
        angle_l1 = mx.stack([b.angle_codes_l1 for b in self.compressed_k_blocks], axis=2)
        angle_deep = mx.stack([b.angle_codes_deep for b in self.compressed_k_blocks], axis=2)
        has_scales = all(b.radii_scales is not None for b in self.compressed_k_blocks)
        radii_scales = mx.stack([b.radii_scales for b in self.compressed_k_blocks], axis=2) if has_scales else None

        partial_quant_v: Optional[QuantizedVBlock] = None
        partial_dense_v: Optional[mx.array] = None
        partial_qjl: Optional[QJLPayload] = None

        if partial is not None:
            radii = mx.concatenate([radii, partial["radii"]], axis=2)
            angle_l1 = mx.concatenate([angle_l1, partial["angle_l1"]], axis=2)
            angle_deep = mx.concatenate([angle_deep, partial["angle_deep"]], axis=2)
            if radii_scales is not None and partial.get("radii_scales") is not None:
                radii_scales = mx.concatenate([radii_scales, partial["radii_scales"]], axis=2)
            partial_quant_v = partial["quant_v"]
            partial_dense_v = partial["dense_v"]
            partial_qjl = partial["qjl"]

        unified_k = PolarKeyBlock(
            radii=radii,
            angle_codes_l1=angle_l1,
            angle_codes_deep=angle_deep,
            radii_scales=radii_scales,
            shape=(radii.shape[0], radii.shape[1], radii.shape[2] * radii.shape[3], radii.shape[4] * 2),
            block_size=self.config.block_size,
            head_dim=self.config.head_dim,
            metadata=self.compressed_k_blocks[0].metadata,
        )

        unified_v: Optional[QuantizedVBlock] = None
        unified_dense_v: Optional[mx.array] = None
        if self.config.storage_mode == "kv_quant" and self.compressed_v_blocks:
            # Each stored block is [B,H,1,L,*]; stack creates [B,H,S,1,L,*], squeeze the singleton.
            v_codes = mx.squeeze(mx.stack([b.codes for b in self.compressed_v_blocks], axis=2), axis=3)
            v_scales = mx.squeeze(mx.stack([b.scales for b in self.compressed_v_blocks], axis=2), axis=3)
            if partial_quant_v is not None:
                v_codes = mx.concatenate([v_codes, partial_quant_v.codes], axis=2)
                v_scales = mx.concatenate([v_scales, partial_quant_v.scales], axis=2)
            unified_v = QuantizedVBlock(
                codes=v_codes, scales=v_scales,
                group_size=self.compressed_v_blocks[0].group_size,
            )
        elif self.config.storage_mode == "dense_v_debug" and self.dense_v_blocks:
            unified_dense_v = mx.squeeze(mx.stack(self.dense_v_blocks, axis=2), axis=3)
            if partial_dense_v is not None:
                unified_dense_v = mx.concatenate([unified_dense_v, partial_dense_v], axis=2)

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

        return unified_k, unified_v, unified_dense_v, unified_qjl, self.actual_seq_len

    def fetch_blocks(self, indices: Optional[List[int]] = None) -> mx.array:
        if not self.compressed_k_blocks:
            return mx.array([])
        if indices is None:
            indices = list(range(len(self.compressed_k_blocks)))
        fetched = []
        for idx in indices:
            block = self.compressed_k_blocks[idx]
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
        if not self.compressed_k_blocks and self.partial_k is None:
            return {}
        D = self.config.head_dim
        if self.compressed_k_blocks:
            B = int(self.compressed_k_blocks[0].radii.shape[0])
            H = int(self.compressed_k_blocks[0].radii.shape[1])
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
        }
