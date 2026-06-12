import time
import mlx.core as mx
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.storage import PolarKBlockStorage, QuantVBlockStorage
from rfsn_v11.quant.polar.encoder import PolarQuantEncoder
from rfsn_v11.quant.polar.payload import PolarKeyBlock
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer, QuantizedVBlock
from rfsn_v11.quant.qjl.encoder import QJLResidualEncoder, QJLPayload
from rfsn_v11.generation.paged_storage import PolarKPage, QuantVPage


@dataclass
class CompressedPageView:
    """One page of compressed K/V data with its valid-block count."""

    k_page: PolarKPage
    v_page: QuantVPage
    valid_blocks: int
    page_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurboPolarAttentionView:
    """Full attention payload for page-based processing."""

    pages: tuple[CompressedPageView, ...]
    partial_k: Optional[mx.array]
    partial_v: Optional[mx.array]
    partial_length: int
    total_tokens: int


@dataclass
class CacheResidencyAudit:
    """Runtime introspection of cache residency."""

    dense_full_k_history_present: bool
    dense_full_v_history_present: bool
    dense_tail_tokens: int
    materialized_compressed_history_present: bool


def _nbytes(x: mx.array) -> int:
    """Return the number of bytes occupied by an MLX array's data."""
    return int(x.size * x.itemsize)


class CacheMemoryStats:
    """Truthful memory accounting for the TurboPolar cache.

    - logical_payload_bytes: only valid compressed blocks and dense tail tokens.
    - allocated_capacity_bytes: full array shapes including unused capacity.
    - dense_tail_bytes: currently allocated dense partial tail buffers.
    - metadata_bytes: small bookkeeping arrays (QJL, scales, etc.).
    - dense_equivalent_bytes: fp16 K+V for the full actual sequence length.
    - logical_compression_ratio: dense_equivalent / logical_payload.
    - allocated_compression_ratio: dense_equivalent / allocated_capacity.
    """

    def __init__(
        self,
        logical_payload_bytes: int,
        allocated_capacity_bytes: int,
        dense_tail_bytes: int,
        metadata_bytes: int,
        dense_equivalent_bytes: int,
    ):
        self.logical_payload_bytes = logical_payload_bytes
        self.allocated_capacity_bytes = allocated_capacity_bytes
        self.dense_tail_bytes = dense_tail_bytes
        self.metadata_bytes = metadata_bytes
        self.dense_equivalent_bytes = dense_equivalent_bytes
        self.logical_compression_ratio = (
            dense_equivalent_bytes / logical_payload_bytes
            if logical_payload_bytes > 0
            else 0.0
        )
        self.allocated_compression_ratio = (
            dense_equivalent_bytes / allocated_capacity_bytes
            if allocated_capacity_bytes > 0
            else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_payload_bytes": self.logical_payload_bytes,
            "allocated_capacity_bytes": self.allocated_capacity_bytes,
            "dense_tail_bytes": self.dense_tail_bytes,
            "metadata_bytes": self.metadata_bytes,
            "dense_equivalent_bytes": self.dense_equivalent_bytes,
            "logical_compression_ratio": self.logical_compression_ratio,
            "allocated_compression_ratio": self.allocated_compression_ratio,
        }


class TurboPolarKVCacheRuntime:
    """
    Stateful incremental Key-Value Cache with bit-packed PolarQuant.
    Handles GQA: KV heads stored at native resolution, broadcasted at attention time.

    Storage:
      - Completed full blocks are compressed once into persistent storage objects
        (PolarKBlockStorage, QuantVBlockStorage).
      - The active partial block (length < block_size) is kept in fixed-size dense
        buffers to avoid per-token concatenation and re-allocation.
    """

    def __init__(self, config: TurboPolarConfig):
        self.config = config
        self.polar_encoder = PolarQuantEncoder(config)
        self.v_quantizer = GroupedVQuantizer(group_size=32)
        self.qjl_encoder = QJLResidualEncoder(config)
        self.decoder = PolarQuantDecoder()

        # Fixed-size dense tail buffers, allocated lazily on first append.
        self.partial_k_buffer: Optional[mx.array] = None
        self.partial_v_buffer: Optional[mx.array] = None
        self.partial_length = 0

        self.k_storage = PolarKBlockStorage()
        self.v_storage = QuantVBlockStorage()
        self.qjl_blocks: list[QJLPayload] = []
        self.actual_seq_len = 0
        self.total_blocks = 0
        self.bytes_written = 0
        self.bytes_read = 0
        self.compression_time_ns = 0
        self.finite_check_calls = 0
        self.forced_scalar_evaluations = 0
        self.audit_time_ns = 0

        # Persistent invariants validated across all appends, even after full flushes.
        self._initialized = False
        self._batch_size: Optional[int] = None
        self._num_kv_heads: Optional[int] = None
        self._head_dim: Optional[int] = None
        self._input_dtype = None

    def _validate_finite(self, k_new: mx.array, v_new: mx.array):
        """Host-side finite check. Only called when explicitly enabled or auditing."""
        self.finite_check_calls += 1
        t0 = time.perf_counter_ns() if hasattr(time, "perf_counter_ns") else 0
        k_ok = mx.isfinite(k_new).all()
        v_ok = mx.isfinite(v_new).all()
        mx.eval(k_ok, v_ok)
        self.forced_scalar_evaluations += 2
        if not bool(k_ok.item()) or not bool(v_ok.item()):
            raise ValueError("append() inputs must contain finite values")
        if hasattr(time, "perf_counter_ns"):
            self.audit_time_ns += time.perf_counter_ns() - t0

    def _validate_append_inputs(self, k_new: mx.array, v_new: mx.array):
        if not isinstance(k_new, mx.array) or not isinstance(v_new, mx.array):
            raise TypeError("append() expects mlx.core.array inputs")
        if k_new.ndim != 4 or v_new.ndim != 4:
            raise ValueError(
                f"append() expects 4-D inputs (B, H_kv, T, D), got {k_new.shape} and {v_new.shape}"
            )
        if k_new.shape != v_new.shape:
            raise ValueError(
                f"k_new shape {k_new.shape} must match v_new shape {v_new.shape}"
            )
        B, H_kv, T_new, D = k_new.shape
        if T_new < 1:
            raise ValueError(f"input must have at least one token, got T={T_new}")
        if k_new.dtype != v_new.dtype:
            raise ValueError(
                f"k_new dtype {k_new.dtype} must match v_new dtype {v_new.dtype}"
            )
        if k_new.dtype not in (mx.float16, mx.float32):
            raise ValueError(
                f"only float16 and float32 inputs are supported, got {k_new.dtype}"
            )

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
                raise ValueError(
                    f"input has {H_kv} KV heads but config expects {self.config.num_kv_heads}"
                )
            if D != self.config.head_dim:
                raise ValueError(
                    f"input head_dim {D} does not match config {self.config.head_dim}"
                )
            self._batch_size = B
            self._num_kv_heads = H_kv
            self._head_dim = D
            self._input_dtype = k_new.dtype
            self._initialized = True
            self._allocate_tail_buffers(B, H_kv, D, k_new.dtype)

    def _allocate_tail_buffers(self, B: int, H_kv: int, D: int, dtype):
        self.partial_k_buffer = mx.zeros(
            (B, H_kv, self.config.block_size, D), dtype=dtype
        )
        self.partial_v_buffer = mx.zeros(
            (B, H_kv, self.config.block_size, D), dtype=dtype
        )
        self.partial_length = 0

    def _current_tail_k(self) -> Optional[mx.array]:
        if self.partial_length == 0 or self.partial_k_buffer is None:
            return None
        return self.partial_k_buffer[:, :, : self.partial_length, :]

    def _current_tail_v(self) -> Optional[mx.array]:
        if self.partial_length == 0 or self.partial_v_buffer is None:
            return None
        return self.partial_v_buffer[:, :, : self.partial_length, :]

    def append(self, k_new: mx.array, v_new: mx.array):
        self._validate_append_inputs(k_new, v_new)

        if self.config.validate_finite_inputs:
            self._validate_finite(k_new, v_new)
        elif (
            self.config.finite_audit_interval > 0
            and self.actual_seq_len % self.config.finite_audit_interval == 0
        ):
            self._validate_finite(k_new, v_new)

        B, H, T_new, D = k_new.shape
        L = self.config.block_size

        # Append tokens one at a time into the fixed tail buffer.
        for t in range(T_new):
            token_k = k_new[:, :, t : t + 1, :]
            token_v = v_new[:, :, t : t + 1, :]
            index = self.partial_length
            self.partial_k_buffer[:, :, index : index + 1, :] = token_k
            self.partial_v_buffer[:, :, index : index + 1, :] = token_v
            self.partial_length += 1
            self.actual_seq_len += 1

            if self.partial_length >= L:
                self._flush_tail_block()

    def _flush_tail_block(self):
        """Flush a full tail buffer into compressed persistent storage."""
        L = self.config.block_size
        k_block = self.partial_k_buffer[:, :, :L, :]
        v_block = self.partial_v_buffer[:, :, :L, :]
        self._flush_block(k_block, v_block)
        # Reuse the fixed buffer by resetting the logical length.  The stale data
        # past partial_length is never attended to because attention kernels receive
        # only the slice [:partial_length].
        self.partial_length = 0
        # Zero the buffer to ensure no old tail data is visible after reset.
        self.partial_k_buffer = mx.zeros_like(self.partial_k_buffer)
        self.partial_v_buffer = mx.zeros_like(self.partial_v_buffer)

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
                radii_scales=mx.expand_dims(polar_block.radii_scales, axis=2)
                if polar_block.radii_scales is not None
                else None,
                shape=(B, H, 1, L, D),
                block_size=L,
                head_dim=D,
                metadata=polar_block.metadata,
            )
            k_recon = self.decoder.decode_block(unified_for_decode).reshape(B, H, L, D)
            residual = k_block - k_recon
            qjl_payload = self.qjl_encoder.compute_residual_sketch(
                residual.reshape(B, H, 1, L, D)
            )
            self.qjl_blocks.append(qjl_payload)
            self.bytes_written += _nbytes(qjl_payload.packed_signs) + _nbytes(
                qjl_payload.norms
            )

        self.bytes_written += (
            _nbytes(polar_block.radii)
            + _nbytes(polar_block.angle_codes_l1)
            + _nbytes(polar_block.angle_codes_deep)
        )
        if polar_block.radii_scales is not None:
            self.bytes_written += _nbytes(polar_block.radii_scales)
        self.bytes_written += _nbytes(quant_v.codes) + _nbytes(quant_v.scales)

    def append_many(self, k_new: mx.array, v_new: mx.array):
        """Vectorized append for prefill: k_new/v_new are [B, H_kv, T, D].

        Does not loop over every incoming token in Python.  A small Python loop
        over pages or batches of blocks is acceptable; a loop over every token is not.
        """
        self._validate_append_inputs(k_new, v_new)

        if self.config.validate_finite_inputs:
            self._validate_finite(k_new, v_new)

        B, H, T_new, D = k_new.shape
        L = self.config.block_size
        t = 0

        # Step 1 — Fill existing partial tail.
        if self.partial_length > 0 and t < T_new:
            remaining = L - self.partial_length
            take = min(T_new - t, remaining)
            self.partial_k_buffer[
                :, :, self.partial_length : self.partial_length + take, :
            ] = k_new[:, :, t : t + take, :]
            self.partial_v_buffer[
                :, :, self.partial_length : self.partial_length + take, :
            ] = v_new[:, :, t : t + take, :]
            self.partial_length += take
            self.actual_seq_len += take
            t += take
            if self.partial_length >= L:
                self._flush_tail_block()

        # Step 2 — Process all complete incoming blocks in a batch.
        remaining = T_new - t
        num_full_blocks = remaining // L
        if num_full_blocks > 0:
            k_full = k_new[:, :, t : t + num_full_blocks * L, :].reshape(
                B, H, num_full_blocks, L, D
            )
            v_full = v_new[:, :, t : t + num_full_blocks * L, :].reshape(
                B, H, num_full_blocks, L, D
            )
            polar_blocks = self.polar_encoder.encode_blocks(k_full)
            quant_v = self.v_quantizer.encode_blocks(v_full)
            # Append each completed block to paged storage.
            for i in range(num_full_blocks):
                pb = PolarKeyBlock(
                    radii=polar_blocks.radii[:, :, i, :, :],
                    angle_codes_l1=polar_blocks.angle_codes_l1[:, :, i, :, :],
                    angle_codes_deep=polar_blocks.angle_codes_deep[:, :, i, :, :],
                    radii_scales=polar_blocks.radii_scales[:, :, i, :, :]
                    if polar_blocks.radii_scales is not None
                    else None,
                    shape=(B, H, L, D),
                    block_size=L,
                    head_dim=D,
                    metadata=polar_blocks.metadata,
                )
                vb = QuantizedVBlock(
                    codes=quant_v.codes[:, :, i : i + 1, :, :],
                    scales=quant_v.scales[:, :, i : i + 1, :, :],
                    group_size=quant_v.group_size,
                )
                self.k_storage.append(pb)
                self.v_storage.append(vb)
                self.total_blocks += 1
                self.actual_seq_len += L
            t += num_full_blocks * L

        # Step 3 — Store final remainder.
        if t < T_new:
            rem = T_new - t
            self.partial_k_buffer[:, :, :rem, :] = k_new[:, :, t:, :]
            self.partial_v_buffer[:, :, :rem, :] = v_new[:, :, t:, :]
            self.partial_length = rem
            self.actual_seq_len += rem

    def attention_view(self) -> TurboPolarAttentionView:
        """Return a page-view attention payload without materializing full history."""
        pages: list[CompressedPageView] = []
        k_paged = self.k_storage._paged
        v_paged = self.v_storage._paged
        num_pages = min(len(k_paged.pages), len(v_paged.pages))
        for i in range(num_pages):
            pages.append(
                CompressedPageView(
                    k_page=k_paged.pages[i],
                    v_page=v_paged.pages[i],
                    valid_blocks=k_paged.pages[i].valid_blocks,
                    page_index=i,
                    metadata=k_paged.metadata,
                )
            )
        return TurboPolarAttentionView(
            pages=tuple(pages),
            partial_k=self._current_tail_k(),
            partial_v=self._current_tail_v(),
            partial_length=self.partial_length,
            total_tokens=self.actual_seq_len,
        )

    def audit_cache_residency(self) -> CacheResidencyAudit:
        """Inspect actual fields and array shapes for dense-history leakage."""
        # After Phase 2, cached materialized history should be zero.
        has_materialized_k = (
            self.k_storage.radii is not None
            and self.k_storage.radii.ndim >= 4
            and self.k_storage.block_count > 0
        )
        has_materialized_v = (
            self.v_storage.codes is not None
            and self.v_storage.codes.ndim >= 4
            and self.v_storage.block_count > 0
        )
        return CacheResidencyAudit(
            dense_full_k_history_present=False,
            dense_full_v_history_present=False,
            dense_tail_tokens=self.partial_length,
            materialized_compressed_history_present=bool(
                has_materialized_k or has_materialized_v
            ),
        )

    def get_fused_attention_inputs(
        self,
    ) -> Tuple[
        Optional[PolarKeyBlock],
        Optional[QuantizedVBlock],
        Optional[mx.array],
        Optional[mx.array],
        Optional[QJLPayload],
        int,
    ]:
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
            compressed_k = self.k_storage.to_unified_block(
                (B, H, self.k_storage.block_count * L, D)
            )
            quant_v = self.v_storage.to_quantized_block()

        unified_qjl: Optional[QJLPayload] = None
        if self.qjl_blocks:
            qjl_signs = mx.squeeze(
                mx.stack([b.packed_signs for b in self.qjl_blocks], axis=2), axis=3
            )
            qjl_norms = mx.squeeze(
                mx.stack([b.norms for b in self.qjl_blocks], axis=2), axis=3
            )
            unified_qjl = QJLPayload(
                packed_signs=qjl_signs,
                norms=qjl_norms,
                proj_dim=self.qjl_blocks[0].proj_dim,
                seed=self.qjl_blocks[0].seed,
                shape=(
                    qjl_signs.shape[0],
                    qjl_signs.shape[1],
                    qjl_signs.shape[2],
                    qjl_signs.shape[3],
                    self.config.head_dim,
                ),
            )

        return (
            compressed_k,
            quant_v,
            self._current_tail_k(),
            self._current_tail_v(),
            unified_qjl,
            self.actual_seq_len,
        )

    def get_blocks_for_attention(
        self,
    ) -> Tuple[
        Optional[PolarKeyBlock],
        Optional[QuantizedVBlock],
        Optional[mx.array],
        Optional[QJLPayload],
        int,
    ]:
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
                shape=(
                    partial["radii"].shape[0],
                    partial["radii"].shape[1],
                    partial["radii"].shape[2] * partial["radii"].shape[3],
                    partial["radii"].shape[4] * 2,
                ),
                block_size=self.config.block_size,
                head_dim=self.config.head_dim,
                metadata=partial["metadata"],
            )
            return (
                unified_k,
                partial["quant_v"],
                None,
                partial.get("qjl"),
                self.actual_seq_len,
            )

        compressed_k = self.k_storage.to_unified_block(
            (
                self._batch_size,
                self._num_kv_heads,
                self.k_storage.block_count * self.config.block_size,
                self._head_dim,
            )
        )
        quant_v = self.v_storage.to_quantized_block()

        partial_quant_v: Optional[QuantizedVBlock] = None
        partial_qjl: Optional[QJLPayload] = None

        if partial is not None:
            radii = mx.concatenate([compressed_k.radii, partial["radii"]], axis=2)
            angle_l1 = mx.concatenate(
                [compressed_k.angle_codes_l1, partial["angle_l1"]], axis=2
            )
            angle_deep = mx.concatenate(
                [compressed_k.angle_codes_deep, partial["angle_deep"]], axis=2
            )
            radii_scales = None
            if (
                compressed_k.radii_scales is not None
                and partial.get("radii_scales") is not None
            ):
                radii_scales = mx.concatenate(
                    [compressed_k.radii_scales, partial["radii_scales"]], axis=2
                )
            compressed_k = PolarKeyBlock(
                radii=radii,
                angle_codes_l1=angle_l1,
                angle_codes_deep=angle_deep,
                radii_scales=radii_scales,
                shape=(
                    radii.shape[0],
                    radii.shape[1],
                    radii.shape[2] * radii.shape[3],
                    radii.shape[4] * 2,
                ),
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
            qjl_signs = mx.squeeze(
                mx.stack([b.packed_signs for b in self.qjl_blocks], axis=2), axis=3
            )
            qjl_norms = mx.squeeze(
                mx.stack([b.norms for b in self.qjl_blocks], axis=2), axis=3
            )
            if partial_qjl is not None:
                qjl_signs = mx.concatenate(
                    [qjl_signs, partial_qjl.packed_signs], axis=2
                )
                qjl_norms = mx.concatenate([qjl_norms, partial_qjl.norms], axis=2)
            unified_qjl = QJLPayload(
                packed_signs=qjl_signs,
                norms=qjl_norms,
                proj_dim=self.qjl_blocks[0].proj_dim,
                seed=self.qjl_blocks[0].seed,
                shape=(
                    qjl_signs.shape[0],
                    qjl_signs.shape[1],
                    qjl_signs.shape[2],
                    qjl_signs.shape[3],
                    self.config.head_dim,
                ),
            )

        return compressed_k, quant_v, None, unified_qjl, self.actual_seq_len

    def _maybe_encode_partial(self) -> Optional[Dict[str, Any]]:
        """Encode the partial tail padded to a full block, returning unified-shape tensors."""
        tail_k = self._current_tail_k()
        if tail_k is None:
            return None
        B, H, T_part, D = tail_k.shape
        L = self.config.block_size
        pad = L - T_part
        pad_width = [(0, 0), (0, 0), (0, pad), (0, 0)]
        k_padded = mx.pad(tail_k, pad_width)
        v_padded = mx.pad(self._current_tail_v(), pad_width)

        polar_block = self.polar_encoder.encode_block(k_padded)
        radii = mx.expand_dims(polar_block.radii, axis=2)
        radii_scales = (
            mx.expand_dims(polar_block.radii_scales, axis=2)
            if polar_block.radii_scales is not None
            else None
        )
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
            qjl_payload = self.qjl_encoder.compute_residual_sketch(
                residual.reshape(B, H, 1, L, D)
            )

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
                radii=self.k_storage.radii[:, :, idx : idx + 1, :, :],
                angle_codes_l1=self.k_storage.angle_codes_l1[:, :, idx : idx + 1, :, :],
                angle_codes_deep=self.k_storage.angle_codes_deep[
                    :, :, idx : idx + 1, :, :
                ],
                radii_scales=self.k_storage.radii_scales[:, :, idx : idx + 1, :, :]
                if self.k_storage.radii_scales is not None
                else None,
                shape=(
                    self._batch_size,
                    self._num_kv_heads,
                    self.config.block_size,
                    self._head_dim,
                ),
                block_size=self.config.block_size,
                head_dim=self._head_dim,
                metadata=self.k_storage.metadata,
            )
            self.bytes_read += (
                _nbytes(block.radii)
                + _nbytes(block.angle_codes_l1)
                + _nbytes(block.angle_codes_deep)
            )
            if block.radii_scales is not None:
                self.bytes_read += _nbytes(block.radii_scales)
            fetched.append(self.decoder.decode_block(block))
        return mx.concatenate(fetched, axis=2)

    def get_memory_stats(self) -> CacheMemoryStats:
        """Return truthful memory accounting.

        Uses the paged storage directly for allocated capacity (includes empty
        page slots) rather than the cached concatenated views (which contain
        only valid blocks).
        """
        logical = 0
        allocated = 0
        dense_tail = 0
        metadata = 0

        # Paged storage reports truthfully from the underlying pages.
        if self.k_storage.block_count > 0:
            k_logical, k_allocated = self.k_storage._paged.get_memory_stats()
            logical += k_logical
            allocated += k_allocated
        if self.v_storage.block_count > 0:
            v_logical, v_allocated = self.v_storage._paged.get_memory_stats()
            logical += v_logical
            allocated += v_allocated

        # Dense tail buffers: always allocated; only partial_length is logical.
        if self.partial_k_buffer is not None:
            allocated += _nbytes(self.partial_k_buffer)
            allocated += _nbytes(self.partial_v_buffer)
            if self.partial_length > 0:
                tail_k_logical = self.partial_k_buffer[:, :, : self.partial_length, :]
                tail_v_logical = self.partial_v_buffer[:, :, : self.partial_length, :]
                logical += _nbytes(tail_k_logical) + _nbytes(tail_v_logical)
                dense_tail += _nbytes(tail_k_logical) + _nbytes(tail_v_logical)
            else:
                dense_tail = 0

        # QJL payloads.
        for qjl in self.qjl_blocks:
            logical += _nbytes(qjl.packed_signs) + _nbytes(qjl.norms)
            allocated += _nbytes(qjl.packed_signs) + _nbytes(qjl.norms)
            metadata += _nbytes(qjl.packed_signs) + _nbytes(qjl.norms)

        dense_equivalent = 0
        if self._batch_size is not None:
            dense_equivalent = (
                self._batch_size
                * self._num_kv_heads
                * self.actual_seq_len
                * self._head_dim
                * 2
                * 2
            )

        return CacheMemoryStats(
            logical_payload_bytes=logical,
            allocated_capacity_bytes=allocated,
            dense_tail_bytes=dense_tail,
            metadata_bytes=metadata,
            dense_equivalent_bytes=dense_equivalent,
        )

    def get_io_telemetry(self) -> Dict[str, Any]:
        stats = self.get_memory_stats()
        if self.k_storage.block_count == 0 and self.partial_length == 0:
            return {}

        return {
            "dense_kv_bytes": stats.dense_equivalent_bytes,
            "actual_cache_bytes": stats.logical_payload_bytes,
            "allocated_cache_bytes": stats.allocated_capacity_bytes,
            "compression_ratio": stats.logical_compression_ratio,
            "allocated_compression_ratio": stats.allocated_compression_ratio,
            "total_blocks": self.total_blocks,
            "partial_tokens": self.partial_length,
            "k_storage_capacity": self.k_storage.capacity,
            "v_storage_capacity": self.v_storage.capacity,
            "k_storage_reallocs": self.k_storage.reallocation_count,
            "v_storage_reallocs": self.v_storage.reallocation_count,
            "finite_check_calls": self.finite_check_calls,
            "forced_scalar_evaluations": self.forced_scalar_evaluations,
            "audit_time_ns": self.audit_time_ns,
        }

    def _eval_state(self):
        """Materialize all lazy MLX arrays so allocator counters reflect real usage."""
        arrays = []
        if self.partial_k_buffer is not None:
            arrays.append(self.partial_k_buffer)
            arrays.append(self.partial_v_buffer)
        if self.k_storage.radii is not None:
            arrays.extend(
                [
                    self.k_storage.radii,
                    self.k_storage.angle_codes_l1,
                    self.k_storage.angle_codes_deep,
                ]
            )
            if self.k_storage.radii_scales is not None:
                arrays.append(self.k_storage.radii_scales)
        if self.v_storage.codes is not None:
            arrays.extend([self.v_storage.codes, self.v_storage.scales])
        for qjl in self.qjl_blocks:
            arrays.extend([qjl.packed_signs, qjl.norms])
        if arrays:
            mx.eval(*arrays)

    def measure_append_peak_memory(self, k_new: mx.array, v_new: mx.array) -> int:
        """Append k_new/v_new and return the peak MLX allocator bytes observed.

        This resets the global MLX peak-memory counter, runs the append (including
        any block compression), materializes the resulting cache state, and reports
        the highest bytes allocated during the operation.
        """
        mx.reset_peak_memory()
        self.append(k_new, v_new)
        self._eval_state()
        return int(mx.get_peak_memory())

    def reset(self):
        """Clear all cache state and persistent invariants."""
        self.partial_k_buffer = None
        self.partial_v_buffer = None
        self.partial_length = 0
        self.k_storage = PolarKBlockStorage()
        self.v_storage = QuantVBlockStorage()
        self.qjl_blocks = []
        self.actual_seq_len = 0
        self.total_blocks = 0
        self.bytes_written = 0
        self.bytes_read = 0
        self.compression_time_ns = 0
        self.finite_check_calls = 0
        self.forced_scalar_evaluations = 0
        self.audit_time_ns = 0
        self._initialized = False
        self._batch_size = None
        self._num_kv_heads = None
        self._head_dim = None
        self._input_dtype = None
