"""CuTe DSL kernels for chunked KDA prefill: prologue, prepare, recurrence.

The prologue (one CTA) validates the packed metadata, builds the tile tables,
and zeroes the error code. The prepare kernel (one CTA per chunk tile and
head) turns raw projections into the per-tile operands of the chunked delta
rule. The recurrence kernel (one CTA per sequence, head, and value split)
walks a sequence's tiles with the state resident in registers.

Workspace tile layout (private to these kernels): the ``[16 x 128]`` bf16 tiles
are stored with their 16-byte chunks XOR-swizzled by ``row & 7`` so the
recurrence kernel's ``ldmatrix`` reads are bank-conflict free;
:func:`workspace_tiles` returns de-swizzled views for tests.
"""

from __future__ import annotations

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    atomic_cas_global_i32,
    bf16_mma_m16n8k16_f32,
    ld_global_nc_v4_u32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_trans_b16,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_v4_u32,
    st_shared_v4_u32,
    warp_reduce,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._impl import MISC_RECORD_ELEMENTS, Binding

_HEAD_DIM = 128
_CHUNK = 16
_PROLOGUE_THREADS = 256
_PREPARE_THREADS = 128
_LOG2E = 1.4426950408889634

_PROLOGUE_CACHE: dict[tuple, Callable[..., None]] = {}
_PREPARE_CACHE: dict[tuple, Callable[..., None]] = {}
_RECURRENCE_CACHE: dict[tuple, Callable[..., None]] = {}
_WARMED: set[tuple] = set()


def _add(left: Float32, right: Float32) -> Float32:
    return left + right


@dsl_user_op
def _exp2_approx_ftz_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    """``ex2.approx.ftz.f32``; every argument here is at least -116."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pointer_address(ptr: cute.Pointer, offset, *, loc=None, ip=None) -> Int64:
    """Return the global address of ``ptr[offset]`` as an Int64."""
    element = ptr + offset
    return Int64(llvm.ptrtoint(T.i64(), element.llvm_ptr, loc=loc, ip=ip))


def _numeric_type(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return BFloat16
    if dtype == torch.float32:
        return Float32
    if dtype == torch.int32:
        return Int32
    if dtype == torch.int64:
        return Int64
    raise TypeError(f"unsupported KDA prefill dtype {dtype}")


def _fake_pointer(dtype: type[cutlass.Numeric]) -> cute.Pointer:
    return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=max(1, dtype.width // 8))


def _pointer(tensor: torch.Tensor, dtype: type[cutlass.Numeric]) -> cute.Pointer:
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=max(1, dtype.width // 8)
    )


def swizzled_column(row: int, column: int) -> int:
    """Physical column of logical ``(row, column)`` in a workspace tile."""
    return ((((column >> 3) ^ (row & 7)) << 3) | (column & 7))


class _PrologueKernel:
    """Validate packed metadata and build the tile tables in one CTA."""

    def __init__(
        self,
        *,
        max_seqs: int,
        tiles_capacity: int,
        table_size: int,
        max_state_slots: int,
        validate: bool,
        null_state_index: int | None,
        index_type: type[cutlass.Numeric],
    ) -> None:
        self.max_seqs = int(max_seqs)
        self.tiles_capacity = int(tiles_capacity)
        self.table_size = int(table_size)
        self.max_state_slots = int(max_state_slots)
        self.validate = bool(validate)
        self.has_null = null_state_index is not None
        self.null_state_index = 0 if null_state_index is None else int(null_state_index)
        self.index_type = index_type
        self.block = (self.max_seqs + _PROLOGUE_THREADS - 1) // _PROLOGUE_THREADS

    @cute.jit
    def __call__(
        self,
        cu_seqlens: cute.Pointer,
        initial_indices: cute.Pointer,
        final_indices: cute.Pointer,
        checkpoint_indices: cute.Pointer,
        checkpoint_offsets: cute.Pointer,
        num_seqs: cute.Pointer,
        num_tokens: cute.Pointer,
        error_code: cute.Pointer,
        table: cute.Pointer,
        seq_tile_base: cute.Pointer,
        tile_seq: cute.Pointer,
        seq_order: cute.Pointer,
        seq_capacity: Int32,
        token_capacity: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            cu_seqlens,
            initial_indices,
            final_indices,
            checkpoint_indices,
            checkpoint_offsets,
            num_seqs,
            num_tokens,
            error_code,
            table,
            seq_tile_base,
            tile_seq,
            seq_order,
            seq_capacity,
            token_capacity,
        ).launch(grid=(1, 1, 1), block=(_PROLOGUE_THREADS, 1, 1), stream=stream)

    @cute.jit
    def _is_null(self, slot: Int64) -> cutlass.Boolean:
        result = slot != slot
        if cutlass.const_expr(self.has_null):
            result = slot == Int64(self.null_state_index)
        return result

    @cute.jit
    def _insert(self, table: cute.Pointer, slot: Int64) -> Int32:
        """Insert ``slot`` into the open-addressing table; 1 when present."""
        key = slot.to(Int32) & Int32(0x7FFFFFFF)
        stored = key + Int32(1)
        position = key & Int32(self.table_size - 1)
        duplicate = Int32(0)
        done = Int32(0)
        for _ in cutlass.range(Int32(self.table_size), unroll=1):
            if done == Int32(0):
                previous = atomic_cas_global_i32(
                    _pointer_address(table, position), Int32(0), stored
                )
                if previous == Int32(0):
                    done = Int32(1)
                elif previous == stored:
                    duplicate = Int32(1)
                    done = Int32(1)
                else:
                    position = (position + Int32(1)) & Int32(self.table_size - 1)
        return duplicate

    @cute.jit
    def _contains(self, table: cute.Pointer, slot: Int64) -> Int32:
        key = slot.to(Int32) & Int32(0x7FFFFFFF)
        stored = key + Int32(1)
        position = key & Int32(self.table_size - 1)
        found = Int32(0)
        done = Int32(0)
        for _ in cutlass.range(Int32(self.table_size), unroll=1):
            if done == Int32(0):
                current = table[position].to(Int32)
                if current == Int32(0):
                    done = Int32(1)
                elif current == stored:
                    found = Int32(1)
                    done = Int32(1)
                else:
                    position = (position + Int32(1)) & Int32(self.table_size - 1)
        return found

    @cute.kernel
    def kernel(
        self,
        cu_seqlens: cute.Pointer,
        initial_indices: cute.Pointer,
        final_indices: cute.Pointer,
        checkpoint_indices: cute.Pointer,
        checkpoint_offsets: cute.Pointer,
        num_seqs: cute.Pointer,
        num_tokens: cute.Pointer,
        error_code: cute.Pointer,
        table: cute.Pointer,
        seq_tile_base: cute.Pointer,
        tile_seq: cute.Pointer,
        seq_order: cute.Pointer,
        seq_capacity: Int32,
        token_capacity: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        thread = Int32(thread)
        allocator = cutlass.utils.SmemAllocator()
        counts = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.max_seqs,), stride=(1,)),
            byte_alignment=16,
        )
        exclusive = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.max_seqs,), stride=(1,)),
            byte_alignment=16,
        )
        block_sums = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((_PROLOGUE_THREADS + 1,), stride=(1,)),
            byte_alignment=16,
        )
        flags = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((8,), stride=(1,)),
            byte_alignment=16,
        )
        if thread < Int32(8):
            flags[thread] = Int32(0)
        position = thread
        while position < Int32(self.table_size):
            table[position] = Int32(0)
            position += Int32(_PROLOGUE_THREADS)
        cute.arch.sync_threads()

        live_seqs = num_seqs[Int32(0)].to(Int32)
        live_tokens = num_tokens[Int32(0)].to(Int32)
        bounded_seqs = cutlass.max(Int32(0), cutlass.min(live_seqs, seq_capacity))
        if cutlass.const_expr(self.validate):
            if thread == Int32(0):
                bad_counts = (
                    (live_seqs < Int32(0))
                    | (live_seqs > seq_capacity)
                    | (live_tokens < Int32(0))
                    | (live_tokens > token_capacity)
                )
                if bad_counts:
                    flags[1] = Int32(1)
                if cu_seqlens[Int32(0)].to(Int32) != Int32(0):
                    flags[1] = Int32(1)
                if cu_seqlens[bounded_seqs].to(Int32) != live_tokens:
                    flags[1] = Int32(1)

        # Per-sequence pass: tile counts, slot checks, write-slot insertion.
        seq = thread
        while seq < Int32(self.max_seqs):
            count = Int32(0)
            if seq < bounded_seqs:
                start = cu_seqlens[seq].to(Int32)
                end = cu_seqlens[seq + Int32(1)].to(Int32)
                length = cutlass.max(Int32(0), end - start)
                count = (length + Int32(_CHUNK - 1)) // Int32(_CHUNK)
                if cutlass.const_expr(self.validate):
                    if (start < Int32(0)) | (end < start) | (end > live_tokens):
                        flags[1] = Int32(1)
                    initial = Int64(initial_indices[seq])
                    final = Int64(final_indices[seq])
                    checkpoint = Int64(checkpoint_indices[seq])
                    offset = checkpoint_offsets[seq].to(Int32)
                    slot_limit = Int64(self.max_state_slots)
                    if not self._is_null(initial):
                        if (initial < Int64(0)) | (initial >= slot_limit):
                            flags[2] = Int32(1)
                    if not self._is_null(final):
                        if (final < Int64(0)) | (final >= slot_limit):
                            flags[2] = Int32(1)
                        elif self._insert(table, final) != Int32(0):
                            flags[0] = Int32(1)
                    if offset > length:
                        flags[3] = Int32(1)
                    if (offset > Int32(0)) & ((offset % Int32(_CHUNK)) != Int32(0)):
                        flags[3] = Int32(1)
                    if offset > Int32(0):
                        if not self._is_null(checkpoint):
                            if (checkpoint < Int64(0)) | (checkpoint >= slot_limit):
                                flags[2] = Int32(1)
                            elif self._insert(table, checkpoint) != Int32(0):
                                flags[0] = Int32(1)
            counts[seq] = count
            seq_order[seq] = seq
            seq += Int32(_PROLOGUE_THREADS)
        cute.arch.sync_threads()

        # Exclusive scan of tile counts: per-thread blocks, then block offsets.
        running = Int32(0)
        for item in cutlass.range_constexpr(self.block):
            index = thread * Int32(self.block) + Int32(item)
            if index < Int32(self.max_seqs):
                exclusive[index] = running
                running += counts[index]
        block_sums[thread] = running
        cute.arch.sync_threads()
        if thread == Int32(0):
            total = Int32(0)
            for item in cutlass.range(Int32(_PROLOGUE_THREADS), unroll=1):
                value = block_sums[item]
                block_sums[item] = total
                total += value
            block_sums[Int32(_PROLOGUE_THREADS)] = total
        cute.arch.sync_threads()
        offset_base = block_sums[thread]
        for item in cutlass.range_constexpr(self.block):
            index = thread * Int32(self.block) + Int32(item)
            if index < Int32(self.max_seqs):
                if index <= bounded_seqs:
                    seq_tile_base[index] = exclusive[index] + offset_base
        total_tiles = block_sums[Int32(_PROLOGUE_THREADS)]
        if thread == Int32(0):
            seq_tile_base[bounded_seqs] = total_tiles
            if cutlass.const_expr(self.validate):
                if total_tiles > Int32(self.tiles_capacity):
                    flags[1] = Int32(1)
        cute.arch.sync_threads()

        # Tile-to-sequence map, unused tail, and initial-slot conflicts.
        bounded_tiles = cutlass.min(total_tiles, Int32(self.tiles_capacity))
        seq = thread
        while seq < bounded_seqs:
            base = seq_tile_base[seq].to(Int32)
            count = counts[seq]
            for item in cutlass.range(count, unroll=1):
                tile = base + item
                if tile < Int32(self.tiles_capacity):
                    tile_seq[tile] = seq
            if cutlass.const_expr(self.validate):
                initial = Int64(initial_indices[seq])
                final = Int64(final_indices[seq])
                if not self._is_null(initial):
                    if (initial >= Int64(0)) & (initial < Int64(self.max_state_slots)):
                        if initial != final:
                            if self._contains(table, initial) != Int32(0):
                                flags[0] = Int32(1)
            seq += Int32(_PROLOGUE_THREADS)
        tile = bounded_tiles + thread
        while tile < Int32(self.tiles_capacity):
            tile_seq[tile] = Int32(-1)
            tile += Int32(_PROLOGUE_THREADS)
        cute.arch.sync_threads()
        if cutlass.const_expr(self.validate):
            if thread == Int32(0):
                code = flags[0] | (flags[1] << Int32(1)) | (flags[2] << Int32(2)) | (flags[3] << Int32(3))
                error_code[Int32(0)] = code


class _PrepareKernel:
    """Per (tile, head): gates, norms, decayed operands, WY inverse, Mqk."""

    def __init__(
        self,
        *,
        heads: int,
        tiles_capacity: int,
        qk_l2norm: bool,
        a_log_type: type[cutlass.Numeric],
        dt_bias_type: type[cutlass.Numeric],
    ) -> None:
        self.heads = int(heads)
        self.tiles_capacity = int(tiles_capacity)
        self.qk_l2norm = bool(qk_l2norm)
        self.a_log_type = a_log_type
        self.dt_bias_type = dt_bias_type

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        k: cute.Pointer,
        raw_g: cute.Pointer,
        raw_beta: cute.Pointer,
        A_log: cute.Pointer,
        dt_bias: cute.Pointer,
        cu_seqlens: cute.Pointer,
        tile_seq: cute.Pointer,
        seq_tile_base: cute.Pointer,
        error_code: cute.Pointer,
        ws_q: cute.Pointer,
        ws_k: cute.Pointer,
        ws_kr: cute.Pointer,
        ws_misc: cute.Pointer,
        ws_inv: cute.Pointer,
        ws_mqk: cute.Pointer,
        q_stride: Int64,
        k_stride: Int64,
        g_stride: Int64,
        beta_token_stride: Int64,
        beta_head_stride: Int64,
        scale: Float32,
        gate_scale: Float32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q, k, raw_g, raw_beta, A_log, dt_bias, cu_seqlens, tile_seq, seq_tile_base,
            error_code, ws_q, ws_k, ws_kr, ws_misc, ws_inv, ws_mqk, q_stride, k_stride,
            g_stride, beta_token_stride, beta_head_stride, scale, gate_scale, eps,
        ).launch(
            grid=(self.tiles_capacity, self.heads, 1),
            block=(_PREPARE_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Pointer,
        k: cute.Pointer,
        raw_g: cute.Pointer,
        raw_beta: cute.Pointer,
        A_log: cute.Pointer,
        dt_bias: cute.Pointer,
        cu_seqlens: cute.Pointer,
        tile_seq: cute.Pointer,
        seq_tile_base: cute.Pointer,
        error_code: cute.Pointer,
        ws_q: cute.Pointer,
        ws_k: cute.Pointer,
        ws_kr: cute.Pointer,
        ws_misc: cute.Pointer,
        ws_inv: cute.Pointer,
        ws_mqk: cute.Pointer,
        q_stride: Int64,
        k_stride: Int64,
        g_stride: Int64,
        beta_token_stride: Int64,
        beta_head_stride: Int64,
        scale: Float32,
        gate_scale: Float32,
        eps: Float32,
    ):
        tile, head, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        tile = Int32(tile)
        head = Int32(head)
        column = Int32(thread)
        warp = column // Int32(32)
        lane = Int32(cute.arch.lane_idx())
        error = error_code[Int32(0)].to(Int32)
        seq = tile_seq[tile].to(Int32)
        if (error == Int32(0)) & (seq >= Int32(0)):
            allocator = cutlass.utils.SmemAllocator()
            tile_elements = _CHUNK * _HEAD_DIM
            s_q = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((tile_elements,), stride=(1,)),
                byte_alignment=16,
            )
            s_k = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((tile_elements,), stride=(1,)),
                byte_alignment=16,
            )
            s_kinv = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((tile_elements,), stride=(1,)),
                byte_alignment=16,
            )
            s_part = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((2 * _CHUNK * 4,), stride=(1,)),
                byte_alignment=16,
            )
            s_beta = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((_CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            s_p = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((_CHUNK * _CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            s_p2 = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((_CHUNK * _CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            s_inv = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((_CHUNK * _CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            s_inv2 = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((_CHUNK * _CHUNK,), stride=(1,)),
                byte_alignment=16,
            )

            base_tile = seq_tile_base[seq].to(Int32)
            start = cu_seqlens[seq].to(Int32) + (tile - base_tile) * Int32(_CHUNK)
            end = cu_seqlens[seq + Int32(1)].to(Int32)
            rows = cutlass.min(Int32(_CHUNK), end - start)
            rate = cute.math.exp(Float32(A_log[head]), fastmath=False)
            bias = Float32(dt_bias[head * Int32(_HEAD_DIM) + column])
            head_offset = head.to(Int64) * Int64(_HEAD_DIM) + column.to(Int64)

            q_values = cute.make_rmem_tensor((_CHUNK,), Float32)
            k_values = cute.make_rmem_tensor((_CHUNK,), Float32)
            g_cum = cute.make_rmem_tensor((_CHUNK,), Float32)
            running = Float32(0.0)
            for t in cutlass.range_constexpr(_CHUNK):
                q_value = Float32(0.0)
                k_value = Float32(0.0)
                g2 = Float32(0.0)
                if Int32(t) < rows:
                    token = (start + Int32(t)).to(Int64)
                    q_value = Float32(q[token * q_stride + head_offset])
                    k_value = Float32(k[token * k_stride + head_offset])
                    g_value = Float32(raw_g[token * g_stride + head_offset])
                    z = rate * (g_value + bias)
                    sigmoid = cute.arch.rcp_approx(
                        Float32(1.0) + _exp2_approx_ftz_f32(-z * Float32(_LOG2E))
                    )
                    g2 = gate_scale * sigmoid
                q_values[t] = q_value
                k_values[t] = k_value
                running += g2
                g_cum[t] = running
                q_sq = warp_reduce(q_value * q_value, _add)
                k_sq = warp_reduce(k_value * k_value, _add)
                if lane == Int32(0):
                    s_part[Int32(t * 4) + warp] = q_sq
                    s_part[Int32(_CHUNK * 4 + t * 4) + warp] = k_sq
            if column < Int32(_CHUNK):
                beta = Float32(0.0)
                if column < rows:
                    beta_offset = (
                        (start + column).to(Int64) * beta_token_stride
                        + head.to(Int64) * beta_head_stride
                    )
                    beta_raw = Float32(raw_beta[beta_offset])
                    beta = cute.arch.rcp_approx(
                        Float32(1.0) + _exp2_approx_ftz_f32(-beta_raw * Float32(_LOG2E))
                    )
                s_beta[column] = beta
            cute.arch.sync_threads()

            record = tile.to(Int64) * Int64(self.heads) + head.to(Int64)
            tile_base = record * Int64(tile_elements)
            misc_base = record * Int64(MISC_RECORD_ELEMENTS)
            square_base = record * Int64(_CHUNK * _CHUNK)
            last = g_cum[_CHUNK - 1]
            lambda_c = _exp2_approx_ftz_f32(last)
            ws_misc[misc_base + column.to(Int64)] = lambda_c
            if column < Int32(_CHUNK):
                ws_misc[misc_base + Int64(_HEAD_DIM) + column.to(Int64)] = s_beta[column]
            for t in cutlass.range_constexpr(_CHUNK):
                rinv_q = Float32(1.0)
                rinv_k = Float32(1.0)
                if cutlass.const_expr(self.qk_l2norm):
                    q_sum = (
                        s_part[Int32(t * 4)]
                        + s_part[Int32(t * 4 + 1)]
                        + s_part[Int32(t * 4 + 2)]
                        + s_part[Int32(t * 4 + 3)]
                    )
                    k_sum = (
                        s_part[Int32(_CHUNK * 4 + t * 4)]
                        + s_part[Int32(_CHUNK * 4 + t * 4 + 1)]
                        + s_part[Int32(_CHUNK * 4 + t * 4 + 2)]
                        + s_part[Int32(_CHUNK * 4 + t * 4 + 3)]
                    )
                    rinv_q = cute.math.rsqrt(q_sum + eps, fastmath=False)
                    rinv_k = cute.math.rsqrt(k_sum + eps, fastmath=False)
                lam = _exp2_approx_ftz_f32(g_cum[t])
                lam_inv = _exp2_approx_ftz_f32(-g_cum[t])
                lam_r = _exp2_approx_ftz_f32(last - g_cum[t])
                q_tilde = BFloat16(q_values[t] * rinv_q * lam * scale)
                k_tilde = BFloat16(k_values[t] * rinv_k * lam)
                k_inv = BFloat16(k_values[t] * rinv_k * lam_inv)
                k_r = BFloat16(k_values[t] * rinv_k * lam_r)
                local = Int32(t * _HEAD_DIM) + column
                s_q[local] = q_tilde
                s_k[local] = k_tilde
                s_kinv[local] = k_inv
                physical = (
                    Int32(t * _HEAD_DIM)
                    + ((((column >> Int32(3)) ^ Int32(t & 7)) << Int32(3)) | (column & Int32(7)))
                )
                ws_q[tile_base + physical.to(Int64)] = q_tilde
                ws_k[tile_base + physical.to(Int64)] = k_tilde
                ws_kr[tile_base + physical.to(Int64)] = k_r
            cute.arch.sync_threads()

            # L = beta_i <k~_i, k_inv_j> (j < i) and Mqk = <q~_i, k_inv_j> (j <= i).
            for entry in cutlass.range_constexpr(2):
                index = column + Int32(entry * _PREPARE_THREADS)
                row = index // Int32(_CHUNK)
                col = index % Int32(_CHUNK)
                acc_l = Float32(0.0)
                acc_m = Float32(0.0)
                for d in cutlass.range_constexpr(_HEAD_DIM):
                    k_inv_value = Float32(s_kinv[col * Int32(_HEAD_DIM) + Int32(d)])
                    acc_l += Float32(s_k[row * Int32(_HEAD_DIM) + Int32(d)]) * k_inv_value
                    acc_m += Float32(s_q[row * Int32(_HEAD_DIM) + Int32(d)]) * k_inv_value
                lower = Float32(0.0)
                if col < row:
                    lower = s_beta[row] * acc_l
                mqk = Float32(0.0)
                if col <= row:
                    mqk = acc_m
                s_p[index] = lower
                identity = Float32(0.0)
                if col == row:
                    identity = Float32(1.0)
                s_inv[index] = identity - lower
                ws_mqk[square_base + index.to(Int64)] = BFloat16(mqk)
            cute.arch.sync_threads()

            # Neumann series: INV = (I - L)(I + L^2)(I + L^4)(I + L^8).
            for _step in cutlass.range_constexpr(3):
                for entry in cutlass.range_constexpr(2):
                    index = column + Int32(entry * _PREPARE_THREADS)
                    row = index // Int32(_CHUNK)
                    col = index % Int32(_CHUNK)
                    acc = Float32(0.0)
                    for j in cutlass.range_constexpr(_CHUNK):
                        acc += s_p[row * Int32(_CHUNK) + Int32(j)] * s_p[Int32(j * _CHUNK) + col]
                    s_p2[index] = acc
                cute.arch.sync_threads()
                for entry in cutlass.range_constexpr(2):
                    index = column + Int32(entry * _PREPARE_THREADS)
                    row = index // Int32(_CHUNK)
                    col = index % Int32(_CHUNK)
                    acc = s_inv[index]
                    for j in cutlass.range_constexpr(_CHUNK):
                        acc += s_inv[row * Int32(_CHUNK) + Int32(j)] * s_p2[Int32(j * _CHUNK) + col]
                    s_inv2[index] = acc
                cute.arch.sync_threads()
                for entry in cutlass.range_constexpr(2):
                    index = column + Int32(entry * _PREPARE_THREADS)
                    s_p[index] = s_p2[index]
                    s_inv[index] = s_inv2[index]
                cute.arch.sync_threads()
            for entry in cutlass.range_constexpr(2):
                index = column + Int32(entry * _PREPARE_THREADS)
                ws_inv[square_base + index.to(Int64)] = BFloat16(s_inv[index])


class _RecurrenceKernel:
    """Per (sequence, head, value split): walk the tiles with the state in registers.

    The state ``S^T`` rows owned by a warp live in sixteen ``m16n8`` fp32
    accumulator fragments (one per 8-column key block) plus a bf16 shadow
    packed as eight k16 A-fragments. Every per-tile product is arranged so that
    the previous accumulator is reused as the next A operand.
    """

    def __init__(
        self,
        *,
        heads: int,
        max_seqs: int,
        v_split: int,
        checkpoint_export: bool,
        null_state_index: int | None,
        index_type: type[cutlass.Numeric],
    ) -> None:
        self.heads = int(heads)
        self.max_seqs = int(max_seqs)
        self.v_split = int(v_split)
        self.splits = _HEAD_DIM // self.v_split
        self.warps = self.v_split // 16
        self.threads = 32 * self.warps
        self.checkpoint_export = bool(checkpoint_export)
        self.has_null = null_state_index is not None
        self.null_state_index = 0 if null_state_index is None else int(null_state_index)
        self.index_type = index_type
        self.v_chunks_per_row = self.v_split // 8

    @cute.jit
    def __call__(
        self,
        v: cute.Pointer,
        cu_seqlens: cute.Pointer,
        seq_tile_base: cute.Pointer,
        seq_order: cute.Pointer,
        initial_indices: cute.Pointer,
        final_indices: cute.Pointer,
        checkpoint_indices: cute.Pointer,
        checkpoint_offsets: cute.Pointer,
        num_seqs: cute.Pointer,
        error_code: cute.Pointer,
        ws_q: cute.Pointer,
        ws_k: cute.Pointer,
        ws_kr: cute.Pointer,
        ws_misc: cute.Pointer,
        ws_inv: cute.Pointer,
        ws_mqk: cute.Pointer,
        recurrent_state: cute.Pointer,
        output: cute.Pointer,
        v_stride: Int64,
        out_stride: Int64,
        slot_stride: Int64,
        token_capacity: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            v, cu_seqlens, seq_tile_base, seq_order, initial_indices, final_indices,
            checkpoint_indices, checkpoint_offsets, num_seqs, error_code, ws_q, ws_k,
            ws_kr, ws_misc, ws_inv, ws_mqk, recurrent_state, output, v_stride, out_stride,
            slot_stride, token_capacity,
        ).launch(
            grid=(self.heads * self.splits, self.max_seqs, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _is_null(self, slot: Int64) -> cutlass.Boolean:
        result = slot != slot
        if cutlass.const_expr(self.has_null):
            result = slot == Int64(self.null_state_index)
        return result

    @cute.jit
    def _store_state(
        self,
        recurrent_state: cute.Pointer,
        acc: cute.Tensor,
        base: Int64,
        row0: Int32,
        row1: Int32,
        tid: Int32,
    ):
        for nb in cutlass.range_constexpr(16):
            kcol = Int32(nb * 8) + tid * Int32(2)
            offset0 = base + row0.to(Int64) * Int64(_HEAD_DIM) + kcol.to(Int64)
            offset1 = base + row1.to(Int64) * Int64(_HEAD_DIM) + kcol.to(Int64)
            recurrent_state[offset0] = acc[nb, 0]
            recurrent_state[offset0 + Int64(1)] = acc[nb, 1]
            recurrent_state[offset1] = acc[nb, 2]
            recurrent_state[offset1 + Int64(1)] = acc[nb, 3]

    @cute.jit
    def _refresh_shadow(self, acc: cute.Tensor, shadow: cute.Tensor):
        for kb in cutlass.range_constexpr(8):
            shadow[kb, 0] = pack_f32x2_to_bfloat2(acc[2 * kb, 0], acc[2 * kb, 1])
            shadow[kb, 1] = pack_f32x2_to_bfloat2(acc[2 * kb, 2], acc[2 * kb, 3])
            shadow[kb, 2] = pack_f32x2_to_bfloat2(acc[2 * kb + 1, 0], acc[2 * kb + 1, 1])
            shadow[kb, 3] = pack_f32x2_to_bfloat2(acc[2 * kb + 1, 2], acc[2 * kb + 1, 3])

    @cute.kernel
    def kernel(
        self,
        v: cute.Pointer,
        cu_seqlens: cute.Pointer,
        seq_tile_base: cute.Pointer,
        seq_order: cute.Pointer,
        initial_indices: cute.Pointer,
        final_indices: cute.Pointer,
        checkpoint_indices: cute.Pointer,
        checkpoint_offsets: cute.Pointer,
        num_seqs: cute.Pointer,
        error_code: cute.Pointer,
        ws_q: cute.Pointer,
        ws_k: cute.Pointer,
        ws_kr: cute.Pointer,
        ws_misc: cute.Pointer,
        ws_inv: cute.Pointer,
        ws_mqk: cute.Pointer,
        recurrent_state: cute.Pointer,
        output: cute.Pointer,
        v_stride: Int64,
        out_stride: Int64,
        slot_stride: Int64,
        token_capacity: Int32,
    ):
        bx, by, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        thread = Int32(thread)
        head = Int32(bx) // Int32(self.splits)
        split = Int32(bx) % Int32(self.splits)
        warp = thread // Int32(32)
        lane = Int32(cute.arch.lane_idx())
        gid = lane >> Int32(2)
        tid = lane & Int32(3)
        matrix = lane >> Int32(3)
        matrix_row = lane & Int32(7)
        error = error_code[Int32(0)].to(Int32)
        live_seqs = num_seqs[Int32(0)].to(Int32)
        vd = self.v_split
        if error != Int32(0):
            # Transactional failure: poison every output row of this CTA's
            # value columns; no state is written.
            if Int32(by) == Int32(0):
                nan_pair = Uint32(0x7FC07FC0)
                chunk = thread
                while chunk < token_capacity * Int32(self.v_chunks_per_row):
                    row = chunk // Int32(self.v_chunks_per_row)
                    col_chunk = chunk % Int32(self.v_chunks_per_row)
                    element = (
                        row.to(Int64) * out_stride
                        + head.to(Int64) * Int64(_HEAD_DIM)
                        + (split * Int32(vd) + col_chunk * Int32(8)).to(Int64)
                    )
                    st_global_v4_u32(
                        _pointer_address(output, element), nan_pair, nan_pair, nan_pair, nan_pair
                    )
                    chunk += Int32(self.threads)
        elif Int32(by) < live_seqs:
            seq = seq_order[Int32(by)].to(Int32)
            start = cu_seqlens[seq].to(Int32)
            end = cu_seqlens[seq + Int32(1)].to(Int32)
            length = cutlass.max(Int32(0), end - start)
            tiles = (length + Int32(_CHUNK - 1)) // Int32(_CHUNK)
            base_tile = seq_tile_base[seq].to(Int32)
            initial = Int64(initial_indices[seq])
            final = Int64(final_indices[seq])
            checkpoint = Int64(checkpoint_indices[seq])
            offset = checkpoint_offsets[seq].to(Int32)

            allocator = cutlass.utils.SmemAllocator()
            tile_elements = _CHUNK * _HEAD_DIM
            s_qt = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((tile_elements,), stride=(1,)),
                byte_alignment=128,
            )
            s_kt = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((tile_elements,), stride=(1,)),
                byte_alignment=128,
            )
            s_kr = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((tile_elements,), stride=(1,)),
                byte_alignment=128,
            )
            s_inv = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((_CHUNK * _CHUNK,), stride=(1,)),
                byte_alignment=128,
            )
            s_mqk = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((_CHUNK * _CHUNK,), stride=(1,)),
                byte_alignment=128,
            )
            s_v = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((_CHUNK * vd,), stride=(1,)),
                byte_alignment=128,
            )
            s_out = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((_CHUNK * vd,), stride=(1,)),
                byte_alignment=128,
            )
            s_lam = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((_HEAD_DIM,), stride=(1,)),
                byte_alignment=128,
            )
            s_beta = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((_CHUNK,), stride=(1,)),
                byte_alignment=128,
            )
            qt_addr = shared_ptr_to_u32(s_qt.iterator)
            kt_addr = shared_ptr_to_u32(s_kt.iterator)
            kr_addr = shared_ptr_to_u32(s_kr.iterator)
            inv_addr = shared_ptr_to_u32(s_inv.iterator)
            mqk_addr = shared_ptr_to_u32(s_mqk.iterator)
            v_addr = shared_ptr_to_u32(s_v.iterator)
            out_addr = shared_ptr_to_u32(s_out.iterator)
            lam_addr = shared_ptr_to_u32(s_lam.iterator)
            beta_addr = shared_ptr_to_u32(s_beta.iterator)

            # State fragments: rows row0/row1 of this head's [V, K] slab.
            row_local0 = warp * Int32(16) + gid
            row_local1 = row_local0 + Int32(8)
            row0 = split * Int32(vd) + row_local0
            row1 = row0 + Int32(8)
            head_base = head.to(Int64) * Int64(_HEAD_DIM * _HEAD_DIM)
            acc = cute.make_rmem_tensor((16, 4), Float32)
            shadow = cute.make_rmem_tensor((8, 4), Uint32)
            for nb in cutlass.range_constexpr(16):
                acc[nb, 0] = Float32(0.0)
                acc[nb, 1] = Float32(0.0)
                acc[nb, 2] = Float32(0.0)
                acc[nb, 3] = Float32(0.0)
            if not self._is_null(initial):
                base = initial * slot_stride + head_base
                for nb in cutlass.range_constexpr(16):
                    kcol = Int32(nb * 8) + tid * Int32(2)
                    offset0 = base + row0.to(Int64) * Int64(_HEAD_DIM) + kcol.to(Int64)
                    offset1 = base + row1.to(Int64) * Int64(_HEAD_DIM) + kcol.to(Int64)
                    acc[nb, 0] = Float32(recurrent_state[offset0])
                    acc[nb, 1] = Float32(recurrent_state[offset0 + Int64(1)])
                    acc[nb, 2] = Float32(recurrent_state[offset1])
                    acc[nb, 3] = Float32(recurrent_state[offset1 + Int64(1)])
            self._refresh_shadow(acc, shadow)

            v_chunks = self.v_chunks_per_row
            v_chunk_total = _CHUNK * v_chunks
            for local in cutlass.range(tiles, unroll=1):
                tile = base_tile + local
                record = tile.to(Int64) * Int64(self.heads) + head.to(Int64)
                tile_base = record * Int64(tile_elements)
                square_base = record * Int64(_CHUNK * _CHUNK)
                misc_base = record * Int64(MISC_RECORD_ELEMENTS)
                token_base = start + local * Int32(_CHUNK)
                rows_live = cutlass.min(Int32(_CHUNK), end - token_base)

                # Stage the tile: three swizzled operand tiles (256 chunks of
                # 16 bytes each), INV and Mqk (32 chunks each), lambda_c and
                # beta (36 chunks), and the value rows (zero past the tail).
                chunk = thread
                while chunk < Int32(3 * 256 + 64 + 36):
                    if chunk < Int32(256):
                        c0, c1, c2, c3 = ld_global_nc_v4_u32(
                            _pointer_address(ws_q, tile_base + chunk.to(Int64) * Int64(8))
                        )
                        st_shared_v4_u32(qt_addr + chunk * Int32(16), c0, c1, c2, c3)
                    elif chunk < Int32(512):
                        local_chunk = chunk - Int32(256)
                        c0, c1, c2, c3 = ld_global_nc_v4_u32(
                            _pointer_address(ws_k, tile_base + local_chunk.to(Int64) * Int64(8))
                        )
                        st_shared_v4_u32(kt_addr + local_chunk * Int32(16), c0, c1, c2, c3)
                    elif chunk < Int32(768):
                        local_chunk = chunk - Int32(512)
                        c0, c1, c2, c3 = ld_global_nc_v4_u32(
                            _pointer_address(ws_kr, tile_base + local_chunk.to(Int64) * Int64(8))
                        )
                        st_shared_v4_u32(kr_addr + local_chunk * Int32(16), c0, c1, c2, c3)
                    elif chunk < Int32(800):
                        local_chunk = chunk - Int32(768)
                        c0, c1, c2, c3 = ld_global_nc_v4_u32(
                            _pointer_address(ws_inv, square_base + local_chunk.to(Int64) * Int64(8))
                        )
                        st_shared_v4_u32(inv_addr + local_chunk * Int32(16), c0, c1, c2, c3)
                    elif chunk < Int32(832):
                        local_chunk = chunk - Int32(800)
                        c0, c1, c2, c3 = ld_global_nc_v4_u32(
                            _pointer_address(ws_mqk, square_base + local_chunk.to(Int64) * Int64(8))
                        )
                        st_shared_v4_u32(mqk_addr + local_chunk * Int32(16), c0, c1, c2, c3)
                    elif chunk < Int32(864):
                        local_chunk = chunk - Int32(832)
                        c0, c1, c2, c3 = ld_global_nc_v4_u32(
                            _pointer_address(ws_misc, misc_base + local_chunk.to(Int64) * Int64(4))
                        )
                        st_shared_v4_u32(lam_addr + local_chunk * Int32(16), c0, c1, c2, c3)
                    else:
                        local_chunk = chunk - Int32(864)
                        c0, c1, c2, c3 = ld_global_nc_v4_u32(
                            _pointer_address(
                                ws_misc, misc_base + Int64(_HEAD_DIM) + local_chunk.to(Int64) * Int64(4)
                            )
                        )
                        st_shared_v4_u32(beta_addr + local_chunk * Int32(16), c0, c1, c2, c3)
                    chunk += Int32(self.threads)
                chunk = thread
                while chunk < Int32(v_chunk_total):
                    row = chunk // Int32(v_chunks)
                    col_chunk = chunk % Int32(v_chunks)
                    c0 = Uint32(0)
                    c1 = Uint32(0)
                    c2 = Uint32(0)
                    c3 = Uint32(0)
                    if row < rows_live:
                        element = (
                            (token_base + row).to(Int64) * v_stride
                            + head.to(Int64) * Int64(_HEAD_DIM)
                            + (split * Int32(vd) + col_chunk * Int32(8)).to(Int64)
                        )
                        c0, c1, c2, c3 = ld_global_nc_v4_u32(_pointer_address(v, element))
                    st_shared_v4_u32(v_addr + chunk * Int32(16), c0, c1, c2, c3)
                    chunk += Int32(self.threads)
                cute.arch.sync_threads()

                # Phase A: v'^T = v^T - S^T k~^T, scaled by beta per token column.
                vp = cute.make_rmem_tensor((2, 4), Float32)
                for half in cutlass.range_constexpr(2):
                    for item in cutlass.range_constexpr(4):
                        vp[half, item] = Float32(0.0)
                for kb in cutlass.range_constexpr(8):
                    tok = (matrix >> Int32(1)) * Int32(8) + matrix_row
                    logical_chunk = Int32(kb * 2) + (matrix & Int32(1))
                    physical = logical_chunk ^ (tok & Int32(7))
                    b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                        kt_addr + tok * Int32(256) + physical * Int32(16)
                    )
                    vp[0, 0], vp[0, 1], vp[0, 2], vp[0, 3] = bf16_mma_m16n8k16_f32(
                        vp[0, 0], vp[0, 1], vp[0, 2], vp[0, 3],
                        shadow[kb, 0], shadow[kb, 1], shadow[kb, 2], shadow[kb, 3], b0, b1,
                    )
                    vp[1, 0], vp[1, 1], vp[1, 2], vp[1, 3] = bf16_mma_m16n8k16_f32(
                        vp[1, 0], vp[1, 1], vp[1, 2], vp[1, 3],
                        shadow[kb, 0], shadow[kb, 1], shadow[kb, 2], shadow[kb, 3], b2, b3,
                    )
                for half in cutlass.range_constexpr(2):
                    tok0 = Int32(half * 8) + tid * Int32(2)
                    tok1 = tok0 + Int32(1)
                    beta0 = s_beta[tok0]
                    beta1 = s_beta[tok1]
                    vp[half, 0] = (Float32(s_v[tok0 * Int32(vd) + row_local0]) - vp[half, 0]) * beta0
                    vp[half, 1] = (Float32(s_v[tok1 * Int32(vd) + row_local0]) - vp[half, 1]) * beta1
                    vp[half, 2] = (Float32(s_v[tok0 * Int32(vd) + row_local1]) - vp[half, 2]) * beta0
                    vp[half, 3] = (Float32(s_v[tok1 * Int32(vd) + row_local1]) - vp[half, 3]) * beta1
                a_vp0 = pack_f32x2_to_bfloat2(vp[0, 0], vp[0, 1])
                a_vp1 = pack_f32x2_to_bfloat2(vp[0, 2], vp[0, 3])
                a_vp2 = pack_f32x2_to_bfloat2(vp[1, 0], vp[1, 1])
                a_vp3 = pack_f32x2_to_bfloat2(vp[1, 2], vp[1, 3])

                # Phase B: U^T = v'^T INV^T.
                square_row = (matrix >> Int32(1)) * Int32(8) + matrix_row
                square_addr = square_row * Int32(32) + (matrix & Int32(1)) * Int32(16)
                b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(inv_addr + square_addr)
                u = cute.make_rmem_tensor((2, 4), Float32)
                u[0, 0], u[0, 1], u[0, 2], u[0, 3] = bf16_mma_m16n8k16_f32(
                    Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                    a_vp0, a_vp1, a_vp2, a_vp3, b0, b1,
                )
                u[1, 0], u[1, 1], u[1, 2], u[1, 3] = bf16_mma_m16n8k16_f32(
                    Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                    a_vp0, a_vp1, a_vp2, a_vp3, b2, b3,
                )
                a_u0 = pack_f32x2_to_bfloat2(u[0, 0], u[0, 1])
                a_u1 = pack_f32x2_to_bfloat2(u[0, 2], u[0, 3])
                a_u2 = pack_f32x2_to_bfloat2(u[1, 0], u[1, 1])
                a_u3 = pack_f32x2_to_bfloat2(u[1, 2], u[1, 3])

                # Phase C: out^T = S^T q~^T + U^T Mqk^T.
                out = cute.make_rmem_tensor((2, 4), Float32)
                for half in cutlass.range_constexpr(2):
                    for item in cutlass.range_constexpr(4):
                        out[half, item] = Float32(0.0)
                for kb in cutlass.range_constexpr(8):
                    tok = (matrix >> Int32(1)) * Int32(8) + matrix_row
                    logical_chunk = Int32(kb * 2) + (matrix & Int32(1))
                    physical = logical_chunk ^ (tok & Int32(7))
                    b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                        qt_addr + tok * Int32(256) + physical * Int32(16)
                    )
                    out[0, 0], out[0, 1], out[0, 2], out[0, 3] = bf16_mma_m16n8k16_f32(
                        out[0, 0], out[0, 1], out[0, 2], out[0, 3],
                        shadow[kb, 0], shadow[kb, 1], shadow[kb, 2], shadow[kb, 3], b0, b1,
                    )
                    out[1, 0], out[1, 1], out[1, 2], out[1, 3] = bf16_mma_m16n8k16_f32(
                        out[1, 0], out[1, 1], out[1, 2], out[1, 3],
                        shadow[kb, 0], shadow[kb, 1], shadow[kb, 2], shadow[kb, 3], b2, b3,
                    )
                b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(mqk_addr + square_addr)
                out[0, 0], out[0, 1], out[0, 2], out[0, 3] = bf16_mma_m16n8k16_f32(
                    out[0, 0], out[0, 1], out[0, 2], out[0, 3], a_u0, a_u1, a_u2, a_u3, b0, b1,
                )
                out[1, 0], out[1, 1], out[1, 2], out[1, 3] = bf16_mma_m16n8k16_f32(
                    out[1, 0], out[1, 1], out[1, 2], out[1, 3], a_u0, a_u1, a_u2, a_u3, b2, b3,
                )
                for half in cutlass.range_constexpr(2):
                    tok0 = Int32(half * 8) + tid * Int32(2)
                    tok1 = tok0 + Int32(1)
                    s_out[tok0 * Int32(vd) + row_local0] = BFloat16(out[half, 0])
                    s_out[tok1 * Int32(vd) + row_local0] = BFloat16(out[half, 1])
                    s_out[tok0 * Int32(vd) + row_local1] = BFloat16(out[half, 2])
                    s_out[tok1 * Int32(vd) + row_local1] = BFloat16(out[half, 3])

                # Phase D: S^T <- S^T * lambda_c[k] + U^T k_r.
                for nb in cutlass.range_constexpr(16):
                    kcol = Int32(nb * 8) + tid * Int32(2)
                    lam0 = s_lam[kcol]
                    lam1 = s_lam[kcol + Int32(1)]
                    acc[nb, 0] = acc[nb, 0] * lam0
                    acc[nb, 1] = acc[nb, 1] * lam1
                    acc[nb, 2] = acc[nb, 2] * lam0
                    acc[nb, 3] = acc[nb, 3] * lam1
                for pair in cutlass.range_constexpr(8):
                    tok = (matrix & Int32(1)) * Int32(8) + matrix_row
                    logical_chunk = Int32(pair * 2) + (matrix >> Int32(1))
                    physical = logical_chunk ^ (tok & Int32(7))
                    b0, b1, b2, b3 = ldmatrix_m8n8x4_trans_b16(
                        kr_addr + tok * Int32(256) + physical * Int32(16)
                    )
                    acc[2 * pair, 0], acc[2 * pair, 1], acc[2 * pair, 2], acc[2 * pair, 3] = (
                        bf16_mma_m16n8k16_f32(
                            acc[2 * pair, 0], acc[2 * pair, 1], acc[2 * pair, 2], acc[2 * pair, 3],
                            a_u0, a_u1, a_u2, a_u3, b0, b1,
                        )
                    )
                    (
                        acc[2 * pair + 1, 0],
                        acc[2 * pair + 1, 1],
                        acc[2 * pair + 1, 2],
                        acc[2 * pair + 1, 3],
                    ) = bf16_mma_m16n8k16_f32(
                        acc[2 * pair + 1, 0], acc[2 * pair + 1, 1],
                        acc[2 * pair + 1, 2], acc[2 * pair + 1, 3],
                        a_u0, a_u1, a_u2, a_u3, b2, b3,
                    )
                self._refresh_shadow(acc, shadow)
                cute.arch.sync_threads()

                # Store the live output rows of this tile.
                chunk = thread
                while chunk < Int32(v_chunk_total):
                    row = chunk // Int32(v_chunks)
                    col_chunk = chunk % Int32(v_chunks)
                    if row < rows_live:
                        c0, c1, c2, c3 = ld_shared_v4_u32(out_addr + chunk * Int32(16))
                        element = (
                            (token_base + row).to(Int64) * out_stride
                            + head.to(Int64) * Int64(_HEAD_DIM)
                            + (split * Int32(vd) + col_chunk * Int32(8)).to(Int64)
                        )
                        st_global_v4_u32(_pointer_address(output, element), c0, c1, c2, c3)
                    chunk += Int32(self.threads)
                if cutlass.const_expr(self.checkpoint_export):
                    if (offset > Int32(0)) & ((local + Int32(1)) * Int32(_CHUNK) == offset):
                        if not self._is_null(checkpoint):
                            self._store_state(
                                recurrent_state, acc, checkpoint * slot_stride + head_base,
                                row0, row1, tid,
                            )
                cute.arch.sync_threads()
            if not self._is_null(final):
                self._store_state(recurrent_state, acc, final * slot_stride + head_base, row0, row1, tid)


def _recurrence_key(binding: Binding) -> tuple[object, ...]:
    caps = binding.plan.caps
    return (
        "recurrence",
        binding.output.device.index,
        caps.heads,
        caps.max_seqs,
        binding.plan.v_split,
        caps.checkpoint_export,
        caps.null_state_index,
        binding.initial_state_indices.dtype,
    )


def _compile_recurrence(binding: Binding) -> tuple[tuple[object, ...], Callable[..., None]]:
    key = _recurrence_key(binding)
    cached = _RECURRENCE_CACHE.get(key)
    if cached is not None:
        return key, cached
    caps = binding.plan.caps
    index_type = _numeric_type(binding.initial_state_indices.dtype)
    kernel = _RecurrenceKernel(
        heads=caps.heads,
        max_seqs=caps.max_seqs,
        v_split=binding.plan.v_split,
        checkpoint_export=caps.checkpoint_export,
        null_state_index=caps.null_state_index,
        index_type=index_type,
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    raw = b12x_compile(
        kernel,
        _fake_pointer(BFloat16),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(index_type),
        _fake_pointer(index_type),
        _fake_pointer(index_type),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        _fake_pointer(Float32),
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        _fake_pointer(Float32),
        _fake_pointer(BFloat16),
        Int64(1),
        Int64(1),
        Int64(1),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("sequence.kda_prefill.recurrence", 1, key),
    )

    def launch(active: Binding) -> None:
        if _recurrence_key(active) != key:
            raise ValueError("compiled KDA recurrence kernel does not match the binding")
        raw(
            _pointer(active.v, BFloat16),
            _pointer(active.cu_seqlens, Int32),
            _pointer(active.seq_tile_base, Int32),
            _pointer(active.seq_order, Int32),
            _pointer(active.initial_state_indices, index_type),
            _pointer(active.final_state_indices, index_type),
            _pointer(active.checkpoint_state_indices, index_type),
            _pointer(active.checkpoint_offsets, Int32),
            _pointer(active.num_seqs, Int32),
            _pointer(active.error_code, Int32),
            _pointer(active.ws_q, BFloat16),
            _pointer(active.ws_k, BFloat16),
            _pointer(active.ws_kr, BFloat16),
            _pointer(active.ws_misc, Float32),
            _pointer(active.ws_inv, BFloat16),
            _pointer(active.ws_mqk, BFloat16),
            _pointer(active.recurrent_state, Float32),
            _pointer(active.output, BFloat16),
            int(active.v.stride(0)),
            int(active.output.stride(0)),
            int(active.recurrent_state.stride(0)),
            int(active.token_capacity),
            current_cuda_stream(),
        )

    _RECURRENCE_CACHE[key] = launch
    return key, launch


def run_recurrence(binding: Binding) -> None:
    """Walk every live sequence's tiles (stage 2); requires stages 0 and 1."""
    with torch.cuda.device(binding.output.device):
        _launch_stage(
            lambda b: (_recurrence_key(b), _RECURRENCE_CACHE.get(_recurrence_key(b))),
            _compile_recurrence,
            binding,
        )


def _prologue_key(binding: Binding) -> tuple[object, ...]:
    caps = binding.plan.caps
    return (
        "prologue",
        binding.output.device.index,
        caps.max_seqs,
        caps.tiles_capacity,
        binding.plan.duplicate_table_size,
        caps.max_state_slots,
        caps.metadata_validation,
        caps.null_state_index,
        binding.initial_state_indices.dtype,
    )


def _compile_prologue(binding: Binding) -> tuple[tuple[object, ...], Callable[..., None]]:
    key = _prologue_key(binding)
    cached = _PROLOGUE_CACHE.get(key)
    if cached is not None:
        return key, cached
    caps = binding.plan.caps
    index_type = _numeric_type(binding.initial_state_indices.dtype)
    kernel = _PrologueKernel(
        max_seqs=caps.max_seqs,
        tiles_capacity=caps.tiles_capacity,
        table_size=binding.plan.duplicate_table_size,
        max_state_slots=caps.max_state_slots,
        validate=caps.metadata_validation == "transactional",
        null_state_index=caps.null_state_index,
        index_type=index_type,
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    raw = b12x_compile(
        kernel,
        _fake_pointer(Int32),
        _fake_pointer(index_type),
        _fake_pointer(index_type),
        _fake_pointer(index_type),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        Int32(1),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("sequence.kda_prefill.prologue", 1, key),
    )

    def launch(active: Binding) -> None:
        if _prologue_key(active) != key:
            raise ValueError("compiled KDA prologue does not match the binding")
        raw(
            _pointer(active.cu_seqlens, Int32),
            _pointer(active.initial_state_indices, index_type),
            _pointer(active.final_state_indices, index_type),
            _pointer(active.checkpoint_state_indices, index_type),
            _pointer(active.checkpoint_offsets, Int32),
            _pointer(active.num_seqs, Int32),
            _pointer(active.num_tokens, Int32),
            _pointer(active.error_code, Int32),
            _pointer(active.duplicate_slots, Int32),
            _pointer(active.seq_tile_base, Int32),
            _pointer(active.tile_seq, Int32),
            _pointer(active.seq_order, Int32),
            int(active.seq_capacity),
            int(active.token_capacity),
            current_cuda_stream(),
        )

    _PROLOGUE_CACHE[key] = launch
    return key, launch


def _prepare_key(binding: Binding) -> tuple[object, ...]:
    caps = binding.plan.caps
    return (
        "prepare",
        binding.output.device.index,
        caps.heads,
        caps.tiles_capacity,
        caps.qk_l2norm,
        binding.A_log.dtype,
        binding.dt_bias.dtype,
    )


def _compile_prepare(binding: Binding) -> tuple[tuple[object, ...], Callable[..., None]]:
    key = _prepare_key(binding)
    cached = _PREPARE_CACHE.get(key)
    if cached is not None:
        return key, cached
    caps = binding.plan.caps
    a_log_type = _numeric_type(binding.A_log.dtype)
    dt_bias_type = _numeric_type(binding.dt_bias.dtype)
    kernel = _PrepareKernel(
        heads=caps.heads,
        tiles_capacity=caps.tiles_capacity,
        qk_l2norm=caps.qk_l2norm,
        a_log_type=a_log_type,
        dt_bias_type=dt_bias_type,
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    raw = b12x_compile(
        kernel,
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        _fake_pointer(a_log_type),
        _fake_pointer(dt_bias_type),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        _fake_pointer(Float32),
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Float32(1.0),
        Float32(1.0),
        Float32(1.0),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("sequence.kda_prefill.prepare", 1, key),
    )

    def launch(active: Binding, scale: float, gate_scale: float, eps: float) -> None:
        if _prepare_key(active) != key:
            raise ValueError("compiled KDA prepare kernel does not match the binding")
        raw(
            _pointer(active.q, BFloat16),
            _pointer(active.k, BFloat16),
            _pointer(active.raw_g, BFloat16),
            _pointer(active.raw_beta, BFloat16),
            _pointer(active.A_log, a_log_type),
            _pointer(active.dt_bias, dt_bias_type),
            _pointer(active.cu_seqlens, Int32),
            _pointer(active.tile_seq, Int32),
            _pointer(active.seq_tile_base, Int32),
            _pointer(active.error_code, Int32),
            _pointer(active.ws_q, BFloat16),
            _pointer(active.ws_k, BFloat16),
            _pointer(active.ws_kr, BFloat16),
            _pointer(active.ws_misc, Float32),
            _pointer(active.ws_inv, BFloat16),
            _pointer(active.ws_mqk, BFloat16),
            int(active.q.stride(0)),
            int(active.k.stride(0)),
            int(active.raw_g.stride(0)),
            int(active.raw_beta.stride(0)),
            int(active.raw_beta.stride(1)),
            float(scale),
            float(gate_scale),
            float(eps),
            current_cuda_stream(),
        )

    _PREPARE_CACHE[key] = launch
    return key, launch


def _launch_stage(cache_lookup, compile_fn, binding: Binding, *args) -> None:
    capturing = torch.cuda.is_current_stream_capturing()
    key, launch = cache_lookup(binding)
    if capturing and (launch is None or key not in _WARMED):
        raise RuntimeError(
            "KDA prefill kernels must be compiled and warm-run before CUDA graph capture"
        )
    if launch is None:
        key, launch = compile_fn(binding)
    launch(binding, *args)
    if not capturing:
        _WARMED.add(key)


def run_prologue(binding: Binding) -> None:
    """Validate metadata and build the tile tables (stage 0)."""
    with torch.cuda.device(binding.output.device):
        _launch_stage(
            lambda b: (_prologue_key(b), _PROLOGUE_CACHE.get(_prologue_key(b))),
            _compile_prologue,
            binding,
        )


def run_prepare(binding: Binding, *, lower_bound: float, scale: float, eps: float) -> None:
    """Fill the per-tile workspace (stage 1); requires the prologue first."""
    with torch.cuda.device(binding.output.device):
        _launch_stage(
            lambda b: (_prepare_key(b), _PREPARE_CACHE.get(_prepare_key(b))),
            _compile_prepare,
            binding,
            float(scale),
            float(lower_bound) * _LOG2E,
            float(eps),
        )


def run_prefill(binding: Binding, *, lower_bound: float, scale: float, eps: float) -> None:
    run_prologue(binding)
    run_prepare(binding, lower_bound=lower_bound, scale=scale, eps=eps)
    run_recurrence(binding)


def prewarm_binding(binding: Binding) -> None:
    """Compile the three stages for ``binding`` without launching them."""
    with torch.cuda.device(binding.output.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("KDA prefill compilation is forbidden during CUDA capture")
        _compile_prologue(binding)
        _compile_prepare(binding)
        _compile_recurrence(binding)


def workspace_tiles(binding: Binding) -> dict[str, torch.Tensor]:
    """Return de-swizzled logical views of the prepared workspace."""
    tiles, heads = binding.ws_q.shape[:2]
    rows = torch.arange(_CHUNK).view(_CHUNK, 1)
    cols = torch.arange(_HEAD_DIM).view(1, _HEAD_DIM)
    physical = ((((cols >> 3) ^ (rows & 7)) << 3) | (cols & 7)).to(binding.ws_q.device)
    index = physical.view(1, 1, _CHUNK, _HEAD_DIM).expand(tiles, heads, _CHUNK, _HEAD_DIM)

    def logical(ws: torch.Tensor) -> torch.Tensor:
        return torch.gather(ws, 3, index)

    return {
        "q_tilde": logical(binding.ws_q),
        "k_tilde": logical(binding.ws_k),
        "k_r": logical(binding.ws_kr),
        "lambda_c": binding.ws_misc[:, :, :_HEAD_DIM],
        "beta": binding.ws_misc[:, :, _HEAD_DIM : _HEAD_DIM + _CHUNK],
        "inv": binding.ws_inv,
        "mqk": binding.ws_mqk,
    }


def clear_caches() -> None:
    _PROLOGUE_CACHE.clear()
    _PREPARE_CACHE.clear()
    _RECURRENCE_CACHE.clear()
    _WARMED.clear()


__all__ = [
    "clear_caches",
    "prewarm_binding",
    "run_prefill",
    "run_prepare",
    "run_prologue",
    "run_recurrence",
    "swizzled_column",
    "workspace_tiles",
]
