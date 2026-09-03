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
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import atomic_cas_global_i32, warp_reduce
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._impl import MISC_RECORD_ELEMENTS, Binding, Plan

_HEAD_DIM = 128
_CHUNK = 16
_PROLOGUE_THREADS = 256
_PREPARE_THREADS = 128
_LOG2E = 1.4426950408889634

_PROLOGUE_CACHE: dict[tuple, Callable[..., None]] = {}
_PREPARE_CACHE: dict[tuple, Callable[..., None]] = {}
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
def _pointer_address(ptr: cute.Pointer, offset: Int32, *, loc=None, ip=None) -> Int64:
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
    raise NotImplementedError("the KDA prefill recurrence kernel is not implemented yet")


def prewarm_plan(plan: Plan) -> None:
    raise NotImplementedError("prewarm requires the recurrence kernel")


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
    _WARMED.clear()


__all__ = [
    "clear_caches",
    "prewarm_plan",
    "run_prefill",
    "run_prepare",
    "run_prologue",
    "swizzled_column",
    "workspace_tiles",
]
