"""Specialized CuTeDSL Qwen3.8 Flash QSA sparse GQA.

The public QSA contract owns split storage. Both the indexed K/V scan and the
final split merge are CuTe kernels, and unsupported layouts are rejected.
Cache strides are runtime values so the scan can consume zero-copy per-layer
views from an interleaved BLHNC allocation. Every page-scaled cache offset is
widened to Int64 before multiplication.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compiler import run_compiled
from b12x._lib.intrinsics import warp_reduce
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from ._sparse_gqa_cute_config import BLOCK_N as _BLOCK_N
from ._sparse_gqa_cute_config import HEAD_DIM as _HEAD_DIM
from ._sparse_gqa_cute_config import NUM_SPLITS as _NUM_SPLITS
from ._sparse_gqa_cute_config import SELECTION_WIDTH as _SELECTION_WIDTH
from ._sparse_gqa_cute_config import clear_device_cache
from ._sparse_gqa_cute_config import is_candidate


_THREADS = 128
_WARPS = _THREADS // 32
_DIMS_PER_LANE = _HEAD_DIM // 32
_TILES_PER_SPLIT = 3
_LOG2_E = 1.4426950408889634
_LOCK = RLock()
_KERNEL_CACHE: dict[tuple[int, int, int, int], Callable[..., None]] = {}
_WARMED: dict[tuple[int, int, int, int], Callable[..., None]] = {}
_MERGE_CACHE: dict[tuple[int, int, int], Callable[..., None]] = {}
_MERGE_WARMED: dict[tuple[int, int, int], Callable[..., None]] = {}


def _add(left: Float32, right: Float32) -> Float32:
    return left + right


@dsl_user_op
def _exp2_approx(value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
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
def _log2_approx(value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class _SparseGqaSplitKernel:
    """One four-warp CTA per ``(row, query_head, split)``.

    Each warp owns one quarter of every 16-position selector tile.  A lane
    retains eight head dimensions and an online-softmax value accumulator in
    registers.  The four independent warp states are merged through 4 KiB of
    shared memory before writing the caller-owned FP32 split partial.
    """

    def __init__(self, *, q_heads: int, kv_heads: int) -> None:
        self.q_heads = int(q_heads)
        self.kv_heads = int(kv_heads)
        self.heads_per_kv = self.q_heads // self.kv_heads

    @cute.jit
    def __call__(
        self,
        query: cute.Pointer,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        block_table: cute.Pointer,
        request_ids: cute.Pointer,
        selected_positions: cute.Pointer,
        query_positions: cute.Pointer,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        num_cache_pages: Int64,
        table_batch: Int64,
        table_width: Int64,
        page_size: Int64,
        key_page_stride: Int64,
        key_token_stride: Int64,
        key_head_stride: Int64,
        value_page_stride: Int64,
        value_token_stride: Int64,
        value_head_stride: Int64,
        softmax_scale: Float32,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            query,
            key_cache,
            value_cache,
            block_table,
            request_ids,
            selected_positions,
            query_positions,
            partial_output,
            partial_lse,
            num_cache_pages,
            table_batch,
            table_width,
            page_size,
            key_page_stride,
            key_token_stride,
            key_head_stride,
            value_page_stride,
            value_token_stride,
            value_head_stride,
            softmax_scale,
        ).launch(
            grid=(_NUM_SPLITS, self.q_heads, rows),
            block=(_THREADS, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        query: cute.Pointer,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        block_table: cute.Pointer,
        request_ids: cute.Pointer,
        selected_positions: cute.Pointer,
        query_positions: cute.Pointer,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        num_cache_pages: Int64,
        table_batch: Int64,
        table_width: Int64,
        page_size: Int64,
        key_page_stride: Int64,
        key_token_stride: Int64,
        key_head_stride: Int64,
        value_page_stride: Int64,
        value_token_stride: Int64,
        value_head_stride: Int64,
        softmax_scale: Float32,
    ) -> None:
        split_idx, query_head_idx, row_idx = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        lane = Int32(thread_idx) & Int32(31)
        warp = Int32(thread_idx) >> Int32(5)
        split = Int32(split_idx)
        query_head = Int32(query_head_idx)
        row = Int32(row_idx)
        kv_head = query_head // Int32(self.heads_per_kv)

        # Pool-scaled products are Int64 even though ordinary benchmark page
        # ids fit in Int32.
        query_base = (
            row.to(Int64) * Int64(self.q_heads) + query_head.to(Int64)
        ) * Int64(_HEAD_DIM)
        request_id = request_ids[row].to(Int64)
        query_position = query_positions[row].to(Int64)
        request_valid = (request_id >= Int64(0)) & (request_id < table_batch)

        query_values = cute.make_rmem_tensor((_DIMS_PER_LANE,), Float32)
        accumulator = cute.make_rmem_tensor((_DIMS_PER_LANE,), Float32)
        for item in cutlass.range_constexpr(_DIMS_PER_LANE):
            dimension = lane + Int32(item * 32)
            query_values[item] = Float32(query[query_base + dimension.to(Int64)])
            accumulator[item] = Float32(0.0)

        running_max = Float32(-Float32.inf)
        running_sum = Float32(0.0)

        # 2051 positions form 129 BLOCK_N tiles.  Split zero sees tile 128's
        # three-position tail; every other split sees two complete tiles.
        for local_tile in cutlass.range_constexpr(_TILES_PER_SPLIT):
            tile = split + Int32(local_tile * _NUM_SPLITS)
            for warp_item in cutlass.range_constexpr(_BLOCK_N // _WARPS):
                column = (
                    tile * Int32(_BLOCK_N)
                    + warp * Int32(_BLOCK_N // _WARPS)
                    + Int32(warp_item)
                )
                valid = request_valid & (column < Int32(_SELECTION_WIDTH))
                logical_position = Int64(-1)
                if valid:
                    selected_offset = row.to(Int64) * Int64(
                        _SELECTION_WIDTH
                    ) + column.to(Int64)
                    logical_position = selected_positions[selected_offset].to(Int64)
                    valid = (logical_position >= Int64(0)) & (
                        logical_position <= query_position
                    )

                physical_page = Int64(-1)
                page_offset = Int64(0)
                if valid:
                    logical_page = logical_position // page_size
                    valid = logical_page < table_width
                    if valid:
                        table_offset = request_id * table_width + logical_page
                        physical_page = block_table[table_offset].to(Int64)
                        valid = (physical_page >= Int64(0)) & (
                            physical_page < num_cache_pages
                        )
                        page_offset = logical_position % page_size

                if valid:
                    key_cache_base = (
                        physical_page * key_page_stride
                        + page_offset * key_token_stride
                        + kv_head.to(Int64) * key_head_stride
                    )
                    value_cache_base = (
                        physical_page * value_page_stride
                        + page_offset * value_token_stride
                        + kv_head.to(Int64) * value_head_stride
                    )
                    score = Float32(0.0)
                    value_values = cute.make_rmem_tensor((_DIMS_PER_LANE,), Float32)
                    for item in cutlass.range_constexpr(_DIMS_PER_LANE):
                        dimension = lane + Int32(item * 32)
                        key_offset = key_cache_base + dimension.to(Int64)
                        value_offset = value_cache_base + dimension.to(Int64)
                        key_value = Float32(key_cache[key_offset])
                        value_values[item] = Float32(value_cache[value_offset])
                        score += query_values[item] * key_value
                    score = warp_reduce(score, _add) * softmax_scale

                    next_max = running_max
                    if score > next_max:
                        next_max = score
                    prior_scale = Float32(0.0)
                    if running_sum > Float32(0.0):
                        prior_scale = _exp2_approx(
                            (running_max - next_max) * Float32(_LOG2_E)
                        )
                    probability = _exp2_approx((score - next_max) * Float32(_LOG2_E))
                    for item in cutlass.range_constexpr(_DIMS_PER_LANE):
                        accumulator[item] = (
                            accumulator[item] * prior_scale
                            + value_values[item] * probability
                        )
                    running_sum = running_sum * prior_scale + probability
                    running_max = next_max

        smem = cutlass.utils.SmemAllocator()
        warp_output = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((_WARPS, _HEAD_DIM), stride=(_HEAD_DIM, 1)),
            byte_alignment=16,
        )
        warp_max = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((_WARPS,), stride=(1,)),
            byte_alignment=16,
        )
        warp_sum = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((_WARPS,), stride=(1,)),
            byte_alignment=16,
        )
        warp_weight = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((_WARPS,), stride=(1,)),
            byte_alignment=16,
        )
        inverse = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((1,), stride=(1,)),
            byte_alignment=4,
        )

        for item in cutlass.range_constexpr(_DIMS_PER_LANE):
            dimension = lane + Int32(item * 32)
            warp_output[warp, dimension] = accumulator[item]
        if lane == Int32(0):
            warp_max[warp] = running_max
            warp_sum[warp] = running_sum
        cute.arch.barrier()

        if Int32(thread_idx) == Int32(0):
            merged_max = Float32(-Float32.inf)
            for source_warp in cutlass.range_constexpr(_WARPS):
                source_sum = Float32(warp_sum[source_warp])
                source_max = Float32(warp_max[source_warp])
                if (source_sum > Float32(0.0)) & (source_max > merged_max):
                    merged_max = source_max

            merged_sum = Float32(0.0)
            for source_warp in cutlass.range_constexpr(_WARPS):
                source_sum = Float32(warp_sum[source_warp])
                weight = Float32(0.0)
                if source_sum > Float32(0.0):
                    weight = _exp2_approx(
                        (Float32(warp_max[source_warp]) - merged_max) * Float32(_LOG2_E)
                    )
                    merged_sum += source_sum * weight
                warp_weight[source_warp] = weight

            inverse_value = Float32(0.0)
            lse = Float32(-Float32.inf)
            if merged_sum > Float32(0.0):
                inverse_value = Float32(1.0) / merged_sum
                lse = merged_max + _log2_approx(merged_sum) / Float32(_LOG2_E)
            inverse[0] = inverse_value
            lse_offset = (row.to(Int64) * Int64(_NUM_SPLITS) + split.to(Int64)) * Int64(
                self.q_heads
            ) + query_head.to(Int64)
            partial_lse[lse_offset] = lse
        cute.arch.barrier()

        output_base = (
            (row.to(Int64) * Int64(_NUM_SPLITS) + split.to(Int64)) * Int64(self.q_heads)
            + query_head.to(Int64)
        ) * Int64(_HEAD_DIM)
        output_dimension = Int32(thread_idx)
        for _item in cutlass.range_constexpr(_HEAD_DIM // _THREADS):
            numerator = Float32(0.0)
            for source_warp in cutlass.range_constexpr(_WARPS):
                numerator += Float32(
                    warp_output[source_warp, output_dimension]
                ) * Float32(warp_weight[source_warp])
            partial_output[output_base + output_dimension.to(Int64)] = (
                numerator * Float32(inverse[0])
            )
            output_dimension += Int32(_THREADS)


class _SparseGqaMergeKernel:
    """Merge 64 FP32 split-softmax partials into caller-owned BF16 output."""

    def __init__(self, *, q_heads: int) -> None:
        self.q_heads = int(q_heads)

    @cute.jit
    def __call__(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(partial_output, partial_lse, output).launch(
            grid=(self.q_heads, rows, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
    ) -> None:
        query_head, row, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        thread_i = Int32(thread)
        row_i = Int64(row)
        head_i = Int64(query_head)
        allocator = cutlass.utils.SmemAllocator()
        weights = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((_NUM_SPLITS,), stride=(1,)),
            byte_alignment=16,
        )
        inverse = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((1,), stride=(1,)),
            byte_alignment=4,
        )
        if thread_i == Int32(0):
            maximum = Float32(-Float32.inf)
            for split in cutlass.range_constexpr(_NUM_SPLITS):
                offset = (
                    (row_i * Int64(_NUM_SPLITS) + Int64(split))
                    * Int64(self.q_heads)
                    + head_i
                )
                value = Float32(partial_lse[offset])
                if value > maximum:
                    maximum = value
            denominator = Float32(0.0)
            for split in cutlass.range_constexpr(_NUM_SPLITS):
                offset = (
                    (row_i * Int64(_NUM_SPLITS) + Int64(split))
                    * Int64(self.q_heads)
                    + head_i
                )
                value = Float32(partial_lse[offset])
                weight = Float32(0.0)
                if value > Float32(-Float32.inf):
                    weight = _exp2_approx((value - maximum) * Float32(_LOG2_E))
                weights[split] = weight
                denominator += weight
            inverse[0] = Float32(0.0)
            if denominator > Float32(0.0):
                inverse[0] = Float32(1.0) / denominator
        cute.arch.sync_threads()

        dimension = thread_i
        for _ in cutlass.range_constexpr(_HEAD_DIM // _THREADS):
            total = Float32(0.0)
            for split in cutlass.range_constexpr(_NUM_SPLITS):
                offset = (
                    (
                        (row_i * Int64(_NUM_SPLITS) + Int64(split))
                        * Int64(self.q_heads)
                        + head_i
                    )
                    * Int64(_HEAD_DIM)
                    + dimension.to(Int64)
                )
                total += Float32(partial_output[offset]) * Float32(weights[split])
            output_offset = (
                (row_i * Int64(self.q_heads) + head_i) * Int64(_HEAD_DIM)
                + dimension.to(Int64)
            )
            output[output_offset] = BFloat16(total * Float32(inverse[0]))
            dimension += Int32(_THREADS)


def is_supported(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    block_n: int,
    splits: int,
) -> bool:
    """Return whether tensors match the qualified Qwen split specialization."""
    return is_candidate(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        partial_output=partial_output,
        partial_lse=partial_lse,
        block_n=block_n,
        splits=splits,
    )


def _cache_key(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    request_ids: torch.Tensor,
) -> tuple[int, int, int, int]:
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        int(device_index),
        int(query.shape[1]),
        int(key_cache.shape[2]),
        int(request_ids.element_size() * 8),
    )


def _pointer(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
) -> cute.Pointer:
    return make_ptr(
        dtype,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=max(1, dtype.width // 8),
    )


def _fake_pointer(dtype: type[cutlass.Numeric]) -> cute.Pointer:
    return make_ptr(
        dtype,
        16,
        cute.AddressSpace.gmem,
        assumed_align=max(1, dtype.width // 8),
    )


def _compile(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    request_ids: torch.Tensor,
) -> tuple[tuple[int, int, int, int], Callable[..., None]]:
    key = _cache_key(query, key_cache, request_ids)
    with _LOCK:
        cached = _KERNEL_CACHE.get(key)
        if cached is not None:
            return key, cached
        _, q_heads, kv_heads, request_id_bits = key
        request_id_type = Int32 if request_id_bits == 32 else Int64
        kernel = _SparseGqaSplitKernel(
            q_heads=q_heads,
            kv_heads=kv_heads,
        )
        with torch.cuda.device(query.device):
            raise_if_kernel_resolution_frozen(
                "cute.compile",
                target=kernel,
                cache_key=key,
            )
            raw = b12x_compile(
                kernel,
                _fake_pointer(BFloat16),
                _fake_pointer(BFloat16),
                _fake_pointer(BFloat16),
                _fake_pointer(Int32),
                _fake_pointer(request_id_type),
                _fake_pointer(Int32),
                _fake_pointer(Int64),
                _fake_pointer(Float32),
                _fake_pointer(Float32),
                Int64(1),
                Int64(1),
                Int64(1),
                Int64(1),
                Int64(1),
                Int64(1),
                Int64(1),
                Int64(1),
                Int64(1),
                Int64(1),
                Float32(1.0),
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.qsa.sparse_gqa_split",
                    3,
                    (q_heads, kv_heads, request_id_bits),
                    labels=("q_heads", "kv_heads", "request_id_bits"),
                ),
            )
        _KERNEL_CACHE[key] = raw
        return key, raw


def precompile_sparse_gqa_split(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    request_ids: torch.Tensor,
) -> None:
    """Compile a supported specialization without touching runtime storage."""
    with torch.cuda.device(query.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CuTe sparse GQA compilation is forbidden during capture"
            )
        _compile(query=query, key_cache=key_cache, request_ids=request_ids)


def launch_sparse_gqa_split(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    softmax_scale: float,
) -> None:
    """Launch the specialized split core without allocation or fallback."""
    key = _cache_key(query, key_cache, request_ids)
    with torch.cuda.device(query.device):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            raw = _KERNEL_CACHE.get(key)
            warmed = raw is not None and _WARMED.get(key) is raw
        if capturing and (raw is None or not warmed):
            raise RuntimeError(
                "CuTe sparse GQA must be compiled and warm-run before CUDA graph "
                "capture"
            )
        if raw is None:
            key, raw = _compile(
                query=query,
                key_cache=key_cache,
                request_ids=request_ids,
            )
        request_id_type = Int32 if request_ids.dtype == torch.int32 else Int64
        run_compiled(
            raw,
            (
                _pointer(query, BFloat16),
                _pointer(key_cache, BFloat16),
                _pointer(value_cache, BFloat16),
                _pointer(block_table, Int32),
                _pointer(request_ids, request_id_type),
                _pointer(selected_positions, Int32),
                _pointer(query_positions, Int64),
                _pointer(partial_output, Float32),
                _pointer(partial_lse, Float32),
                int(key_cache.shape[0]),
                int(block_table.shape[0]),
                int(block_table.shape[1]),
                int(key_cache.shape[1]),
                int(key_cache.stride(0)),
                int(key_cache.stride(1)),
                int(key_cache.stride(2)),
                int(value_cache.stride(0)),
                int(value_cache.stride(1)),
                int(value_cache.stride(2)),
                float(softmax_scale),
                int(query.shape[0]),
                current_cuda_stream(),
            ),
        )
    if not capturing:
        with _LOCK:
            if _KERNEL_CACHE.get(key) is raw:
                _WARMED[key] = raw


def launch_sparse_gqa_merge(
    *,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    output: torch.Tensor,
    rows: int,
) -> None:
    """Merge the fixed Qwen split layout without allocation or fallback."""
    device_index = partial_output.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    q_heads = int(partial_output.shape[2])
    key = (int(device_index), q_heads, _HEAD_DIM)
    with torch.cuda.device(partial_output.device):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            raw = _MERGE_CACHE.get(key)
            warmed = raw is not None and _MERGE_WARMED.get(key) is raw
        if capturing and (raw is None or not warmed):
            raise RuntimeError(
                "CuTe sparse GQA merge must be compiled and warm-run before "
                "CUDA graph capture"
            )
        if raw is None:
            kernel = _SparseGqaMergeKernel(q_heads=q_heads)
            raise_if_kernel_resolution_frozen(
                "cute.compile",
                target=kernel,
                cache_key=key,
            )
            raw = b12x_compile(
                kernel,
                _fake_pointer(Float32),
                _fake_pointer(Float32),
                _fake_pointer(BFloat16),
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.qsa.sparse_gqa_merge",
                    2,
                    key[1:],
                ),
            )
            with _LOCK:
                _MERGE_CACHE[key] = raw
        run_compiled(
            raw,
            (
                _pointer(partial_output, Float32),
                _pointer(partial_lse, Float32),
                _pointer(output, BFloat16),
                int(rows),
                current_cuda_stream(),
            ),
        )
    if not capturing:
        with _LOCK:
            if _MERGE_CACHE.get(key) is raw:
                _MERGE_WARMED[key] = raw


def clear_caches() -> None:
    """Clear process-local compiled and warm-launch state for focused tests."""
    with _LOCK:
        _KERNEL_CACHE.clear()
        _WARMED.clear()
        _MERGE_CACHE.clear()
        _MERGE_WARMED.clear()
    clear_device_cache()


__all__ = [
    "clear_caches",
    "is_supported",
    "launch_sparse_gqa_split",
    "launch_sparse_gqa_merge",
    "precompile_sparse_gqa_split",
]
