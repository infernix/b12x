"""Indexed sparse paged causal GQA for QSA.

This private stage reads exact BF16 main-cache K/V at caller-selected logical
token positions. It never writes either cache. Split policy and all storage
are caller-owned; launches perform no allocation and never dispatch a Torch
reference fallback.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOG2_E = tl.constexpr(1.4426950408889634)


@triton.jit
def _sparse_gqa_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    request_ids_ptr,
    selected_positions_ptr,
    query_positions_ptr,
    output_ptr,
    partial_lse_ptr,
    q_stride_r: tl.constexpr,
    q_stride_h: tl.constexpr,
    k_stride_p: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    v_stride_p: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    table_stride_b: tl.constexpr,
    table_stride_p: tl.constexpr,
    selected_stride_r: tl.constexpr,
    selected_stride_k: tl.constexpr,
    out_stride_r: tl.constexpr,
    out_stride_s: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    lse_stride_r: tl.constexpr,
    lse_stride_s: tl.constexpr,
    lse_stride_h: tl.constexpr,
    softmax_scale,
    ROWS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_CACHE_PAGES: tl.constexpr,
    TABLE_BATCH: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    SELECTION_WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    WRITE_PARTIAL: tl.constexpr,
):
    row = tl.program_id(0)
    query_head = tl.program_id(1)
    split = tl.program_id(2)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM

    query_offsets = row * q_stride_r + query_head * q_stride_h + dims
    query = tl.load(query_ptr + query_offsets, mask=dim_mask, other=0.0).to(tl.float32)
    heads_per_kv = Q_HEADS // KV_HEADS
    kv_head = query_head // heads_per_kv
    request_id = tl.load(request_ids_ptr + row).to(tl.int64)
    query_position = tl.load(query_positions_ptr + row).to(tl.int64)
    request_valid = (request_id >= 0) & (request_id < TABLE_BATCH)

    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    key_tiles = tl.cdiv(SELECTION_WIDTH, BLOCK_N)
    tiles_per_split = tl.cdiv(key_tiles, NUM_SPLITS)

    for local_tile in tl.range(0, tiles_per_split, num_stages=2):
        tile = split + local_tile * NUM_SPLITS
        columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        slot_valid = (tile < key_tiles) & (columns < SELECTION_WIDTH)
        selected_offsets = row * selected_stride_r + columns * selected_stride_k
        logical_position = tl.load(
            selected_positions_ptr + selected_offsets,
            mask=slot_valid,
            other=-1,
        ).to(tl.int64)
        valid = (
            slot_valid
            & request_valid
            & (logical_position >= 0)
            & (logical_position <= query_position)
        )

        logical_page = logical_position // PAGE_SIZE
        valid &= (logical_page >= 0) & (logical_page < TABLE_WIDTH)
        safe_request = tl.where(request_valid, request_id, 0).to(tl.int64)
        safe_logical_page = tl.where(valid, logical_page, 0).to(tl.int64)
        table_offsets = safe_request * tl.full(
            (), table_stride_b, tl.int64
        ) + safe_logical_page * tl.full((), table_stride_p, tl.int64)
        physical_page = tl.load(
            block_table_ptr + table_offsets,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid &= (physical_page >= 0) & (physical_page < NUM_CACHE_PAGES)

        # Keep the page id in Int64 before every page-scaled product. This is
        # required for serving pools whose live physical ids cross Int32.
        safe_page = tl.where(valid, physical_page, 0).to(tl.int64)
        page_offset = tl.where(valid, logical_position % PAGE_SIZE, 0).to(tl.int64)
        key_offsets = (
            safe_page[:, None] * tl.full((), k_stride_p, tl.int64)
            + page_offset[:, None] * tl.full((), k_stride_t, tl.int64)
            + tl.full((), kv_head * k_stride_h, tl.int64)
            + dims[None, :].to(tl.int64)
        )
        value_offsets = (
            safe_page[:, None] * tl.full((), v_stride_p, tl.int64)
            + page_offset[:, None] * tl.full((), v_stride_t, tl.int64)
            + tl.full((), kv_head * v_stride_h, tl.int64)
            + dims[None, :].to(tl.int64)
        )
        cache_mask = valid[:, None] & dim_mask[None, :]
        key = tl.load(key_cache_ptr + key_offsets, mask=cache_mask, other=0.0).to(
            tl.float32
        )
        value = tl.load(value_cache_ptr + value_offsets, mask=cache_mask, other=0.0).to(
            tl.float32
        )

        scores = tl.sum(key * query[None, :], axis=1) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))
        tile_has_value = tl.sum(valid.to(tl.int32), axis=0) > 0
        tile_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, tile_max)
        prior_scale = tl.where(
            tile_has_value,
            tl.exp2((running_max - next_max) * _LOG2_E),
            1.0,
        )
        probabilities = tl.where(
            valid,
            tl.exp2((scores - next_max) * _LOG2_E),
            0.0,
        )
        accumulator = accumulator * prior_scale + tl.sum(
            probabilities[:, None] * value, axis=0
        )
        running_sum = running_sum * prior_scale + tl.sum(probabilities, axis=0)
        running_max = tl.where(tile_has_value, next_max, running_max)

    denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
    normalized = tl.where(running_sum > 0.0, accumulator / denominator, 0.0)
    if WRITE_PARTIAL:
        output_offsets = (
            row * out_stride_r
            + split * out_stride_s
            + query_head * out_stride_h
            + dims * out_stride_d
        )
        lse_offset = (
            row * lse_stride_r + split * lse_stride_s + query_head * lse_stride_h
        )
        lse = tl.where(
            running_sum > 0.0,
            running_max + tl.log2(running_sum) / _LOG2_E,
            -float("inf"),
        )
        tl.store(output_ptr + output_offsets, normalized, mask=dim_mask)
        tl.store(partial_lse_ptr + lse_offset, lse)
    else:
        output_offsets = (
            row * out_stride_r + query_head * out_stride_h + dims * out_stride_d
        )
        tl.store(output_ptr + output_offsets, normalized, mask=dim_mask)


@triton.jit
def _merge_sparse_gqa_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    partial_stride_r: tl.constexpr,
    partial_stride_s: tl.constexpr,
    partial_stride_h: tl.constexpr,
    partial_stride_d: tl.constexpr,
    lse_stride_r: tl.constexpr,
    lse_stride_s: tl.constexpr,
    lse_stride_h: tl.constexpr,
    out_stride_r: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    query_head = tl.program_id(1)
    splits = tl.arange(0, BLOCK_SPLITS)
    split_mask = splits < NUM_SPLITS
    lse_offsets = row * lse_stride_r + splits * lse_stride_s + query_head * lse_stride_h
    lse = tl.load(
        partial_lse_ptr + lse_offsets,
        mask=split_mask,
        other=-float("inf"),
    ).to(tl.float32)
    finite = split_mask & (lse != -float("inf"))
    maximum = tl.max(lse, axis=0)
    weights = tl.where(
        finite,
        tl.exp2((lse - maximum) * _LOG2_E),
        0.0,
    )
    weight_sum = tl.sum(weights, axis=0)

    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    partial_offsets = (
        row * partial_stride_r
        + splits[:, None] * partial_stride_s
        + query_head * partial_stride_h
        + dims[None, :] * partial_stride_d
    )
    partial = tl.load(
        partial_output_ptr + partial_offsets,
        mask=finite[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    merged = tl.sum(weights[:, None] * partial, axis=0)
    merged = tl.where(weight_sum > 0.0, merged / weight_sum, 0.0)
    output_offsets = row * out_stride_r + query_head * out_stride_h + dims
    tl.store(output_ptr + output_offsets, merged, mask=dim_mask)


def _launch_direct(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    block_n: int,
) -> None:
    rows, q_heads, head_dim = map(int, query.shape)
    kv_heads = int(key_cache.shape[2])
    _sparse_gqa_kernel[(rows, q_heads, 1)](
        query,
        key_cache,
        value_cache,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        output,
        output,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        selected_positions.stride(0),
        selected_positions.stride(1),
        output.stride(0),
        0,
        output.stride(1),
        output.stride(2),
        0,
        0,
        0,
        float(softmax_scale),
        ROWS=rows,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=int(key_cache.shape[1]),
        NUM_CACHE_PAGES=int(key_cache.shape[0]),
        TABLE_BATCH=int(block_table.shape[0]),
        TABLE_WIDTH=int(block_table.shape[1]),
        SELECTION_WIDTH=int(selected_positions.shape[1]),
        BLOCK_N=int(block_n),
        BLOCK_D=triton.next_power_of_2(head_dim),
        NUM_SPLITS=1,
        WRITE_PARTIAL=False,
        num_warps=4 if int(block_n) == 16 else 2,
        num_stages=2,
    )


def _launch_split(
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
    block_n: int,
    splits: int,
) -> None:
    rows, q_heads, head_dim = map(int, query.shape)
    kv_heads = int(key_cache.shape[2])
    _sparse_gqa_kernel[(rows, q_heads, splits)](
        query,
        key_cache,
        value_cache,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        partial_output,
        partial_lse,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        selected_positions.stride(0),
        selected_positions.stride(1),
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        float(softmax_scale),
        ROWS=rows,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=int(key_cache.shape[1]),
        NUM_CACHE_PAGES=int(key_cache.shape[0]),
        TABLE_BATCH=int(block_table.shape[0]),
        TABLE_WIDTH=int(block_table.shape[1]),
        SELECTION_WIDTH=int(selected_positions.shape[1]),
        BLOCK_N=int(block_n),
        BLOCK_D=triton.next_power_of_2(head_dim),
        NUM_SPLITS=int(splits),
        WRITE_PARTIAL=True,
        num_warps=4 if int(block_n) == 16 else 2,
        num_stages=2,
    )


def _launch_merge(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    splits: int,
) -> None:
    q_heads = int(output.shape[1])
    head_dim = int(output.shape[2])
    _merge_sparse_gqa_kernel[(rows, q_heads)](
        partial_output,
        partial_lse,
        output,
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        HEAD_DIM=head_dim,
        NUM_SPLITS=int(splits),
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_SPLITS=triton.next_power_of_2(splits),
        num_warps=4,
    )


@torch.library.custom_op("b12x::qsa_sparse_paged_gqa_direct", mutates_args=("output",))
def _direct_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    block_n: int,
) -> None:
    _launch_direct(
        query,
        key_cache,
        value_cache,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        output,
        softmax_scale,
        block_n,
    )


@_direct_op.register_fake
def _direct_fake(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    block_n: int,
) -> None:
    del query, key_cache, value_cache, block_table, request_ids
    del selected_positions, query_positions, output, softmax_scale, block_n


@torch.library.custom_op(
    "b12x::qsa_sparse_paged_gqa_split",
    mutates_args=("partial_output", "partial_lse"),
)
def _split_op(
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
    block_n: int,
    splits: int,
) -> None:
    _launch_split(
        query,
        key_cache,
        value_cache,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        partial_output,
        partial_lse,
        softmax_scale,
        block_n,
        splits,
    )


@_split_op.register_fake
def _split_fake(
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
    block_n: int,
    splits: int,
) -> None:
    del query, key_cache, value_cache, block_table, request_ids
    del selected_positions, query_positions, partial_output, partial_lse
    del softmax_scale, block_n, splits


@torch.library.custom_op("b12x::qsa_sparse_paged_gqa_merge", mutates_args=("output",))
def _merge_op(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    splits: int,
) -> None:
    _launch_merge(partial_output, partial_lse, output, rows, splits)


@_merge_op.register_fake
def _merge_fake(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    splits: int,
) -> None:
    del partial_output, partial_lse, output, rows, splits


def _require_unit_inner_stride(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim == 0 or int(tensor.stride(-1)) != 1:
        raise ValueError(f"{name} must have unit inner stride")


def _validate_launch(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    partial_output: torch.Tensor | None,
    partial_lse: torch.Tensor | None,
    softmax_scale: float,
    block_n: int,
    splits: int,
) -> tuple[int, int, int]:
    if not query.is_cuda:
        raise ValueError("QSA sparse GQA requires CUDA tensors")
    device = query.device
    if query.ndim != 3:
        raise ValueError("query must have shape [rows, q_heads, head_dim]")
    rows, q_heads, head_dim = map(int, query.shape)
    if rows <= 0:
        raise ValueError("query must contain at least one active row")
    if query.dtype != torch.bfloat16:
        raise TypeError(f"query must be torch.bfloat16, got {query.dtype}")
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two at least 16")
    _require_unit_inner_stride(query, "query")

    if key_cache.ndim != 4:
        raise ValueError("key_cache must have shape [pages, page, kv_heads, dim]")
    pages, page_size, kv_heads, cache_dim = map(int, key_cache.shape)
    if pages <= 0 or page_size <= 0 or kv_heads <= 0:
        raise ValueError("key_cache dimensions must be positive")
    if cache_dim != head_dim:
        raise ValueError("key_cache head dimension must match query")
    if q_heads % kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")
    if key_cache.dtype != torch.bfloat16:
        raise TypeError("key_cache must be torch.bfloat16")
    if value_cache.shape != key_cache.shape or value_cache.dtype != torch.bfloat16:
        raise ValueError("value_cache must match the BF16 key_cache shape")
    _require_unit_inner_stride(key_cache, "key_cache")
    _require_unit_inner_stride(value_cache, "value_cache")

    tensors = (
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        output,
        key_cache,
        value_cache,
    )
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all QSA sparse GQA tensors must share one device")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise TypeError("block_table must be rank-2 torch.int32")
    if not block_table.is_contiguous():
        raise ValueError("block_table must be contiguous")
    if request_ids.shape != (rows,) or request_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("request_ids must be contiguous int32/int64 shape [rows]")
    if not request_ids.is_contiguous():
        raise ValueError("request_ids must be contiguous")
    if query_positions.shape != (rows,) or query_positions.dtype != torch.int64:
        raise TypeError("query_positions must be contiguous int64 shape [rows]")
    if not query_positions.is_contiguous():
        raise ValueError("query_positions must be contiguous")
    if (
        selected_positions.ndim != 2
        or int(selected_positions.shape[0]) < rows
        or int(selected_positions.shape[1]) <= 0
        or selected_positions.dtype != torch.int32
    ):
        raise TypeError(
            "selected_positions must be int32 [capacity_rows, selection_width]"
        )
    if not selected_positions.is_contiguous():
        raise ValueError("selected_positions must be contiguous")
    if (
        output.ndim != 3
        or int(output.shape[0]) < rows
        or tuple(output.shape[1:]) != (q_heads, head_dim)
        or output.dtype != torch.bfloat16
    ):
        raise ValueError("output must be BF16 [capacity_rows, q_heads, head_dim]")
    _require_unit_inner_stride(output, "output")

    block_n = int(block_n)
    splits = int(splits)
    if block_n not in (16, 64):
        raise ValueError("block_n must be 16 or 64")
    key_tiles = math.ceil(int(selected_positions.shape[1]) / block_n)
    if splits <= 0 or splits & (splits - 1) or splits > key_tiles:
        raise ValueError(
            "splits must be a positive power of two not exceeding key tiles"
        )
    if not math.isfinite(float(softmax_scale)) or float(softmax_scale) <= 0:
        raise ValueError("softmax_scale must be finite and positive")

    if splits > 1:
        if partial_output is None or partial_lse is None:
            raise ValueError("split sparse GQA requires both partial tensors")
        if partial_output.device != device or partial_lse.device != device:
            raise ValueError("partial tensors must share the query device")
        if (
            partial_output.ndim != 4
            or int(partial_output.shape[0]) < rows
            or int(partial_output.shape[1]) < splits
            or tuple(partial_output.shape[2:]) != (q_heads, head_dim)
            or partial_output.dtype != torch.float32
        ):
            raise ValueError(
                "partial_output must be float32 [capacity_rows, >=splits, "
                "q_heads, head_dim]"
            )
        if (
            partial_lse.ndim != 3
            or int(partial_lse.shape[0]) < rows
            or int(partial_lse.shape[1]) < splits
            or int(partial_lse.shape[2]) != q_heads
            or partial_lse.dtype != torch.float32
        ):
            raise ValueError(
                "partial_lse must be float32 [capacity_rows, >=splits, q_heads]"
            )
        _require_unit_inner_stride(partial_output, "partial_output")
        _require_unit_inner_stride(partial_lse, "partial_lse")
    return rows, q_heads, head_dim


def launch_sparse_paged_gqa(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    partial_output: torch.Tensor | None,
    partial_lse: torch.Tensor | None,
    softmax_scale: float,
    block_n: int,
    splits: int,
) -> torch.Tensor:
    """Launch allocation-free indexed sparse causal GQA into ``output``."""
    rows, _, _ = _validate_launch(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=output,
        partial_output=partial_output,
        partial_lse=partial_lse,
        softmax_scale=softmax_scale,
        block_n=block_n,
        splits=splits,
    )
    if int(splits) == 1:
        torch.ops.b12x.qsa_sparse_paged_gqa_direct(
            query,
            key_cache,
            value_cache,
            block_table,
            request_ids,
            selected_positions,
            query_positions,
            output,
            float(softmax_scale),
            int(block_n),
        )
    else:
        assert partial_output is not None and partial_lse is not None
        torch.ops.b12x.qsa_sparse_paged_gqa_split(
            query,
            key_cache,
            value_cache,
            block_table,
            request_ids,
            selected_positions,
            query_positions,
            partial_output,
            partial_lse,
            float(softmax_scale),
            int(block_n),
            int(splits),
        )
        torch.ops.b12x.qsa_sparse_paged_gqa_merge(
            partial_output,
            partial_lse,
            output,
            rows,
            int(splits),
        )
    return output[:rows]


__all__ = ["launch_sparse_paged_gqa"]
