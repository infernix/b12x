"""Portable Triton stages for GLM pooled sparse-index selection."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ..qsa._kernels import (
    _validate_packed_boundaries_kernel,
    launch_expand_selected_groups,
    launch_propagate_request_errors,
    launch_remap_topk_group_ids,
    launch_stabilize_topk,
    launch_stage_topk_carry,
    launch_topk_groups,
    launch_validate_completed_groups as _launch_qsa_validate_completed_groups,
    launch_validate_decode_rows as _launch_qsa_validate_decode_rows,
)


@triton.jit
def _clear_i32_kernel(values, count, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(values + offsets, 0, mask=offsets < count)


@triton.jit
def _validate_prefill_rows_kernel(
    request_ids,
    query_positions,
    sequence_lengths,
    query_start_loc,
    raw_state_slot_ids,
    raw_interval_start_positions,
    request_errors,
    state_errors,
    raw_state_slot_stride,
    raw_interval_start_stride,
    ROWS: tl.constexpr,
    MAX_BATCH: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    MAX_RAW_STATE_SLOTS: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    row = tl.program_id(0)
    terminal = tl.load(query_start_loc + MAX_BATCH).to(tl.int32)
    global_error = tl.load(request_errors + MAX_BATCH).to(tl.int32)
    request = tl.load(request_ids + row).to(tl.int64)
    position = tl.load(query_positions + row).to(tl.int64)
    live = row < terminal
    valid_request = live & (request >= 0) & (request < MAX_BATCH)
    start = tl.load(
        query_start_loc + request, mask=valid_request, other=0
    ).to(tl.int64)
    end = tl.load(
        query_start_loc + request + 1, mask=valid_request, other=0
    ).to(tl.int64)
    query_length = end - start
    sequence_length = tl.load(
        sequence_lengths + request, mask=valid_request, other=0
    ).to(tl.int64)
    expected_position = sequence_length - query_length + row - start
    state_slot = tl.load(
        raw_state_slot_ids + request * raw_state_slot_stride,
        mask=valid_request,
        other=-1,
    ).to(tl.int64)
    valid_slot = (state_slot >= 0) & (state_slot < MAX_RAW_STATE_SLOTS)
    prior_anchor = tl.load(
        raw_interval_start_positions
        + state_slot * raw_interval_start_stride,
        mask=valid_request & valid_slot,
        other=-2,
    ).to(tl.int64)
    expected_prior_anchor = expected_position - (row - start) - 1

    owners = tl.arange(0, BLOCK_BATCH)
    owner_mask = owners < MAX_BATCH
    owner_slots = tl.load(
        raw_state_slot_ids + owners * raw_state_slot_stride,
        mask=owner_mask,
        other=-1,
    ).to(tl.int64)
    invalid_slot_map = (
        tl.sum(
            owner_mask & ((owner_slots < -1) | (owner_slots >= MAX_RAW_STATE_SLOTS)),
            axis=0,
        )
        != 0
    )
    duplicate_slot = (
        valid_request
        & valid_slot
        & (
            tl.sum(
                owner_mask
                & (owners != request)
                & (owner_slots == state_slot)
                & (owner_slots >= 0),
                axis=0,
            )
            != 0
        )
    )
    error = global_error
    error = error | tl.where(live & ~valid_request, 2, 0)
    error = error | tl.where(
        valid_request & ((row < start) | (row >= end) | (query_length <= 0)),
        4,
        0,
    )
    error = error | tl.where(
        valid_request
        & (
            (sequence_length <= 0)
            | (sequence_length > MAX_SEQ_LEN)
            | (expected_position < 0)
            | (position != expected_position)
        ),
        8,
        0,
    )
    error = error | tl.where(
        valid_request & (~valid_slot | invalid_slot_map | duplicate_slot), 16, 0
    )
    error = error | tl.where(
        valid_request
        & valid_slot
        & ((prior_anchor < -1) | (prior_anchor != expected_prior_anchor)),
        64,
        0,
    )
    error = error | tl.where((~live) & ((request != -1) | (position != -1)), 32, 0)
    tl.store(state_errors + row, error)


@triton.jit
def _finalize_prefill_anchors_kernel(
    request_ids,
    query_positions,
    query_start_loc,
    raw_state_slot_ids,
    raw_interval_start_positions,
    request_errors,
    raw_state_slot_stride,
    raw_interval_start_stride,
    MAX_BATCH: tl.constexpr,
    MAX_RAW_STATE_SLOTS: tl.constexpr,
):
    request = tl.program_id(0)
    start = tl.load(query_start_loc + request).to(tl.int64)
    end = tl.load(query_start_loc + request + 1).to(tl.int64)
    active = (end > start) & (tl.load(request_errors + request) == 0)
    observed_request = tl.load(
        request_ids + end - 1, mask=active, other=-1
    ).to(tl.int64)
    state_slot = tl.load(
        raw_state_slot_ids + request * raw_state_slot_stride,
        mask=active,
        other=-1,
    ).to(tl.int64)
    valid_slot = (state_slot >= 0) & (state_slot < MAX_RAW_STATE_SLOTS)
    last_position = tl.load(
        query_positions + end - 1,
        mask=active & valid_slot & (observed_request == request),
        other=-1,
    ).to(tl.int64)
    if active & valid_slot & (observed_request == request):
        anchor_offset = (state_slot * raw_interval_start_stride).to(tl.int64)
        tl.store(raw_interval_start_positions + anchor_offset, last_position)


@triton.jit
def _validate_compressed_page_table_kernel(
    request_ids,
    sequence_lengths,
    compressed_block_table,
    state_errors,
    compressed_table_stride,
    num_compressed_pages,
    COMPRESSED_TABLE_WIDTH: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    page_block = tl.program_id(1)
    request = tl.load(request_ids + row).to(tl.int64)
    real_request = (tl.load(state_errors + row) == 0) & (request >= 0)
    sequence_length = tl.load(
        sequence_lengths + request, mask=real_request, other=0
    ).to(tl.int64)
    completed_groups = sequence_length // COMPRESS_RATIO
    required_pages = tl.cdiv(completed_groups, COMPRESSED_PAGE_SIZE)
    pages = page_block * BLOCK_P + tl.arange(0, BLOCK_P)
    active = real_request & (pages < required_pages)
    table_offsets = (
        request * compressed_table_stride + pages.to(tl.int64)
    ).to(tl.int64)
    physical_pages = tl.load(
        compressed_block_table + table_offsets,
        mask=active,
        other=-1,
    ).to(tl.int64)
    invalid = active & (
        (physical_pages < 0) | (physical_pages >= num_compressed_pages)
    )
    if tl.sum(invalid.to(tl.int32), axis=0) != 0:
        tl.atomic_or(state_errors + row, 1024)


@triton.jit
def _pool_completed_groups_kernel(
    normalized_index_key,
    index_gate_logits,
    query_positions,
    request_ids,
    query_start_loc,
    raw_state_slot_ids,
    raw_k_ring,
    raw_gate_ring,
    position_embedding,
    compressed_cache,
    compressed_block_table,
    state_errors,
    raw_state_slot_stride,
    raw_k_slot_stride,
    raw_k_ring_stride,
    raw_gate_slot_stride,
    raw_gate_ring_stride,
    position_embedding_row_stride,
    compressed_page_stride,
    compressed_token_stride,
    compressed_table_stride,
    num_compressed_pages,
    INDEX_HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    MAX_RAW_STATE_SLOTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    status = tl.load(state_errors + row).to(tl.int32)
    request = tl.load(request_ids + row).to(tl.int64)
    position = tl.load(query_positions + row).to(tl.int64)
    real_request = (status == 0) & (request >= 0)
    state_slot = tl.load(
        raw_state_slot_ids + request * raw_state_slot_stride,
        mask=real_request,
        other=-1,
    ).to(tl.int64)
    valid_state_slot = (state_slot >= 0) & (state_slot < MAX_RAW_STATE_SLOTS)
    complete = ((position + 1) % COMPRESS_RATIO) == 0
    if real_request & complete & valid_state_slot:
        request_start = tl.load(query_start_loc + request).to(tl.int64)
        current_first = tl.load(query_positions + request_start).to(tl.int64)
        group_first = position - COMPRESS_RATIO + 1
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < INDEX_HEAD_DIM

        max_logit = tl.full((BLOCK_D,), -float("inf"), tl.float32)
        for offset in tl.static_range(0, COMPRESS_RATIO):
            source_position = group_first + offset
            from_current = source_position >= current_first
            current_row = request_start + source_position - current_first
            current_base = (current_row * INDEX_HEAD_DIM).to(tl.int64)
            current_gate = tl.load(
                index_gate_logits + current_base + dims,
                mask=from_current & dim_mask,
                other=0.0,
            ).to(tl.float32)
            ring_slot = source_position % RING_CAPACITY
            gate_base = (
                state_slot * raw_gate_slot_stride
                + ring_slot * raw_gate_ring_stride
            ).to(tl.int64)
            prior_gate = tl.load(
                raw_gate_ring + gate_base + dims,
                mask=(~from_current) & dim_mask,
                other=0.0,
            ).to(tl.float32)
            ape = tl.load(
                position_embedding + offset * position_embedding_row_stride + dims,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            logit = tl.where(from_current, current_gate, prior_gate) + ape
            max_logit = tl.maximum(max_logit, logit)

        denominator = tl.zeros((BLOCK_D,), tl.float32)
        for offset in tl.static_range(0, COMPRESS_RATIO):
            source_position = group_first + offset
            from_current = source_position >= current_first
            current_row = request_start + source_position - current_first
            current_base = (current_row * INDEX_HEAD_DIM).to(tl.int64)
            current_gate = tl.load(
                index_gate_logits + current_base + dims,
                mask=from_current & dim_mask,
                other=0.0,
            ).to(tl.float32)
            ring_slot = source_position % RING_CAPACITY
            gate_base = (
                state_slot * raw_gate_slot_stride
                + ring_slot * raw_gate_ring_stride
            ).to(tl.int64)
            prior_gate = tl.load(
                raw_gate_ring + gate_base + dims,
                mask=(~from_current) & dim_mask,
                other=0.0,
            ).to(tl.float32)
            ape = tl.load(
                position_embedding + offset * position_embedding_row_stride + dims,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            logit = tl.where(from_current, current_gate, prior_gate) + ape
            denominator += tl.exp(logit - max_logit)

        representative = tl.zeros((BLOCK_D,), tl.float32)
        for offset in tl.static_range(0, COMPRESS_RATIO):
            source_position = group_first + offset
            from_current = source_position >= current_first
            current_row = request_start + source_position - current_first
            current_base = (current_row * INDEX_HEAD_DIM).to(tl.int64)
            current_gate = tl.load(
                index_gate_logits + current_base + dims,
                mask=from_current & dim_mask,
                other=0.0,
            ).to(tl.float32)
            current_key = tl.load(
                normalized_index_key + current_base + dims,
                mask=from_current & dim_mask,
                other=0.0,
            ).to(tl.float32)
            ring_slot = source_position % RING_CAPACITY
            gate_base = (
                state_slot * raw_gate_slot_stride
                + ring_slot * raw_gate_ring_stride
            ).to(tl.int64)
            key_base = (
                state_slot * raw_k_slot_stride + ring_slot * raw_k_ring_stride
            ).to(tl.int64)
            prior_gate = tl.load(
                raw_gate_ring + gate_base + dims,
                mask=(~from_current) & dim_mask,
                other=0.0,
            ).to(tl.float32)
            prior_key = tl.load(
                raw_k_ring + key_base + dims,
                mask=(~from_current) & dim_mask,
                other=0.0,
            ).to(tl.float32)
            ape = tl.load(
                position_embedding + offset * position_embedding_row_stride + dims,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            logit = tl.where(from_current, current_gate, prior_gate) + ape
            weight = tl.exp(logit - max_logit) / denominator
            key = tl.where(from_current, current_key, prior_key)
            representative += weight * key

        group_id = position // COMPRESS_RATIO
        logical_page = group_id // COMPRESSED_PAGE_SIZE
        page_offset = group_id % COMPRESSED_PAGE_SIZE
        table_offset = (
            request * compressed_table_stride + logical_page
        ).to(tl.int64)
        physical_page = tl.load(compressed_block_table + table_offset).to(tl.int64)
        valid_page = (physical_page >= 0) & (
            physical_page < num_compressed_pages
        )
        if valid_page:
            cache_base = (
                physical_page * compressed_page_stride
                + page_offset * compressed_token_stride
            ).to(tl.int64)
            tl.store(
                compressed_cache + cache_base + dims,
                representative,
                mask=dim_mask,
            )
        else:
            tl.atomic_or(state_errors + row, 512)


@triton.jit
def _update_raw_rings_kernel(
    normalized_index_key,
    index_gate_logits,
    query_positions,
    request_ids,
    query_start_loc,
    raw_state_slot_ids,
    raw_k_ring,
    raw_gate_ring,
    raw_logical_positions,
    raw_interval_start_positions,
    state_errors,
    raw_state_slot_stride,
    raw_k_slot_stride,
    raw_k_ring_stride,
    raw_gate_slot_stride,
    raw_gate_ring_stride,
    raw_position_slot_stride,
    raw_interval_start_stride,
    INDEX_HEAD_DIM: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MAX_RAW_STATE_SLOTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    status = tl.load(state_errors + row).to(tl.int32)
    request = tl.load(request_ids + row).to(tl.int64)
    real_request = (status == 0) & (request >= 0)
    state_slot = tl.load(
        raw_state_slot_ids + request * raw_state_slot_stride,
        mask=real_request,
        other=-1,
    ).to(tl.int64)
    valid_slot = (state_slot >= 0) & (state_slot < MAX_RAW_STATE_SLOTS)
    if real_request & valid_slot:
        position = tl.load(query_positions + row).to(tl.int64)
        ring_slot = position % RING_CAPACITY
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < INDEX_HEAD_DIM
        source_base = (row.to(tl.int64) * INDEX_HEAD_DIM).to(tl.int64)
        key_base = (
            state_slot * raw_k_slot_stride + ring_slot * raw_k_ring_stride
        ).to(tl.int64)
        gate_base = (
            state_slot * raw_gate_slot_stride + ring_slot * raw_gate_ring_stride
        ).to(tl.int64)
        key = tl.load(
            normalized_index_key + source_base + dims,
            mask=dim_mask,
            other=0.0,
        )
        gate = tl.load(
            index_gate_logits + source_base + dims,
            mask=dim_mask,
            other=0.0,
        )
        tl.store(raw_k_ring + key_base + dims, key, mask=dim_mask)
        tl.store(raw_gate_ring + gate_base + dims, gate, mask=dim_mask)
        tag_offset = (
            state_slot * raw_position_slot_stride + ring_slot
        ).to(tl.int64)
        tl.store(raw_logical_positions + tag_offset, position)
        request_start = tl.load(query_start_loc + request).to(tl.int64)
        if row == request_start:
            anchor_offset = (
                state_slot * raw_interval_start_stride
            ).to(tl.int64)
            tl.store(raw_interval_start_positions + anchor_offset, position)
    elif real_request:
        tl.atomic_or(state_errors + row, 32)


@triton.jit
def _update_prefill_raw_rings_kernel(
    normalized_index_key,
    index_gate_logits,
    query_positions,
    request_ids,
    query_start_loc,
    raw_state_slot_ids,
    raw_k_ring,
    raw_gate_ring,
    raw_logical_positions,
    state_errors,
    raw_state_slot_stride,
    raw_k_slot_stride,
    raw_k_ring_stride,
    raw_gate_slot_stride,
    raw_gate_ring_stride,
    raw_position_slot_stride,
    INDEX_HEAD_DIM: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MAX_RAW_STATE_SLOTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = tl.program_id(0)
    tail_offset = tl.program_id(1)
    start = tl.load(query_start_loc + request).to(tl.int64)
    end = tl.load(query_start_loc + request + 1).to(tl.int64)
    count = tl.minimum(end - start, RING_CAPACITY)
    active = tail_offset < count
    row = end - count + tail_offset
    observed_request = tl.load(
        request_ids + row, mask=active, other=-1
    ).to(tl.int64)
    status = tl.load(state_errors + row, mask=active, other=1).to(tl.int32)
    state_slot = tl.load(
        raw_state_slot_ids + request * raw_state_slot_stride,
        mask=active,
        other=-1,
    ).to(tl.int64)
    valid_slot = (state_slot >= 0) & (state_slot < MAX_RAW_STATE_SLOTS)
    active = active & (observed_request == request) & (status == 0) & valid_slot
    if active:
        position = tl.load(query_positions + row).to(tl.int64)
        ring_slot = position % RING_CAPACITY
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < INDEX_HEAD_DIM
        source_base = (row * INDEX_HEAD_DIM).to(tl.int64)
        key_base = (
            state_slot * raw_k_slot_stride + ring_slot * raw_k_ring_stride
        ).to(tl.int64)
        gate_base = (
            state_slot * raw_gate_slot_stride + ring_slot * raw_gate_ring_stride
        ).to(tl.int64)
        key = tl.load(
            normalized_index_key + source_base + dims,
            mask=dim_mask,
            other=0.0,
        )
        gate = tl.load(
            index_gate_logits + source_base + dims,
            mask=dim_mask,
            other=0.0,
        )
        tl.store(raw_k_ring + key_base + dims, key, mask=dim_mask)
        tl.store(raw_gate_ring + gate_base + dims, gate, mask=dim_mask)
        tag_offset = (
            state_slot * raw_position_slot_stride + ring_slot
        ).to(tl.int64)
        tl.store(raw_logical_positions + tag_offset, position)


@triton.jit
def _reset_selector_state_kernel(
    reset_mask,
    prefix_lengths,
    raw_state_slot_ids,
    raw_logical_positions,
    raw_interval_start_positions,
    raw_state_slot_stride,
    raw_position_slot_stride,
    raw_interval_start_stride,
    MAX_BATCH: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    MAX_RAW_STATE_SLOTS: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_RING: tl.constexpr,
):
    request = tl.program_id(0)
    reset = tl.load(reset_mask + request).to(tl.int1)
    prefix_length = tl.load(prefix_lengths + request).to(tl.int64)
    state_slot = tl.load(
        raw_state_slot_ids + request * raw_state_slot_stride,
        mask=reset,
        other=-1,
    ).to(tl.int64)
    valid_slot = (state_slot >= 0) & (state_slot < MAX_RAW_STATE_SLOTS)
    valid_prefix = (
        (prefix_length >= 0)
        & (prefix_length <= MAX_SEQ_LEN)
        & ((prefix_length % 4) == 0)
    )
    owners = tl.arange(0, BLOCK_BATCH)
    owner_mask = owners < MAX_BATCH
    owner_slots = tl.load(
        raw_state_slot_ids + owners * raw_state_slot_stride,
        mask=owner_mask,
        other=-1,
    ).to(tl.int64)
    duplicate = (
        tl.sum(
            owner_mask
            & (owners != request)
            & (owner_slots == state_slot)
            & (owner_slots >= 0),
            axis=0,
        )
        != 0
    )
    if reset & valid_slot:
        slots = tl.arange(0, BLOCK_RING)
        tag_offsets = (
            state_slot * raw_position_slot_stride + slots
        ).to(tl.int64)
        tl.store(
            raw_logical_positions + tag_offsets,
            -1,
            mask=slots < RING_CAPACITY,
        )
        anchor = tl.where(valid_prefix & ~duplicate, prefix_length - 1, -2)
        anchor_offset = (
            state_slot * raw_interval_start_stride
        ).to(tl.int64)
        tl.store(raw_interval_start_positions + anchor_offset, anchor)


@triton.jit
def _score_representatives_kernel(
    index_query,
    index_head_weights,
    query_positions,
    request_ids,
    sequence_lengths,
    compressed_cache,
    compressed_block_table,
    state_errors,
    scores,
    eligible_counts,
    merge_lengths,
    compressed_page_stride,
    compressed_token_stride,
    compressed_table_stride,
    score_row_stride,
    num_compressed_pages,
    MAX_GROUPS: tl.constexpr,
    GROUP_OFFSET: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    GROUP_BUDGET: tl.constexpr,
    INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    group_block = tl.program_id(1)
    status = tl.load(state_errors + row).to(tl.int32)
    request = tl.load(request_ids + row).to(tl.int64)
    position = tl.load(query_positions + row).to(tl.int64)
    real_request = (status == 0) & (request >= 0)
    sequence_length = tl.load(
        sequence_lengths + request, mask=real_request, other=0
    ).to(tl.int64)
    eligible = tl.minimum(
        (position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    eligible = tl.minimum(eligible, MAX_GROUPS)
    eligible = tl.where(real_request, eligible, 0)
    prior_eligible = tl.minimum(eligible, GROUP_OFFSET)
    carry_count = tl.minimum(prior_eligible, GROUP_BUDGET)
    chunk_eligible = tl.minimum(
        tl.maximum(eligible - GROUP_OFFSET, 0), GROUP_COUNT
    )
    if group_block == 0:
        tl.store(eligible_counts + row, eligible)
        tl.store(merge_lengths + row, carry_count + chunk_eligible)

    local_groups = group_block * BLOCK_G + tl.arange(0, BLOCK_G)
    groups = GROUP_OFFSET + local_groups
    group_mask = local_groups < GROUP_COUNT
    active = group_mask & (groups < eligible) & real_request
    logical_pages = groups // COMPRESSED_PAGE_SIZE
    page_offsets = groups % COMPRESSED_PAGE_SIZE
    table_offsets = (
        request * compressed_table_stride + logical_pages.to(tl.int64)
    ).to(tl.int64)
    physical_pages = tl.load(
        compressed_block_table + table_offsets,
        mask=active,
        other=-1,
    ).to(tl.int64)
    valid_pages = (physical_pages >= 0) & (
        physical_pages < num_compressed_pages
    )
    if tl.sum((active & ~valid_pages).to(tl.int32), axis=0) != 0:
        tl.atomic_or(state_errors + row, 512)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < INDEX_HEAD_DIM
    cache_offsets = (
        physical_pages[:, None] * compressed_page_stride
        + page_offsets[:, None].to(tl.int64) * compressed_token_stride
        + dims[None, :]
    ).to(tl.int64)
    keys = tl.load(
        compressed_cache + cache_offsets,
        mask=active[:, None] & valid_pages[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    score = tl.zeros((BLOCK_G,), tl.float32)
    row64 = row.to(tl.int64)
    for head in tl.static_range(0, INDEX_HEADS):
        query_offsets = (
            (row64 * INDEX_HEADS + head) * INDEX_HEAD_DIM + dims
        ).to(tl.int64)
        query = tl.load(
            index_query + query_offsets, mask=dim_mask, other=0.0
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1) / math.sqrt(INDEX_HEAD_DIM)
        head_weight = tl.load(
            index_head_weights + row64 * INDEX_HEADS + head
        ).to(tl.float32)
        score += tl.maximum(dot, 0.0) * head_weight
    score *= 1.0 / math.sqrt(INDEX_HEADS)
    score = tl.where(active & valid_pages, score, -float("inf"))
    output_columns = carry_count + local_groups
    score_offsets = (
        row64 * score_row_stride + output_columns.to(tl.int64)
    ).to(tl.int64)
    tl.store(scores + score_offsets, score, mask=group_mask)


@triton.jit
def _sort_signed_topk_kernel(
    topk_values,
    topk_ids,
    merge_lengths,
    GROUP_BUDGET: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_K)
    count = tl.minimum(tl.load(merge_lengths + row), GROUP_BUDGET)
    active = columns < count
    values = tl.load(
        topk_values + row * GROUP_BUDGET + columns,
        mask=active,
        other=-float("inf"),
    ).to(tl.float32)
    ids = tl.load(
        topk_ids + row * GROUP_BUDGET + columns,
        mask=active,
        other=-1,
    ).to(tl.int32)
    raw_bits = values.to(tl.uint32, bitcast=True)
    ordered_bits = tl.where(
        (raw_bits & 0x80000000) != 0,
        ~raw_bits,
        raw_bits ^ 0x80000000,
    ).to(tl.uint64)
    id_key = (0xFFFFFFFF - ids.to(tl.uint32)).to(tl.uint64)
    keys = tl.where(active & (ids >= 0), (ordered_bits << 32) | id_key, 0)
    keys = tl.sort(keys, dim=0, descending=True)
    sorted_ids = (0xFFFFFFFF - (keys & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    sorted_ordered = (keys >> 32).to(tl.uint32)
    sorted_raw = tl.where(
        (sorted_ordered & 0x80000000) != 0,
        sorted_ordered ^ 0x80000000,
        ~sorted_ordered,
    )
    sorted_values = sorted_raw.to(tl.float32, bitcast=True)
    valid = (columns < count) & (keys != 0)
    tl.store(
        topk_values + row * GROUP_BUDGET + columns,
        tl.where(valid, sorted_values, -float("inf")),
        mask=columns < GROUP_BUDGET,
    )
    tl.store(
        topk_ids + row * GROUP_BUDGET + columns,
        tl.where(valid, sorted_ids, -1),
        mask=columns < GROUP_BUDGET,
    )


def launch_validate_decode_rows(
    *,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    request_errors: torch.Tensor,
    state_errors: torch.Tensor,
    caps,
) -> None:
    logical_positions_2d = query_positions[:, None]
    _launch_qsa_validate_decode_rows(
        request_ids=request_ids,
        query_positions=query_positions,
        rope_positions=logical_positions_2d,
        sequence_lengths=sequence_lengths,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_interval_start_positions=raw_interval_start_positions,
        request_errors=request_errors,
        state_errors=state_errors,
        rope_position_rows=int(caps.max_seq_len),
        caps=caps,
    )


def launch_validate_prefill_rows(
    *,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    request_errors: torch.Tensor,
    state_errors: torch.Tensor,
    caps,
) -> None:
    rows = int(request_ids.shape[0])
    block_batch = triton.next_power_of_2(int(caps.max_batch))
    block = 256
    _clear_i32_kernel[(triton.cdiv(int(caps.max_batch) + 1, block),)](
        request_errors,
        int(caps.max_batch) + 1,
        BLOCK=block,
        num_warps=4,
    )
    _validate_packed_boundaries_kernel[(1,)](
        query_start_loc,
        request_errors,
        ROWS=rows,
        MAX_BATCH=int(caps.max_batch),
        BLOCK_BATCH=block_batch,
        num_warps=1,
    )
    _validate_prefill_rows_kernel[(rows,)](
        request_ids,
        query_positions,
        sequence_lengths,
        query_start_loc,
        raw_state_slot_ids,
        raw_interval_start_positions,
        request_errors,
        state_errors,
        int(raw_state_slot_ids.stride(0)),
        int(raw_interval_start_positions.stride(0)),
        ROWS=rows,
        MAX_BATCH=int(caps.max_batch),
        MAX_SEQ_LEN=int(caps.max_seq_len),
        MAX_RAW_STATE_SLOTS=int(caps.max_raw_state_slots),
        BLOCK_BATCH=block_batch,
        num_warps=1,
    )


def launch_validate_compressed_page_table(
    *,
    request_ids: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compressed_block_table: torch.Tensor,
    compressed_cache: torch.Tensor,
    state_errors: torch.Tensor,
    caps,
) -> None:
    rows = int(request_ids.shape[0])
    block_p = 32
    _validate_compressed_page_table_kernel[
        (rows, triton.cdiv(int(compressed_block_table.shape[1]), block_p))
    ](
        request_ids,
        sequence_lengths,
        compressed_block_table,
        state_errors,
        int(compressed_block_table.stride(0)),
        int(compressed_cache.shape[0]),
        COMPRESSED_TABLE_WIDTH=int(compressed_block_table.shape[1]),
        COMPRESSED_PAGE_SIZE=int(caps.compressed_page_size),
        COMPRESS_RATIO=int(caps.compress_ratio),
        BLOCK_P=block_p,
        num_warps=1,
    )


def launch_validate_completed_groups(
    *,
    query_positions: torch.Tensor,
    request_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    state_errors: torch.Tensor,
    caps,
) -> None:
    _launch_qsa_validate_completed_groups(
        query_positions=query_positions,
        rope_positions=query_positions[:, None],
        request_ids=request_ids,
        query_start_loc=query_start_loc,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_logical_positions=raw_logical_positions,
        raw_rope_positions=raw_logical_positions[:, :, None],
        state_errors=state_errors,
        rope_position_rows=int(caps.max_seq_len),
        caps=caps,
    )


def launch_pool_completed_groups(
    *,
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    query_positions: torch.Tensor,
    request_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    position_embedding: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    state_errors: torch.Tensor,
    caps,
) -> None:
    rows = int(normalized_index_key.shape[0])
    _pool_completed_groups_kernel[(rows,)](
        normalized_index_key,
        index_gate_logits,
        query_positions,
        request_ids,
        query_start_loc,
        raw_state_slot_ids,
        raw_k_ring,
        raw_gate_ring,
        position_embedding,
        compressed_cache,
        compressed_block_table,
        state_errors,
        int(raw_state_slot_ids.stride(0)),
        int(raw_k_ring.stride(0)),
        int(raw_k_ring.stride(1)),
        int(raw_gate_ring.stride(0)),
        int(raw_gate_ring.stride(1)),
        int(position_embedding.stride(0)),
        int(compressed_cache.stride(0)),
        int(compressed_cache.stride(1)),
        int(compressed_block_table.stride(0)),
        int(compressed_cache.shape[0]),
        INDEX_HEAD_DIM=int(caps.index_head_dim),
        COMPRESS_RATIO=int(caps.compress_ratio),
        RING_CAPACITY=int(caps.raw_ring_capacity),
        COMPRESSED_PAGE_SIZE=int(caps.compressed_page_size),
        MAX_RAW_STATE_SLOTS=int(caps.max_raw_state_slots),
        BLOCK_D=triton.next_power_of_2(int(caps.index_head_dim)),
        num_warps=4,
    )


def launch_update_raw_rings(
    *,
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    query_positions: torch.Tensor,
    request_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    state_errors: torch.Tensor,
    caps,
) -> None:
    rows = int(normalized_index_key.shape[0])
    _update_raw_rings_kernel[(rows,)](
        normalized_index_key,
        index_gate_logits,
        query_positions,
        request_ids,
        query_start_loc,
        raw_state_slot_ids,
        raw_k_ring,
        raw_gate_ring,
        raw_logical_positions,
        raw_interval_start_positions,
        state_errors,
        int(raw_state_slot_ids.stride(0)),
        int(raw_k_ring.stride(0)),
        int(raw_k_ring.stride(1)),
        int(raw_gate_ring.stride(0)),
        int(raw_gate_ring.stride(1)),
        int(raw_logical_positions.stride(0)),
        int(raw_interval_start_positions.stride(0)),
        INDEX_HEAD_DIM=int(caps.index_head_dim),
        RING_CAPACITY=int(caps.raw_ring_capacity),
        MAX_RAW_STATE_SLOTS=int(caps.max_raw_state_slots),
        BLOCK_D=triton.next_power_of_2(int(caps.index_head_dim)),
        num_warps=4,
    )


def launch_update_prefill_raw_rings(
    *,
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    query_positions: torch.Tensor,
    request_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    state_errors: torch.Tensor,
    caps,
) -> None:
    _update_prefill_raw_rings_kernel[
        (int(caps.max_batch), int(caps.raw_ring_capacity))
    ](
        normalized_index_key,
        index_gate_logits,
        query_positions,
        request_ids,
        query_start_loc,
        raw_state_slot_ids,
        raw_k_ring,
        raw_gate_ring,
        raw_logical_positions,
        state_errors,
        int(raw_state_slot_ids.stride(0)),
        int(raw_k_ring.stride(0)),
        int(raw_k_ring.stride(1)),
        int(raw_gate_ring.stride(0)),
        int(raw_gate_ring.stride(1)),
        int(raw_logical_positions.stride(0)),
        INDEX_HEAD_DIM=int(caps.index_head_dim),
        RING_CAPACITY=int(caps.raw_ring_capacity),
        MAX_RAW_STATE_SLOTS=int(caps.max_raw_state_slots),
        BLOCK_D=triton.next_power_of_2(int(caps.index_head_dim)),
        num_warps=4,
    )


def launch_reset_selector_state(
    *,
    reset_mask: torch.Tensor,
    prefix_lengths: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    caps,
) -> None:
    block_batch = triton.next_power_of_2(int(caps.max_batch))
    _reset_selector_state_kernel[(int(caps.max_batch),)](
        reset_mask,
        prefix_lengths,
        raw_state_slot_ids,
        raw_logical_positions,
        raw_interval_start_positions,
        int(raw_state_slot_ids.stride(0)),
        int(raw_logical_positions.stride(0)),
        int(raw_interval_start_positions.stride(0)),
        MAX_BATCH=int(caps.max_batch),
        MAX_SEQ_LEN=int(caps.max_seq_len),
        MAX_RAW_STATE_SLOTS=int(caps.max_raw_state_slots),
        RING_CAPACITY=int(caps.raw_ring_capacity),
        BLOCK_BATCH=block_batch,
        BLOCK_RING=triton.next_power_of_2(int(caps.raw_ring_capacity)),
        num_warps=1,
    )


def launch_finalize_prefill_anchors(
    *,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    request_errors: torch.Tensor,
    caps,
) -> None:
    _finalize_prefill_anchors_kernel[(int(caps.max_batch),)](
        request_ids,
        query_positions,
        query_start_loc,
        raw_state_slot_ids,
        raw_interval_start_positions,
        request_errors,
        int(raw_state_slot_ids.stride(0)),
        int(raw_interval_start_positions.stride(0)),
        MAX_BATCH=int(caps.max_batch),
        MAX_RAW_STATE_SLOTS=int(caps.max_raw_state_slots),
        num_warps=1,
    )


def launch_score_representatives(
    *,
    index_query: torch.Tensor,
    index_head_weights: torch.Tensor,
    query_positions: torch.Tensor,
    request_ids: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    state_errors: torch.Tensor,
    scores: torch.Tensor,
    eligible_counts: torch.Tensor,
    merge_lengths: torch.Tensor,
    group_offset: int,
    group_count: int,
    caps,
) -> None:
    rows = int(index_query.shape[0])
    block_g = 32
    _score_representatives_kernel[
        (rows, triton.cdiv(int(group_count), block_g))
    ](
        index_query,
        index_head_weights,
        query_positions,
        request_ids,
        sequence_lengths,
        compressed_cache,
        compressed_block_table,
        state_errors,
        scores,
        eligible_counts,
        merge_lengths,
        int(compressed_cache.stride(0)),
        int(compressed_cache.stride(1)),
        int(compressed_block_table.stride(0)),
        int(scores.stride(0)),
        int(compressed_cache.shape[0]),
        MAX_GROUPS=int(caps.max_groups),
        GROUP_OFFSET=int(group_offset),
        GROUP_COUNT=int(group_count),
        GROUP_BUDGET=int(caps.group_budget),
        INDEX_HEADS=int(caps.index_heads),
        INDEX_HEAD_DIM=int(caps.index_head_dim),
        COMPRESS_RATIO=int(caps.compress_ratio),
        COMPRESSED_PAGE_SIZE=int(caps.compressed_page_size),
        BLOCK_G=block_g,
        BLOCK_D=triton.next_power_of_2(int(caps.index_head_dim)),
        num_warps=4,
    )


def launch_sort_signed_topk(
    *,
    topk_values: torch.Tensor,
    topk_group_ids: torch.Tensor,
    merge_lengths: torch.Tensor,
    group_budget: int,
) -> None:
    rows = int(topk_values.shape[0])
    _sort_signed_topk_kernel[(rows,)](
        topk_values,
        topk_group_ids,
        merge_lengths,
        GROUP_BUDGET=int(group_budget),
        BLOCK_K=triton.next_power_of_2(int(group_budget)),
        num_warps=8,
    )


__all__ = [
    "launch_expand_selected_groups",
    "launch_finalize_prefill_anchors",
    "launch_pool_completed_groups",
    "launch_propagate_request_errors",
    "launch_remap_topk_group_ids",
    "launch_score_representatives",
    "launch_sort_signed_topk",
    "launch_reset_selector_state",
    "launch_stabilize_topk",
    "launch_stage_topk_carry",
    "launch_topk_groups",
    "launch_update_raw_rings",
    "launch_update_prefill_raw_rings",
    "launch_validate_completed_groups",
    "launch_validate_compressed_page_table",
    "launch_validate_decode_rows",
    "launch_validate_prefill_rows",
]
