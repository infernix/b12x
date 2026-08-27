"""Capacity, binding, and fail-closed execution contract for GLM pooling."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from ..qsa._contract import (
    _canonical_device,
    _check_tensor,
    _require_mutation_alias_contract,
    _scratch_view,
)

_ALIGN_BYTES = 256
_SCORE_WORKSPACE_LIMIT_BYTES = 128 * 1024 * 1024
_MAX_SCORE_CHUNK_GROUPS = 65536
_MIN_TOPK_WORKSPACE_BYTES = 1024 * 1024
_STABLE_TOPK_BLOCK = 512


def _align_up(value: int, alignment: int = _ALIGN_BYTES) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


@dataclass(frozen=True, kw_only=True)
class CacheRequirements:
    """Pure cache and persistent-state geometry for one selector layer."""

    dtype: torch.dtype
    compressed_page_shape: tuple[int, int]
    raw_k_ring_shape: tuple[int, int]
    raw_gate_ring_shape: tuple[int, int]
    raw_logical_positions_shape: tuple[int]
    raw_interval_start_positions_shape: tuple[int]
    compressed_page_nbytes: int
    raw_k_ring_nbytes: int
    raw_gate_ring_nbytes: int
    raw_logical_positions_nbytes: int
    raw_interval_start_positions_nbytes: int
    raw_state_slot_nbytes: int
    raw_ring_capacity: int
    selection_width: int
    alignment_bytes: int


def cache_requirements(
    *,
    compressed_page_size: int,
    max_speculative_tokens: int = 0,
    index_head_dim: int = 128,
    compress_ratio: int = 4,
    budget: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
) -> CacheRequirements:
    """Describe selector storage without requiring a device or pool capacity."""
    if int(compressed_page_size) <= 0:
        raise ValueError("compressed_page_size must be positive")
    if int(index_head_dim) != 128:
        raise ValueError("GLM pooled selection requires index_head_dim=128")
    if int(compress_ratio) != 4:
        raise ValueError("GLM pooled selection requires compress_ratio=4")
    if int(budget) != 2048:
        raise ValueError("GLM pooled selection requires budget=2048")
    if int(max_speculative_tokens) < 0:
        raise ValueError("max_speculative_tokens must be nonnegative")
    if dtype != torch.bfloat16:
        raise TypeError("GLM pooled selector state must use torch.bfloat16")

    raw_ring_capacity = int(compress_ratio) * math.ceil(
        (int(compress_ratio) + int(max_speculative_tokens)) / int(compress_ratio)
    )
    key_elements = raw_ring_capacity * int(index_head_dim)
    payload_nbytes = key_elements * dtype.itemsize
    logical_nbytes = raw_ring_capacity * torch.int64.itemsize
    anchor_nbytes = torch.int64.itemsize
    return CacheRequirements(
        dtype=dtype,
        compressed_page_shape=(int(compressed_page_size), int(index_head_dim)),
        raw_k_ring_shape=(raw_ring_capacity, int(index_head_dim)),
        raw_gate_ring_shape=(raw_ring_capacity, int(index_head_dim)),
        raw_logical_positions_shape=(raw_ring_capacity,),
        raw_interval_start_positions_shape=(1,),
        compressed_page_nbytes=(
            int(compressed_page_size) * int(index_head_dim) * dtype.itemsize
        ),
        raw_k_ring_nbytes=payload_nbytes,
        raw_gate_ring_nbytes=payload_nbytes,
        raw_logical_positions_nbytes=logical_nbytes,
        raw_interval_start_positions_nbytes=anchor_nbytes,
        raw_state_slot_nbytes=(2 * payload_nbytes + logical_nbytes + anchor_nbytes),
        raw_ring_capacity=raw_ring_capacity,
        selection_width=int(budget) + int(compress_ratio) - 1,
        alignment_bytes=_ALIGN_BYTES,
    )


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Static GLM selector geometry and serving capacities."""

    device: torch.device | str
    max_batch: int
    max_raw_state_slots: int
    max_q_rows: int
    max_seq_len: int
    num_compressed_cache_pages: int
    compressed_page_size: int
    max_speculative_tokens: int = 0
    index_heads: int = 32
    index_head_dim: int = 128
    compress_ratio: int = 4
    budget: int = 2048
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        if self.device.type != "cuda":
            raise ValueError(
                f"GLM pooled decode requires a CUDA device, got {self.device}"
            )
        capability = torch.cuda.get_device_capability(self.device)
        if capability not in ((12, 0), (12, 1)):
            raise ValueError(
                "GLM pooled decode in b12x requires SM120 or SM121, got "
                f"compute capability {capability[0]}.{capability[1]}"
            )
        for name in (
            "max_batch",
            "max_raw_state_slots",
            "max_q_rows",
            "max_seq_len",
            "num_compressed_cache_pages",
            "compressed_page_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.max_raw_state_slots) < int(self.max_batch):
            raise ValueError("max_raw_state_slots must be at least max_batch")
        if int(self.max_q_rows) < int(self.max_batch):
            raise ValueError("max_q_rows must be at least max_batch")
        if int(self.max_seq_len) < 4:
            raise ValueError("max_seq_len must cover at least one four-token group")
        if int(self.max_seq_len) > torch.iinfo(torch.int32).max:
            raise ValueError("max_seq_len must fit in positive int32 positions")
        max_page_count = torch.iinfo(torch.int32).max + 1
        if int(self.num_compressed_cache_pages) > max_page_count:
            raise ValueError(
                "num_compressed_cache_pages must fit in nonnegative int32 IDs"
            )
        if int(self.max_speculative_tokens) < 0:
            raise ValueError("max_speculative_tokens must be nonnegative")
        if int(self.index_heads) != 32:
            raise ValueError("GLM pooled selection requires index_heads=32")
        if int(self.index_head_dim) != 128:
            raise ValueError("GLM pooled selection requires index_head_dim=128")
        if int(self.compress_ratio) != 4:
            raise ValueError("GLM pooled selection requires compress_ratio=4")
        if int(self.budget) != 2048:
            raise ValueError("GLM pooled selection requires budget=2048")
        if self.dtype != torch.bfloat16:
            raise TypeError("GLM pooled selector tensors must use torch.bfloat16")
        if (
            int(self.num_compressed_cache_pages) * int(self.compressed_page_size)
            < self.max_groups
        ):
            raise ValueError("compressed cache capacity cannot cover max_seq_len")

    @property
    def position_axes(self) -> int:
        """Compatibility property for shared packed-interval validation."""
        return 1

    @property
    def group_budget(self) -> int:
        return 512

    @property
    def selection_width(self) -> int:
        return 2051

    @property
    def max_groups(self) -> int:
        return int(self.max_seq_len) // 4

    @property
    def compressed_table_width(self) -> int:
        return math.ceil(self.max_groups / int(self.compressed_page_size))

    @property
    def cache_requirements(self) -> CacheRequirements:
        return cache_requirements(
            compressed_page_size=int(self.compressed_page_size),
            max_speculative_tokens=int(self.max_speculative_tokens),
            index_head_dim=int(self.index_head_dim),
            compress_ratio=int(self.compress_ratio),
            budget=int(self.budget),
            dtype=self.dtype,
        )

    @property
    def raw_ring_capacity(self) -> int:
        return self.cache_requirements.raw_ring_capacity


@dataclass(frozen=True)
class _ScratchLayout:
    score_offset_bytes: int
    eligible_counts_offset_bytes: int
    merge_lengths_offset_bytes: int
    topk_values_offset_bytes: int
    topk_indices_offset_bytes: int
    topk_values_b_offset_bytes: int
    topk_indices_b_offset_bytes: int
    state_errors_offset_bytes: int
    request_errors_offset_bytes: int
    topk_offset_bytes: int
    topk_nbytes: int
    total_nbytes: int


@dataclass(frozen=True)
class Plan:
    """Fixed-capacity selector policy and caller-allocated scratch contract."""

    caps: Caps
    score_chunk_groups: int
    score_workspace_width: int
    num_score_chunks: int
    _layout: _ScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(self, **kwargs: object) -> Binding:
        return bind(self, **kwargs)


@dataclass(frozen=True)
class Binding:
    """Caller-owned selector cache, persistent state, output, and scratch."""

    plan: Plan
    scratch: torch.Tensor
    compressed_k_cache: torch.Tensor
    compressed_block_table: torch.Tensor
    raw_k_ring: torch.Tensor
    raw_gate_ring: torch.Tensor
    raw_logical_positions: torch.Tensor
    raw_interval_start_positions: torch.Tensor
    raw_state_slot_ids: torch.Tensor
    position_embedding: torch.Tensor
    selected_positions: torch.Tensor
    scores: torch.Tensor
    eligible_group_counts: torch.Tensor
    merge_lengths: torch.Tensor
    topk_values: torch.Tensor
    topk_group_ids: torch.Tensor
    topk_values_b: torch.Tensor
    topk_group_ids_b: torch.Tensor
    state_errors: torch.Tensor
    request_errors: torch.Tensor


@dataclass(frozen=True)
class _KernelCaps:
    max_batch: int
    max_raw_state_slots: int
    max_seq_len: int
    compressed_page_size: int
    max_speculative_tokens: int
    index_heads: int
    index_head_dim: int
    compress_ratio: int
    budget: int
    raw_ring_capacity: int

    @property
    def position_axes(self) -> int:
        return 1

    @property
    def group_budget(self) -> int:
        return self.budget // self.compress_ratio

    @property
    def selection_width(self) -> int:
        return self.budget + self.compress_ratio - 1

    @property
    def max_groups(self) -> int:
        return self.max_seq_len // self.compress_ratio


def _scratch_layout(caps: Caps) -> tuple[_ScratchLayout, int, int, int]:
    score_width_limit = max(
        1,
        _SCORE_WORKSPACE_LIMIT_BYTES
        // (int(caps.max_q_rows) * torch.float32.itemsize),
    )
    score_width_limit = min(
        score_width_limit, int(caps.group_budget) + _MAX_SCORE_CHUNK_GROUPS
    )
    if int(caps.max_groups) <= score_width_limit:
        score_chunk_groups = int(caps.max_groups)
        score_workspace_width = score_chunk_groups
    else:
        score_chunk_groups = score_width_limit - int(caps.group_budget)
        if score_chunk_groups <= 0:
            raise ValueError(
                "GLM selector score workspace cannot hold one top-k carry row"
            )
        score_workspace_width = int(caps.group_budget) + score_chunk_groups
    num_score_chunks = math.ceil(int(caps.max_groups) / score_chunk_groups)

    rows = int(caps.max_q_rows)
    group_budget = int(caps.group_budget)
    score_nbytes = rows * score_workspace_width * torch.float32.itemsize
    counts_nbytes = rows * torch.int32.itemsize
    topk_values_nbytes = rows * group_budget * torch.float32.itemsize
    topk_indices_nbytes = rows * group_budget * torch.int32.itemsize
    state_errors_nbytes = rows * torch.int32.itemsize
    request_errors_nbytes = (int(caps.max_batch) + 1) * torch.int32.itemsize
    stable_topk_blocks = math.ceil(score_workspace_width / _STABLE_TOPK_BLOCK)
    stable_topk_nbytes = (
        2 * rows * stable_topk_blocks * torch.int32.itemsize
        + rows
        * group_budget
        * (torch.float32.itemsize + torch.int32.itemsize)
        + rows * (torch.float32.itemsize + torch.int32.itemsize)
    )
    topk_workspace_nbytes = _align_up(
        max(_MIN_TOPK_WORKSPACE_BYTES, stable_topk_nbytes)
    )

    offset = 0
    score_offset = _align_up(offset)
    offset = score_offset + score_nbytes
    eligible_offset = _align_up(offset)
    offset = eligible_offset + counts_nbytes
    merge_offset = _align_up(offset)
    offset = merge_offset + counts_nbytes
    topk_values_offset = _align_up(offset)
    offset = topk_values_offset + topk_values_nbytes
    topk_indices_offset = _align_up(offset)
    offset = topk_indices_offset + topk_indices_nbytes
    topk_values_b_offset = _align_up(offset)
    offset = topk_values_b_offset + topk_values_nbytes
    topk_indices_b_offset = _align_up(offset)
    offset = topk_indices_b_offset + topk_indices_nbytes
    state_errors_offset = _align_up(offset)
    offset = state_errors_offset + state_errors_nbytes
    request_errors_offset = _align_up(offset)
    offset = request_errors_offset + request_errors_nbytes
    topk_offset = _align_up(offset)
    offset = topk_offset + topk_workspace_nbytes
    return (
        _ScratchLayout(
            score_offset_bytes=score_offset,
            eligible_counts_offset_bytes=eligible_offset,
            merge_lengths_offset_bytes=merge_offset,
            topk_values_offset_bytes=topk_values_offset,
            topk_indices_offset_bytes=topk_indices_offset,
            topk_values_b_offset_bytes=topk_values_b_offset,
            topk_indices_b_offset_bytes=topk_indices_b_offset,
            state_errors_offset_bytes=state_errors_offset,
            request_errors_offset_bytes=request_errors_offset,
            topk_offset_bytes=topk_offset,
            topk_nbytes=topk_workspace_nbytes,
            total_nbytes=_align_up(offset),
        ),
        score_chunk_groups,
        score_workspace_width,
        num_score_chunks,
    )


def plan(caps: Caps) -> Plan:
    """Plan chunked scoring and one caller-owned scratch allocation."""
    layout, score_chunk_groups, score_workspace_width, num_score_chunks = (
        _scratch_layout(caps)
    )
    return Plan(
        caps=caps,
        score_chunk_groups=score_chunk_groups,
        score_workspace_width=score_workspace_width,
        num_score_chunks=num_score_chunks,
        _layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "glm_pooled_indexer.scratch",
                nbytes=layout.total_nbytes,
                device=caps.device,
            ),
        ),
    )


def bind(
    plan: Plan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    position_embedding: torch.Tensor,
    selected_positions: torch.Tensor,
) -> Binding:
    """Bind caller-owned storage without allocating or initializing state."""
    if not isinstance(plan, Plan):
        raise TypeError("plan must be a glm_pooled_indexer.Plan")
    caps = plan.caps
    scratch_storage = scratch_tensor(
        scratch, plan.scratch_specs(), owner="glm_pooled_indexer"
    )
    if compressed_k_cache.ndim != 3 or tuple(compressed_k_cache.shape[1:]) != (
        int(caps.compressed_page_size),
        int(caps.index_head_dim),
    ):
        raise ValueError(
            "compressed_k_cache must have shape "
            f"[pages, {caps.compressed_page_size}, {caps.index_head_dim}]"
        )
    if not 0 < int(compressed_k_cache.shape[0]) <= int(
        caps.num_compressed_cache_pages
    ):
        raise ValueError("compressed_k_cache page count exceeds planned capacity")
    _check_tensor(
        compressed_k_cache,
        name="compressed_k_cache",
        device=caps.device,
        dtype=caps.dtype,
        unit_inner_stride=True,
    )
    expected_token_stride = int(caps.index_head_dim)
    minimum_page_stride = int(caps.compressed_page_size) * expected_token_stride
    if int(compressed_k_cache.stride(1)) != expected_token_stride:
        raise ValueError(
            "compressed_k_cache tokens must have dense index-head payloads"
        )
    if int(compressed_k_cache.stride(0)) < minimum_page_stride:
        raise ValueError(
            "compressed_k_cache physical-page stride cannot overlap page payloads"
        )
    if (
        compressed_block_table.ndim != 2
        or int(compressed_block_table.shape[0]) != int(caps.max_batch)
        or int(compressed_block_table.shape[1])
        < int(caps.compressed_table_width)
    ):
        raise ValueError(
            "compressed_block_table must have shape [max_batch, width] with "
            f"width >= {caps.compressed_table_width}"
        )
    _check_tensor(
        compressed_block_table,
        name="compressed_block_table",
        device=caps.device,
        dtype=torch.int32,
        contiguous=True,
    )
    raw_shape = (
        int(caps.max_raw_state_slots),
        int(caps.raw_ring_capacity),
        int(caps.index_head_dim),
    )
    for tensor, name in (
        (raw_k_ring, "raw_k_ring"),
        (raw_gate_ring, "raw_gate_ring"),
    ):
        _check_tensor(
            tensor,
            name=name,
            device=caps.device,
            shape=raw_shape,
            dtype=caps.dtype,
            contiguous=True,
        )
    _check_tensor(
        raw_logical_positions,
        name="raw_logical_positions",
        device=caps.device,
        shape=(int(caps.max_raw_state_slots), int(caps.raw_ring_capacity)),
        dtype=torch.int64,
        contiguous=True,
    )
    _check_tensor(
        raw_interval_start_positions,
        name="raw_interval_start_positions",
        device=caps.device,
        shape=(int(caps.max_raw_state_slots),),
        dtype=torch.int64,
        contiguous=True,
    )
    _check_tensor(
        raw_state_slot_ids,
        name="raw_state_slot_ids",
        device=caps.device,
        shape=(int(caps.max_batch),),
        dtype=(torch.int32, torch.int64),
        contiguous=True,
    )
    _check_tensor(
        position_embedding,
        name="position_embedding",
        device=caps.device,
        shape=(4, int(caps.index_head_dim)),
        dtype=(torch.bfloat16, torch.float32),
        contiguous=True,
    )
    _check_tensor(
        selected_positions,
        name="selected_positions",
        device=caps.device,
        shape=(int(caps.max_q_rows), int(caps.selection_width)),
        dtype=torch.int32,
        contiguous=True,
    )
    _require_mutation_alias_contract(
        mutable=(
            ("scratch", scratch_storage),
            ("compressed_k_cache", compressed_k_cache),
            ("raw_k_ring", raw_k_ring),
            ("raw_gate_ring", raw_gate_ring),
            ("raw_logical_positions", raw_logical_positions),
            ("raw_interval_start_positions", raw_interval_start_positions),
            ("selected_positions", selected_positions),
        ),
        read_only=(
            ("compressed_block_table", compressed_block_table),
            ("raw_state_slot_ids", raw_state_slot_ids),
            ("position_embedding", position_embedding),
        ),
    )

    layout = plan._layout
    rows = int(caps.max_q_rows)
    group_budget = int(caps.group_budget)
    return Binding(
        plan=plan,
        scratch=scratch_storage,
        compressed_k_cache=compressed_k_cache,
        compressed_block_table=compressed_block_table,
        raw_k_ring=raw_k_ring,
        raw_gate_ring=raw_gate_ring,
        raw_logical_positions=raw_logical_positions,
        raw_interval_start_positions=raw_interval_start_positions,
        raw_state_slot_ids=raw_state_slot_ids,
        position_embedding=position_embedding,
        selected_positions=selected_positions,
        scores=_scratch_view(
            scratch_storage,
            offset_bytes=layout.score_offset_bytes,
            shape=(rows, int(plan.score_workspace_width)),
            dtype=torch.float32,
        ),
        eligible_group_counts=_scratch_view(
            scratch_storage,
            offset_bytes=layout.eligible_counts_offset_bytes,
            shape=(rows,),
            dtype=torch.int32,
        ),
        merge_lengths=_scratch_view(
            scratch_storage,
            offset_bytes=layout.merge_lengths_offset_bytes,
            shape=(rows,),
            dtype=torch.int32,
        ),
        topk_values=_scratch_view(
            scratch_storage,
            offset_bytes=layout.topk_values_offset_bytes,
            shape=(rows, group_budget),
            dtype=torch.float32,
        ),
        topk_group_ids=_scratch_view(
            scratch_storage,
            offset_bytes=layout.topk_indices_offset_bytes,
            shape=(rows, group_budget),
            dtype=torch.int32,
        ),
        topk_values_b=_scratch_view(
            scratch_storage,
            offset_bytes=layout.topk_values_b_offset_bytes,
            shape=(rows, group_budget),
            dtype=torch.float32,
        ),
        topk_group_ids_b=_scratch_view(
            scratch_storage,
            offset_bytes=layout.topk_indices_b_offset_bytes,
            shape=(rows, group_budget),
            dtype=torch.int32,
        ),
        state_errors=_scratch_view(
            scratch_storage,
            offset_bytes=layout.state_errors_offset_bytes,
            shape=(rows,),
            dtype=torch.int32,
        ),
        request_errors=_scratch_view(
            scratch_storage,
            offset_bytes=layout.request_errors_offset_bytes,
            shape=(int(caps.max_batch) + 1,),
            dtype=torch.int32,
        ),
    )


def _stable_topk_views(
    scratch: torch.Tensor,
    *,
    rows: int,
    max_rows: int,
    score_workspace_width: int,
    group_budget: int,
    topk_offset_bytes: int,
) -> tuple[torch.Tensor, ...]:
    stable_blocks = math.ceil(int(score_workspace_width) / _STABLE_TOPK_BLOCK)
    offset = int(topk_offset_bytes)
    count_nbytes = max_rows * stable_blocks * torch.int32.itemsize
    tie_counts = _scratch_view(
        scratch,
        offset_bytes=offset,
        shape=(max_rows, stable_blocks),
        dtype=torch.int32,
    )[:rows]
    offset += count_nbytes
    greater_counts = _scratch_view(
        scratch,
        offset_bytes=offset,
        shape=(max_rows, stable_blocks),
        dtype=torch.int32,
    )[:rows]
    offset += count_nbytes
    stable_values = _scratch_view(
        scratch,
        offset_bytes=offset,
        shape=(max_rows, group_budget),
        dtype=torch.float32,
    )[:rows]
    offset += max_rows * group_budget * torch.float32.itemsize
    stable_ids = _scratch_view(
        scratch,
        offset_bytes=offset,
        shape=(max_rows, group_budget),
        dtype=torch.int32,
    )[:rows]
    offset += max_rows * group_budget * torch.int32.itemsize
    thresholds = _scratch_view(
        scratch,
        offset_bytes=offset,
        shape=(max_rows,),
        dtype=torch.float32,
    )[:rows]
    offset += max_rows * torch.float32.itemsize
    greater_totals = _scratch_view(
        scratch,
        offset_bytes=offset,
        shape=(max_rows,),
        dtype=torch.int32,
    )[:rows]
    return (
        tie_counts,
        greater_counts,
        stable_values,
        stable_ids,
        thresholds,
        greater_totals,
    )


def _selection_scratch_views(
    scratch: torch.Tensor,
    *,
    rows: int,
    max_rows: int,
    score_workspace_width: int,
    group_budget: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    state_errors_offset_bytes: int,
    topk_offset_bytes: int,
) -> tuple[torch.Tensor, ...]:
    scores = _scratch_view(
        scratch,
        offset_bytes=int(score_offset_bytes),
        shape=(int(max_rows), int(score_workspace_width)),
        dtype=torch.float32,
    )[:rows]
    eligible_counts = _scratch_view(
        scratch,
        offset_bytes=int(eligible_counts_offset_bytes),
        shape=(int(max_rows),),
        dtype=torch.int32,
    )[:rows]
    merge_lengths = _scratch_view(
        scratch,
        offset_bytes=int(merge_lengths_offset_bytes),
        shape=(int(max_rows),),
        dtype=torch.int32,
    )[:rows]
    topk_values = _scratch_view(
        scratch,
        offset_bytes=int(topk_values_offset_bytes),
        shape=(int(max_rows), int(group_budget)),
        dtype=torch.float32,
    )[:rows]
    topk_ids = _scratch_view(
        scratch,
        offset_bytes=int(topk_indices_offset_bytes),
        shape=(int(max_rows), int(group_budget)),
        dtype=torch.int32,
    )[:rows]
    topk_values_b = _scratch_view(
        scratch,
        offset_bytes=int(topk_values_b_offset_bytes),
        shape=(int(max_rows), int(group_budget)),
        dtype=torch.float32,
    )[:rows]
    topk_ids_b = _scratch_view(
        scratch,
        offset_bytes=int(topk_indices_b_offset_bytes),
        shape=(int(max_rows), int(group_budget)),
        dtype=torch.int32,
    )[:rows]
    state_errors = _scratch_view(
        scratch,
        offset_bytes=int(state_errors_offset_bytes),
        shape=(int(max_rows),),
        dtype=torch.int32,
    )[:rows]
    stable_views = _stable_topk_views(
        scratch,
        rows=int(rows),
        max_rows=int(max_rows),
        score_workspace_width=int(score_workspace_width),
        group_budget=int(group_budget),
        topk_offset_bytes=int(topk_offset_bytes),
    )
    return (
        scores,
        eligible_counts,
        merge_lengths,
        topk_values,
        topk_ids,
        topk_values_b,
        topk_ids_b,
        state_errors,
        *stable_views,
    )


def _run_selection_stages(
    *,
    index_query: torch.Tensor,
    index_head_weights: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    state_errors: torch.Tensor,
    scores: torch.Tensor,
    eligible_counts: torch.Tensor,
    merge_lengths: torch.Tensor,
    topk_values: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_values_b: torch.Tensor,
    topk_ids_b: torch.Tensor,
    tie_counts: torch.Tensor,
    greater_counts: torch.Tensor,
    stable_values: torch.Tensor,
    stable_ids: torch.Tensor,
    thresholds: torch.Tensor,
    greater_totals: torch.Tensor,
    selected_positions: torch.Tensor,
    score_chunk_groups: int,
    num_score_chunks: int,
    caps: _KernelCaps,
) -> None:
    from ._kernels import (
        launch_expand_selected_groups,
        launch_remap_topk_group_ids,
        launch_score_representatives,
        launch_sort_signed_topk,
        launch_stabilize_topk,
        launch_stage_topk_carry,
        launch_topk_groups,
    )

    group_budget = int(caps.group_budget)
    prior_values = topk_values_b
    prior_ids = topk_ids_b
    final_ids = prior_ids
    for chunk_index in range(int(num_score_chunks)):
        group_offset = chunk_index * int(score_chunk_groups)
        group_count = min(
            int(score_chunk_groups), int(caps.max_groups) - group_offset
        )
        output_values = topk_values if chunk_index % 2 == 0 else topk_values_b
        output_ids = topk_ids if chunk_index % 2 == 0 else topk_ids_b
        if group_offset:
            launch_stage_topk_carry(
                prior_values=prior_values,
                eligible_counts=eligible_counts,
                scores=scores,
                group_offset=group_offset,
                group_budget=group_budget,
            )
        launch_score_representatives(
            index_query=index_query,
            index_head_weights=index_head_weights,
            query_positions=query_positions,
            request_ids=request_ids,
            sequence_lengths=sequence_lengths,
            compressed_cache=compressed_k_cache,
            compressed_block_table=compressed_block_table,
            state_errors=state_errors,
            scores=scores,
            eligible_counts=eligible_counts,
            merge_lengths=merge_lengths,
            group_offset=group_offset,
            group_count=group_count,
            caps=caps,
        )
        launch_topk_groups(
            scores=scores,
            eligible_counts=merge_lengths,
            topk_values=output_values,
            topk_group_ids=output_ids,
            group_budget=group_budget,
        )
        launch_remap_topk_group_ids(
            local_ids=output_ids,
            prior_ids=prior_ids,
            eligible_counts=eligible_counts,
            merge_lengths=merge_lengths,
            group_offset=group_offset,
            group_budget=group_budget,
        )
        launch_stabilize_topk(
            scores=scores,
            merge_lengths=merge_lengths,
            prior_ids=prior_ids,
            eligible_counts=eligible_counts,
            topk_values=output_values,
            topk_group_ids=output_ids,
            tie_counts=tie_counts,
            greater_counts=greater_counts,
            stable_values=stable_values,
            stable_ids=stable_ids,
            thresholds=thresholds,
            greater_totals=greater_totals,
            group_offset=group_offset,
            group_budget=group_budget,
        )
        launch_sort_signed_topk(
            topk_values=output_values,
            topk_group_ids=output_ids,
            merge_lengths=merge_lengths,
            group_budget=group_budget,
        )
        prior_values, prior_ids = output_values, output_ids
        final_ids = output_ids

    launch_expand_selected_groups(
        topk_group_ids=final_ids,
        eligible_counts=eligible_counts,
        query_positions=query_positions,
        state_errors=state_errors,
        selected_positions=selected_positions,
        caps=caps,
    )


def _decode_impl(
    index_query: torch.Tensor,
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    index_head_weights: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    scratch: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    position_embedding: torch.Tensor,
    selected_positions: torch.Tensor,
    output_row_start: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
    topk_offset_bytes: int,
) -> None:
    rows = int(index_query.shape[0])
    max_rows = int(selected_positions.shape[0])
    index_heads = int(index_query.shape[1])
    index_head_dim = int(index_query.shape[2])
    group_budget = int(budget) // int(compress_ratio)
    caps = _KernelCaps(
        max_batch=int(compressed_block_table.shape[0]),
        max_raw_state_slots=int(raw_k_ring.shape[0]),
        max_seq_len=int(max_seq_len),
        compressed_page_size=int(compressed_k_cache.shape[1]),
        max_speculative_tokens=int(max_speculative_tokens),
        index_heads=index_heads,
        index_head_dim=index_head_dim,
        compress_ratio=int(compress_ratio),
        budget=int(budget),
        raw_ring_capacity=int(raw_k_ring.shape[1]),
    )
    scores = _scratch_view(
        scratch,
        offset_bytes=int(score_offset_bytes),
        shape=(max_rows, int(score_workspace_width)),
        dtype=torch.float32,
    )[:rows]
    eligible_counts = _scratch_view(
        scratch,
        offset_bytes=int(eligible_counts_offset_bytes),
        shape=(max_rows,),
        dtype=torch.int32,
    )[:rows]
    merge_lengths = _scratch_view(
        scratch,
        offset_bytes=int(merge_lengths_offset_bytes),
        shape=(max_rows,),
        dtype=torch.int32,
    )[:rows]
    topk_values = _scratch_view(
        scratch,
        offset_bytes=int(topk_values_offset_bytes),
        shape=(max_rows, group_budget),
        dtype=torch.float32,
    )[:rows]
    topk_ids = _scratch_view(
        scratch,
        offset_bytes=int(topk_indices_offset_bytes),
        shape=(max_rows, group_budget),
        dtype=torch.int32,
    )[:rows]
    topk_values_b = _scratch_view(
        scratch,
        offset_bytes=int(topk_values_b_offset_bytes),
        shape=(max_rows, group_budget),
        dtype=torch.float32,
    )[:rows]
    topk_ids_b = _scratch_view(
        scratch,
        offset_bytes=int(topk_indices_b_offset_bytes),
        shape=(max_rows, group_budget),
        dtype=torch.int32,
    )[:rows]
    state_errors = _scratch_view(
        scratch,
        offset_bytes=int(state_errors_offset_bytes),
        shape=(max_rows,),
        dtype=torch.int32,
    )[:rows]
    request_errors = _scratch_view(
        scratch,
        offset_bytes=int(request_errors_offset_bytes),
        shape=(int(caps.max_batch) + 1,),
        dtype=torch.int32,
    )
    (
        tie_counts,
        greater_counts,
        stable_values,
        stable_ids,
        thresholds,
        greater_totals,
    ) = _stable_topk_views(
        scratch,
        rows=rows,
        max_rows=max_rows,
        score_workspace_width=int(score_workspace_width),
        group_budget=group_budget,
        topk_offset_bytes=int(topk_offset_bytes),
    )

    from ._kernels import (
        launch_pool_completed_groups,
        launch_propagate_request_errors,
        launch_update_raw_rings,
        launch_validate_completed_groups,
        launch_validate_compressed_page_table,
        launch_validate_decode_rows,
    )

    launch_validate_decode_rows(
        request_ids=request_ids,
        query_positions=query_positions,
        sequence_lengths=sequence_lengths,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_interval_start_positions=raw_interval_start_positions,
        request_errors=request_errors,
        state_errors=state_errors,
        caps=caps,
    )
    launch_validate_compressed_page_table(
        request_ids=request_ids,
        sequence_lengths=sequence_lengths,
        compressed_block_table=compressed_block_table,
        compressed_cache=compressed_k_cache,
        state_errors=state_errors,
        caps=caps,
    )
    launch_validate_completed_groups(
        query_positions=query_positions,
        request_ids=request_ids,
        query_start_loc=query_start_loc,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_logical_positions=raw_logical_positions,
        state_errors=state_errors,
        caps=caps,
    )
    launch_propagate_request_errors(
        request_ids=request_ids,
        request_errors=request_errors,
        state_errors=state_errors,
        caps=caps,
    )
    # Pooling consumes the committed ring before the current interval can wrap it.
    launch_pool_completed_groups(
        normalized_index_key=normalized_index_key,
        index_gate_logits=index_gate_logits,
        query_positions=query_positions,
        request_ids=request_ids,
        query_start_loc=query_start_loc,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_k_ring=raw_k_ring,
        raw_gate_ring=raw_gate_ring,
        position_embedding=position_embedding,
        compressed_cache=compressed_k_cache,
        compressed_block_table=compressed_block_table,
        state_errors=state_errors,
        caps=caps,
    )
    launch_update_raw_rings(
        normalized_index_key=normalized_index_key,
        index_gate_logits=index_gate_logits,
        query_positions=query_positions,
        request_ids=request_ids,
        query_start_loc=query_start_loc,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_k_ring=raw_k_ring,
        raw_gate_ring=raw_gate_ring,
        raw_logical_positions=raw_logical_positions,
        raw_interval_start_positions=raw_interval_start_positions,
        state_errors=state_errors,
        caps=caps,
    )

    _run_selection_stages(
        index_query=index_query,
        index_head_weights=index_head_weights,
        request_ids=request_ids,
        query_positions=query_positions,
        sequence_lengths=sequence_lengths,
        compressed_k_cache=compressed_k_cache,
        compressed_block_table=compressed_block_table,
        state_errors=state_errors,
        scores=scores,
        eligible_counts=eligible_counts,
        merge_lengths=merge_lengths,
        topk_values=topk_values,
        topk_ids=topk_ids,
        topk_values_b=topk_values_b,
        topk_ids_b=topk_ids_b,
        tie_counts=tie_counts,
        greater_counts=greater_counts,
        stable_values=stable_values,
        stable_ids=stable_ids,
        thresholds=thresholds,
        greater_totals=greater_totals,
        selected_positions=selected_positions[
            int(output_row_start) : int(output_row_start) + rows
        ],
        score_chunk_groups=int(score_chunk_groups),
        num_score_chunks=int(num_score_chunks),
        caps=caps,
    )


_MUTATED_ARGUMENTS = (
    "scratch",
    "compressed_k_cache",
    "raw_k_ring",
    "raw_gate_ring",
    "raw_logical_positions",
    "raw_interval_start_positions",
    "selected_positions",
)


_PREFILL_MUTATED_ARGUMENTS = (
    "scratch",
    "compressed_k_cache",
    "raw_k_ring",
    "raw_gate_ring",
    "raw_logical_positions",
    "raw_interval_start_positions",
)


def _prefill_impl(
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    scratch: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    position_embedding: torch.Tensor,
    max_q_rows: int,
    max_seq_len: int,
    compress_ratio: int,
    budget: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
) -> None:
    rows = int(normalized_index_key.shape[0])
    caps = _KernelCaps(
        max_batch=int(compressed_block_table.shape[0]),
        max_raw_state_slots=int(raw_k_ring.shape[0]),
        max_seq_len=int(max_seq_len),
        compressed_page_size=int(compressed_k_cache.shape[1]),
        max_speculative_tokens=0,
        index_heads=32,
        index_head_dim=int(normalized_index_key.shape[1]),
        compress_ratio=int(compress_ratio),
        budget=int(budget),
        raw_ring_capacity=int(raw_k_ring.shape[1]),
    )
    state_errors = _scratch_view(
        scratch,
        offset_bytes=int(state_errors_offset_bytes),
        shape=(int(max_q_rows),),
        dtype=torch.int32,
    )[:rows]
    request_errors = _scratch_view(
        scratch,
        offset_bytes=int(request_errors_offset_bytes),
        shape=(int(caps.max_batch) + 1,),
        dtype=torch.int32,
    )
    from ._kernels import (
        launch_finalize_prefill_anchors,
        launch_pool_completed_groups,
        launch_propagate_request_errors,
        launch_update_prefill_raw_rings,
        launch_validate_completed_groups,
        launch_validate_compressed_page_table,
        launch_validate_prefill_rows,
    )

    launch_validate_prefill_rows(
        request_ids=request_ids,
        query_positions=query_positions,
        sequence_lengths=sequence_lengths,
        query_start_loc=query_start_loc,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_interval_start_positions=raw_interval_start_positions,
        request_errors=request_errors,
        state_errors=state_errors,
        caps=caps,
    )
    launch_validate_compressed_page_table(
        request_ids=request_ids,
        sequence_lengths=sequence_lengths,
        compressed_block_table=compressed_block_table,
        compressed_cache=compressed_k_cache,
        state_errors=state_errors,
        caps=caps,
    )
    launch_validate_completed_groups(
        query_positions=query_positions,
        request_ids=request_ids,
        query_start_loc=query_start_loc,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_logical_positions=raw_logical_positions,
        state_errors=state_errors,
        caps=caps,
    )
    launch_propagate_request_errors(
        request_ids=request_ids,
        request_errors=request_errors,
        state_errors=state_errors,
        caps=caps,
    )
    launch_pool_completed_groups(
        normalized_index_key=normalized_index_key,
        index_gate_logits=index_gate_logits,
        query_positions=query_positions,
        request_ids=request_ids,
        query_start_loc=query_start_loc,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_k_ring=raw_k_ring,
        raw_gate_ring=raw_gate_ring,
        position_embedding=position_embedding,
        compressed_cache=compressed_k_cache,
        compressed_block_table=compressed_block_table,
        state_errors=state_errors,
        caps=caps,
    )
    launch_update_prefill_raw_rings(
        normalized_index_key=normalized_index_key,
        index_gate_logits=index_gate_logits,
        query_positions=query_positions,
        request_ids=request_ids,
        query_start_loc=query_start_loc,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_k_ring=raw_k_ring,
        raw_gate_ring=raw_gate_ring,
        raw_logical_positions=raw_logical_positions,
        state_errors=state_errors,
        caps=caps,
    )
    launch_finalize_prefill_anchors(
        request_ids=request_ids,
        query_positions=query_positions,
        query_start_loc=query_start_loc,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_interval_start_positions=raw_interval_start_positions,
        request_errors=request_errors,
        caps=caps,
    )


def _prefill_select_impl(
    index_query: torch.Tensor,
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    index_head_weights: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    scratch: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    position_embedding: torch.Tensor,
    selected_positions: torch.Tensor,
    output_row_start: int,
    max_q_rows: int,
    max_seq_len: int,
    compress_ratio: int,
    budget: int,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
    topk_offset_bytes: int,
) -> None:
    _prefill_impl(
        normalized_index_key,
        index_gate_logits,
        request_ids,
        query_positions,
        sequence_lengths,
        query_start_loc,
        scratch,
        compressed_k_cache,
        compressed_block_table,
        raw_k_ring,
        raw_gate_ring,
        raw_logical_positions,
        raw_interval_start_positions,
        raw_state_slot_ids,
        position_embedding,
        max_q_rows,
        max_seq_len,
        compress_ratio,
        budget,
        state_errors_offset_bytes,
        request_errors_offset_bytes,
    )
    rows = int(index_query.shape[0])
    group_budget = int(budget) // int(compress_ratio)
    caps = _KernelCaps(
        max_batch=int(compressed_block_table.shape[0]),
        max_raw_state_slots=int(raw_k_ring.shape[0]),
        max_seq_len=int(max_seq_len),
        compressed_page_size=int(compressed_k_cache.shape[1]),
        max_speculative_tokens=0,
        index_heads=int(index_query.shape[1]),
        index_head_dim=int(index_query.shape[2]),
        compress_ratio=int(compress_ratio),
        budget=int(budget),
        raw_ring_capacity=int(raw_k_ring.shape[1]),
    )
    (
        scores,
        eligible_counts,
        merge_lengths,
        topk_values,
        topk_ids,
        topk_values_b,
        topk_ids_b,
        state_errors,
        tie_counts,
        greater_counts,
        stable_values,
        stable_ids,
        thresholds,
        greater_totals,
    ) = _selection_scratch_views(
        scratch,
        rows=rows,
        max_rows=int(max_q_rows),
        score_workspace_width=int(score_workspace_width),
        group_budget=group_budget,
        score_offset_bytes=int(score_offset_bytes),
        eligible_counts_offset_bytes=int(eligible_counts_offset_bytes),
        merge_lengths_offset_bytes=int(merge_lengths_offset_bytes),
        topk_values_offset_bytes=int(topk_values_offset_bytes),
        topk_indices_offset_bytes=int(topk_indices_offset_bytes),
        topk_values_b_offset_bytes=int(topk_values_b_offset_bytes),
        topk_indices_b_offset_bytes=int(topk_indices_b_offset_bytes),
        state_errors_offset_bytes=int(state_errors_offset_bytes),
        topk_offset_bytes=int(topk_offset_bytes),
    )
    _run_selection_stages(
        index_query=index_query,
        index_head_weights=index_head_weights,
        request_ids=request_ids,
        query_positions=query_positions,
        sequence_lengths=sequence_lengths,
        compressed_k_cache=compressed_k_cache,
        compressed_block_table=compressed_block_table,
        state_errors=state_errors,
        scores=scores,
        eligible_counts=eligible_counts,
        merge_lengths=merge_lengths,
        topk_values=topk_values,
        topk_ids=topk_ids,
        topk_values_b=topk_values_b,
        topk_ids_b=topk_ids_b,
        tie_counts=tie_counts,
        greater_counts=greater_counts,
        stable_values=stable_values,
        stable_ids=stable_ids,
        thresholds=thresholds,
        greater_totals=greater_totals,
        selected_positions=selected_positions[
            int(output_row_start) : int(output_row_start) + rows
        ],
        score_chunk_groups=int(score_chunk_groups),
        num_score_chunks=int(num_score_chunks),
        caps=caps,
    )


@torch.library.custom_op(
    "b12x::glm_pooled_indexer_prefill",
    mutates_args=_PREFILL_MUTATED_ARGUMENTS,
)
def _prefill_op(
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    scratch: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    position_embedding: torch.Tensor,
    max_q_rows: int,
    max_seq_len: int,
    compress_ratio: int,
    budget: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
) -> None:
    _require_mutation_alias_contract(
        mutable=(
            ("scratch", scratch),
            ("compressed_k_cache", compressed_k_cache),
            ("raw_k_ring", raw_k_ring),
            ("raw_gate_ring", raw_gate_ring),
            ("raw_logical_positions", raw_logical_positions),
            ("raw_interval_start_positions", raw_interval_start_positions),
        ),
        read_only=(
            ("normalized_index_key", normalized_index_key),
            ("index_gate_logits", index_gate_logits),
            ("request_ids", request_ids),
            ("query_positions", query_positions),
            ("sequence_lengths", sequence_lengths),
            ("query_start_loc", query_start_loc),
            ("compressed_block_table", compressed_block_table),
            ("raw_state_slot_ids", raw_state_slot_ids),
            ("position_embedding", position_embedding),
        ),
    )
    _prefill_impl(
        normalized_index_key,
        index_gate_logits,
        request_ids,
        query_positions,
        sequence_lengths,
        query_start_loc,
        scratch,
        compressed_k_cache,
        compressed_block_table,
        raw_k_ring,
        raw_gate_ring,
        raw_logical_positions,
        raw_interval_start_positions,
        raw_state_slot_ids,
        position_embedding,
        max_q_rows,
        max_seq_len,
        compress_ratio,
        budget,
        state_errors_offset_bytes,
        request_errors_offset_bytes,
    )


@_prefill_op.register_fake
def _prefill_fake(
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    scratch: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    position_embedding: torch.Tensor,
    max_q_rows: int,
    max_seq_len: int,
    compress_ratio: int,
    budget: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
) -> None:
    return None


@torch.library.custom_op(
    "b12x::glm_pooled_indexer_prefill_select",
    mutates_args=_MUTATED_ARGUMENTS,
)
def _prefill_select_op(
    index_query: torch.Tensor,
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    index_head_weights: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    scratch: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    position_embedding: torch.Tensor,
    selected_positions: torch.Tensor,
    output_row_start: int,
    max_q_rows: int,
    max_seq_len: int,
    compress_ratio: int,
    budget: int,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
    topk_offset_bytes: int,
) -> None:
    _require_mutation_alias_contract(
        mutable=(
            ("scratch", scratch),
            ("compressed_k_cache", compressed_k_cache),
            ("raw_k_ring", raw_k_ring),
            ("raw_gate_ring", raw_gate_ring),
            ("raw_logical_positions", raw_logical_positions),
            ("raw_interval_start_positions", raw_interval_start_positions),
            ("selected_positions", selected_positions),
        ),
        read_only=(
            ("index_query", index_query),
            ("normalized_index_key", normalized_index_key),
            ("index_gate_logits", index_gate_logits),
            ("index_head_weights", index_head_weights),
            ("request_ids", request_ids),
            ("query_positions", query_positions),
            ("sequence_lengths", sequence_lengths),
            ("query_start_loc", query_start_loc),
            ("compressed_block_table", compressed_block_table),
            ("raw_state_slot_ids", raw_state_slot_ids),
            ("position_embedding", position_embedding),
        ),
    )
    _prefill_select_impl(
        index_query,
        normalized_index_key,
        index_gate_logits,
        index_head_weights,
        request_ids,
        query_positions,
        sequence_lengths,
        query_start_loc,
        scratch,
        compressed_k_cache,
        compressed_block_table,
        raw_k_ring,
        raw_gate_ring,
        raw_logical_positions,
        raw_interval_start_positions,
        raw_state_slot_ids,
        position_embedding,
        selected_positions,
        output_row_start,
        max_q_rows,
        max_seq_len,
        compress_ratio,
        budget,
        score_chunk_groups,
        score_workspace_width,
        num_score_chunks,
        score_offset_bytes,
        eligible_counts_offset_bytes,
        merge_lengths_offset_bytes,
        topk_values_offset_bytes,
        topk_indices_offset_bytes,
        topk_values_b_offset_bytes,
        topk_indices_b_offset_bytes,
        state_errors_offset_bytes,
        request_errors_offset_bytes,
        topk_offset_bytes,
    )


@_prefill_select_op.register_fake
def _prefill_select_fake(
    index_query: torch.Tensor,
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    index_head_weights: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    scratch: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    position_embedding: torch.Tensor,
    selected_positions: torch.Tensor,
    output_row_start: int,
    max_q_rows: int,
    max_seq_len: int,
    compress_ratio: int,
    budget: int,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
    topk_offset_bytes: int,
) -> None:
    return None


_RESET_MUTATED_ARGUMENTS = (
    "raw_logical_positions",
    "raw_interval_start_positions",
)


@torch.library.custom_op(
    "b12x::glm_pooled_indexer_reset_state",
    mutates_args=_RESET_MUTATED_ARGUMENTS,
)
def _reset_state_op(
    reset_mask: torch.Tensor,
    prefix_lengths: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    max_seq_len: int,
    max_raw_state_slots: int,
    raw_ring_capacity: int,
) -> None:
    _require_mutation_alias_contract(
        mutable=(
            ("raw_logical_positions", raw_logical_positions),
            ("raw_interval_start_positions", raw_interval_start_positions),
        ),
        read_only=(
            ("reset_mask", reset_mask),
            ("prefix_lengths", prefix_lengths),
            ("raw_state_slot_ids", raw_state_slot_ids),
        ),
    )
    caps = _KernelCaps(
        max_batch=int(reset_mask.shape[0]),
        max_raw_state_slots=int(max_raw_state_slots),
        max_seq_len=int(max_seq_len),
        compressed_page_size=1,
        max_speculative_tokens=0,
        index_heads=32,
        index_head_dim=128,
        compress_ratio=4,
        budget=2048,
        raw_ring_capacity=int(raw_ring_capacity),
    )
    from ._kernels import launch_reset_selector_state

    launch_reset_selector_state(
        reset_mask=reset_mask,
        prefix_lengths=prefix_lengths,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_logical_positions=raw_logical_positions,
        raw_interval_start_positions=raw_interval_start_positions,
        caps=caps,
    )


@_reset_state_op.register_fake
def _reset_state_fake(
    reset_mask: torch.Tensor,
    prefix_lengths: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    max_seq_len: int,
    max_raw_state_slots: int,
    raw_ring_capacity: int,
) -> None:
    return None


@torch.library.custom_op(
    "b12x::glm_pooled_indexer_decode", mutates_args=_MUTATED_ARGUMENTS
)
def _decode_op(
    index_query: torch.Tensor,
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    index_head_weights: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    scratch: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    position_embedding: torch.Tensor,
    selected_positions: torch.Tensor,
    output_row_start: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
    topk_offset_bytes: int,
) -> None:
    _require_mutation_alias_contract(
        mutable=(
            ("scratch", scratch),
            ("compressed_k_cache", compressed_k_cache),
            ("raw_k_ring", raw_k_ring),
            ("raw_gate_ring", raw_gate_ring),
            ("raw_logical_positions", raw_logical_positions),
            ("raw_interval_start_positions", raw_interval_start_positions),
            ("selected_positions", selected_positions),
        ),
        read_only=(
            ("index_query", index_query),
            ("normalized_index_key", normalized_index_key),
            ("index_gate_logits", index_gate_logits),
            ("index_head_weights", index_head_weights),
            ("request_ids", request_ids),
            ("query_positions", query_positions),
            ("sequence_lengths", sequence_lengths),
            ("query_start_loc", query_start_loc),
            ("num_accepted_tokens", num_accepted_tokens),
            ("compressed_block_table", compressed_block_table),
            ("raw_state_slot_ids", raw_state_slot_ids),
            ("position_embedding", position_embedding),
        ),
    )
    _decode_impl(
        index_query,
        normalized_index_key,
        index_gate_logits,
        index_head_weights,
        request_ids,
        query_positions,
        sequence_lengths,
        query_start_loc,
        num_accepted_tokens,
        scratch,
        compressed_k_cache,
        compressed_block_table,
        raw_k_ring,
        raw_gate_ring,
        raw_logical_positions,
        raw_interval_start_positions,
        raw_state_slot_ids,
        position_embedding,
        selected_positions,
        output_row_start,
        max_seq_len,
        max_speculative_tokens,
        compress_ratio,
        budget,
        score_chunk_groups,
        score_workspace_width,
        num_score_chunks,
        score_offset_bytes,
        eligible_counts_offset_bytes,
        merge_lengths_offset_bytes,
        topk_values_offset_bytes,
        topk_indices_offset_bytes,
        topk_values_b_offset_bytes,
        topk_indices_b_offset_bytes,
        state_errors_offset_bytes,
        request_errors_offset_bytes,
        topk_offset_bytes,
    )


@_decode_op.register_fake
def _decode_fake(
    index_query: torch.Tensor,
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    index_head_weights: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    scratch: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    position_embedding: torch.Tensor,
    selected_positions: torch.Tensor,
    output_row_start: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
    topk_offset_bytes: int,
) -> None:
    return None


def reset_state(
    binding: Binding,
    *,
    reset_mask: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> None:
    """Reset recycled raw-state slots for fresh or aligned-prefix requests.

    Inputs have one element per request-table row.  A true ``reset_mask`` entry
    invalidates every raw logical tag owned by that request and sets its
    interval anchor to ``prefix_lengths[request] - 1``.  Prefix lengths must be
    nonnegative, no greater than ``max_seq_len``, and divisible by four.  Thus
    zero initializes a fresh request, while an aligned prefix-cache hit may
    begin prefill or decode at the first uncached token without copying raw
    selector state.  Invalid device metadata writes a fail-closed anchor that
    the next execution call rejects without an out-of-bounds access.
    """
    if not isinstance(binding, Binding):
        raise TypeError("binding must be a glm_pooled_indexer.Binding")
    caps = binding.plan.caps
    for tensor, name, dtypes in (
        (reset_mask, "reset_mask", (torch.bool,)),
        (prefix_lengths, "prefix_lengths", (torch.int32, torch.int64)),
    ):
        _check_tensor(
            tensor,
            name=name,
            device=caps.device,
            shape=(int(caps.max_batch),),
            dtype=dtypes,
            contiguous=True,
        )
    if not torch.compiler.is_compiling():
        _require_mutation_alias_contract(
            mutable=(
                ("raw_logical_positions", binding.raw_logical_positions),
                (
                    "raw_interval_start_positions",
                    binding.raw_interval_start_positions,
                ),
            ),
            read_only=(
                ("reset_mask", reset_mask),
                ("prefix_lengths", prefix_lengths),
            ),
        )
    _reset_state_op(
        reset_mask,
        prefix_lengths,
        binding.raw_state_slot_ids,
        binding.raw_logical_positions,
        binding.raw_interval_start_positions,
        int(caps.max_seq_len),
        int(caps.max_raw_state_slots),
        int(caps.raw_ring_capacity),
    )


def update_prefill_cache(
    binding: Binding,
    *,
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    """Commit one packed prefill chunk to the selector cache and raw state.

    ``sequence_lengths`` includes the supplied chunk and ``query_start_loc``
    partitions the rows by request.  The first chunk for a request must begin
    at logical position zero with its persistent interval anchor initialized to
    ``-1``.  Later chunks must begin immediately after the committed anchor.
    Complete four-token groups are pooled into ``compressed_k_cache``; the
    exact trailing raw keys, gate logits, and logical tags remain in the raw
    rings.  The anchor is committed to the last row of each active request, so
    its first decode interval uses one accepted token.

    The key is the caller-computed BF16 LayerNorm output and the gate input is
    the raw BF16 linear projection.  This operation applies neither key
    normalization nor selector RoPE and does not produce prefill selections.
    """
    if not isinstance(binding, Binding):
        raise TypeError("binding must be a glm_pooled_indexer.Binding")
    caps = binding.plan.caps
    if normalized_index_key.device.type != "cuda":
        raise RuntimeError("GLM pooled selector requires a CUDA device")
    rows = int(normalized_index_key.shape[0])
    if not 0 < rows <= int(caps.max_q_rows):
        raise ValueError("prefill rows must be within planned capacity")
    specs = (
        (
            normalized_index_key,
            "normalized_index_key",
            (rows, int(caps.index_head_dim)),
            (torch.bfloat16,),
        ),
        (
            index_gate_logits,
            "index_gate_logits",
            (rows, int(caps.index_head_dim)),
            (torch.bfloat16,),
        ),
        (request_ids, "request_ids", (rows,), (torch.int32, torch.int64)),
        (query_positions, "query_positions", (rows,), (torch.int64,)),
        (
            sequence_lengths,
            "sequence_lengths",
            (int(caps.max_batch),),
            (torch.int32,),
        ),
        (
            query_start_loc,
            "query_start_loc",
            (int(caps.max_batch) + 1,),
            (torch.int32,),
        ),
    )
    for tensor, name, shape, dtypes in specs:
        _check_tensor(
            tensor,
            name=name,
            device=caps.device,
            shape=shape,
            dtype=dtypes,
            contiguous=True,
        )
    if not torch.compiler.is_compiling():
        _require_mutation_alias_contract(
            mutable=(
                ("scratch", binding.scratch),
                ("compressed_k_cache", binding.compressed_k_cache),
                ("raw_k_ring", binding.raw_k_ring),
                ("raw_gate_ring", binding.raw_gate_ring),
                ("raw_logical_positions", binding.raw_logical_positions),
                (
                    "raw_interval_start_positions",
                    binding.raw_interval_start_positions,
                ),
            ),
            read_only=tuple(
                (name, tensor) for tensor, name, _shape, _dtypes in specs
            ),
        )

    layout = binding.plan._layout
    _prefill_op(
        normalized_index_key,
        index_gate_logits,
        request_ids,
        query_positions,
        sequence_lengths,
        query_start_loc,
        binding.scratch,
        binding.compressed_k_cache,
        binding.compressed_block_table,
        binding.raw_k_ring,
        binding.raw_gate_ring,
        binding.raw_logical_positions,
        binding.raw_interval_start_positions,
        binding.raw_state_slot_ids,
        binding.position_embedding,
        int(caps.max_q_rows),
        int(caps.max_seq_len),
        int(caps.compress_ratio),
        int(caps.budget),
        int(layout.state_errors_offset_bytes),
        int(layout.request_errors_offset_bytes),
    )


def run_prefill(
    binding: Binding,
    *,
    index_query: torch.Tensor,
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    index_head_weights: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    output_row_start: int = 0,
) -> torch.Tensor:
    """Commit packed prefill rows and select sparse-MLA positions for each row.

    Cache maintenance follows :func:`update_prefill_cache`.  After all newly
    completed representatives are materialized, every query row scores only
    the groups causally complete at its own logical position, selects the
    stable highest-scoring 512 groups, expands them to 2048 token positions,
    and appends its raw incomplete-group tail.  The returned caller-owned view
    always has width 2051.  ``output_row_start`` permits prefill and decode
    calls to fill disjoint rows of one mixed-batch output buffer.
    """
    if not isinstance(binding, Binding):
        raise TypeError("binding must be a glm_pooled_indexer.Binding")
    caps = binding.plan.caps
    if index_query.device.type != "cuda":
        raise RuntimeError("GLM pooled selector requires a CUDA device")
    rows = int(index_query.shape[0])
    if not 0 < rows <= int(caps.max_q_rows):
        raise ValueError("prefill query rows must be within planned capacity")
    output_row_start = int(output_row_start)
    if not 0 <= output_row_start <= int(caps.max_q_rows) - rows:
        raise ValueError("prefill output rows exceed selected-position capacity")
    specs = (
        (
            index_query,
            "index_query",
            (rows, int(caps.index_heads), int(caps.index_head_dim)),
            (torch.bfloat16,),
        ),
        (
            normalized_index_key,
            "normalized_index_key",
            (rows, int(caps.index_head_dim)),
            (torch.bfloat16,),
        ),
        (
            index_gate_logits,
            "index_gate_logits",
            (rows, int(caps.index_head_dim)),
            (torch.bfloat16,),
        ),
        (
            index_head_weights,
            "index_head_weights",
            (rows, int(caps.index_heads)),
            (torch.bfloat16, torch.float32),
        ),
        (request_ids, "request_ids", (rows,), (torch.int32, torch.int64)),
        (query_positions, "query_positions", (rows,), (torch.int64,)),
        (
            sequence_lengths,
            "sequence_lengths",
            (int(caps.max_batch),),
            (torch.int32,),
        ),
        (
            query_start_loc,
            "query_start_loc",
            (int(caps.max_batch) + 1,),
            (torch.int32,),
        ),
    )
    for tensor, name, shape, dtypes in specs:
        _check_tensor(
            tensor,
            name=name,
            device=caps.device,
            shape=shape,
            dtype=dtypes,
            contiguous=True,
        )
    if not torch.compiler.is_compiling():
        _require_mutation_alias_contract(
            mutable=(
                ("scratch", binding.scratch),
                ("compressed_k_cache", binding.compressed_k_cache),
                ("raw_k_ring", binding.raw_k_ring),
                ("raw_gate_ring", binding.raw_gate_ring),
                ("raw_logical_positions", binding.raw_logical_positions),
                (
                    "raw_interval_start_positions",
                    binding.raw_interval_start_positions,
                ),
                ("selected_positions", binding.selected_positions),
            ),
            read_only=tuple(
                (name, tensor) for tensor, name, _shape, _dtypes in specs
            ),
        )

    layout = binding.plan._layout
    _prefill_select_op(
        index_query,
        normalized_index_key,
        index_gate_logits,
        index_head_weights,
        request_ids,
        query_positions,
        sequence_lengths,
        query_start_loc,
        binding.scratch,
        binding.compressed_k_cache,
        binding.compressed_block_table,
        binding.raw_k_ring,
        binding.raw_gate_ring,
        binding.raw_logical_positions,
        binding.raw_interval_start_positions,
        binding.raw_state_slot_ids,
        binding.position_embedding,
        binding.selected_positions,
        output_row_start,
        int(caps.max_q_rows),
        int(caps.max_seq_len),
        int(caps.compress_ratio),
        int(caps.budget),
        int(binding.plan.score_chunk_groups),
        int(binding.plan.score_workspace_width),
        int(binding.plan.num_score_chunks),
        int(layout.score_offset_bytes),
        int(layout.eligible_counts_offset_bytes),
        int(layout.merge_lengths_offset_bytes),
        int(layout.topk_values_offset_bytes),
        int(layout.topk_indices_offset_bytes),
        int(layout.topk_values_b_offset_bytes),
        int(layout.topk_indices_b_offset_bytes),
        int(layout.state_errors_offset_bytes),
        int(layout.request_errors_offset_bytes),
        int(layout.topk_offset_bytes),
    )
    return binding.selected_positions[output_row_start : output_row_start + rows]


def run(
    binding: Binding,
    *,
    index_query: torch.Tensor,
    normalized_index_key: torch.Tensor,
    index_gate_logits: torch.Tensor,
    index_head_weights: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    output_row_start: int = 0,
) -> torch.Tensor:
    """Run one packed decode/speculative-verification selector transaction.

    ``sequence_lengths`` includes the current packed interval.  Each active
    ``num_accepted_tokens`` value commits the accepted prefix of the preceding
    interval, including its guaranteed or recovered token.  Before the first
    decode row at position ``N``, prefill must seed the trailing open group in
    both raw rings with exact logical tags and set the persistent interval
    anchor to ``N - num_accepted_tokens``.  A fresh position-zero request uses
    anchor ``-1`` and one accepted token.

    The normalized key input is the per-token LayerNorm result consumed by the
    GLM indexer.  The operation deliberately does not apply selector RoPE or a
    post-pooling normalization.  Invalid dynamic metadata suppresses mutation
    for the entire request and fills its selected-position rows with ``-1``.
    """
    if not isinstance(binding, Binding):
        raise TypeError("binding must be a glm_pooled_indexer.Binding")
    caps = binding.plan.caps
    if index_query.device.type != "cuda":
        raise RuntimeError("GLM pooled selector requires a CUDA device")
    rows = int(index_query.shape[0])
    if not 0 < rows <= int(caps.max_q_rows):
        raise ValueError("index query rows must be within planned capacity")
    output_row_start = int(output_row_start)
    if not 0 <= output_row_start <= int(caps.max_q_rows) - rows:
        raise ValueError("decode output rows exceed selected-position capacity")
    specs = (
        (
            index_query,
            "index_query",
            (rows, int(caps.index_heads), int(caps.index_head_dim)),
            (torch.bfloat16,),
        ),
        (
            normalized_index_key,
            "normalized_index_key",
            (rows, int(caps.index_head_dim)),
            (torch.bfloat16,),
        ),
        (
            index_gate_logits,
            "index_gate_logits",
            (rows, int(caps.index_head_dim)),
            (torch.bfloat16,),
        ),
        (
            index_head_weights,
            "index_head_weights",
            (rows, int(caps.index_heads)),
            (torch.bfloat16, torch.float32),
        ),
        (request_ids, "request_ids", (rows,), (torch.int32, torch.int64)),
        (query_positions, "query_positions", (rows,), (torch.int64,)),
        (
            sequence_lengths,
            "sequence_lengths",
            (int(caps.max_batch),),
            (torch.int32,),
        ),
        (
            query_start_loc,
            "query_start_loc",
            (int(caps.max_batch) + 1,),
            (torch.int32,),
        ),
        (
            num_accepted_tokens,
            "num_accepted_tokens",
            (int(caps.max_batch),),
            (torch.int32,),
        ),
    )
    for tensor, name, shape, dtypes in specs:
        _check_tensor(
            tensor,
            name=name,
            device=caps.device,
            shape=shape,
            dtype=dtypes,
            contiguous=True,
        )
    if not torch.compiler.is_compiling():
        _require_mutation_alias_contract(
            mutable=(
                ("scratch", binding.scratch),
                ("compressed_k_cache", binding.compressed_k_cache),
                ("raw_k_ring", binding.raw_k_ring),
                ("raw_gate_ring", binding.raw_gate_ring),
                ("raw_logical_positions", binding.raw_logical_positions),
                (
                    "raw_interval_start_positions",
                    binding.raw_interval_start_positions,
                ),
                ("selected_positions", binding.selected_positions),
            ),
            read_only=tuple(
                (name, tensor) for tensor, name, _shape, _dtypes in specs
            ),
        )

    layout = binding.plan._layout
    _decode_op(
        index_query,
        normalized_index_key,
        index_gate_logits,
        index_head_weights,
        request_ids,
        query_positions,
        sequence_lengths,
        query_start_loc,
        num_accepted_tokens,
        binding.scratch,
        binding.compressed_k_cache,
        binding.compressed_block_table,
        binding.raw_k_ring,
        binding.raw_gate_ring,
        binding.raw_logical_positions,
        binding.raw_interval_start_positions,
        binding.raw_state_slot_ids,
        binding.position_embedding,
        binding.selected_positions,
        output_row_start,
        int(caps.max_seq_len),
        int(caps.max_speculative_tokens),
        int(caps.compress_ratio),
        int(caps.budget),
        int(binding.plan.score_chunk_groups),
        int(binding.plan.score_workspace_width),
        int(binding.plan.num_score_chunks),
        int(layout.score_offset_bytes),
        int(layout.eligible_counts_offset_bytes),
        int(layout.merge_lengths_offset_bytes),
        int(layout.topk_values_offset_bytes),
        int(layout.topk_indices_offset_bytes),
        int(layout.topk_values_b_offset_bytes),
        int(layout.topk_indices_b_offset_bytes),
        int(layout.state_errors_offset_bytes),
        int(layout.request_errors_offset_bytes),
        int(layout.topk_offset_bytes),
    )
    return binding.selected_positions[output_row_start : output_row_start + rows]


def is_supported(device: torch.device | str | None = None) -> bool:
    """Return whether the SM120/SM121 Triton selector path is available."""
    from ..._lib.gating import default_is_supported
    from . import META

    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Binding",
    "CacheRequirements",
    "Caps",
    "Plan",
    "bind",
    "cache_requirements",
    "is_supported",
    "plan",
    "reset_state",
    "run",
    "run_prefill",
    "update_prefill_cache",
]
