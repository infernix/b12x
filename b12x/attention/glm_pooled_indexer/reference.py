"""Allocation-heavy mathematical oracles for the GLM pooled indexer.

Nothing in the GPU execution path dispatches functions from this module.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pool_group_reference(
    normalized_keys: torch.Tensor,
    gate_logits: torch.Tensor,
    position_embedding: torch.Tensor,
) -> torch.Tensor:
    """Pool one four-token group with a feature-wise learned softmax."""
    if normalized_keys.ndim != 2 or int(normalized_keys.shape[0]) != 4:
        raise ValueError("normalized_keys must have shape [4, index_head_dim]")
    if tuple(gate_logits.shape) != tuple(normalized_keys.shape):
        raise ValueError("gate_logits must match normalized_keys")
    if tuple(position_embedding.shape) != tuple(normalized_keys.shape):
        raise ValueError("position_embedding must match normalized_keys")
    if normalized_keys.dtype != torch.bfloat16:
        raise TypeError("normalized_keys must use torch.bfloat16")
    weights = F.softmax(
        gate_logits.float() + position_embedding.float(), dim=0
    )
    return (normalized_keys.float() * weights).sum(dim=0).to(torch.bfloat16)


def packed_stream_pool_reference(
    normalized_keys: torch.Tensor,
    gate_logits: torch.Tensor,
    logical_positions: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_gate_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    *,
    prior_interval_start_position: int,
    num_accepted_tokens: int,
    position_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pool newly completed groups and commit one packed decode interval.

    The old ring is consumed before any current rows are committed.  Exact
    logical tags make replacement intervals safe after speculative rejection.
    """
    if normalized_keys.ndim != 2:
        raise ValueError("normalized_keys must have shape [rows, index_head_dim]")
    rows, head_dim = map(int, normalized_keys.shape)
    if rows <= 0:
        raise ValueError("the packed interval must contain at least one row")
    if tuple(gate_logits.shape) != (rows, head_dim):
        raise ValueError("gate_logits must match normalized_keys")
    if tuple(logical_positions.shape) != (rows,):
        raise ValueError("logical_positions must have shape [rows]")
    if logical_positions.dtype != torch.int64:
        raise TypeError("logical_positions must use torch.int64")
    capacity = int(raw_k_ring.shape[0])
    if tuple(raw_k_ring.shape) != (capacity, head_dim):
        raise ValueError("raw_k_ring must have shape [capacity, index_head_dim]")
    if tuple(raw_gate_ring.shape) != (capacity, head_dim):
        raise ValueError("raw_gate_ring must match raw_k_ring")
    if tuple(raw_logical_positions.shape) != (capacity,):
        raise ValueError("raw_logical_positions must have shape [capacity]")
    if capacity < 4:
        raise ValueError("raw rings must cover one four-token group")
    if tuple(position_embedding.shape) != (4, head_dim):
        raise ValueError("position_embedding must have shape [4, index_head_dim]")
    if not 1 <= int(num_accepted_tokens) <= rows:
        raise ValueError("num_accepted_tokens must be within the packed interval")

    current_first = int(logical_positions[0].item())
    expected_positions = torch.arange(
        current_first,
        current_first + rows,
        dtype=torch.int64,
        device=logical_positions.device,
    )
    if current_first < 0 or not torch.equal(logical_positions, expected_positions):
        raise ValueError("logical positions must be nonnegative and contiguous")
    if int(prior_interval_start_position) == -1 and (
        int(num_accepted_tokens) != 1 or current_first != 0
    ):
        raise ValueError(
            "prior interval start -1 is reserved for one accepted token at position zero"
        )
    if current_first != int(prior_interval_start_position) + int(
        num_accepted_tokens
    ):
        raise ValueError(
            "current interval start must equal prior interval start plus num_accepted_tokens"
        )

    group_ids: list[int] = []
    representatives: list[torch.Tensor] = []
    for row in range(rows):
        position = int(logical_positions[row].item())
        if (position + 1) % 4:
            continue
        group_first = position - 3
        source_keys: list[torch.Tensor] = []
        source_gates: list[torch.Tensor] = []
        for source_position in range(group_first, position + 1):
            if source_position >= current_first:
                source_row = source_position - current_first
                source_keys.append(normalized_keys[source_row])
                source_gates.append(gate_logits[source_row])
            else:
                source_slot = source_position % capacity
                if (
                    int(raw_logical_positions[source_slot].item())
                    != source_position
                ):
                    raise RuntimeError(
                        "raw selector ring is missing accepted tagged history for a completed group"
                    )
                source_keys.append(raw_k_ring[source_slot])
                source_gates.append(raw_gate_ring[source_slot])
        representatives.append(
            pool_group_reference(
                torch.stack(source_keys),
                torch.stack(source_gates),
                position_embedding,
            )
        )
        group_ids.append(position // 4)

    for row in range(rows):
        position = int(logical_positions[row].item())
        slot = position % capacity
        raw_k_ring[slot].copy_(normalized_keys[row])
        raw_gate_ring[slot].copy_(gate_logits[row])
        raw_logical_positions[slot] = position

    if representatives:
        representative_tensor = torch.stack(representatives)
        group_id_tensor = torch.tensor(
            group_ids, dtype=torch.int64, device=normalized_keys.device
        )
    else:
        representative_tensor = normalized_keys.new_empty((0, head_dim))
        group_id_tensor = logical_positions.new_empty((0,), dtype=torch.int64)
    return group_id_tensor, representative_tensor, current_first


def paged_store_compressed_reference(
    compressed_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_id: int,
    group_ids: torch.Tensor,
    representatives: torch.Tensor,
) -> None:
    """Store group representatives through a request-relative page table."""
    if compressed_cache.ndim != 3:
        raise ValueError("compressed_cache must have shape [pages, page, dim]")
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [requests, logical_pages]")
    if group_ids.dtype != torch.int64:
        raise TypeError("group_ids must use int64")
    if tuple(representatives.shape) != (
        int(group_ids.numel()),
        int(compressed_cache.shape[-1]),
    ):
        raise ValueError("representatives shape does not match group_ids/cache")
    page_size = int(compressed_cache.shape[1])
    for index, group_id_value in enumerate(group_ids):
        group_id = int(group_id_value.item())
        logical_page, page_offset = divmod(group_id, page_size)
        if not 0 <= logical_page < int(block_table.shape[1]):
            raise IndexError("compressed logical page is outside the page table")
        physical_page = int(block_table[int(request_id), logical_page].item())
        if not 0 <= physical_page < int(compressed_cache.shape[0]):
            raise IndexError("compressed page table contains an invalid physical page")
        compressed_cache[physical_page, page_offset].copy_(representatives[index])


def score_select_reference(
    index_query: torch.Tensor,
    index_head_weights: torch.Tensor,
    compressed_keys: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_length: int,
    *,
    budget: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score complete groups and emit fixed-width original-token positions."""
    if index_query.ndim != 3:
        raise ValueError("index_query must have shape [rows, heads, dim]")
    rows, heads, head_dim = map(int, index_query.shape)
    if tuple(index_head_weights.shape) != (rows, heads):
        raise ValueError("index_head_weights must have shape [rows, heads]")
    if compressed_keys.ndim != 2 or int(compressed_keys.shape[1]) != head_dim:
        raise ValueError("compressed_keys must have shape [groups, index_head_dim]")
    if tuple(query_positions.shape) != (rows,):
        raise ValueError("query_positions must have shape [rows]")
    if heads <= 0 or head_dim <= 0:
        raise ValueError("index query geometry must be positive")
    if int(budget) != 2048:
        raise ValueError("GLM pooled selection requires a 2048-token budget")

    group_budget = 512
    selected = torch.full(
        (rows, 2051), -1, dtype=torch.int32, device=index_query.device
    )
    scores_out = torch.full(
        (rows, int(compressed_keys.shape[0])),
        -torch.inf,
        dtype=torch.float32,
        device=index_query.device,
    )
    for row in range(rows):
        position = int(query_positions[row].item())
        eligible = min(
            (position + 1) // 4,
            int(sequence_length) // 4,
            int(compressed_keys.shape[0]),
        )
        if eligible:
            dots = torch.einsum(
                "hd,gd->hg",
                index_query[row].float(),
                compressed_keys[:eligible].float(),
            ) / math.sqrt(head_dim)
            scores = (
                F.relu(dots) * index_head_weights[row].float()[:, None]
            ).sum(dim=0) / math.sqrt(heads)
            scores_out[row, :eligible] = scores
            count = min(group_budget, eligible)
            group_ids = torch.argsort(scores, descending=True, stable=True)[:count]
            expanded = (
                group_ids[:, None] * 4
                + torch.arange(4, dtype=torch.int64, device=index_query.device)[
                    None, :
                ]
            ).flatten()
        else:
            expanded = query_positions.new_empty((0,), dtype=torch.int64)
        tail_start = ((position + 1) // 4) * 4
        tail = torch.arange(
            tail_start,
            position + 1,
            dtype=torch.int64,
            device=index_query.device,
        )
        positions = torch.cat((expanded, tail))
        selected[row, : int(positions.numel())] = positions.to(torch.int32)
    return scores_out, selected


def physical_element_offsets_reference(
    physical_pages: torch.Tensor,
    page_offsets: torch.Tensor,
    *,
    page_stride_elements: int,
    token_stride_elements: int,
) -> torch.Tensor:
    """Return page-scaled offsets using the serving-required Int64 math."""
    if physical_pages.shape != page_offsets.shape:
        raise ValueError("physical_pages and page_offsets must have the same shape")
    return physical_pages.to(torch.int64) * int(
        page_stride_elements
    ) + page_offsets.to(torch.int64) * int(token_stride_elements)


__all__ = [
    "packed_stream_pool_reference",
    "paged_store_compressed_reference",
    "physical_element_offsets_reference",
    "pool_group_reference",
    "score_select_reference",
]
