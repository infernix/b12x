"""Lightweight dispatch contract for the QSA CuTe sparse GQA core."""

from __future__ import annotations

from functools import lru_cache

import torch

HEAD_DIM = 256
PAGE_SIZE = 16
SELECTION_WIDTH = 2051
BLOCK_N = 16
NUM_SPLITS = 64
SUPPORTED_HEAD_LAYOUTS = frozenset({(6, 1), (12, 1), (24, 2)})
SUPPORTED_ROW_LIMITS = {
    (6, 1): 8,
    (12, 1): 4,
    (24, 2): 2,
}


@lru_cache(maxsize=None)
def _is_sm120(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (12, 0)


def _is_page_token_head_layout(tensor: torch.Tensor) -> bool:
    """Accept a strided outer page with non-overlapping inner cache rows."""
    if tensor.ndim != 4 or int(tensor.stride(3)) != 1:
        return False
    _, page_size, kv_heads, head_dim = map(int, tensor.shape)
    page_stride, token_stride, head_stride, _ = map(int, tensor.stride())
    if min(page_stride, token_stride, head_stride) <= 0:
        return False
    head_span = head_dim
    token_span = (kv_heads - 1) * head_stride + head_span
    page_span = (page_size - 1) * token_stride + token_span
    return (
        head_stride >= head_span
        and token_stride >= token_span
        and page_stride >= page_span
    )


def is_candidate(
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
    """Reject unqualified geometry without importing the CuTe implementation."""
    if (
        query.ndim != 3
        or key_cache.ndim != 4
        or value_cache.ndim != 4
        or block_table.ndim != 2
        or request_ids.ndim != 1
        or selected_positions.ndim != 2
        or query_positions.ndim != 1
        or partial_output.ndim != 4
        or partial_lse.ndim != 3
    ):
        return False
    rows, q_heads, head_dim = map(int, query.shape)
    head_layout = (q_heads, int(key_cache.shape[2]))
    if not (
        rows > 0
        and head_layout in SUPPORTED_HEAD_LAYOUTS
        and rows <= SUPPORTED_ROW_LIMITS[head_layout]
        and head_dim == HEAD_DIM
        and int(key_cache.shape[0]) > 0
        and int(key_cache.shape[1]) == PAGE_SIZE
        and int(key_cache.shape[3]) == HEAD_DIM
        and int(selected_positions.shape[0]) >= rows
        and tuple(selected_positions.shape[1:]) == (SELECTION_WIDTH,)
        and int(block_n) == BLOCK_N
        and int(splits) == NUM_SPLITS
    ):
        return False
    if not query.is_cuda or not _is_sm120(query.device):
        return False
    if (
        query.dtype != torch.bfloat16
        or key_cache.dtype != torch.bfloat16
        or value_cache.dtype != torch.bfloat16
        or block_table.dtype != torch.int32
        or request_ids.dtype not in (torch.int32, torch.int64)
        or selected_positions.dtype != torch.int32
        or query_positions.dtype != torch.int64
        or partial_output.dtype != torch.float32
        or partial_lse.dtype != torch.float32
    ):
        return False
    contiguous_tensors = (
        query,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        partial_output,
        partial_lse,
    )
    return (
        all(
            tensor.device == query.device and tensor.is_contiguous()
            for tensor in contiguous_tensors
        )
        and key_cache.device == query.device
        and value_cache.device == query.device
        and tuple(value_cache.shape) == tuple(key_cache.shape)
        and _is_page_token_head_layout(key_cache)
        and _is_page_token_head_layout(value_cache)
        and tuple(block_table.shape)[0] > 0
        and tuple(request_ids.shape) == (rows,)
        and tuple(query_positions.shape) == (rows,)
        and int(partial_output.shape[0]) >= rows
        and tuple(partial_output.shape[1:]) == (NUM_SPLITS, q_heads, HEAD_DIM)
        and int(partial_lse.shape[0]) >= rows
        and tuple(partial_lse.shape[1:]) == (NUM_SPLITS, q_heads)
    )


def clear_device_cache() -> None:
    _is_sm120.cache_clear()


__all__ = [
    "BLOCK_N",
    "HEAD_DIM",
    "NUM_SPLITS",
    "PAGE_SIZE",
    "SELECTION_WIDTH",
    "SUPPORTED_HEAD_LAYOUTS",
    "SUPPORTED_ROW_LIMITS",
    "clear_device_cache",
    "is_candidate",
]
