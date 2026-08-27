"""Lightweight dispatch contract for the MTP CuTe prefill projections."""

from __future__ import annotations

from functools import lru_cache

import torch


TMA_ALIGNMENT_BYTES = 16
QUALIFIED_TOKENS = frozenset({1, 2, 4, 8, 16, 64, 96, 128, 512, 4_096})


@lru_cache(maxsize=None)
def _is_sm120(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (12, 0)


def supports_prefill(
    *,
    tokens: int,
    streams: int,
    hidden_size: int,
    device: torch.device | None = None,
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Return whether a measured Qwen3.8 projection geometry applies."""
    token_count = int(tokens)
    if (
        token_count not in QUALIFIED_TOKENS
        or int(streams) != 4
        or int(hidden_size) != 2_560
    ):
        return False
    if compute_capability is None:
        if device is None:
            raise ValueError("device or compute_capability is required")
        return _is_sm120(device)
    return tuple(map(int, compute_capability)) == (12, 0)


def tensors_support_prefill(*tensors: torch.Tensor) -> bool:
    """Return whether tensors satisfy the contiguous TMA pointer contract."""
    return all(
        type(tensor) is torch.Tensor
        and tensor.is_cuda
        and tensor.is_contiguous()
        and int(tensor.data_ptr()) % TMA_ALIGNMENT_BYTES == 0
        for tensor in tensors
    )


__all__ = [
    "QUALIFIED_TOKENS",
    "TMA_ALIGNMENT_BYTES",
    "supports_prefill",
    "tensors_support_prefill",
]
