from __future__ import annotations

import math

import pytest
import torch

from b12x.attention.qsa._sparse_gqa import launch_sparse_paged_gqa
from b12x.attention.qsa import _sparse_gqa_cute_config as cute_config

from ..conftest import require_b12x as require_sm120


def _dense_gathered_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    rows, q_heads, head_dim = map(int, query.shape)
    page_size = int(key_cache.shape[1])
    kv_heads = int(key_cache.shape[2])
    heads_per_kv = q_heads // kv_heads
    result = torch.zeros_like(query)
    for row in range(rows):
        request_id = int(request_ids[row].item())
        query_position = int(query_positions[row].item())
        if request_id < 0 or request_id >= int(block_table.shape[0]):
            continue
        logical_positions = [
            int(position)
            for position in selected_positions[row].detach().cpu().tolist()
            if 0 <= int(position) <= query_position
        ]
        for query_head in range(q_heads):
            kv_head = query_head // heads_per_kv
            keys = []
            values = []
            for logical_position in logical_positions:
                logical_page = logical_position // page_size
                if logical_page >= int(block_table.shape[1]):
                    continue
                physical_page = int(block_table[request_id, logical_page].item())
                if physical_page < 0 or physical_page >= int(key_cache.shape[0]):
                    continue
                page_offset = logical_position % page_size
                keys.append(key_cache[physical_page, page_offset, kv_head])
                values.append(value_cache[physical_page, page_offset, kv_head])
            if not keys:
                continue
            gathered_key = torch.stack(keys).float()
            gathered_value = torch.stack(values).float()
            scores = (query[row, query_head].float() @ gathered_key.T) * softmax_scale
            result[row, query_head] = (
                torch.softmax(scores, dim=-1) @ gathered_value
            ).to(torch.bfloat16)
    return result


def _cache_layout(
    *,
    pages: int,
    page_size: int,
    kv_heads: int,
    head_dim: int,
    layout: str,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    def make() -> torch.Tensor:
        if layout == "contiguous":
            return torch.randn(
                (pages, page_size, kv_heads, head_dim),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=device, dtype=torch.bfloat16)
        if layout == "page_transposed":
            storage = torch.randn(
                (page_size, pages, kv_heads, head_dim),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=device, dtype=torch.bfloat16)
            return storage.permute(1, 0, 2, 3)
        if layout == "padded_inner":
            storage = torch.randn(
                (pages, page_size, kv_heads, head_dim + 8),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=device, dtype=torch.bfloat16)
            return storage[..., :head_dim]
        if layout == "interleaved_page":
            storage = torch.randn(
                (pages, 3, page_size, kv_heads, head_dim),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=device, dtype=torch.bfloat16)
            return storage[:, 1]
        raise AssertionError(f"unknown cache layout {layout}")

    return make(), make()


class _CandidateTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        strides: tuple[int, ...] | None = None,
        contiguous: bool = True,
    ) -> None:
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = torch.device("cuda", 0)
        self.is_cuda = True
        self._contiguous = contiguous
        if strides is None:
            running = 1
            reversed_strides = []
            for extent in reversed(shape):
                reversed_strides.append(running)
                running *= extent
            strides = tuple(reversed(reversed_strides))
        self._strides = strides

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def stride(self, dim: int | None = None) -> tuple[int, ...] | int:
        if dim is None:
            return self._strides
        return self._strides[dim]


@pytest.mark.parametrize(
    ("rows", "q_heads", "kv_heads", "expected"),
    [
        (2, 6, 1, True),
        (8, 6, 1, True),
        (9, 6, 1, False),
        (4, 12, 1, True),
        (5, 12, 1, False),
        (2, 24, 2, True),
        (4, 24, 2, False),
    ],
)
def test_cute_candidate_uses_qualified_rows_for_interleaved_blh_cache_views(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    q_heads: int,
    kv_heads: int,
    expected: bool,
) -> None:
    monkeypatch.setattr(cute_config, "_is_sm120", lambda _device: True)
    pages, layers = 128, 64
    allocation = torch.empty(
        (
            pages,
            layers,
            2,
            cute_config.PAGE_SIZE,
            kv_heads * cute_config.HEAD_DIM,
        ),
        dtype=torch.bfloat16,
        device="meta",
    )
    layer_cache = allocation[:, layers // 2]
    key_view, value_view = (
        side.unflatten(-1, (kv_heads, cute_config.HEAD_DIM))
        for side in layer_cache.unbind(1)
    )
    assert not key_view.is_contiguous()
    assert not value_view.is_contiguous()
    query = _CandidateTensor((rows, q_heads, cute_config.HEAD_DIM), torch.bfloat16)
    key_cache = _CandidateTensor(
        (pages, cute_config.PAGE_SIZE, kv_heads, cute_config.HEAD_DIM),
        torch.bfloat16,
        strides=key_view.stride(),
        contiguous=False,
    )
    value_cache = _CandidateTensor(
        (pages, cute_config.PAGE_SIZE, kv_heads, cute_config.HEAD_DIM),
        torch.bfloat16,
        strides=value_view.stride(),
        contiguous=False,
    )

    assert cute_config.is_candidate(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=_CandidateTensor((4, 128), torch.int32),
        request_ids=_CandidateTensor((rows,), torch.int64),
        selected_positions=_CandidateTensor(
            (rows, cute_config.SELECTION_WIDTH), torch.int32
        ),
        query_positions=_CandidateTensor((rows,), torch.int64),
        partial_output=_CandidateTensor(
            (
                rows,
                cute_config.NUM_SPLITS,
                q_heads,
                cute_config.HEAD_DIM,
            ),
            torch.float32,
        ),
        partial_lse=_CandidateTensor(
            (rows, cute_config.NUM_SPLITS, q_heads), torch.float32
        ),
        block_n=cute_config.BLOCK_N,
        splits=cute_config.NUM_SPLITS,
    ) is expected


@pytest.mark.parametrize(
    (
        "rows",
        "q_heads",
        "kv_heads",
        "head_dim",
        "page_size",
        "selection_width",
        "block_n",
        "splits",
        "layout",
    ),
    [
        (1, 24, 2, 256, 16, 2051, 16, 64, "contiguous"),
        (1, 6, 1, 256, 16, 2051, 16, 64, "interleaved_page"),
        (3, 8, 2, 64, 4, 67, 16, 4, "page_transposed"),
        (2, 6, 3, 32, 8, 65, 64, 1, "padded_inner"),
    ],
)
def test_sparse_gqa_matches_gathered_dense_reference(
    rows: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    selection_width: int,
    block_n: int,
    splits: int,
    layout: str,
) -> None:
    device = require_sm120()
    generator = torch.Generator(device="cpu").manual_seed(
        92000 + rows + q_heads + head_dim
    )
    batches = 3
    table_width = 6
    pages = batches * table_width
    key_cache, value_cache = _cache_layout(
        pages=pages,
        page_size=page_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        layout=layout,
        device=device,
        generator=generator,
    )
    block_table = torch.stack(
        [
            torch.randperm(pages, generator=generator, dtype=torch.int64)[:table_width]
            for _ in range(batches)
        ]
    ).to(device=device, dtype=torch.int32)
    query = torch.randn(
        (rows, q_heads, head_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=torch.bfloat16)
    request_ids = (
        torch.arange(rows, dtype=torch.int64, device=device) % batches
    ).contiguous()
    query_positions = torch.tensor(
        [table_width * page_size - 2 - row for row in range(rows)],
        dtype=torch.int64,
        device=device,
    )
    selected_positions = torch.full(
        (rows + 2, selection_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    logical_capacity = table_width * page_size
    for row in range(rows):
        candidates = torch.randperm(
            logical_capacity, generator=generator, dtype=torch.int64
        )
        count = min(selection_width, logical_capacity)
        selected_positions[row, :count].copy_(
            candidates[:count].to(device=device, dtype=torch.int32)
        )

    output = torch.empty(
        (rows + 2, q_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    partial_output = (
        torch.empty(
            (rows, splits, q_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )
        if splits > 1
        else None
    )
    partial_lse = (
        torch.empty((rows, splits, q_heads), dtype=torch.float32, device=device)
        if splits > 1
        else None
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)
    actual = launch_sparse_paged_gqa(
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
    expected = _dense_gathered_reference(
        query,
        key_cache,
        value_cache,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        softmax_scale,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-2)


def test_sparse_gqa_zeroes_padded_request_and_all_masked_rows() -> None:
    device = require_sm120()
    rows, q_heads, kv_heads, head_dim = 2, 24, 2, 256
    query = torch.randn((rows, q_heads, head_dim), dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(
        (2, 4, kv_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    request_ids = torch.tensor([-1, 0], dtype=torch.int64, device=device)
    query_positions = torch.tensor([7, 0], dtype=torch.int64, device=device)
    selected_positions = torch.full((rows, 17), -1, dtype=torch.int32, device=device)
    selected_positions[1, :2] = torch.tensor([2, 3], dtype=torch.int32, device=device)
    output = torch.full(
        (rows, q_heads, head_dim),
        17,
        dtype=torch.bfloat16,
        device=device,
    )
    actual = launch_sparse_paged_gqa(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=output,
        partial_output=None,
        partial_lse=None,
        softmax_scale=1.0 / math.sqrt(head_dim),
        block_n=16,
        splits=1,
    )
    assert torch.count_nonzero(actual).item() == 0


def test_sparse_gqa_split_path_is_cuda_graph_replay_safe() -> None:
    device = require_sm120()
    rows, q_heads, kv_heads, head_dim = 1, 24, 2, 256
    page_size, selection_width, splits = 16, 2051, 64
    query = torch.randn((rows, q_heads, head_dim), dtype=torch.bfloat16, device=device)
    key_cache = torch.randn(
        (8, page_size, kv_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(8, dtype=torch.int32, device=device).view(1, 8)
    request_ids = torch.zeros((rows,), dtype=torch.int64, device=device)
    query_positions = torch.tensor([95], dtype=torch.int64, device=device)
    selected_positions = torch.full(
        (rows, selection_width), -1, dtype=torch.int32, device=device
    )
    selected_positions[0, :96] = torch.randperm(
        96, dtype=torch.int64, device=device
    ).to(torch.int32)
    output = torch.empty_like(query)
    partial_output = torch.empty(
        (rows, splits, q_heads, head_dim), dtype=torch.float32, device=device
    )
    partial_lse = torch.empty(
        (rows, splits, q_heads), dtype=torch.float32, device=device
    )
    scale = 1.0 / math.sqrt(head_dim)

    launch_sparse_paged_gqa(
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
        softmax_scale=scale,
        block_n=16,
        splits=splits,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = launch_sparse_paged_gqa(
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
            softmax_scale=scale,
            block_n=16,
            splits=splits,
        )

    query.copy_(torch.randn_like(query))
    expected = _dense_gathered_reference(
        query,
        key_cache,
        value_cache,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        scale,
    )
    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_output.data_ptr() == output.data_ptr()
    torch.testing.assert_close(captured_output, expected, rtol=0.0, atol=2e-2)


def test_sparse_gqa_uses_int64_for_high_physical_page_offsets() -> None:
    device = require_sm120()
    rows, q_heads, kv_heads, head_dim = 1, 24, 2, 256
    page_size = 16
    page_stride_elements = page_size * kv_heads * head_dim
    tail_page = math.ceil((1 << 31) / page_stride_elements)
    num_pages = tail_page + 1
    required_bytes = num_pages * page_stride_elements * torch.bfloat16.itemsize
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reserve_bytes = 2 * 1024**3
    if free_bytes < required_bytes + reserve_bytes:
        pytest.skip(
            "high-page-id live allocation requires "
            f"{required_bytes + reserve_bytes} bytes free, found {free_bytes}"
        )
    try:
        cache = torch.empty(
            (num_pages, page_size, kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
    except torch.OutOfMemoryError:
        pytest.skip(
            "CUDA allocator could not reserve the required mostly-uninitialized "
            f"{required_bytes}-byte high-page-id cache"
        )

    live_value = (
        torch.linspace(
            -1.0,
            1.0,
            kv_heads * head_dim,
            dtype=torch.float32,
            device=device,
        )
        .view(kv_heads, head_dim)
        .to(torch.bfloat16)
    )
    cache[tail_page, 0].copy_(live_value)
    query = torch.randn((rows, q_heads, head_dim), dtype=torch.bfloat16, device=device)
    block_table = torch.tensor([[tail_page]], dtype=torch.int32, device=device)
    request_ids = torch.zeros((rows,), dtype=torch.int64, device=device)
    query_positions = torch.zeros((rows,), dtype=torch.int64, device=device)
    selected_positions = torch.full((rows, 2051), -1, dtype=torch.int32, device=device)
    selected_positions[0, 0] = 0
    output = torch.empty_like(query)
    splits = 64
    partial_output = torch.empty(
        (rows, splits, q_heads, head_dim), dtype=torch.float32, device=device
    )
    partial_lse = torch.empty(
        (rows, splits, q_heads), dtype=torch.float32, device=device
    )
    actual = launch_sparse_paged_gqa(
        query=query,
        key_cache=cache,
        value_cache=cache,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=output,
        partial_output=partial_output,
        partial_lse=partial_lse,
        softmax_scale=1.0 / math.sqrt(head_dim),
        block_n=16,
        splits=splits,
    )
    expected = torch.stack(
        [live_value[head // (q_heads // kv_heads)] for head in range(q_heads)]
    ).unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
