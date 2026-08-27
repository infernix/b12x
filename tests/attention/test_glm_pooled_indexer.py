from __future__ import annotations

import math

import pytest
import torch

from b12x.attention import glm_pooled_indexer
from b12x.attention.glm_pooled_indexer.reference import (
    packed_stream_pool_reference,
    physical_element_offsets_reference,
    pool_group_reference,
    score_select_reference,
)

from ..conftest import require_b12x as require_sm120


def test_pool_group_uses_feature_wise_gate_plus_position_softmax() -> None:
    keys = torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=torch.bfloat16,
    )
    gate = torch.zeros_like(keys)
    ape = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 8.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )

    pooled = pool_group_reference(keys, gate, ape)
    expected_weights = torch.softmax(ape.float(), dim=0)
    expected = (keys.float() * expected_weights).sum(dim=0).to(torch.bfloat16)

    assert torch.equal(pooled, expected)
    assert float(pooled[0]) < 1.01
    assert 19.9 < float(pooled[1]) < 20.1
    assert 299.0 < float(pooled[2]) < 301.0


def test_packed_stream_pool_replaces_rejected_group_from_exact_tagged_state() -> None:
    dim = 8
    ring = torch.empty((8, dim), dtype=torch.bfloat16)
    gate_ring = torch.empty_like(ring)
    tags = torch.full((8,), -1, dtype=torch.int64)
    ape = torch.linspace(-0.4, 0.4, 4 * dim).reshape(4, dim).to(torch.bfloat16)
    original_keys = torch.arange(6 * dim, dtype=torch.float32).reshape(6, dim).to(
        torch.bfloat16
    )
    original_gates = torch.linspace(-1, 1, 6 * dim).reshape(6, dim).to(
        torch.bfloat16
    )

    group_ids, original_groups, anchor = packed_stream_pool_reference(
        original_keys,
        original_gates,
        torch.arange(6, dtype=torch.int64),
        ring,
        gate_ring,
        tags,
        prior_interval_start_position=-1,
        num_accepted_tokens=1,
        position_embedding=ape,
    )
    assert torch.equal(group_ids, torch.tensor([0], dtype=torch.int64))
    assert anchor == 0

    replacement_keys = torch.flip(original_keys[1:4], dims=(1,))
    replacement_gates = -original_gates[1:4]
    replacement_ids, replacement_groups, replacement_anchor = (
        packed_stream_pool_reference(
            replacement_keys,
            replacement_gates,
            torch.arange(1, 4, dtype=torch.int64),
            ring,
            gate_ring,
            tags,
            prior_interval_start_position=anchor,
            num_accepted_tokens=1,
            position_embedding=ape,
        )
    )

    assert torch.equal(replacement_ids, torch.tensor([0], dtype=torch.int64))
    assert replacement_anchor == 1
    expected = pool_group_reference(
        torch.cat((original_keys[:1], replacement_keys)),
        torch.cat((original_gates[:1], replacement_gates)),
        ape,
    )
    assert torch.equal(replacement_groups[0], expected)
    assert not torch.equal(replacement_groups[0], original_groups[0])
    assert torch.equal(tags[:4], torch.arange(4, dtype=torch.int64))


def test_score_selection_applies_learned_head_weights_and_both_scales() -> None:
    query = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]], dtype=torch.bfloat16)
    weights = torch.tensor([[3.0, -2.0]], dtype=torch.bfloat16)
    keys = torch.tensor([[1.0, 1.0], [2.0, -1.0]], dtype=torch.bfloat16)

    scores, selected = score_select_reference(
        query,
        weights,
        keys,
        torch.tensor([8], dtype=torch.int64),
        sequence_length=9,
    )
    dots = torch.einsum("hd,gd->hg", query[0].float(), keys.float()) / math.sqrt(2)
    expected = (torch.relu(dots) * weights[0, :, None].float()).sum(0) / math.sqrt(2)

    assert torch.allclose(scores[0], expected)
    assert torch.equal(selected[0, :9], torch.tensor([4, 5, 6, 7, 0, 1, 2, 3, 8]))
    assert torch.all(selected[0, 9:] == -1)


def test_score_selection_is_stable_at_512_groups_and_appends_raw_tail() -> None:
    query = torch.zeros((1, 32, 128), dtype=torch.bfloat16)
    weights = torch.ones((1, 32), dtype=torch.bfloat16)
    keys = torch.zeros((513, 128), dtype=torch.bfloat16)

    scores, selected = score_select_reference(
        query,
        weights,
        keys,
        torch.tensor([2052], dtype=torch.int64),
        sequence_length=2053,
    )

    assert torch.equal(scores, torch.zeros_like(scores))
    assert torch.equal(
        selected[0, :2048], torch.arange(2048, dtype=torch.int32)
    )
    assert int(selected[0, 2048]) == 2052
    assert torch.all(selected[0, 2049:] == -1)


def test_pool_scaled_physical_offsets_remain_int64_past_int32_boundary() -> None:
    pages = torch.tensor([0, 8_388_609], dtype=torch.int32)
    offsets = torch.tensor([0, 15], dtype=torch.int32)

    result = physical_element_offsets_reference(
        pages,
        offsets,
        page_stride_elements=4096,
        token_stride_elements=128,
    )

    assert result.dtype == torch.int64
    assert int(result[1]) == 8_388_609 * 4096 + 15 * 128
    assert int(result[1]) > torch.iinfo(torch.int32).max


def test_cache_requirements_fix_glm_geometry_and_speculative_capacity() -> None:
    requirements = glm_pooled_indexer.cache_requirements(
        compressed_page_size=16,
        max_speculative_tokens=4,
    )

    assert requirements.compressed_page_shape == (16, 128)
    assert requirements.raw_ring_capacity == 8
    assert requirements.raw_k_ring_shape == (8, 128)
    assert requirements.raw_gate_ring_shape == (8, 128)
    assert requirements.selection_width == 2051
    assert requirements.raw_state_slot_nbytes > 2 * 8 * 128 * 2


def _allocate_binding(
    *,
    device: torch.device,
    max_seq_len: int,
    compressed_page_size: int,
    max_q_rows: int = 1,
    block_table_width: int | None = None,
    cache_page_stride: int | None = None,
) -> glm_pooled_indexer.Binding:
    max_groups = max_seq_len // 4
    pages = math.ceil(max_groups / compressed_page_size)
    required_table_width = math.ceil(max_groups / compressed_page_size)
    table_width = (
        required_table_width
        if block_table_width is None
        else int(block_table_width)
    )
    page_stride = (
        compressed_page_size * 128
        if cache_page_stride is None
        else int(cache_page_stride)
    )
    caps = glm_pooled_indexer.Caps(
        device=device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=max_q_rows,
        max_seq_len=max_seq_len,
        num_compressed_cache_pages=pages,
        compressed_page_size=compressed_page_size,
    )
    plan = glm_pooled_indexer.plan(caps)
    (spec,) = plan.scratch_specs()
    cache_storage = torch.empty(
        pages * page_stride,
        dtype=torch.bfloat16,
        device=device,
    )
    compressed_k_cache = torch.as_strided(
        cache_storage,
        (pages, compressed_page_size, 128),
        (page_stride, 128, 1),
    )
    block_table = torch.full(
        (1, table_width), -1, dtype=torch.int32, device=device
    )
    block_table[0, :pages] = torch.arange(
        pages, dtype=torch.int32, device=device
    )
    return glm_pooled_indexer.bind(
        plan,
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device),
        compressed_k_cache=compressed_k_cache,
        compressed_block_table=block_table,
        raw_k_ring=torch.empty((1, 4, 128), dtype=torch.bfloat16, device=device),
        raw_gate_ring=torch.empty((1, 4, 128), dtype=torch.bfloat16, device=device),
        raw_logical_positions=torch.full(
            (1, 4), -1, dtype=torch.int64, device=device
        ),
        raw_interval_start_positions=torch.full(
            (1,), -1, dtype=torch.int64, device=device
        ),
        raw_state_slot_ids=torch.zeros((1,), dtype=torch.int64, device=device),
        position_embedding=torch.randn(
            (4, 128), dtype=torch.bfloat16, device=device
        ),
        selected_positions=torch.empty(
            (max_q_rows, 2051), dtype=torch.int32, device=device
        ),
    )


def test_gpu_decode_scores_513_groups_and_emits_fixed_width() -> None:
    device = require_sm120()
    binding = _allocate_binding(
        device=device, max_seq_len=2053, compressed_page_size=16
    )
    flat_keys = torch.randn((513, 128), dtype=torch.bfloat16, device=device)
    binding.compressed_k_cache.flatten(0, 1)[:513].copy_(flat_keys)
    binding.raw_interval_start_positions.fill_(2051)
    query = torch.randn((1, 32, 128), dtype=torch.bfloat16, device=device)
    head_weights = torch.randn((1, 32), dtype=torch.float32, device=device)

    selected = glm_pooled_indexer.run(
        binding,
        index_query=query,
        normalized_index_key=torch.randn(
            (1, 128), dtype=torch.bfloat16, device=device
        ),
        index_gate_logits=torch.randn(
            (1, 128), dtype=torch.bfloat16, device=device
        ),
        index_head_weights=head_weights,
        request_ids=torch.zeros((1,), dtype=torch.int32, device=device),
        query_positions=torch.tensor([2052], dtype=torch.int64, device=device),
        sequence_lengths=torch.tensor([2053], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        num_accepted_tokens=torch.ones((1,), dtype=torch.int32, device=device),
    )
    _, expected = score_select_reference(
        query.cpu(),
        head_weights.cpu(),
        flat_keys.cpu(),
        torch.tensor([2052], dtype=torch.int64),
        sequence_length=2053,
    )

    assert selected.shape == (1, 2051)
    assert torch.equal(selected.cpu(), expected)


def test_gpu_decode_pools_across_calls_without_rope_or_post_norm() -> None:
    device = require_sm120()
    binding = _allocate_binding(
        device=device, max_seq_len=8, compressed_page_size=2
    )
    keys = torch.randn((4, 128), dtype=torch.bfloat16, device=device)
    gates = torch.randn((4, 128), dtype=torch.bfloat16, device=device)
    query = torch.zeros((1, 32, 128), dtype=torch.bfloat16, device=device)
    head_weights = torch.ones((1, 32), dtype=torch.bfloat16, device=device)

    for position in range(4):
        selected = glm_pooled_indexer.run(
            binding,
            index_query=query,
            normalized_index_key=keys[position : position + 1],
            index_gate_logits=gates[position : position + 1],
            index_head_weights=head_weights,
            request_ids=torch.zeros((1,), dtype=torch.int32, device=device),
            query_positions=torch.tensor(
                [position], dtype=torch.int64, device=device
            ),
            sequence_lengths=torch.tensor(
                [position + 1], dtype=torch.int32, device=device
            ),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
            num_accepted_tokens=torch.ones(
                (1,), dtype=torch.int32, device=device
            ),
        )

    expected_pool = pool_group_reference(
        keys.cpu(), gates.cpu(), binding.position_embedding.cpu()
    )
    assert torch.allclose(
        binding.compressed_k_cache[0, 0].float().cpu(),
        expected_pool.float(),
        atol=2e-2,
        rtol=2e-2,
    )
    assert torch.equal(selected[0, :4].cpu(), torch.arange(4, dtype=torch.int32))
    assert torch.all(selected[0, 4:] == -1)


def test_gpu_prefill_seeds_cache_tail_and_decode_anchor_with_padded_pages() -> None:
    device = require_sm120()
    binding = _allocate_binding(
        device=device,
        max_seq_len=9,
        compressed_page_size=2,
        max_q_rows=6,
        block_table_width=2,
        cache_page_stride=1024,
    )
    keys = torch.randn((7, 128), dtype=torch.bfloat16, device=device)
    gates = torch.randn((7, 128), dtype=torch.bfloat16, device=device)
    query = torch.zeros((3, 32, 128), dtype=torch.bfloat16, device=device)
    head_weights = torch.ones((3, 32), dtype=torch.float32, device=device)
    binding.raw_logical_positions.fill_(1234)
    binding.raw_interval_start_positions.fill_(1234)
    glm_pooled_indexer.reset_state(
        binding,
        reset_mask=torch.ones((1,), dtype=torch.bool, device=device),
        prefix_lengths=torch.zeros((1,), dtype=torch.int32, device=device),
    )
    assert torch.all(binding.raw_logical_positions == -1)
    assert int(binding.raw_interval_start_positions[0]) == -1

    first_selected = glm_pooled_indexer.run_prefill(
        binding,
        index_query=query,
        normalized_index_key=keys[:3],
        index_gate_logits=gates[:3],
        index_head_weights=head_weights,
        request_ids=torch.zeros((3,), dtype=torch.int32, device=device),
        query_positions=torch.arange(3, dtype=torch.int64, device=device),
        sequence_lengths=torch.tensor([3], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32, device=device),
    )
    second_selected = glm_pooled_indexer.run_prefill(
        binding,
        index_query=query,
        normalized_index_key=keys[3:6],
        index_gate_logits=gates[3:6],
        index_head_weights=head_weights,
        request_ids=torch.zeros((3,), dtype=torch.int32, device=device),
        query_positions=torch.arange(3, 6, dtype=torch.int64, device=device),
        sequence_lengths=torch.tensor([6], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32, device=device),
        output_row_start=3,
    )

    for position in range(3):
        assert torch.equal(
            first_selected[position, : position + 1].cpu(),
            torch.arange(position + 1, dtype=torch.int32),
        )
        assert torch.all(first_selected[position, position + 1 :] == -1)
    for row, position in enumerate(range(3, 6)):
        assert torch.equal(
            second_selected[row, : position + 1].cpu(),
            torch.arange(position + 1, dtype=torch.int32),
        )
        assert torch.all(second_selected[row, position + 1 :] == -1)
    expected_pool = pool_group_reference(
        keys[:4].cpu(), gates[:4].cpu(), binding.position_embedding.cpu()
    )
    assert torch.allclose(
        binding.compressed_k_cache[0, 0].float().cpu(),
        expected_pool.float(),
        atol=2e-2,
        rtol=2e-2,
    )
    assert torch.equal(
        binding.raw_logical_positions[0].cpu(),
        torch.tensor([4, 5, 2, 3], dtype=torch.int64),
    )
    assert int(binding.raw_interval_start_positions[0]) == 5

    selected = glm_pooled_indexer.run(
        binding,
        index_query=torch.zeros((1, 32, 128), dtype=torch.bfloat16, device=device),
        normalized_index_key=keys[6:7],
        index_gate_logits=gates[6:7],
        index_head_weights=torch.ones((1, 32), dtype=torch.float32, device=device),
        request_ids=torch.zeros((1,), dtype=torch.int32, device=device),
        query_positions=torch.tensor([6], dtype=torch.int64, device=device),
        sequence_lengths=torch.tensor([7], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        num_accepted_tokens=torch.ones((1,), dtype=torch.int32, device=device),
    )
    assert torch.equal(
        selected[0, :7].cpu(), torch.arange(7, dtype=torch.int32)
    )
    assert torch.all(selected[0, 7:] == -1)


def test_gpu_pooled_cache_uses_int64_for_high_physical_page_offsets() -> None:
    device = require_sm120()
    compressed_page_size = 1
    page_stride_elements = compressed_page_size * 128
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
        compressed_cache = torch.empty(
            (num_pages, compressed_page_size, 128),
            dtype=torch.bfloat16,
            device=device,
        )
    except torch.OutOfMemoryError:
        pytest.skip(
            "CUDA allocator could not reserve the required mostly-uninitialized "
            f"{required_bytes}-byte high-page-id selector cache"
        )

    caps = glm_pooled_indexer.Caps(
        device=device,
        max_batch=1,
        max_raw_state_slots=1,
        max_q_rows=1,
        max_seq_len=4,
        num_compressed_cache_pages=num_pages,
        compressed_page_size=compressed_page_size,
    )
    plan = glm_pooled_indexer.plan(caps)
    (spec,) = plan.scratch_specs()
    prior_keys = torch.randn((3, 128), dtype=torch.bfloat16, device=device)
    prior_gates = torch.randn((3, 128), dtype=torch.bfloat16, device=device)
    raw_k_ring = torch.empty((1, 4, 128), dtype=torch.bfloat16, device=device)
    raw_gate_ring = torch.empty_like(raw_k_ring)
    raw_k_ring[0, :3].copy_(prior_keys)
    raw_gate_ring[0, :3].copy_(prior_gates)
    position_embedding = torch.randn(
        (4, 128), dtype=torch.bfloat16, device=device
    )
    binding = glm_pooled_indexer.bind(
        plan,
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device),
        compressed_k_cache=compressed_cache,
        compressed_block_table=torch.tensor(
            [[tail_page]], dtype=torch.int32, device=device
        ),
        raw_k_ring=raw_k_ring,
        raw_gate_ring=raw_gate_ring,
        raw_logical_positions=torch.tensor(
            [[0, 1, 2, -1]], dtype=torch.int64, device=device
        ),
        raw_interval_start_positions=torch.tensor(
            [2], dtype=torch.int64, device=device
        ),
        raw_state_slot_ids=torch.zeros((1,), dtype=torch.int64, device=device),
        position_embedding=position_embedding,
        selected_positions=torch.empty(
            (1, 2051), dtype=torch.int32, device=device
        ),
    )
    current_key = torch.randn((1, 128), dtype=torch.bfloat16, device=device)
    current_gate = torch.randn((1, 128), dtype=torch.bfloat16, device=device)
    selected = glm_pooled_indexer.run(
        binding,
        index_query=torch.zeros((1, 32, 128), dtype=torch.bfloat16, device=device),
        normalized_index_key=current_key,
        index_gate_logits=current_gate,
        index_head_weights=torch.ones((1, 32), dtype=torch.bfloat16, device=device),
        request_ids=torch.zeros((1,), dtype=torch.int32, device=device),
        query_positions=torch.tensor([3], dtype=torch.int64, device=device),
        sequence_lengths=torch.tensor([4], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        num_accepted_tokens=torch.ones((1,), dtype=torch.int32, device=device),
    )
    expected_pool = pool_group_reference(
        torch.cat((prior_keys, current_key)).cpu(),
        torch.cat((prior_gates, current_gate)).cpu(),
        position_embedding.cpu(),
    )

    assert tail_page * page_stride_elements >= 1 << 31
    torch.testing.assert_close(
        compressed_cache[tail_page, 0].float().cpu(),
        expected_pool.float(),
        rtol=2e-2,
        atol=2e-2,
    )
    assert torch.equal(selected[0, :4].cpu(), torch.arange(4, dtype=torch.int32))


def test_gpu_reset_state_starts_from_aligned_prefix_cache() -> None:
    device = require_sm120()
    binding = _allocate_binding(
        device=device,
        max_seq_len=8,
        compressed_page_size=2,
        max_q_rows=2,
    )
    binding.compressed_k_cache[0, 0].zero_()
    binding.raw_logical_positions.fill_(999)
    binding.raw_interval_start_positions.fill_(999)
    glm_pooled_indexer.reset_state(
        binding,
        reset_mask=torch.ones((1,), dtype=torch.bool, device=device),
        prefix_lengths=torch.tensor([4], dtype=torch.int64, device=device),
    )

    selected = glm_pooled_indexer.run_prefill(
        binding,
        index_query=torch.zeros((2, 32, 128), dtype=torch.bfloat16, device=device),
        normalized_index_key=torch.randn(
            (2, 128), dtype=torch.bfloat16, device=device
        ),
        index_gate_logits=torch.randn(
            (2, 128), dtype=torch.bfloat16, device=device
        ),
        index_head_weights=torch.ones(
            (2, 32), dtype=torch.bfloat16, device=device
        ),
        request_ids=torch.zeros((2,), dtype=torch.int32, device=device),
        query_positions=torch.tensor([4, 5], dtype=torch.int64, device=device),
        sequence_lengths=torch.tensor([6], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device=device),
    )

    assert torch.equal(selected[0, :5].cpu(), torch.arange(5, dtype=torch.int32))
    assert torch.equal(selected[1, :6].cpu(), torch.arange(6, dtype=torch.int32))
    assert torch.all(selected[0, 5:] == -1)
    assert torch.all(selected[1, 6:] == -1)
    assert int(binding.raw_interval_start_positions[0]) == 5
    assert torch.equal(
        binding.raw_logical_positions[0].cpu(),
        torch.tensor([4, 5, -1, -1], dtype=torch.int64),
    )


@pytest.mark.parametrize("bad", [127, 256])
def test_cache_requirements_reject_non_glm_index_dimensions(bad: int) -> None:
    with pytest.raises(ValueError, match="index_head_dim=128"):
        glm_pooled_indexer.cache_requirements(
            compressed_page_size=16, index_head_dim=bad
        )
