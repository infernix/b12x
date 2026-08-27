from __future__ import annotations

import gc

import pytest
import torch

from b12x.sequence import gdn_decode as gdn

from ..conftest import require_b12x as require_sm120


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    scale: float = 0.25,
) -> torch.Tensor:
    return (
        torch.randn(shape, dtype=torch.float32, device=device)
        .mul_(scale)
        .to(dtype)
        .contiguous()
    )


def _make_case(
    *,
    device: torch.device,
    query_lengths: tuple[int, ...] = (1, 1),
    max_tokens: int | None = None,
    max_seqs: int | None = None,
    state_slots: int | None = None,
    key_heads: int = 1,
    value_heads: int = 3,
    columns: int | None = None,
    accepted: tuple[int, ...] | None = None,
    activation: str = "sigmoid",
    state_dtype: torch.dtype = torch.float32,
    norm_dtype: torch.dtype = torch.bfloat16,
    qk_l2norm: bool = True,
) -> tuple[gdn.Binding, dict[str, torch.Tensor]]:
    live_seqs = len(query_lengths)
    columns = max(1, max(query_lengths, default=0)) if columns is None else columns
    max_seqs = live_seqs if max_seqs is None else max_seqs
    live_tokens = sum(query_lengths)
    max_tokens = live_tokens if max_tokens is None else max_tokens
    state_slots = max_seqs * columns + 1 if state_slots is None else state_slots
    if accepted is None:
        accepted = (1,) * live_seqs
    caps = gdn.Caps(
        device=device,
        max_tokens=max_tokens,
        max_seqs=max_seqs,
        max_state_slots=state_slots,
        key_heads=key_heads,
        value_heads=value_heads,
        state_index_columns=columns,
        state_dtype=state_dtype,
        gate_activation=activation,
        qk_l2norm=qk_l2norm,
    )
    plan = gdn.plan(caps)
    (scratch_spec,) = plan.scratch_specs()
    query_start_loc = torch.full(
        (max_seqs + 1,), live_tokens, dtype=torch.int32, device=device
    )
    query_start_loc[0] = 0
    if live_seqs:
        query_start_loc[1 : live_seqs + 1].copy_(
            torch.tensor(query_lengths, dtype=torch.int32, device=device).cumsum(0)
        )
    num_accepted_tokens = torch.ones(max_seqs, dtype=torch.int32, device=device)
    if live_seqs:
        num_accepted_tokens[:live_seqs].copy_(
            torch.tensor(accepted, dtype=torch.int32, device=device)
        )
    state_indices = torch.arange(
        max_seqs * columns, dtype=torch.int32, device=device
    ).view(max_seqs, columns)
    state_indices.remainder_(state_slots)
    tensors = {
        "scratch": torch.empty(
            scratch_spec.shape,
            dtype=scratch_spec.dtype,
            device=device,
        ),
        "mixed_qkv": _randn((max_tokens, caps.packed_qkv_width), device=device),
        "a": _randn((max_tokens, value_heads), device=device),
        "b": _randn((max_tokens, value_heads), device=device),
        "z": _randn((max_tokens, value_heads, 128), device=device),
        "A_log": _randn((value_heads,), device=device, dtype=torch.float32, scale=0.1),
        "dt_bias": _randn(
            (value_heads,), device=device, dtype=torch.float32, scale=0.1
        ),
        "norm_weight": (
            1.0 + _randn((128,), device=device, dtype=norm_dtype, scale=0.05)
        ).contiguous(),
        "recurrent_state": _randn(
            (state_slots, value_heads, 128, 128),
            device=device,
            dtype=state_dtype,
            scale=0.1,
        ),
        "query_start_loc": query_start_loc,
        "num_accepted_tokens": num_accepted_tokens,
        "state_indices": state_indices,
        "num_seqs": torch.tensor([live_seqs], dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([live_tokens], dtype=torch.int32, device=device),
        "output": torch.full(
            (max_tokens, value_heads, 128),
            7.0,
            dtype=torch.bfloat16,
            device=device,
        ),
    }
    binding = gdn.bind(plan, **tensors)
    return binding, tensors


def _reference(binding: gdn.Binding, state: torch.Tensor) -> torch.Tensor:
    caps = binding.plan.caps
    return gdn.reference.decode(
        binding.mixed_qkv,
        binding.a,
        binding.b,
        binding.z,
        binding.A_log,
        binding.dt_bias,
        binding.norm_weight,
        state,
        binding.query_start_loc,
        binding.num_accepted_tokens,
        binding.state_indices,
        binding.num_seqs,
        binding.num_tokens,
        key_heads=caps.key_heads,
        value_heads=caps.value_heads,
        gate_activation=caps.gate_activation,
        qk_l2norm=caps.qk_l2norm,
    )


def test_research_qwen_cute_stages_are_graph_safe_and_correct() -> None:
    from b12x.sequence.gdn_decode._cute_kernels import (
        run_gated_rmsnorm,
        run_packed_recurrent_qwen,
        run_qwen_validation,
    )

    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(1, 1, 1, 1),
        key_heads=8,
        value_heads=24,
        activation="sigmoid",
        state_dtype=torch.float32,
    )
    initial_state = binding.recurrent_state.clone()
    expected_state = initial_state.clone()
    expected = _reference(binding, expected_state)

    def launch() -> None:
        run_qwen_validation(binding)
        run_packed_recurrent_qwen(binding)
        run_gated_rmsnorm(binding, eps=1.0e-6)

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    binding.recurrent_state.copy_(initial_state)
    binding.output.fill_(float("nan"))
    binding.scratch.fill_(0xFF)
    addresses = (
        binding.recurrent_state.data_ptr(),
        binding.output.data_ptr(),
        binding.scratch.data_ptr(),
    )
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert allocated_after == allocated_before
    assert addresses == (
        binding.recurrent_state.data_ptr(),
        binding.output.data_ptr(),
        binding.scratch.data_ptr(),
    )
    assert binding.error_code.item() == 0
    torch.testing.assert_close(binding.output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, expected_state, rtol=1e-5, atol=2e-5
    )


def test_research_qwen_cute_launcher_caches_are_device_scoped() -> None:
    from b12x.sequence.gdn_decode._cute_kernels import (
        _binding_key,
        _norm_key,
        _validation_key,
    )

    device = require_sm120()
    binding, _ = _make_case(device=device)
    expected_device = binding.output.device.index

    assert _binding_key(binding)[0] == expected_device
    assert _validation_key(binding)[0] == expected_device
    assert _norm_key(binding, norm_fp32=False)[0] == expected_device


def _transformers_kv_state_reference(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    key_heads: int,
    value_heads: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Decode a mathematical ``[slot, head, key_dim, value_dim]`` state."""
    key_dim = value_dim = 128
    ratio = value_heads // key_heads
    q_width = key_heads * key_dim
    q = mixed_qkv[:, :q_width].view(-1, key_heads, key_dim).float()
    k = mixed_qkv[:, q_width : 2 * q_width].view(-1, key_heads, key_dim).float()
    v = mixed_qkv[:, 2 * q_width :].view(-1, value_heads, value_dim).float()
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    q *= key_dim**-0.5
    output = torch.zeros(
        (mixed_qkv.shape[0], value_heads, value_dim), dtype=torch.bfloat16
    )

    for token in range(int(mixed_qkv.shape[0])):
        slot = int(state_indices[token, 0])
        for value_head in range(value_heads):
            key_head = value_head // ratio
            state = recurrent_state[slot, value_head].float()
            softplus_input = a[token, value_head].float() + dt_bias[value_head].float()
            softplus = torch.where(
                softplus_input <= 20.0,
                torch.log1p(torch.exp(softplus_input)),
                softplus_input,
            )
            decay = torch.exp(-torch.exp(A_log[value_head].float()) * softplus)
            beta = (
                torch.sigmoid(b[token, value_head].float()).to(torch.bfloat16).float()
            )
            state *= decay
            delta = v[token, value_head] - (
                state * k[token, key_head].unsqueeze(-1)
            ).sum(dim=-2)
            state += k[token, key_head].unsqueeze(-1) * (delta * beta).unsqueeze(-2)
            decoded = (state * q[token, key_head].unsqueeze(-1)).sum(dim=-2)
            output[token, value_head].copy_(decoded.to(torch.bfloat16))
            recurrent_state[slot, value_head].copy_(state.to(recurrent_state.dtype))

    values = output.float()
    values *= torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    values = values.to(torch.bfloat16) * norm_weight
    return (values * torch.sigmoid(z.float())).to(torch.bfloat16)


def test_v_by_k_padded_slot_layout_matches_transposed_mathematical_oracle() -> None:
    torch.manual_seed(17)
    tokens, slots, key_heads, value_heads = 2, 2, 1, 3
    width = 2 * key_heads * 128 + value_heads * 128
    cpu = torch.device("cpu")
    mixed_qkv = _randn((tokens, width), device=cpu)
    a = _randn((tokens, value_heads), device=cpu)
    b = _randn((tokens, value_heads), device=cpu)
    z = _randn((tokens, value_heads, 128), device=cpu)
    A_log = _randn((value_heads,), device=cpu, dtype=torch.float32, scale=0.1)
    dt_bias = _randn((value_heads,), device=cpu, dtype=torch.float32, scale=0.1)
    norm_weight = 1.0 + _randn((128,), device=cpu, scale=0.05)
    slot_elements = value_heads * 128 * 128
    slot_stride = slot_elements + 37
    storage_offset = 19
    state_storage = torch.full(
        (storage_offset + (slots - 1) * slot_stride + slot_elements,),
        91.0,
        dtype=torch.float32,
        device=cpu,
    )
    state_vk = torch.as_strided(
        state_storage,
        size=(slots, value_heads, 128, 128),
        stride=(slot_stride, 128 * 128, 128, 1),
        storage_offset=storage_offset,
    )
    state_vk.copy_(
        _randn(
            (slots, value_heads, 128, 128),
            device=cpu,
            dtype=torch.float32,
            scale=0.1,
        )
    )
    state_kv = state_vk.transpose(-1, -2).contiguous()
    state_indices = torch.tensor([[0], [1]], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    accepted = torch.ones(2, dtype=torch.int32)

    expected = _transformers_kv_state_reference(
        mixed_qkv,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state_kv,
        state_indices,
        key_heads=key_heads,
        value_heads=value_heads,
    )
    actual = gdn.reference.decode(
        mixed_qkv,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state_vk,
        query_start_loc,
        accepted,
        state_indices,
        2,
        2,
        key_heads=key_heads,
        value_heads=value_heads,
        gate_activation="sigmoid",
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        state_vk.transpose(-1, -2), state_kv, rtol=1e-5, atol=2e-8
    )
    assert torch.count_nonzero(state_storage[:storage_offset] != 91.0) == 0
    gap_start = storage_offset + slot_elements
    gap_end = storage_offset + slot_stride
    assert torch.count_nonzero(state_storage[gap_start:gap_end] != 91.0) == 0


@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_qwen3_8_flash_next_output_norm_preserves_parameter_dtype_rounding(
    norm_dtype: torch.dtype,
) -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(1,),
        key_heads=1,
        value_heads=1,
        activation="sigmoid",
        norm_dtype=norm_dtype,
        qk_l2norm=False,
    )
    binding.mixed_qkv.zero_()
    binding.mixed_qkv[0, 0] = 1.0
    binding.mixed_qkv[0, 128] = 1.0
    value = torch.linspace(-3.5, 4.5, 128, device=device).to(torch.bfloat16)
    binding.mixed_qkv[0, 256:].copy_(value)
    binding.a.zero_()
    binding.b.zero_()
    binding.z.copy_(
        torch.linspace(-3.0, 3.0, 128, device=device).to(torch.bfloat16)[None, None]
    )
    binding.A_log.zero_()
    binding.dt_bias.zero_()
    binding.norm_weight.copy_(
        torch.linspace(0.251, 1.749, 128, device=device).to(norm_dtype)
    )
    binding.recurrent_state.zero_()

    core_output = (value.float() * 0.5).to(torch.bfloat16)
    normalized = core_output.float()
    normalized *= torch.rsqrt(normalized.square().mean() + 1e-6)
    normalized_bf16 = normalized.to(torch.bfloat16)
    if norm_dtype == torch.bfloat16:
        weighted = (normalized_bf16 * binding.norm_weight).to(torch.bfloat16)
    else:
        weighted = normalized_bf16.float() * binding.norm_weight
    expected = (weighted * torch.sigmoid(binding.z[0, 0].float())).to(torch.bfloat16)

    actual = gdn.run(binding, scale=1.0)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual[0, 0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_ordinary_decode_head_ratios_match_reference(
    ratio: int, activation: str
) -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(1, 1),
        key_heads=1,
        value_heads=ratio,
        activation=activation,
    )
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    assert actual.data_ptr() == binding.output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_packed_sequential_state_dtypes_match_reference(
    state_dtype: torch.dtype,
) -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(3, 1),
        max_tokens=5,
        max_seqs=2,
        columns=3,
        accepted=(2, 1),
        key_heads=1,
        value_heads=3,
        state_dtype=state_dtype,
    )
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state,
        state_reference,
        rtol=1e-2 if state_dtype == torch.bfloat16 else 1e-5,
        atol=8e-3 if state_dtype == torch.bfloat16 else 2e-5,
    )
    assert torch.count_nonzero(actual[4:]) == 0


def test_vllm_aligned_page_state_stride_matches_reference_without_copy() -> None:
    device = require_sm120()
    state_slots = 4
    key_heads = 4
    value_heads = 12
    binding, tensors = _make_case(
        device=device,
        query_lengths=(2, 1),
        max_tokens=4,
        max_seqs=3,
        state_slots=state_slots,
        key_heads=key_heads,
        value_heads=value_heads,
        columns=2,
        state_dtype=torch.float32,
    )

    # Qwen3.8-Flash-Next's aligned vLLM page is 818176 bytes: a 25600-byte BF16
    # convolution state precedes this FP32 recurrent state in every page.
    storage_offset = 25_600 // torch.float32.itemsize
    slot_stride = 818_176 // torch.float32.itemsize
    slot_elements = value_heads * 128 * 128
    state_storage = torch.full(
        (storage_offset + (state_slots - 1) * slot_stride + slot_elements,),
        91.0,
        dtype=torch.float32,
        device=device,
    )
    recurrent_state = torch.as_strided(
        state_storage,
        size=(state_slots, value_heads, 128, 128),
        stride=(slot_stride, 128 * 128, 128, 1),
        storage_offset=storage_offset,
    )
    recurrent_state.copy_(
        _randn(
            tuple(recurrent_state.shape),
            device=device,
            dtype=torch.float32,
            scale=0.1,
        )
    )
    tensors["recurrent_state"] = recurrent_state
    binding = gdn.bind(binding.plan, **tensors)

    state_reference = recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    assert binding.recurrent_state.data_ptr() == recurrent_state.data_ptr()
    assert binding.recurrent_state.stride(0) == slot_stride
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(recurrent_state, state_reference, rtol=1e-5, atol=2e-5)
    assert torch.count_nonzero(state_storage[:storage_offset] != 91.0) == 0
    for slot in range(state_slots - 1):
        gap_start = storage_offset + slot * slot_stride + slot_elements
        gap_end = storage_offset + (slot + 1) * slot_stride
        assert torch.count_nonzero(state_storage[gap_start:gap_end] != 91.0) == 0


def test_bind_rejects_noncontiguous_state_slot_contents() -> None:
    device = require_sm120()
    binding, tensors = _make_case(device=device)
    tensors["recurrent_state"] = tensors["recurrent_state"].transpose(-1, -2)

    with pytest.raises(ValueError, match="contiguous within each state slot"):
        gdn.bind(binding.plan, **tensors)


def test_bind_rejects_overlapping_state_slots() -> None:
    device = require_sm120()
    binding, tensors = _make_case(device=device)
    shape = tuple(tensors["recurrent_state"].shape)
    slots, value_heads, value_dim, key_dim = shape
    slot_elements = value_heads * value_dim * key_dim
    state_storage = torch.empty(
        slots * slot_elements,
        dtype=binding.recurrent_state.dtype,
        device=device,
    )
    tensors["recurrent_state"] = torch.as_strided(
        state_storage,
        size=shape,
        stride=(slot_elements - 1, value_dim * key_dim, key_dim, 1),
    )

    with pytest.raises(ValueError, match="slots must not overlap"):
        gdn.bind(binding.plan, **tensors)


def test_accepted_column_selects_rollback_checkpoint() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(3,),
        columns=3,
        accepted=(3,),
        key_heads=1,
        value_heads=1,
    )
    binding.recurrent_state[0].fill_(0.25)
    binding.recurrent_state[1].fill_(-0.5)
    binding.recurrent_state[2].fill_(0.75)
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_rejected_draft_restarts_next_iteration_from_accepted_checkpoint() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(3,),
        columns=3,
        accepted=(1,),
        key_heads=1,
        value_heads=1,
    )
    gdn.run(binding)
    torch.cuda.synchronize(device)
    accepted_checkpoint = binding.recurrent_state[1].clone()

    binding.num_accepted_tokens.fill_(2)
    binding.recurrent_state[2].fill_(73.0)
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.a.copy_(torch.randn_like(binding.a).mul_(0.2))
    binding.b.copy_(torch.randn_like(binding.b).mul_(0.2))
    binding.z.copy_(torch.randn_like(binding.z).mul_(0.2))
    state_reference = binding.recurrent_state.clone()
    torch.testing.assert_close(
        state_reference[1], accepted_checkpoint, rtol=0, atol=0
    )
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_qwen3_8_flash_next_ratio3_sigmoid_fp32_state_shape() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(1,),
        max_seqs=2,
        state_slots=2,
        key_heads=16,
        value_heads=48,
        activation="sigmoid",
        state_dtype=torch.float32,
    )
    assert binding.mixed_qkv.shape == (1, 10240)
    assert binding.output.shape == (1, 48, 128)
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_plan_exposes_caller_owned_scratch_and_error_view() -> None:
    device = require_sm120()
    binding, tensors = _make_case(device=device, query_lengths=(1, 1, 1))
    (spec,) = binding.plan.scratch_specs()

    assert spec.name == "gdn_decode"
    assert binding.plan.shapes_and_dtypes() == ((spec.shape, torch.uint8),)
    assert binding.scratch.data_ptr() == tensors["scratch"].data_ptr()
    assert binding.error_code.shape == (1,)
    assert binding.error_code.dtype == torch.int32
    assert binding.duplicate_slots.shape == (binding.plan.duplicate_table_size,)
    assert binding.duplicate_slots.dtype == torch.int64


def test_bind_rejects_scratch_alias_with_mutable_output() -> None:
    device = require_sm120()
    binding, tensors = _make_case(device=device)
    output_nbytes = binding.output.numel() * binding.output.element_size()
    scratch_nbytes = binding.plan.scratch_specs()[0].nbytes
    shared = torch.empty(
        max(output_nbytes, scratch_nbytes), dtype=torch.uint8, device=device
    )
    aliased_output = shared[:output_nbytes].view(torch.bfloat16).view_as(binding.output)
    tensors.update(scratch=shared, output=aliased_output)

    with pytest.raises(ValueError, match="scratch and output"):
        gdn.bind(binding.plan, **tensors)


@pytest.mark.parametrize("bad_slot", [-1, 2])
def test_invalid_active_slot_poisons_output_without_state_mutation(
    bad_slot: int,
) -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device, query_lengths=(1,), state_slots=2, key_heads=1, value_heads=1
    )
    binding.state_indices.fill_(bad_slot)
    before = binding.recurrent_state.clone()
    binding.output.fill_(13.0)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    assert binding.error_code.item() & 4
    assert torch.isnan(actual).all()
    torch.testing.assert_close(binding.recurrent_state, before, rtol=0, atol=0)


def test_duplicate_active_state_slot_is_transactional() -> None:
    device = require_sm120()
    binding, _ = _make_case(device=device, query_lengths=(2,), columns=2, state_slots=3)
    binding.state_indices[0].fill_(1)
    before = binding.recurrent_state.clone()
    binding.output.fill_(13.0)

    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    assert binding.error_code.item() & 1
    assert torch.isnan(actual).all()
    torch.testing.assert_close(binding.recurrent_state, before, rtol=0, atol=0)


def test_cross_request_duplicate_state_slot_is_transactional() -> None:
    device = require_sm120()
    binding, _ = _make_case(device=device, query_lengths=(1, 1), state_slots=3)
    binding.state_indices[:, 0].fill_(1)
    before = binding.recurrent_state.clone()

    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    assert binding.error_code.item() & 1
    assert torch.isnan(actual).all()
    torch.testing.assert_close(binding.recurrent_state, before, rtol=0, atol=0)


def test_unique_hash_collision_does_not_report_duplicate() -> None:
    device = require_sm120()
    binding, _ = _make_case(device=device, query_lengths=(1, 1), state_slots=5)
    assert binding.plan.duplicate_table_size == 4
    binding.state_indices[:, 0].copy_(
        torch.tensor([0, 4], dtype=torch.int32, device=device)
    )
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    assert binding.error_code.item() == 0
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


@pytest.mark.parametrize("accepted", [0, 4])
def test_invalid_accepted_count_is_transactional(accepted: int) -> None:
    device = require_sm120()
    binding, _ = _make_case(device=device, query_lengths=(2,), columns=3, accepted=(1,))
    binding.num_accepted_tokens[0] = accepted
    before = binding.recurrent_state.clone()
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    assert binding.error_code.item() & 2
    assert torch.isnan(actual).all()
    torch.testing.assert_close(binding.recurrent_state, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "query_start,num_tokens",
    [
        ([1, 2, 3], 3),
        ([0, 2, 1], 1),
        ([0, 1, 2], 1),
    ],
)
def test_invalid_query_metadata_is_transactional(
    query_start: list[int], num_tokens: int
) -> None:
    device = require_sm120()
    binding, _ = _make_case(device=device, query_lengths=(1, 1))
    binding.query_start_loc.copy_(
        torch.tensor(query_start, dtype=torch.int32, device=device)
    )
    binding.num_tokens.fill_(num_tokens)
    before = binding.recurrent_state.clone()
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    assert binding.error_code.item() & 2
    assert torch.isnan(actual).all()
    torch.testing.assert_close(binding.recurrent_state, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "count_name,count_value",
    [
        ("num_seqs", -1),
        ("num_seqs", 3),
        ("num_tokens", -1),
        ("num_tokens", 3),
    ],
)
def test_invalid_device_count_is_transactional(
    count_name: str, count_value: int
) -> None:
    device = require_sm120()
    binding, _ = _make_case(device=device, query_lengths=(1, 1))
    getattr(binding, count_name).fill_(count_value)
    before = binding.recurrent_state.clone()

    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    assert binding.error_code.item() & 2
    assert torch.isnan(actual).all()
    torch.testing.assert_close(binding.recurrent_state, before, rtol=0, atol=0)


def test_zero_length_request_and_capacity_tail_are_zero() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(2, 0),
        max_tokens=4,
        columns=2,
        max_seqs=2,
    )
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    assert binding.error_code.item() == 0
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    assert torch.count_nonzero(actual[2:]) == 0


def test_cuda_graph_replay_uses_device_counts_and_fixed_addresses() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(2, 1),
        max_tokens=6,
        max_seqs=3,
        columns=2,
        state_slots=8,
        accepted=(2, 1),
    )
    gdn.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = gdn.run(binding)

    binding.query_start_loc.copy_(
        torch.tensor([0, 1, 3, 3], dtype=torch.int32, device=device)
    )
    binding.num_seqs.fill_(2)
    binding.num_tokens.fill_(3)
    binding.num_accepted_tokens[:2].copy_(
        torch.tensor([1, 2], dtype=torch.int32, device=device)
    )
    binding.state_indices[:2].copy_(
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    )
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.25))
    binding.a.copy_(torch.randn_like(binding.a).mul_(0.25))
    binding.b.copy_(torch.randn_like(binding.b).mul_(0.25))
    binding.z.copy_(torch.randn_like(binding.z).mul_(0.25))
    binding.recurrent_state.copy_(torch.randn_like(binding.recurrent_state).mul_(0.1))
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert captured_output.data_ptr() == binding.output.data_ptr()
    assert allocated_after == allocated_before
    assert binding.error_code.item() == 0
    torch.testing.assert_close(captured_output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_tp4_target_packed_graph_replays_dynamic_device_counts() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(2, 3),
        max_tokens=10,
        max_seqs=3,
        columns=5,
        state_slots=16,
        accepted=(2, 3),
        key_heads=4,
        value_heads=12,
        activation="sigmoid",
        state_dtype=torch.float32,
    )
    assert binding.mixed_qkv.shape == (10, 2560)
    assert binding.output.shape == (10, 12, 128)
    gdn.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = gdn.run(binding)
    output_address = captured_output.data_ptr()

    binding.query_start_loc.copy_(
        torch.tensor([0, 1, 5, 8], dtype=torch.int32, device=device)
    )
    binding.num_seqs.fill_(3)
    binding.num_tokens.fill_(8)
    binding.num_accepted_tokens.copy_(
        torch.tensor([1, 4, 2], dtype=torch.int32, device=device)
    )
    binding.state_indices.copy_(
        torch.arange(15, dtype=torch.int32, device=device).view(3, 5)
    )
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.a.copy_(torch.randn_like(binding.a).mul_(0.2))
    binding.b.copy_(torch.randn_like(binding.b).mul_(0.2))
    binding.z.copy_(torch.randn_like(binding.z).mul_(0.2))
    binding.recurrent_state.copy_(torch.randn_like(binding.recurrent_state).mul_(0.1))
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert binding.error_code.item() == 0
    assert captured_output.data_ptr() == output_address == binding.output.data_ptr()
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured_output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )
    assert torch.count_nonzero(captured_output[8:]) == 0


def test_cuda_graph_invalid_metadata_is_allocation_free_and_transactional() -> None:
    device = require_sm120()
    binding, _ = _make_case(device=device)
    gdn.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = gdn.run(binding)

    binding.num_accepted_tokens[0] = 0
    binding.recurrent_state.copy_(torch.randn_like(binding.recurrent_state).mul_(0.1))
    before = binding.recurrent_state.clone()
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert captured_output.data_ptr() == binding.output.data_ptr()
    assert allocated_after == allocated_before
    assert binding.error_code.item() & 2
    assert torch.isnan(captured_output).all()
    torch.testing.assert_close(binding.recurrent_state, before, rtol=0, atol=0)


def test_torch_compile_fullgraph_keeps_outer_op_opaque() -> None:
    device = require_sm120()
    binding, _ = _make_case(device=device, query_lengths=(2,), columns=2, accepted=(2,))

    def launch() -> torch.Tensor:
        return gdn.run(binding)

    launch()
    compiled = torch.compile(launch, fullgraph=True)
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.25))
    binding.recurrent_state.copy_(torch.randn_like(binding.recurrent_state).mul_(0.1))
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = compiled()
    torch.cuda.synchronize(device)
    assert actual.data_ptr() == binding.output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_padded_state_slot_offset_past_int32_element_boundary() -> None:
    device = require_sm120()
    key_heads, value_heads = 16, 48
    slot_elements = value_heads * 128 * 128
    slot_stride = slot_elements + 2_048
    tail_slot = (1 << 31) // slot_stride + 1
    assert tail_slot * slot_stride > 1 << 31
    caps = gdn.Caps(
        device=device,
        max_tokens=1,
        max_seqs=1,
        max_state_slots=tail_slot + 1,
        key_heads=key_heads,
        value_heads=value_heads,
        state_dtype=torch.bfloat16,
        gate_activation="sigmoid",
    )
    mixed_qkv = _randn((1, caps.packed_qkv_width), device=device)
    a = _randn((1, value_heads), device=device)
    b = _randn((1, value_heads), device=device)
    z = _randn((1, value_heads, 128), device=device)
    A_log = _randn((value_heads,), device=device, dtype=torch.float32, scale=0.1)
    dt_bias = _randn((value_heads,), device=device, dtype=torch.float32, scale=0.1)
    norm_weight = (1.0 + _randn((128,), device=device, scale=0.05)).contiguous()
    state_storage = torch.empty(
        tail_slot * slot_stride + slot_elements,
        dtype=torch.bfloat16,
        device=device,
    )
    recurrent_state = torch.as_strided(
        state_storage,
        size=(tail_slot + 1, value_heads, 128, 128),
        stride=(slot_stride, 128 * 128, 128, 1),
    )
    recurrent_state[tail_slot].copy_(
        _randn((value_heads, 128, 128), device=device, scale=0.1)
    )
    compact_reference_state = recurrent_state[tail_slot : tail_slot + 1].clone()
    indices = torch.tensor([[tail_slot]], dtype=torch.int64, device=device)
    compact_indices = torch.zeros((1, 1), dtype=torch.int64, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    accepted = torch.ones(1, dtype=torch.int32, device=device)
    num_seqs = torch.ones(1, dtype=torch.int32, device=device)
    num_tokens = torch.ones(1, dtype=torch.int32, device=device)
    output = torch.empty((1, value_heads, 128), dtype=torch.bfloat16, device=device)
    planned = gdn.plan(caps)
    (scratch_spec,) = planned.scratch_specs()
    scratch = torch.empty(scratch_spec.shape, dtype=scratch_spec.dtype, device=device)
    binding = gdn.bind(
        planned,
        scratch=scratch,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        z=z,
        A_log=A_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        recurrent_state=recurrent_state,
        query_start_loc=query_start_loc,
        num_accepted_tokens=accepted,
        state_indices=indices,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        output=output,
    )
    expected = gdn.reference.decode(
        mixed_qkv,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        compact_reference_state,
        query_start_loc,
        accepted,
        compact_indices,
        num_seqs,
        num_tokens,
        key_heads=key_heads,
        value_heads=value_heads,
        gate_activation="sigmoid",
    )
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        recurrent_state[tail_slot],
        compact_reference_state[0],
        rtol=1e-2,
        atol=8e-3,
    )

    del binding, recurrent_state, state_storage
    gc.collect()
    torch.cuda.empty_cache()


def test_cute_fp32_state_slot_offset_past_int32_element_boundary() -> None:
    from b12x.sequence.gdn_decode._cute_kernels import (
        run_gated_rmsnorm,
        run_packed_recurrent_qwen,
        run_qwen_validation,
    )

    device = require_sm120()
    key_heads, value_heads = 1, 3
    slot_elements = value_heads * 128 * 128
    slot_stride = slot_elements + 2_048
    tail_slot = (1 << 31) // slot_stride + 1
    assert tail_slot * slot_stride > 1 << 31
    caps = gdn.Caps(
        device=device,
        max_tokens=1,
        max_seqs=1,
        max_state_slots=tail_slot + 1,
        key_heads=key_heads,
        value_heads=value_heads,
        state_dtype=torch.float32,
        gate_activation="sigmoid",
    )
    mixed_qkv = _randn((1, caps.packed_qkv_width), device=device)
    a = _randn((1, value_heads), device=device)
    b = _randn((1, value_heads), device=device)
    z = _randn((1, value_heads, 128), device=device)
    A_log = _randn((value_heads,), device=device, dtype=torch.float32, scale=0.1)
    dt_bias = _randn((value_heads,), device=device, dtype=torch.float32, scale=0.1)
    norm_weight = (1.0 + _randn((128,), device=device, scale=0.05)).contiguous()
    state_storage = torch.empty(
        tail_slot * slot_stride + slot_elements,
        dtype=torch.float32,
        device=device,
    )
    recurrent_state = torch.as_strided(
        state_storage,
        size=(tail_slot + 1, value_heads, 128, 128),
        stride=(slot_stride, 128 * 128, 128, 1),
    )
    recurrent_state[tail_slot].copy_(
        _randn(
            (value_heads, 128, 128),
            device=device,
            dtype=torch.float32,
            scale=0.1,
        )
    )
    initial_state = recurrent_state[tail_slot : tail_slot + 1].clone()
    expected_state = initial_state.clone()
    indices = torch.tensor([[tail_slot]], dtype=torch.int32, device=device)
    compact_indices = torch.zeros((1, 1), dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    accepted = torch.ones(1, dtype=torch.int32, device=device)
    num_seqs = torch.ones(1, dtype=torch.int32, device=device)
    num_tokens = torch.ones(1, dtype=torch.int32, device=device)
    output = torch.empty((1, value_heads, 128), dtype=torch.bfloat16, device=device)
    planned = gdn.plan(caps)
    (scratch_spec,) = planned.scratch_specs()
    scratch = torch.empty(scratch_spec.shape, dtype=scratch_spec.dtype, device=device)
    binding = gdn.bind(
        planned,
        scratch=scratch,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        z=z,
        A_log=A_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        recurrent_state=recurrent_state,
        query_start_loc=query_start_loc,
        num_accepted_tokens=accepted,
        state_indices=indices,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        output=output,
    )
    expected = gdn.reference.decode(
        mixed_qkv,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        expected_state,
        query_start_loc,
        accepted,
        compact_indices,
        num_seqs,
        num_tokens,
        key_heads=key_heads,
        value_heads=value_heads,
        gate_activation="sigmoid",
    )

    def launch_cute() -> None:
        run_qwen_validation(binding)
        run_packed_recurrent_qwen(binding)
        run_gated_rmsnorm(binding, eps=1.0e-6)

    launch_cute()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_cute()

    recurrent_state[tail_slot].copy_(initial_state[0])
    output.fill_(float("nan"))
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert allocated_after == allocated_before
    assert binding.error_code.item() == 0
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        recurrent_state[tail_slot], expected_state[0], rtol=1e-5, atol=2e-5
    )

    del binding, recurrent_state, state_storage
    gc.collect()
    torch.cuda.empty_cache()


def test_caps_reject_unsupported_ratio_columns_and_packing_capacity() -> None:
    device = require_sm120()
    with pytest.raises(ValueError, match="ratio"):
        gdn.Caps(
            device=device,
            max_tokens=1,
            max_seqs=1,
            max_state_slots=1,
            key_heads=1,
            value_heads=5,
        )
    with pytest.raises(ValueError, match="at most 8"):
        gdn.Caps(
            device=device,
            max_tokens=1,
            max_seqs=1,
            max_state_slots=1,
            key_heads=1,
            value_heads=1,
            state_index_columns=9,
        )
    with pytest.raises(ValueError, match="must fit"):
        gdn.Caps(
            device=device,
            max_tokens=3,
            max_seqs=2,
            max_state_slots=2,
            key_heads=1,
            value_heads=1,
            state_index_columns=1,
        )
