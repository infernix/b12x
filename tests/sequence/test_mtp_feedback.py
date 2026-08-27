from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from b12x.sequence import mtp_feedback as mtp
from b12x.sequence.mtp_feedback import _cute_norm

from ..conftest import require_b12x as require_sm120


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    scale: float = 0.25,
) -> torch.Tensor:
    return (
        torch.randn(shape, dtype=torch.float32, device=device)
        .mul_(scale)
        .to(torch.bfloat16)
        .contiguous()
    )


def _make_case(
    *,
    device: torch.device,
    max_tokens: int = 3,
    tokens: int | None = None,
    streams: int = 4,
    hidden_size: int = 64,
) -> tuple[mtp.Binding, dict[str, torch.Tensor]]:
    caps = mtp.Caps(
        device=device,
        max_tokens=max_tokens,
        streams=streams,
        hidden_size=hidden_size,
    )
    planned = mtp.plan(caps)
    (scratch_spec,) = planned.scratch_specs()
    tensors = {
        "scratch": torch.empty(
            scratch_spec.shape,
            dtype=scratch_spec.dtype,
            device=device,
        ),
        "token_embedding": _randn((max_tokens, hidden_size), device=device, scale=0.4),
        "multi_state": _randn(
            (max_tokens, streams, hidden_size), device=device, scale=0.4
        ),
        "token_norm_weight": _randn((hidden_size,), device=device, scale=0.05),
        "state_norm_weight": _randn(
            (streams * hidden_size,), device=device, scale=0.05
        ),
        "embedding_fc_weight": _randn(
            (hidden_size, hidden_size),
            device=device,
            scale=hidden_size**-0.5,
        ),
        "hidden_fc_weight": _randn(
            (hidden_size, hidden_size),
            device=device,
            scale=hidden_size**-0.5,
        ),
        "output": torch.full(
            (max_tokens, streams, hidden_size),
            7.0,
            dtype=torch.bfloat16,
            device=device,
        ),
    }
    binding = mtp.bind(planned, **tensors, tokens=tokens)
    return binding, tensors


def _reference(binding: mtp.Binding) -> torch.Tensor:
    return mtp.reference.feedback(
        binding.token_embedding,
        binding.multi_state,
        binding.token_norm_weight,
        binding.state_norm_weight,
        binding.embedding_fc_weight,
        binding.hidden_fc_weight,
    )


def test_reference_matches_explicit_transformers_cast_points() -> None:
    torch.manual_seed(19)
    device = torch.device("cpu")
    tokens, streams, hidden = 2, 3, 32
    token_embedding = _randn((tokens, hidden), device=device, scale=0.7)
    multi_state = _randn((tokens, streams, hidden), device=device, scale=0.7)
    token_norm_weight = _randn((hidden,), device=device, scale=0.1)
    state_norm_weight = _randn((streams * hidden,), device=device, scale=0.1)
    embedding_fc_weight = _randn((hidden, hidden), device=device, scale=hidden**-0.5)
    hidden_fc_weight = _randn((hidden, hidden), device=device, scale=hidden**-0.5)

    actual = mtp.reference.feedback(
        token_embedding,
        multi_state,
        token_norm_weight,
        state_norm_weight,
        embedding_fc_weight,
        hidden_fc_weight,
    )
    token_normalized = mtp.reference.gemma_rmsnorm(token_embedding, token_norm_weight)
    state_normalized = mtp.reference.gemma_rmsnorm(
        multi_state.flatten(-2), state_norm_weight
    ).view(tokens, streams, hidden)
    token_path = F.linear(token_normalized, embedding_fc_weight).to(torch.bfloat16)
    state_path = F.linear(state_normalized, hidden_fc_weight).to(torch.bfloat16)
    expected = (state_path + token_path.unsqueeze(1)).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    unrounded = (
        F.linear(state_normalized.float(), hidden_fc_weight.float())
        + F.linear(token_normalized.float(), embedding_fc_weight.float()).unsqueeze(1)
    ).to(torch.bfloat16)
    assert torch.count_nonzero(actual != unrounded).item() > 0


def test_state_norm_uses_one_flattened_stream_group() -> None:
    hidden = 16
    state = (
        torch.stack(
            (
                torch.full((hidden,), 0.25),
                torch.full((hidden,), 4.0),
            )
        )
        .to(torch.bfloat16)[None]
        .contiguous()
    )
    weight = torch.zeros((2 * hidden,), dtype=torch.bfloat16)
    flattened = mtp.reference.gemma_rmsnorm(state.flatten(-2), weight).view_as(state)
    per_stream = torch.stack(
        [
            mtp.reference.gemma_rmsnorm(
                state[:, stream], weight[stream * hidden : (stream + 1) * hidden]
            )
            for stream in range(2)
        ],
        dim=1,
    )

    assert not torch.equal(flattened, per_stream)
    assert flattened[0, 0, 0].abs() < flattened[0, 1, 0].abs()
    torch.testing.assert_close(per_stream[0, 0], per_stream[0, 1], rtol=0, atol=0)


def test_plan_and_bind_expose_only_caller_owned_storage() -> None:
    device = require_sm120()
    binding, tensors = _make_case(
        device=device,
        max_tokens=4,
        tokens=2,
        streams=3,
        hidden_size=64,
    )
    planned = binding.plan
    scratch_start = tensors["scratch"].data_ptr()
    scratch_end = scratch_start + tensors["scratch"].numel()
    views = (
        binding.token_normalized,
        binding.state_partial_sums,
        binding.state_normalized,
        binding.token_path,
    )

    assert binding.tokens == 2
    assert planned.output_shape() == (4, 3, 64)
    assert planned.output_shape(2) == (2, 3, 64)
    assert binding.scratch.data_ptr() == scratch_start
    assert binding.output.data_ptr() == tensors["output"].data_ptr()
    assert all(scratch_start <= view.data_ptr() < scratch_end for view in views)
    assert binding.output.shape == (2, 3, 64)
    assert all(
        offset % 1024 == 0
        for offset in (
            planned.token_normalized_offset_bytes,
            planned.state_partial_sums_offset_bytes,
            planned.state_normalized_offset_bytes,
            planned.token_path_offset_bytes,
        )
    )


def test_bind_rejects_bad_shapes_dtypes_and_mutable_aliases() -> None:
    device = require_sm120()
    binding, tensors = _make_case(device=device, max_tokens=2)
    planned = binding.plan
    bad = dict(tensors)
    bad["token_norm_weight"] = torch.empty((63,), dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="token_norm_weight must have shape"):
        mtp.bind(planned, **bad)

    bad = dict(tensors)
    bad["embedding_fc_weight"] = torch.empty(
        (64, 64), dtype=torch.float32, device=device
    )
    with pytest.raises(TypeError, match="embedding_fc_weight must have dtype"):
        mtp.bind(planned, **bad)

    bad = dict(tensors)
    bad["output"] = tensors["multi_state"]
    with pytest.raises(ValueError, match="output.*multi_state"):
        mtp.bind(planned, **bad)

    bad = dict(tensors)
    bad["output"] = (
        tensors["scratch"][: 2 * 4 * 64 * torch.bfloat16.itemsize]
        .view(torch.bfloat16)
        .view(2, 4, 64)
    )
    with pytest.raises(ValueError, match="scratch and output"):
        mtp.bind(planned, **bad)

    bad = dict(tensors)
    bad["token_embedding"] = (
        tensors["scratch"][: 2 * 64 * torch.bfloat16.itemsize]
        .view(torch.bfloat16)
        .view(2, 64)
    )
    with pytest.raises(ValueError, match="scratch.*token_embedding"):
        mtp.bind(planned, **bad)

    bad = dict(tensors)
    bad["scratch"] = tensors["scratch"][:-1]
    with pytest.raises(ValueError, match="scratch"):
        mtp.bind(planned, **bad)


def test_zero_tokens_is_a_noop_and_live_count_is_capacity_checked() -> None:
    device = require_sm120()
    binding, tensors = _make_case(device=device, max_tokens=3, tokens=0)
    output_before = tensors["output"].clone()

    actual = mtp.run(binding)

    assert actual.shape == (0, 4, 64)
    torch.testing.assert_close(tensors["output"], output_before, rtol=0, atol=0)
    for tokens in (-1, 4):
        with pytest.raises(ValueError, match="tokens="):
            mtp.bind(binding.plan, **tensors, tokens=tokens)


def test_small_geometry_matches_reference() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        max_tokens=3,
        tokens=2,
        streams=3,
        hidden_size=64,
    )
    expected = _reference(binding)
    actual = mtp.run(binding)
    torch.cuda.synchronize(device)

    assert actual.data_ptr() == binding.output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)


def test_target_s4_h2560_geometry_matches_reference() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        max_tokens=1,
        streams=4,
        hidden_size=2560,
    )
    expected = _reference(binding)
    actual = mtp.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=4e-2)


def test_non_tile_aligned_geometry_preserves_inputs_and_output_tail() -> None:
    device = require_sm120()
    binding, tensors = _make_case(
        device=device,
        max_tokens=19,
        tokens=17,
        streams=3,
        hidden_size=80,
    )
    read_only_before = {
        name: tensor.clone()
        for name, tensor in tensors.items()
        if name not in {"scratch", "output"}
    }
    output_tail_before = tensors["output"][binding.tokens :].clone()
    expected = _reference(binding)

    actual = mtp.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)
    torch.testing.assert_close(
        tensors["output"][binding.tokens :], output_tail_before, rtol=0, atol=0
    )
    for name, before in read_only_before.items():
        torch.testing.assert_close(tensors[name], before, rtol=0, atol=0)


def test_cuda_graph_replay_uses_bound_scratch_and_output() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        max_tokens=2,
        streams=2,
        hidden_size=64,
    )
    mtp.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mtp.run(binding)

    assert captured.data_ptr() == binding.output.data_ptr()
    for _ in range(3):
        binding.token_embedding.copy_(
            torch.randn_like(binding.token_embedding).mul_(0.3)
        )
        binding.multi_state.copy_(torch.randn_like(binding.multi_state).mul_(0.3))
        expected = _reference(binding)
        allocated_before = torch.cuda.memory_allocated(device)
        graph.replay()
        torch.cuda.synchronize(device)
        allocated_after = torch.cuda.memory_allocated(device)

        assert allocated_after == allocated_before
        torch.testing.assert_close(captured, expected, rtol=2e-2, atol=3e-2)


def test_target_geometry_cuda_graph_replay_uses_bound_storage() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        max_tokens=3,
        tokens=3,
        streams=4,
        hidden_size=2560,
    )
    mtp.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mtp.run(binding)
    output_address = captured.data_ptr()
    scratch_address = binding.scratch.data_ptr()

    binding.token_embedding.copy_(torch.randn_like(binding.token_embedding).mul_(0.3))
    binding.multi_state.copy_(torch.randn_like(binding.multi_state).mul_(0.3))
    expected = _reference(binding)
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert captured.data_ptr() == output_address == binding.output.data_ptr()
    assert binding.scratch.data_ptr() == scratch_address
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured, expected, rtol=2e-2, atol=4e-2)


@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty.*:UserWarning")
def test_standalone_cute_norm_rejects_cold_cuda_graph_capture() -> None:
    device = require_sm120()
    hidden_size = 2560
    source = _randn((1, hidden_size), device=device, scale=0.4)
    weight = _randn((hidden_size,), device=device, scale=0.05)
    output = torch.empty_like(source)
    _cute_norm.clear_caches()

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="compiled and warm-run"):
        with torch.cuda.graph(graph):
            _cute_norm.token_norm(
                source,
                weight,
                output,
                eps=1.0e-6,
                hidden_size=hidden_size,
            )


def test_standalone_cute_norm_uses_source_device_when_non_current() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("two visible SM120 GPUs are required")
    original_device = torch.cuda.current_device()
    target_index = next(
        index for index in range(torch.cuda.device_count()) if index != original_device
    )
    target = torch.device("cuda", target_index)
    if torch.cuda.get_device_capability(target) not in ((12, 0), (12, 1)):
        pytest.skip("the non-current visible GPU must be SM120 or SM121")

    tokens, streams, hidden_size = 2, 4, 2560
    token_source = _randn((tokens, hidden_size), device=target, scale=0.4)
    state_source = _randn(
        (tokens, streams, hidden_size), device=target, scale=0.4
    )
    token_weight = _randn((hidden_size,), device=target, scale=0.05)
    state_weight = _randn((streams * hidden_size,), device=target, scale=0.05)
    token_output = torch.empty_like(token_source)
    state_output = torch.empty_like(state_source)
    _cute_norm.clear_caches()

    _cute_norm.token_norm(
        token_source,
        token_weight,
        token_output,
        eps=1.0e-6,
        hidden_size=hidden_size,
    )
    _cute_norm.state_norm(
        state_source,
        state_weight,
        state_output,
        eps=1.0e-6,
        streams=streams,
        hidden_size=hidden_size,
    )
    torch.cuda.synchronize(target)

    assert torch.cuda.current_device() == original_device
    expected_token = mtp.reference.gemma_rmsnorm(token_source, token_weight)
    expected_state = mtp.reference.gemma_rmsnorm(
        state_source.flatten(-2), state_weight
    ).view_as(state_source)
    torch.testing.assert_close(token_output, expected_token, rtol=2e-2, atol=4e-2)
    torch.testing.assert_close(state_output, expected_state, rtol=2e-2, atol=4e-2)


def test_standalone_cute_norm_correctness_and_graph_stability() -> None:
    device = require_sm120()
    tokens, streams, hidden_size = 4, 4, 2560
    token_source = _randn((tokens, hidden_size), device=device, scale=0.4)
    state_source = _randn(
        (tokens, streams, hidden_size), device=device, scale=0.4
    )
    token_weight = _randn((hidden_size,), device=device, scale=0.05)
    state_weight = _randn((streams * hidden_size,), device=device, scale=0.05)
    token_output = torch.empty_like(token_source)
    state_output = torch.empty_like(state_source)

    def launch() -> None:
        _cute_norm.token_norm(
            token_source,
            token_weight,
            token_output,
            eps=1.0e-6,
            hidden_size=hidden_size,
        )
        _cute_norm.state_norm(
            state_source,
            state_weight,
            state_output,
            eps=1.0e-6,
            streams=streams,
            hidden_size=hidden_size,
        )

    launch()
    torch.cuda.synchronize(device)
    expected_token = mtp.reference.gemma_rmsnorm(token_source, token_weight)
    expected_state = mtp.reference.gemma_rmsnorm(
        state_source.flatten(-2), state_weight
    ).view_as(state_source)
    torch.testing.assert_close(token_output, expected_token, rtol=2e-2, atol=4e-2)
    torch.testing.assert_close(state_output, expected_state, rtol=2e-2, atol=4e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    token_address = token_output.data_ptr()
    state_address = state_output.data_ptr()

    token_source.copy_(torch.randn_like(token_source).mul_(0.3))
    state_source.copy_(torch.randn_like(state_source).mul_(0.3))
    token_output.fill_(float("nan"))
    state_output.fill_(float("nan"))
    expected_token = mtp.reference.gemma_rmsnorm(token_source, token_weight)
    expected_state = mtp.reference.gemma_rmsnorm(
        state_source.flatten(-2), state_weight
    ).view_as(state_source)
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert token_output.data_ptr() == token_address
    assert state_output.data_ptr() == state_address
    assert allocated_after == allocated_before
    torch.testing.assert_close(token_output, expected_token, rtol=2e-2, atol=4e-2)
    torch.testing.assert_close(state_output, expected_state, rtol=2e-2, atol=4e-2)


def test_torch_compile_fullgraph_keeps_feedback_op_opaque() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        max_tokens=2,
        streams=2,
        hidden_size=64,
    )

    def launch() -> torch.Tensor:
        return mtp.run(binding)

    launch()
    compiled = torch.compile(launch, fullgraph=True)
    binding.token_embedding.copy_(torch.randn_like(binding.token_embedding).mul_(0.3))
    binding.multi_state.copy_(torch.randn_like(binding.multi_state).mul_(0.3))
    expected = _reference(binding)
    actual = compiled()
    torch.cuda.synchronize(device)

    assert actual.data_ptr() == binding.output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)


def test_caps_and_run_validate_contract() -> None:
    device = require_sm120()
    with pytest.raises(ValueError, match="divisible by 16"):
        mtp.Caps(device=device, max_tokens=1, hidden_size=63)
    with pytest.raises(TypeError, match="torch.bfloat16"):
        mtp.Caps(device=device, max_tokens=1, dtype=torch.float16)
    binding, _ = _make_case(device=device, max_tokens=1)
    for eps in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="eps must be finite and positive"):
            mtp.run(binding, eps=eps)
