from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from b12x.norm import hyperconnection as hc

from ..conftest import require_b12x as require_sm120


def _allocate_binding(
    *,
    device: torch.device | str,
    tokens: int,
    max_tokens: int | None = None,
    hidden_size: int = 2560,
    streams: int = 4,
    lowrank: int = 320,
) -> hc.Binding:
    device = torch.device(device)
    capacity = tokens if max_tokens is None else max_tokens
    plan = hc.plan(
        hc.Caps(
            device=device,
            max_tokens=capacity,
            hidden_size=hidden_size,
            streams=streams,
            lowrank=lowrank,
        )
    )
    width = streams * hidden_size
    return hc.bind(
        plan,
        tokens=tokens,
        normalized=torch.empty((capacity, width), dtype=torch.bfloat16, device=device),
        bottleneck=torch.empty(
            (capacity, lowrank), dtype=torch.bfloat16, device=device
        ),
        block_input=torch.empty(
            (capacity, hidden_size), dtype=torch.bfloat16, device=device
        ),
        combined=torch.empty((capacity, width), dtype=torch.bfloat16, device=device),
    )


def _direct_grouped_rmsnorm(
    state: torch.Tensor,
    weight: torch.Tensor,
    *,
    streams: int,
    eps: float,
) -> torch.Tensor:
    tokens, width = state.shape
    hidden_size = width // streams
    grouped = state.float().view(tokens, streams, hidden_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = grouped * torch.rsqrt(variance + eps)
    return (normalized.flatten(1) * (1.0 + weight.float())).to(state.dtype)


def test_reference_matches_explicit_target_expressions() -> None:
    generator = torch.Generator().manual_seed(20260825)
    tokens, streams, hidden_size, lowrank = 2, 4, 2560, 320
    width = streams * hidden_size
    state = torch.randn((tokens, width), generator=generator, dtype=torch.bfloat16)
    norm_weight = torch.randn((width,), generator=generator, dtype=torch.bfloat16) / 16
    projected_down = torch.randn(
        (tokens, lowrank), generator=generator, dtype=torch.bfloat16
    )
    gate_logits = torch.randn(
        (tokens, width), generator=generator, dtype=torch.bfloat16
    )
    block_output = torch.randn(
        (tokens, hidden_size), generator=generator, dtype=torch.bfloat16
    )
    injection_logits = torch.randn(
        (tokens, streams), generator=generator, dtype=torch.bfloat16
    )
    eps = 1e-6

    normalized = hc.reference.grouped_rmsnorm(
        state, norm_weight, streams=streams, eps=eps
    )
    expected_normalized = _direct_grouped_rmsnorm(
        state, norm_weight, streams=streams, eps=eps
    )
    torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=0)

    bottleneck = hc.reference.scaled_silu(projected_down, streams=streams)
    torch.testing.assert_close(
        bottleneck,
        F.silu(projected_down / streams),
        rtol=0,
        atol=0,
    )

    block_input = hc.reference.gate_mean(normalized, gate_logits, streams=streams)
    expected_input = (
        torch.sigmoid(gate_logits).view(tokens, streams, hidden_size)
        * normalized.view(tokens, streams, hidden_size)
    ).mean(dim=1)
    torch.testing.assert_close(block_input, expected_input, rtol=0, atol=0)

    combined, next_normalized = hc.reference.combine_norm(
        state,
        block_output,
        injection_logits,
        norm_weight,
        streams=streams,
        eps=eps,
    )
    scale = 2.0 * torch.sigmoid(injection_logits.float() / streams)
    expected_combined = (
        (
            state.float().view(tokens, streams, hidden_size)
            + block_output.float().unsqueeze(1) * scale.unsqueeze(-1)
        )
        .to(torch.bfloat16)
        .flatten(1)
    )
    expected_next_normalized = _direct_grouped_rmsnorm(
        expected_combined,
        norm_weight,
        streams=streams,
        eps=eps,
    )
    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=0)
    torch.testing.assert_close(
        next_normalized, expected_next_normalized, rtol=0, atol=0
    )


def test_plan_bind_uses_live_views_and_no_scratch() -> None:
    binding = _allocate_binding(
        device="cpu",
        tokens=3,
        max_tokens=8,
        hidden_size=64,
        lowrank=16,
    )
    assert binding.plan.scratch_specs() == ()
    assert binding.plan.output_shapes(tokens=3) == {
        "normalized": (3, 256),
        "bottleneck": (3, 16),
        "block_input": (3, 64),
        "combined": (3, 256),
    }
    assert binding.normalized.shape == (3, 256)
    assert binding.bottleneck.shape == (3, 16)
    assert binding.block_input.shape == (3, 64)
    assert binding.combined.shape == (3, 256)


def test_production_entry_point_never_falls_back_to_torch_on_cpu() -> None:
    binding = _allocate_binding(device="cpu", tokens=1, hidden_size=64, lowrank=16)
    state = torch.zeros((1, 256), dtype=torch.bfloat16)
    weight = torch.zeros((256,), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="require CUDA"):
        hc.run_grouped_rmsnorm(state, weight, eps=1e-6, binding=binding)


def test_bind_rejects_overlapping_outputs() -> None:
    caps = hc.Caps(device="cpu", max_tokens=2, hidden_size=8, lowrank=8)
    plan = hc.plan(caps)
    storage = torch.empty((64,), dtype=torch.bfloat16)
    normalized = storage.view(2, 32)
    with pytest.raises(ValueError, match="must not overlap"):
        hc.bind(
            plan,
            normalized=normalized,
            bottleneck=torch.empty((2, 8), dtype=torch.bfloat16),
            block_input=torch.empty((2, 8), dtype=torch.bfloat16),
            combined=storage.view(2, 32),
        )


@pytest.mark.parametrize("overlap_name", ["bottleneck", "block_input"])
def test_bind_rejects_normalized_live_range_alias(overlap_name: str) -> None:
    caps = hc.Caps(device="cpu", max_tokens=2, hidden_size=8, lowrank=8)
    plan = hc.plan(caps)
    storage = torch.empty((64,), dtype=torch.bfloat16)
    normalized = storage.view(2, 32)
    outputs = {
        "normalized": normalized,
        "bottleneck": torch.empty((2, 8), dtype=torch.bfloat16),
        "block_input": torch.empty((2, 8), dtype=torch.bfloat16),
        "combined": torch.empty((2, 32), dtype=torch.bfloat16),
    }
    outputs[overlap_name] = storage[:16].view(2, 8)
    with pytest.raises(ValueError, match=f"normalized and {overlap_name}"):
        hc.bind(plan, **outputs)


def test_bind_allows_outputs_with_disjoint_live_ranges_to_share_storage() -> None:
    caps = hc.Caps(device="cpu", max_tokens=2, hidden_size=8, lowrank=8)
    plan = hc.plan(caps)
    shared = torch.empty((2, 8), dtype=torch.bfloat16)
    binding = hc.bind(
        plan,
        normalized=torch.empty((2, 32), dtype=torch.bfloat16),
        bottleneck=shared,
        block_input=shared,
        combined=torch.empty((2, 32), dtype=torch.bfloat16),
    )
    assert binding.bottleneck.data_ptr() == binding.block_input.data_ptr()


def test_live_launches_preserve_capacity_tails_and_read_only_inputs() -> None:
    device = require_sm120()
    tokens, capacity, streams, hidden_size, lowrank = 2, 5, 3, 64, 16
    width = streams * hidden_size
    plan = hc.plan(
        hc.Caps(
            device=device,
            max_tokens=capacity,
            hidden_size=hidden_size,
            streams=streams,
            lowrank=lowrank,
        )
    )

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, dtype=torch.bfloat16, device=device).contiguous()

    outputs = {
        "normalized": torch.full(
            (capacity, width), 7.0, dtype=torch.bfloat16, device=device
        ),
        "bottleneck": torch.full(
            (capacity, lowrank), 7.0, dtype=torch.bfloat16, device=device
        ),
        "block_input": torch.full(
            (capacity, hidden_size), 7.0, dtype=torch.bfloat16, device=device
        ),
        "combined": torch.full(
            (capacity, width), 7.0, dtype=torch.bfloat16, device=device
        ),
    }
    binding = hc.bind(plan, tokens=tokens, **outputs)
    state = randn((tokens, width))
    norm_weight = randn((width,))
    projected_down = randn((tokens, lowrank))
    gate_logits = randn((tokens, width))
    block_output = randn((tokens, hidden_size))
    injection_logits = randn((tokens, streams))
    read_only = {
        "state": state,
        "norm_weight": norm_weight,
        "projected_down": projected_down,
        "gate_logits": gate_logits,
        "block_output": block_output,
        "injection_logits": injection_logits,
    }
    read_only_before = {name: tensor.clone() for name, tensor in read_only.items()}
    tails_before = {name: tensor[tokens:].clone() for name, tensor in outputs.items()}

    normalized = hc.run_grouped_rmsnorm(state, norm_weight, eps=1e-6, binding=binding)
    hc.run_scaled_silu(projected_down, binding=binding)
    hc.run_gate_mean(normalized, gate_logits, binding=binding)
    hc.run_combine(state, block_output, injection_logits, binding=binding)
    hc.run_combine_norm(
        state,
        block_output,
        injection_logits,
        norm_weight,
        eps=1e-6,
        binding=binding,
    )
    torch.cuda.synchronize(device)

    for name, before in read_only_before.items():
        torch.testing.assert_close(read_only[name], before, rtol=0, atol=0)
    for name, before in tails_before.items():
        torch.testing.assert_close(outputs[name][tokens:], before, rtol=0, atol=0)


def test_grouped_norm_reuses_distinct_stream_weights_for_every_token() -> None:
    device = require_sm120()
    tokens, streams, hidden_size = 3, 4, 2560
    width = streams * hidden_size
    state = (
        torch.linspace(
            -1.0,
            1.0,
            tokens * width,
            dtype=torch.float32,
            device=device,
        )
        .view(tokens, width)
        .to(torch.bfloat16)
    )
    stream_weights = torch.tensor(
        [0.0, 0.125, -0.25, 0.5],
        dtype=torch.bfloat16,
        device=device,
    )
    weight = stream_weights.repeat_interleave(hidden_size).contiguous()
    binding = _allocate_binding(
        device=device,
        tokens=tokens,
        hidden_size=hidden_size,
        streams=streams,
    )
    actual = hc.run_grouped_rmsnorm(state, weight, eps=1e-6, binding=binding)
    expected = hc.reference.grouped_rmsnorm(state, weight, streams=streams, eps=1e-6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)


@pytest.mark.parametrize("tokens", [1, 3])
def test_target_kernels_match_reference(tokens: int) -> None:
    device = require_sm120()
    generator = torch.Generator(device="cpu").manual_seed(83400 + tokens)
    streams, hidden_size, lowrank = 4, 2560, 320
    width = streams * hidden_size

    def randn(shape: tuple[int, ...], divisor: float = 1.0) -> torch.Tensor:
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .div(divisor)
            .to(device=device, dtype=torch.bfloat16)
            .contiguous()
        )

    state = randn((tokens, width), 3.0)
    norm_weight = randn((width,), 32.0)
    projected_down = randn((tokens, lowrank), 2.0)
    gate_logits = randn((tokens, width), 2.0)
    block_output = randn((tokens, hidden_size), 4.0)
    injection_logits = randn((tokens, streams), 2.0)
    binding = _allocate_binding(
        device=device,
        tokens=tokens,
        hidden_size=hidden_size,
        streams=streams,
        lowrank=lowrank,
    )
    eps = 1e-6

    normalized = hc.run_grouped_rmsnorm(state, norm_weight, eps=eps, binding=binding)
    normalized_ref = hc.reference.grouped_rmsnorm(
        state, norm_weight, streams=streams, eps=eps
    )
    torch.testing.assert_close(normalized, normalized_ref, rtol=0, atol=2e-2)

    bottleneck = hc.run_scaled_silu(projected_down, binding=binding)
    bottleneck_ref = hc.reference.scaled_silu(projected_down, streams=streams)
    torch.testing.assert_close(bottleneck, bottleneck_ref, rtol=0, atol=8e-3)

    block_input = hc.run_gate_mean(normalized, gate_logits, binding=binding)
    block_input_ref = hc.reference.gate_mean(normalized, gate_logits, streams=streams)
    torch.testing.assert_close(block_input, block_input_ref, rtol=0, atol=8e-3)

    combined = hc.run_combine(state, block_output, injection_logits, binding=binding)
    combined_ref = hc.reference.combine(
        state, block_output, injection_logits, streams=streams
    )
    torch.testing.assert_close(combined, combined_ref, rtol=0, atol=8e-3)

    combined, next_normalized = hc.run_combine_norm(
        state,
        block_output,
        injection_logits,
        norm_weight,
        eps=eps,
        binding=binding,
    )
    combined_ref, next_normalized_ref = hc.reference.combine_norm(
        state,
        block_output,
        injection_logits,
        norm_weight,
        streams=streams,
        eps=eps,
    )
    torch.testing.assert_close(combined, combined_ref, rtol=0, atol=8e-3)
    torch.testing.assert_close(next_normalized, next_normalized_ref, rtol=0, atol=2e-2)


def test_combine_norm_cuda_graph_replay_uses_bound_outputs() -> None:
    device = require_sm120()
    tokens, streams, hidden_size = 2, 4, 2560
    width = streams * hidden_size
    state = torch.randn(
        (tokens, width), dtype=torch.bfloat16, device=device
    ).contiguous()
    block_output = torch.randn(
        (tokens, hidden_size), dtype=torch.bfloat16, device=device
    ).contiguous()
    injection_logits = torch.randn(
        (tokens, streams), dtype=torch.bfloat16, device=device
    ).contiguous()
    weight = (
        torch.randn((width,), dtype=torch.bfloat16, device=device).div_(32).contiguous()
    )
    binding = _allocate_binding(
        device=device,
        tokens=tokens,
        hidden_size=hidden_size,
        streams=streams,
    )

    hc.run_combine_norm(
        state,
        block_output,
        injection_logits,
        weight,
        eps=1e-6,
        binding=binding,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        combined, normalized = hc.run_combine_norm(
            state,
            block_output,
            injection_logits,
            weight,
            eps=1e-6,
            binding=binding,
        )

    state.copy_(torch.randn_like(state))
    block_output.copy_(torch.randn_like(block_output))
    injection_logits.copy_(torch.randn_like(injection_logits))
    expected_combined, expected_normalized = hc.reference.combine_norm(
        state,
        block_output,
        injection_logits,
        weight,
        streams=streams,
        eps=1e-6,
    )
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert combined.data_ptr() == binding.combined.data_ptr()
    assert normalized.data_ptr() == binding.normalized.data_ptr()
    assert allocated_after == allocated_before
    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=8e-3)
    torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=2e-2)


def test_target_full_chain_cuda_graph_replay_uses_bound_outputs() -> None:
    device = require_sm120()
    tokens, streams, hidden_size, lowrank = 3, 4, 2560, 320
    width = streams * hidden_size

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, dtype=torch.bfloat16, device=device).contiguous()

    binding = _allocate_binding(
        device=device,
        tokens=tokens,
        hidden_size=hidden_size,
        streams=streams,
        lowrank=lowrank,
    )
    state = randn((tokens, width))
    norm_weight = randn((width,)).div_(32)
    projected_down = randn((tokens, lowrank))
    gate_logits = randn((tokens, width))
    block_output = randn((tokens, hidden_size))
    injection_logits = randn((tokens, streams))

    def launch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = hc.run_grouped_rmsnorm(
            state, norm_weight, eps=1e-6, binding=binding
        )
        bottleneck = hc.run_scaled_silu(projected_down, binding=binding)
        block_input = hc.run_gate_mean(normalized, gate_logits, binding=binding)
        combined, next_normalized = hc.run_combine_norm(
            state,
            block_output,
            injection_logits,
            norm_weight,
            eps=1e-6,
            binding=binding,
        )
        return bottleneck, block_input, combined, next_normalized

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = launch()
    output_addresses = tuple(tensor.data_ptr() for tensor in captured)

    state.copy_(randn(state.shape))
    projected_down.copy_(randn(projected_down.shape))
    gate_logits.copy_(randn(gate_logits.shape))
    block_output.copy_(randn(block_output.shape))
    injection_logits.copy_(randn(injection_logits.shape))
    normalized_ref = hc.reference.grouped_rmsnorm(
        state, norm_weight, streams=streams, eps=1e-6
    )
    bottleneck_ref = hc.reference.scaled_silu(projected_down, streams=streams)
    block_input_ref = hc.reference.gate_mean(
        normalized_ref, gate_logits, streams=streams
    )
    combined_ref, next_normalized_ref = hc.reference.combine_norm(
        state,
        block_output,
        injection_logits,
        norm_weight,
        streams=streams,
        eps=1e-6,
    )
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert tuple(tensor.data_ptr() for tensor in captured) == output_addresses
    assert output_addresses == (
        binding.bottleneck.data_ptr(),
        binding.block_input.data_ptr(),
        binding.combined.data_ptr(),
        binding.normalized.data_ptr(),
    )
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured[0], bottleneck_ref, rtol=0, atol=8e-3)
    torch.testing.assert_close(captured[1], block_input_ref, rtol=0, atol=8e-3)
    torch.testing.assert_close(captured[2], combined_ref, rtol=0, atol=8e-3)
    torch.testing.assert_close(captured[3], next_normalized_ref, rtol=0, atol=2e-2)


def test_combine_norm_torch_compile_fullgraph_uses_bound_outputs() -> None:
    device = require_sm120()
    tokens, streams, hidden_size = 2, 4, 2560
    width = streams * hidden_size
    state = torch.randn(
        (tokens, width), dtype=torch.bfloat16, device=device
    ).contiguous()
    block_output = torch.randn(
        (tokens, hidden_size), dtype=torch.bfloat16, device=device
    ).contiguous()
    injection_logits = torch.randn(
        (tokens, streams), dtype=torch.bfloat16, device=device
    ).contiguous()
    weight = (
        torch.randn((width,), dtype=torch.bfloat16, device=device).div_(32).contiguous()
    )
    binding = _allocate_binding(
        device=device,
        tokens=tokens,
        hidden_size=hidden_size,
        streams=streams,
    )

    def run(
        live_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hc.run_combine_norm(
            live_state,
            block_output,
            injection_logits,
            weight,
            eps=1e-6,
            binding=binding,
        )

    # Warm the Triton specialization before compiling, matching serving setup.
    run(state)
    compiled = torch.compile(run, fullgraph=True)

    state.copy_(torch.randn_like(state))
    expected_combined, expected_normalized = hc.reference.combine_norm(
        state,
        block_output,
        injection_logits,
        weight,
        streams=streams,
        eps=1e-6,
    )
    combined, normalized = compiled(state)
    torch.cuda.synchronize(device)

    assert combined.data_ptr() == binding.combined.data_ptr()
    assert normalized.data_ptr() == binding.normalized.data_ptr()
    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=8e-3)
    torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=2e-2)


def test_bind_inside_torch_compile_fullgraph_uses_live_views() -> None:
    device = require_sm120()
    tokens, capacity, streams, hidden_size, lowrank = 2, 8, 4, 64, 16
    width = streams * hidden_size
    plan = hc.plan(
        hc.Caps(
            device=device,
            max_tokens=capacity,
            hidden_size=hidden_size,
            streams=streams,
            lowrank=lowrank,
        )
    )
    outputs = {
        "normalized": torch.empty(
            (capacity, width), dtype=torch.bfloat16, device=device
        ),
        "bottleneck": torch.empty(
            (capacity, lowrank), dtype=torch.bfloat16, device=device
        ),
        "block_input": torch.empty(
            (capacity, hidden_size), dtype=torch.bfloat16, device=device
        ),
        "combined": torch.empty((capacity, width), dtype=torch.bfloat16, device=device),
    }
    block_output = torch.randn(
        (tokens, hidden_size), dtype=torch.bfloat16, device=device
    ).contiguous()
    injection_logits = torch.randn(
        (tokens, streams), dtype=torch.bfloat16, device=device
    ).contiguous()
    weight = torch.randn((width,), dtype=torch.bfloat16, device=device).contiguous()

    # Validate the fixed workspace before compiling its live-prefix binding.
    hc.bind(plan, tokens=capacity, **outputs)

    def run(live_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        binding = hc.bind(plan, tokens=live_state.shape[0], **outputs)
        return hc.run_combine_norm(
            live_state,
            block_output,
            injection_logits,
            weight,
            eps=1e-6,
            binding=binding,
        )

    state = torch.randn(
        (tokens, width), dtype=torch.bfloat16, device=device
    ).contiguous()
    run(state)
    compiled = torch.compile(run, fullgraph=True)
    combined, normalized = compiled(state)
    torch.cuda.synchronize(device)

    assert combined.data_ptr() == outputs["combined"].data_ptr()
    assert normalized.data_ptr() == outputs["normalized"].data_ptr()


def test_torch_compile_rejects_dynamic_input_aliasing_bound_output() -> None:
    device = require_sm120()
    binding = _allocate_binding(
        device=device,
        tokens=1,
        hidden_size=64,
        lowrank=16,
    )
    projected_down = torch.randn(
        (1, 16), dtype=torch.bfloat16, device=device
    ).contiguous()

    def launch(value: torch.Tensor) -> torch.Tensor:
        return hc.run_scaled_silu(value, binding=binding)

    launch(projected_down)
    compiled = torch.compile(launch, fullgraph=True)
    compiled(projected_down)
    with pytest.raises(ValueError, match="bottleneck must not overlap projected_down"):
        compiled(binding.bottleneck)
