"""Capacity planning, binding, and validation for MTP feedback fusion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from b12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)
from b12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_view,
)


_MAX_TRITON_REDUCTION_WIDTH = 65_536


def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


def _positive(name: str, value: int) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Token capacity and static token/multi-stream feedback geometry."""

    device: torch.device | str
    max_tokens: int
    hidden_size: int = 2560
    streams: int = 4
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        device = _canonical_device(self.device)
        if device.type != "cuda":
            raise ValueError(f"MTP feedback requires a CUDA device, got {device}")
        capability = torch.cuda.get_device_capability(device)
        if capability not in ((12, 0), (12, 1)):
            raise ValueError(
                "MTP feedback in b12x requires SM120 or SM121, got "
                f"compute capability {capability[0]}.{capability[1]}"
            )
        max_tokens = _positive("max_tokens", self.max_tokens)
        hidden_size = _positive("hidden_size", self.hidden_size)
        streams = _positive("streams", self.streams)
        if hidden_size > _MAX_TRITON_REDUCTION_WIDTH:
            raise ValueError(
                f"hidden_size must be at most {_MAX_TRITON_REDUCTION_WIDTH}, "
                f"got {hidden_size}"
            )
        if hidden_size % 16:
            raise ValueError(
                f"hidden_size must be divisible by 16 for BF16 dot, got {hidden_size}"
            )
        if streams > _MAX_TRITON_REDUCTION_WIDTH:
            raise ValueError(
                f"streams must be at most {_MAX_TRITON_REDUCTION_WIDTH}, got {streams}"
            )
        if self.dtype != torch.bfloat16:
            raise TypeError(f"MTP feedback requires torch.bfloat16, got {self.dtype}")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "hidden_size", hidden_size)
        object.__setattr__(self, "streams", streams)


@dataclass(frozen=True)
class Plan:
    """Fixed MTP feedback launch policy and scratch-buffer contract."""

    caps: Caps
    token_normalized_offset_bytes: int
    state_partial_sums_offset_bytes: int
    state_normalized_offset_bytes: int
    token_path_offset_bytes: int
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    norm_block_h: int
    norm_block_s: int
    norm_num_warps: int
    matmul_block_m: int = 16
    matmul_block_n: int = 32
    matmul_block_k: int = 32
    matmul_num_warps: int = 4

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def output_shape(self, tokens: int | None = None) -> tuple[int, int, int]:
        live_tokens = self._live_tokens(tokens)
        return (live_tokens, self.caps.streams, self.caps.hidden_size)

    def _live_tokens(self, tokens: int | None) -> int:
        live_tokens = self.caps.max_tokens if tokens is None else int(tokens)
        if live_tokens < 0 or live_tokens > self.caps.max_tokens:
            raise ValueError(
                f"tokens={live_tokens} exceeds capacity {self.caps.max_tokens}"
            )
        return live_tokens

    def bind(self, **kwargs) -> "Binding":
        return bind(self, **kwargs)


@dataclass(frozen=True)
class Binding:
    """Caller-owned MTP feedback inputs, output, and scratch views.

    Model inputs and weights are read-only; ``output`` is the mutable result
    buffer and all intermediate tensors are views of ``scratch``.
    """

    plan: Plan
    tokens: int
    scratch: torch.Tensor
    token_normalized: torch.Tensor
    state_partial_sums: torch.Tensor
    state_normalized: torch.Tensor
    token_path: torch.Tensor
    token_embedding: torch.Tensor
    multi_state: torch.Tensor
    token_norm_weight: torch.Tensor
    state_norm_weight: torch.Tensor
    embedding_fc_weight: torch.Tensor
    hidden_fc_weight: torch.Tensor
    output: torch.Tensor


def plan(caps: Caps) -> Plan:
    """Plan MTP feedback fusion for a fixed token capacity."""

    if not isinstance(caps, Caps):
        raise TypeError(f"caps must be Caps, got {type(caps)!r}")
    h = caps.hidden_size
    s = caps.streams
    token_normalized_offset_bytes = align_up(0, SCRATCH_ALIGN_BYTES)
    cursor = token_normalized_offset_bytes + (
        caps.max_tokens * h * dtype_nbytes(caps.dtype)
    )
    state_partial_sums_offset_bytes = align_up(cursor, SCRATCH_ALIGN_BYTES)
    cursor = state_partial_sums_offset_bytes + (
        caps.max_tokens * s * dtype_nbytes(torch.float32)
    )
    state_normalized_offset_bytes = align_up(cursor, SCRATCH_ALIGN_BYTES)
    cursor = state_normalized_offset_bytes + (
        caps.max_tokens * s * h * dtype_nbytes(caps.dtype)
    )
    token_path_offset_bytes = align_up(cursor, SCRATCH_ALIGN_BYTES)
    cursor = token_path_offset_bytes + (caps.max_tokens * h * dtype_nbytes(caps.dtype))
    spec = scratch_buffer_spec("mtp_feedback", nbytes=cursor, device=caps.device)
    norm_block_h = _next_power_of_two(h)
    return Plan(
        caps=caps,
        token_normalized_offset_bytes=token_normalized_offset_bytes,
        state_partial_sums_offset_bytes=state_partial_sums_offset_bytes,
        state_normalized_offset_bytes=state_normalized_offset_bytes,
        token_path_offset_bytes=token_path_offset_bytes,
        _scratch_specs=(spec,),
        norm_block_h=norm_block_h,
        norm_block_s=_next_power_of_two(s),
        norm_num_warps=8 if norm_block_h >= 2048 else 4,
    )


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    caps: Caps,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != caps.dtype:
        raise TypeError(f"{name} must have dtype {caps.dtype}, got {tensor.dtype}")
    if tensor.device != caps.device:
        raise ValueError(f"{name} must be on {caps.device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.untyped_storage().data_ptr()) + int(
        tensor.storage_offset()
    ) * int(tensor.element_size())
    return start, start + int(tensor.numel()) * int(tensor.element_size())


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start, left_end = _byte_interval(left)
    right_start, right_end = _byte_interval(right)
    return left_start < right_end and right_start < left_end


def bind(
    plan: Plan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    token_norm_weight: torch.Tensor,
    state_norm_weight: torch.Tensor,
    embedding_fc_weight: torch.Tensor,
    hidden_fc_weight: torch.Tensor,
    output: torch.Tensor,
    tokens: int | None = None,
) -> Binding:
    """Bind fixed-capacity tensors without allocating runtime storage."""
    if not isinstance(plan, Plan):
        raise TypeError(f"plan must be Plan, got {type(plan)!r}")
    caps = plan.caps
    live_tokens = plan._live_tokens(tokens)
    scratch_storage = scratch_tensor(
        scratch,
        plan.scratch_specs(),
        owner="MTP feedback",
    )
    token_normalized, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.token_normalized_offset_bytes,
        shape=(caps.max_tokens, caps.hidden_size),
        dtype=caps.dtype,
    )
    state_partial_sums, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.state_partial_sums_offset_bytes,
        shape=(caps.max_tokens, caps.streams),
        dtype=torch.float32,
    )
    state_normalized, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.state_normalized_offset_bytes,
        shape=(caps.max_tokens, caps.streams, caps.hidden_size),
        dtype=caps.dtype,
    )
    token_path, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.token_path_offset_bytes,
        shape=(caps.max_tokens, caps.hidden_size),
        dtype=caps.dtype,
    )
    capacity_shapes = {
        "token_embedding": (caps.max_tokens, caps.hidden_size),
        "multi_state": (caps.max_tokens, caps.streams, caps.hidden_size),
        "token_norm_weight": (caps.hidden_size,),
        "state_norm_weight": (caps.streams * caps.hidden_size,),
        "embedding_fc_weight": (caps.hidden_size, caps.hidden_size),
        "hidden_fc_weight": (caps.hidden_size, caps.hidden_size),
        "output": (caps.max_tokens, caps.streams, caps.hidden_size),
    }
    tensors = {
        "token_embedding": token_embedding,
        "multi_state": multi_state,
        "token_norm_weight": token_norm_weight,
        "state_norm_weight": state_norm_weight,
        "embedding_fc_weight": embedding_fc_weight,
        "hidden_fc_weight": hidden_fc_weight,
        "output": output,
    }
    for name, shape in capacity_shapes.items():
        _require_tensor(name, tensors[name], shape=shape, caps=caps)

    mutable = (("scratch", scratch_storage), ("output", output))
    read_only = tuple(
        (name, tensor) for name, tensor in tensors.items() if name != "output"
    )
    for index, (left_name, left) in enumerate(mutable):
        for right_name, right in mutable[index + 1 :]:
            if _overlaps(left, right):
                raise ValueError(
                    f"mutable buffers {left_name} and {right_name} must not overlap"
                )
        for right_name, right in read_only:
            if _overlaps(left, right):
                raise ValueError(
                    f"mutable buffer {left_name} must not overlap read-only "
                    f"tensor {right_name}"
                )

    return Binding(
        plan=plan,
        tokens=live_tokens,
        scratch=scratch_storage,
        token_normalized=token_normalized[:live_tokens],
        state_partial_sums=state_partial_sums[:live_tokens],
        state_normalized=state_normalized[:live_tokens],
        token_path=token_path[:live_tokens],
        token_embedding=token_embedding[:live_tokens],
        multi_state=multi_state[:live_tokens],
        token_norm_weight=token_norm_weight,
        state_norm_weight=state_norm_weight,
        embedding_fc_weight=embedding_fc_weight,
        hidden_fc_weight=hidden_fc_weight,
        output=output[:live_tokens],
    )


def run(binding: Binding, *, eps: float = 1e-6) -> torch.Tensor:
    """Write and return the caller-owned BF16 ``[T,S,H]`` draft input."""
    if not isinstance(binding, Binding):
        raise TypeError(f"binding must be Binding, got {type(binding)!r}")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps_value}")
    if binding.tokens == 0:
        return binding.output
    from ._kernels import run_mtp_feedback

    plan_value = binding.plan
    caps = plan_value.caps
    run_mtp_feedback(
        binding.token_embedding,
        binding.multi_state,
        binding.token_norm_weight,
        binding.state_norm_weight,
        binding.embedding_fc_weight,
        binding.hidden_fc_weight,
        binding.scratch,
        binding.output,
        eps=eps_value,
        max_tokens=caps.max_tokens,
        streams=caps.streams,
        hidden_size=caps.hidden_size,
        token_normalized_offset_bytes=plan_value.token_normalized_offset_bytes,
        state_partial_sums_offset_bytes=plan_value.state_partial_sums_offset_bytes,
        state_normalized_offset_bytes=plan_value.state_normalized_offset_bytes,
        token_path_offset_bytes=plan_value.token_path_offset_bytes,
        norm_block_h=plan_value.norm_block_h,
        norm_block_s=plan_value.norm_block_s,
        norm_num_warps=plan_value.norm_num_warps,
        matmul_block_m=plan_value.matmul_block_m,
        matmul_block_n=plan_value.matmul_block_n,
        matmul_block_k=plan_value.matmul_block_k,
        matmul_num_warps=plan_value.matmul_num_warps,
    )
    return binding.output


__all__ = ["Caps", "Plan", "Binding", "plan", "bind", "run"]
