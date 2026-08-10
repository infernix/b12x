"""End-to-end QSRT W4A8 MoE path.

The route-major path keeps the encoded weights compact throughout: activation
transforms feed MXFP8 operands, and the fixed-rate trellis payload is decoded
directly into E4M3 MMA registers. The caller owns all scratch so the complete
sequence is CUDA-graph safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.utils import cuda_stream_to_int
from b12x.gemm._shared.wo_mxfp8 import (
    MXFP8Rows,
    empty_mxfp8_rows_for_dense_gemm,
)
from b12x.moe._shared.kernels.trellis_w4a8_pair import (
    run_trellis_w4a8_fc1_routes,
    run_trellis_w4a8_fc2_routes,
)
from b12x.moe._shared.kernels.trellis_w4a8_uniform import (
    run_trellis_w4a8_uniform_fc1,
    run_trellis_w4a8_uniform_fc2,
)
from b12x.moe._shared.kernels.trellis_w4a8_transform import (
    run_trellis_w4a8_activation_rotation_quant,
    run_trellis_w4a8_coupled_activation_quant,
    run_trellis_w4a8_coupled_input_quant,
    run_trellis_w4a8_input_rotation_quant,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    _w4a16_topk_sum_launch_flat,
)


@dataclass(frozen=True)
class TrellisW4A8MoeScratch:
    """Caller-owned intermediate storage for one fixed ``(M, top-k)`` shape."""

    route_experts: torch.Tensor
    # Shared-su checkpoints need one transformed/quantized FC1 operand per
    # token, not one duplicate per routed expert.
    gate_quantized: MXFP8Rows
    up_quantized: MXFP8Rows
    gate_fc1: torch.Tensor
    up_fc1: torch.Tensor
    activated_quantized: MXFP8Rows
    fc2: torch.Tensor
    output: torch.Tensor


def make_trellis_w4a8_moe_scratch(
    *,
    m: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device | str,
    coupled_hadamard: bool = False,
) -> TrellisW4A8MoeScratch:
    """Allocate the fixed buffers needed by :func:`run_trellis_w4a8_moe`."""

    m = int(m)
    topk = int(topk)
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    if m <= 0 or topk <= 0:
        raise ValueError("m and topk must be positive")
    if hidden_size <= 0 or hidden_size % 128:
        raise ValueError("hidden_size must be a positive multiple of 128")
    if intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError("intermediate_size must be a positive multiple of 128")
    device = torch.device(device)
    routes = m * topk
    gate_fc1 = torch.empty(
        (routes, intermediate_size), dtype=torch.float16, device=device
    )
    up_fc1 = torch.empty_like(gate_fc1)
    fc2 = torch.empty(
        (routes, hidden_size), dtype=torch.float16, device=device
    )
    gate_quantized = empty_mxfp8_rows_for_dense_gemm(
        m, hidden_size, device=device
    )
    up_quantized = (
        gate_quantized
        if coupled_hadamard
        else empty_mxfp8_rows_for_dense_gemm(m, hidden_size, device=device)
    )
    return TrellisW4A8MoeScratch(
        route_experts=torch.empty((routes,), dtype=torch.int32, device=device),
        gate_quantized=gate_quantized,
        up_quantized=up_quantized,
        gate_fc1=gate_fc1,
        up_fc1=up_fc1,
        activated_quantized=empty_mxfp8_rows_for_dense_gemm(
            routes, intermediate_size, device=device
        ),
        fc2=fc2,
        output=torch.empty((m, hidden_size), dtype=torch.float32, device=device),
    )


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def run_trellis_w4a8_moe(
    source: torch.Tensor,
    prepared,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    scratch: TrellisW4A8MoeScratch,
    *,
    expert_map: torch.Tensor | None = None,
    fast_math: bool = False,
) -> torch.Tensor:
    """Run a homogeneous compact E4M3 trellis tier through native W4A8 MMA.

    ``topk_ids`` are local indices into ``prepared`` unless ``expert_map`` is
    supplied.  In that case it is an int32 global-to-local table; ``-1`` map
    entries are inactive routes and contribute exactly zero.
    """

    if source.ndim != 2 or source.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("source must be contiguous BF16/FP16 [M,H]")
    if not source.is_contiguous():
        raise ValueError("source must be contiguous")
    if topk_ids.ndim != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be contiguous int32/int64 [M,top-k]")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if topk_weights.shape != topk_ids.shape or topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must be contiguous FP32 with topk_ids shape")
    if not topk_weights.is_contiguous():
        raise ValueError("topk_weights must be contiguous")
    if topk_ids.device != source.device or topk_weights.device != source.device:
        raise ValueError("source and routing tensors must share one CUDA device")

    m, hidden_size = map(int, source.shape)
    topk = int(topk_ids.shape[1])
    routes = m * topk
    intermediate_size = int(prepared.intermediate_size)
    if int(prepared.hidden_size) != hidden_size:
        raise ValueError(
            f"source hidden size {hidden_size} != prepared {prepared.hidden_size}"
        )
    if intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError("W4A8 requires intermediate_size divisible by 128")
    trellis_codebook = str(getattr(prepared, "trellis_codebook", "")).lower()
    if trellis_codebook != "sqg_xor_cheb_t12":
        raise NotImplementedError(
            "native W4A8 trellis execution requires SQG-XOR-Cheb-T12, "
            f"got {trellis_codebook!r}"
        )
    if int(prepared.gate_suh.shape[0]) != 1 or int(prepared.up_suh.shape[0]) != 1:
        raise ValueError(
            "the direct QSRT W4A8 path requires shared gate/up input rotations"
        )
    coupled_hadamard = bool(getattr(prepared, "coupled_hadamard", False))
    if coupled_hadamard:
        if hidden_size % 512:
            raise ValueError("coupled W4A8 requires hidden_size divisible by 512")
        if prepared.gate_suh.data_ptr() != prepared.up_suh.data_ptr():
            raise ValueError("coupled W4A8 requires one shared gate/up input scale")
        if int(prepared.intermediate_rotations.shape[1]) != 6 * intermediate_size:
            raise ValueError(
                "coupled W4A8 requires six intermediate rotation vectors"
            )
        if scratch.gate_quantized is not scratch.up_quantized:
            raise ValueError(
                "coupled W4A8 scratch must share one quantized FC1 operand"
            )
    pair_fc1 = getattr(prepared, "fc1_trellis_pair_kind", None) is not None
    pair_fc2 = getattr(prepared, "fc2_trellis_pair_kind", None) is not None
    if pair_fc1 != pair_fc2:
        raise ValueError("W4A8 requires both matrices to use pair or uniform storage")
    if coupled_hadamard and pair_fc1:
        raise ValueError("coupled W4A8 requires native fixed-rate storage")
    for name, operand in (
        ("gate_quantized", scratch.gate_quantized),
        ("up_quantized", scratch.up_quantized),
    ):
        if tuple(operand.values.shape[:2]) != (m, hidden_size):
            raise ValueError(
                f"scratch.{name}.values must have leading shape "
                f"{(m, hidden_size)}, got {tuple(operand.values.shape)}"
            )

    _check_tensor(
        "scratch.route_experts",
        scratch.route_experts,
        shape=(routes,),
        dtype=torch.int32,
        device=source.device,
    )
    for name, tensor, shape in (
        ("gate_fc1", scratch.gate_fc1, (routes, intermediate_size)),
        ("up_fc1", scratch.up_fc1, (routes, intermediate_size)),
        ("fc2", scratch.fc2, (routes, hidden_size)),
    ):
        _check_tensor(
            f"scratch.{name}",
            tensor,
            shape=shape,
            dtype=torch.float16,
            device=source.device,
        )
    _check_tensor(
        "scratch.output",
        scratch.output,
        shape=(m, hidden_size),
        dtype=torch.float32,
        device=source.device,
    )

    if expert_map is None:
        scratch.route_experts.copy_(topk_ids.reshape(-1))
    else:
        if (
            expert_map.ndim != 1
            or expert_map.dtype != torch.int32
            or expert_map.device != source.device
            or not expert_map.is_contiguous()
        ):
            raise TypeError(
                "expert_map must be contiguous int32 [global experts] on source device"
            )
        torch.index_select(
            expert_map,
            0,
            topk_ids.reshape(-1),
            out=scratch.route_experts,
        )
    if coupled_hadamard:
        run_trellis_w4a8_coupled_input_quant(
            source, prepared, scratch.gate_quantized
        )
    else:
        run_trellis_w4a8_input_rotation_quant(
            source,
            scratch.route_experts,
            prepared,
            scratch.gate_quantized,
            scratch.up_quantized,
            topk=topk,
        )
    if pair_fc1:
        run_trellis_w4a8_fc1_routes(
            scratch.gate_quantized,
            scratch.up_quantized,
            prepared,
            scratch.route_experts,
            scratch.gate_fc1,
            scratch.up_fc1,
            topk=topk,
            shared_input=True,
            decode_both=False,
            sqg_direct_lut=True,
        )
    else:
        run_trellis_w4a8_uniform_fc1(
            scratch.gate_quantized,
            scratch.up_quantized,
            prepared,
            scratch.route_experts,
            scratch.gate_fc1,
            scratch.up_fc1,
            topk=topk,
            shared_input=True,
        )
    if coupled_hadamard:
        run_trellis_w4a8_coupled_activation_quant(
            scratch.gate_fc1,
            scratch.up_fc1,
            scratch.route_experts,
            prepared,
            scratch.activated_quantized,
            fast_math=fast_math,
        )
    else:
        run_trellis_w4a8_activation_rotation_quant(
            scratch.gate_fc1,
            scratch.up_fc1,
            scratch.route_experts,
            prepared,
            scratch.activated_quantized,
            fast_math=fast_math,
        )
    if pair_fc2:
        run_trellis_w4a8_fc2_routes(
            scratch.activated_quantized,
            prepared,
            scratch.route_experts,
            scratch.fc2,
            sqg_direct_lut=True,
        )
    else:
        run_trellis_w4a8_uniform_fc2(
            scratch.activated_quantized,
            prepared,
            scratch.route_experts,
            scratch.fc2,
        )
    _w4a16_topk_sum_launch_flat(
        scratch.fc2,
        scratch.output,
        m,
        topk,
        hidden_size,
        "fp16",
        int(cuda_stream_to_int(torch.cuda.current_stream())),
        full_rotation=True,
        num_experts=int(prepared.num_experts),
        topk_weights=topk_weights,
        route_expert_ids=scratch.route_experts,
        svh_table=prepared.down_svh,
        coupled_hadamard=coupled_hadamard,
    )
    return scratch.output


__all__ = [
    "TrellisW4A8MoeScratch",
    "make_trellis_w4a8_moe_scratch",
    "run_trellis_w4a8_moe",
]
