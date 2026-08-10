"""Route-major native-rate QSRT W4A8 matrix products.

The kernels consume projection-major native trellis tiles at one fixed rate.
They decode SQG-XOR-Cheb-T12 values directly into E4M3 MMA fragments while
keeping activations in native MXFP8 row order.  No dense weight tensor is
materialized.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint16, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import (
    get_ptr_as_int64,
    mxfp8_mma_m16n8k32_f32_e4m3,
)
from b12x._lib.quant.sqg_e4m3 import sqg_xor_cheb_t12_lut
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows
from b12x.moe._shared.kernels.trellis_decode import _NativeTrellisDecode


_FC1_THREADS = 256
_FC2_THREADS = 128
_CODEBOOK = "sqg_xor_cheb_t12"


class TrellisW4A8UniformFC1Kernel(_NativeTrellisDecode):
    """Compute one route/projection/N32 slab with K-split reduction."""

    threads = _FC1_THREADS

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        bits: int,
        topk: int,
        shared_input: bool,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.bits = int(bits)
        self.topk = int(topk)
        self.shared_input = bool(shared_input)
        if self.bits not in (2, 3, 4):
            raise ValueError("uniform W4A8 supports K2, K3, or K4")
        if self.hidden_size % 32 or self.intermediate_size % 32:
            raise ValueError("uniform W4A8 dimensions must be divisible by 32")
        self.k16_count = self.hidden_size // 16
        self.n16_count = self.intermediate_size // 16
        self.expert_u32 = (
            self.k16_count * self.n16_count * 8 * self.bits
        )

    @cute.jit
    def __call__(
        self,
        gate_values_ptr: cute.Pointer,
        gate_scale_ptr: cute.Pointer,
        up_values_ptr: cute.Pointer,
        up_scale_ptr: cute.Pointer,
        w13_ptr: cute.Pointer,
        rank_lut_ptr: cute.Pointer,
        route_experts_ptr: cute.Pointer,
        gate_out_ptr: cute.Pointer,
        up_out_ptr: cute.Pointer,
        routes: Int32,
        num_experts: Int32,
        stream: cuda.CUstream,
    ) -> None:
        input_rows = routes // Int32(self.topk) if self.shared_input else routes
        gate_values = cute.make_tensor(
            gate_values_ptr,
            cute.make_ordered_layout(
                (input_rows, self.hidden_size // 4), order=(1, 0)
            ),
        )
        gate_scales = cute.make_tensor(
            gate_scale_ptr,
            cute.make_ordered_layout(
                (input_rows, self.hidden_size // 32), order=(1, 0)
            ),
        )
        up_values = cute.make_tensor(
            up_values_ptr,
            cute.make_ordered_layout(
                (input_rows, self.hidden_size // 4), order=(1, 0)
            ),
        )
        up_scales = cute.make_tensor(
            up_scale_ptr,
            cute.make_ordered_layout(
                (input_rows, self.hidden_size // 32), order=(1, 0)
            ),
        )
        w13 = cute.make_tensor(
            w13_ptr,
            cute.make_layout(
                (Int64(2 * self.expert_u32) * Int64(num_experts),)
            ),
        )
        rank_lut = cute.make_tensor(rank_lut_ptr, cute.make_layout((4096,)))
        route_experts = cute.make_tensor(
            route_experts_ptr, cute.make_layout((routes,))
        )
        gate_out = cute.make_tensor(
            gate_out_ptr,
            cute.make_ordered_layout(
                (routes, self.intermediate_size), order=(1, 0)
            ),
        )
        up_out = cute.make_tensor(
            up_out_ptr,
            cute.make_ordered_layout(
                (routes, self.intermediate_size), order=(1, 0)
            ),
        )
        self.kernel(
            gate_values,
            gate_scales,
            up_values,
            up_scales,
            w13,
            rank_lut,
            route_experts,
            gate_out,
            up_out,
            routes,
            num_experts,
        ).launch(
            grid=(routes, 2, self.intermediate_size // 32),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _tile_base(
        self,
        projection: Int32,
        expert: Int32,
        k16: Int32,
        n16: Int32,
        num_experts: Int32,
    ) -> Int64:
        return (
            Int64(projection)
            * Int64(num_experts)
            * Int64(self.expert_u32)
            + Int64(expert) * Int64(self.expert_u32)
            + Int64(k16) * Int64(self.n16_count * 8 * self.bits)
            + Int64(n16) * Int64(8 * self.bits)
        )

    @cute.kernel
    def kernel(
        self,
        gate_values: cute.Tensor,
        gate_scales: cute.Tensor,
        up_values: cute.Tensor,
        up_scales: cute.Tensor,
        w13: cute.Tensor,
        rank_lut: cute.Tensor,
        route_experts: cute.Tensor,
        gate_out: cute.Tensor,
        up_out: cute.Tensor,
        routes: Int32,
        num_experts: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        route_idx, projection_idx, n32_idx = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        route = Int32(route_idx)
        projection = Int32(projection_idx)
        n32 = Int32(n32_idx)
        raw_expert = route_experts[route].to(Int32)
        valid = Int32(0)
        expert = Int32(0)
        if raw_expert >= Int32(0) and raw_expert < num_experts:
            valid = Int32(1)
            expert = raw_expert
        rank_lut_addr = get_ptr_as_int64(rank_lut, Int32(0))
        input_row = route // Int32(self.topk) if self.shared_input else route

        acc = tuple(cute.make_rmem_tensor((4,), Float32) for _ in range(4))
        for nt in cutlass.range_constexpr(4):
            acc[nt].fill(0.0)

        k32_count = Int32(self.hidden_size // 32)
        k32_slice = Int32(
            (self.hidden_size // 32 + self.threads // 32 - 1)
            // (self.threads // 32)
        )
        k32 = warp * k32_slice
        k32_end = k32 + k32_slice
        if k32_end > k32_count:
            k32_end = k32_count
        if valid == Int32(0):
            k32 = k32_end
        while k32 < k32_end:
            a0, a1, a2, a3, sfa = self._a_fragment(
                gate_values, gate_scales, input_row, k32, lane
            )
            if projection != Int32(0):
                a0, a1, a2, a3, sfa = self._a_fragment(
                    up_values, up_scales, input_row, k32, lane
                )
            for npair in cutlass.range_constexpr(2):
                n16 = n32 * Int32(2) + Int32(npair)
                base0 = self._tile_base(
                    projection,
                    expert,
                    k32 * Int32(2),
                    n16,
                    num_experts,
                )
                base1 = self._tile_base(
                    projection,
                    expert,
                    k32 * Int32(2) + Int32(1),
                    n16,
                    num_experts,
                )
                b0_lo, b1_lo, b0_hi, b1_hi = self._pair_native_words_both(
                    w13,
                    lane,
                    base0,
                    base1,
                    self.bits,
                    rank_lut_addr,
                )
                for half in cutlass.range_constexpr(2):
                    nt = npair * 2 + half
                    b0 = b0_lo
                    b1 = b1_lo
                    if cutlass.const_expr(half == 1):
                        b0 = b0_hi
                        b1 = b1_hi
                    frag = acc[nt]
                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        frag[0],
                        frag[1],
                        frag[2],
                        frag[3],
                        a0,
                        a1,
                        a2,
                        a3,
                        b0,
                        b1,
                        sfa,
                        Uint32(0x7F7F7F7F),
                    )
                    frag[0] = d0
                    frag[1] = d1
                    frag[2] = d2
                    frag[3] = d3
            k32 += Int32(1)

        red_ptr = cute.arch.alloc_smem(
            Float32, (self.threads // 32) * 32
        )
        red = cute.make_tensor(
            red_ptr, cute.make_layout(((self.threads // 32) * 32,))
        )
        for nt in cutlass.range_constexpr(4):
            for half in cutlass.range_constexpr(2):
                cute.arch.sync_threads()
                if lane < Int32(4):
                    red[warp * Int32(4) + lane] = acc[nt][half]
                cute.arch.sync_threads()
                if warp == Int32(0) and lane < Int32(4):
                    total = Float32(0.0)
                    for source in cutlass.range_constexpr(self.threads // 32):
                        total += red[Int32(source * 4) + lane]
                    acc[nt][half] = total

        if warp == Int32(0) and lane < Int32(4):
            c = lane & Int32(3)
            for nt in cutlass.range_constexpr(4):
                col = n32 * Int32(32) + Int32(nt * 8) + c * Int32(2)
                out0 = Float32(0.0)
                out1 = Float32(0.0)
                if valid != Int32(0):
                    out0 = acc[nt][0]
                    out1 = acc[nt][1]
                if projection == Int32(0):
                    gate_out[route, col] = cutlass.Float16(out0)
                    gate_out[route, col + Int32(1)] = cutlass.Float16(out1)
                else:
                    up_out[route, col] = cutlass.Float16(out0)
                    up_out[route, col + Int32(1)] = cutlass.Float16(out1)


class TrellisW4A8UniformFC2Kernel(_NativeTrellisDecode):
    """Compute one route/output-N128 slab at one fixed trellis rate."""

    threads = _FC2_THREADS

    def __init__(
        self, *, hidden_size: int, intermediate_size: int, bits: int
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.bits = int(bits)
        if self.bits not in (2, 3, 4):
            raise ValueError("uniform W4A8 supports K2, K3, or K4")
        if self.hidden_size % 128 or self.intermediate_size % 32:
            raise ValueError("uniform W4A8 FC2 requires H%128=0 and I%32=0")
        self.k16_count = self.intermediate_size // 16
        self.n16_count = self.hidden_size // 16
        self.expert_u32 = (
            self.k16_count * self.n16_count * 8 * self.bits
        )

    @cute.jit
    def __call__(
        self,
        values_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        rank_lut_ptr: cute.Pointer,
        route_experts_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        routes: Int32,
        num_experts: Int32,
        stream: cuda.CUstream,
    ) -> None:
        values = cute.make_tensor(
            values_ptr,
            cute.make_ordered_layout(
                (routes, self.intermediate_size // 4), order=(1, 0)
            ),
        )
        scales = cute.make_tensor(
            scale_ptr,
            cute.make_ordered_layout(
                (routes, self.intermediate_size // 32), order=(1, 0)
            ),
        )
        w2 = cute.make_tensor(
            w2_ptr,
            cute.make_layout((Int64(num_experts) * Int64(self.expert_u32),)),
        )
        rank_lut = cute.make_tensor(rank_lut_ptr, cute.make_layout((4096,)))
        route_experts = cute.make_tensor(
            route_experts_ptr, cute.make_layout((routes,))
        )
        output = cute.make_tensor(
            output_ptr,
            cute.make_ordered_layout((routes, self.hidden_size), order=(1, 0)),
        )
        self.kernel(
            values,
            scales,
            w2,
            rank_lut,
            route_experts,
            output,
            routes,
            num_experts,
        ).launch(
            grid=(routes, self.hidden_size // 128, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _tile_base(
        self, expert: Int32, k16: Int32, n16: Int32
    ) -> Int64:
        return (
            Int64(expert) * Int64(self.expert_u32)
            + Int64(k16) * Int64(self.n16_count * 8 * self.bits)
            + Int64(n16) * Int64(8 * self.bits)
        )

    @cute.kernel
    def kernel(
        self,
        values: cute.Tensor,
        scales: cute.Tensor,
        w2: cute.Tensor,
        rank_lut: cute.Tensor,
        route_experts: cute.Tensor,
        output: cute.Tensor,
        routes: Int32,
        num_experts: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        route_idx, output_tile_idx, _ = cute.arch.block_idx()
        lane = Int32(tidx) & Int32(31)
        warp = Int32(tidx) >> Int32(5)
        route = Int32(route_idx)
        output_tile = Int32(output_tile_idx)
        raw_expert = route_experts[route].to(Int32)
        valid = Int32(0)
        expert = Int32(0)
        if raw_expert >= Int32(0) and raw_expert < num_experts:
            valid = Int32(1)
            expert = raw_expert
        rank_lut_addr = get_ptr_as_int64(rank_lut, Int32(0))
        acc = tuple(cute.make_rmem_tensor((4,), Float32) for _ in range(4))
        for nt in cutlass.range_constexpr(4):
            acc[nt].fill(0.0)

        k32 = Int32(0)
        if valid == Int32(0):
            k32 = Int32(self.intermediate_size // 32)
        while k32 < Int32(self.intermediate_size // 32):
            a0, a1, a2, a3, sfa = self._a_fragment(
                values, scales, route, k32, lane
            )
            for npair in cutlass.range_constexpr(2):
                n16 = (
                    output_tile * Int32(8)
                    + warp * Int32(2)
                    + Int32(npair)
                )
                base0 = self._tile_base(
                    expert, k32 * Int32(2), n16
                )
                base1 = self._tile_base(
                    expert, k32 * Int32(2) + Int32(1), n16
                )
                b0_lo, b1_lo, b0_hi, b1_hi = self._pair_native_words_both(
                    w2,
                    lane,
                    base0,
                    base1,
                    self.bits,
                    rank_lut_addr,
                )
                lo_idx = npair * 2
                hi_idx = lo_idx + 1
                for half in cutlass.range_constexpr(2):
                    nt = lo_idx
                    b0 = b0_lo
                    b1 = b1_lo
                    if cutlass.const_expr(half == 1):
                        nt = hi_idx
                        b0 = b0_hi
                        b1 = b1_hi
                    frag = acc[nt]
                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        frag[0],
                        frag[1],
                        frag[2],
                        frag[3],
                        a0,
                        a1,
                        a2,
                        a3,
                        b0,
                        b1,
                        sfa,
                        Uint32(0x7F7F7F7F),
                    )
                    frag[0] = d0
                    frag[1] = d1
                    frag[2] = d2
                    frag[3] = d3
            k32 += Int32(1)

        if lane < Int32(4):
            c = lane & Int32(3)
            for nt in cutlass.range_constexpr(4):
                col = (
                    output_tile * Int32(128)
                    + warp * Int32(32)
                    + Int32(nt * 8)
                    + c * Int32(2)
                )
                out0 = Float32(0.0)
                out1 = Float32(0.0)
                if valid != Int32(0):
                    out0 = acc[nt][0]
                    out1 = acc[nt][1]
                output[route, col] = cutlass.Float16(out0)
                output[route, col + Int32(1)] = cutlass.Float16(out1)


def _ptr(dtype, address: int):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=16)


@functools.cache
def _compile_fc1(
    hidden_size: int,
    intermediate_size: int,
    bits: int,
    topk: int,
    shared_input: bool,
    device_index: int,
):
    launch = TrellisW4A8UniformFC1Kernel(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bits=bits,
        topk=topk,
        shared_input=shared_input,
    )
    key = (
        hidden_size,
        intermediate_size,
        bits,
        topk,
        shared_input,
        device_index,
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Float16, 16),
        _ptr(cutlass.Float16, 16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.trellis_w4a8_uniform_fc1", 1, key
        ),
    )


@functools.cache
def _compile_fc2(
    hidden_size: int,
    intermediate_size: int,
    bits: int,
    device_index: int,
):
    launch = TrellisW4A8UniformFC2Kernel(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bits=bits,
    )
    key = (hidden_size, intermediate_size, bits, device_index)
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Uint32, 16),
        _ptr(cutlass.Uint8, 16),
        _ptr(cutlass.Int32, 16),
        _ptr(cutlass.Float16, 16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.trellis_w4a8_uniform_fc2", 1, key
        ),
    )


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else int(device.index)


def run_trellis_w4a8_uniform_fc1(
    gate_a: MXFP8Rows,
    up_a: MXFP8Rows,
    prepared,
    route_experts: torch.Tensor,
    gate_out: torch.Tensor,
    up_out: torch.Tensor,
    *,
    topk: int,
    shared_input: bool,
) -> None:
    if str(getattr(prepared, "trellis_codebook", "")).lower() != _CODEBOOK:
        raise NotImplementedError("uniform W4A8 requires SQG-XOR-Cheb-T12")
    bits = int(prepared.trellis_bits)
    compiled = _compile_fc1(
        int(prepared.hidden_size),
        int(prepared.intermediate_size),
        bits,
        int(topk),
        bool(shared_input),
        _device_index(route_experts.device),
    )
    rank_lut = sqg_xor_cheb_t12_lut(route_experts.device)
    compiled(
        _ptr(cutlass.Uint32, gate_a.values.data_ptr()),
        _ptr(cutlass.Uint8, gate_a.scale_rows.data_ptr()),
        _ptr(cutlass.Uint32, up_a.values.data_ptr()),
        _ptr(cutlass.Uint8, up_a.scale_rows.data_ptr()),
        _ptr(cutlass.Uint32, prepared.w13.data_ptr()),
        _ptr(cutlass.Uint8, rank_lut.data_ptr()),
        _ptr(cutlass.Int32, route_experts.data_ptr()),
        _ptr(cutlass.Float16, gate_out.data_ptr()),
        _ptr(cutlass.Float16, up_out.data_ptr()),
        int(route_experts.numel()),
        int(prepared.num_experts),
        current_cuda_stream(),
    )


def run_trellis_w4a8_uniform_fc2(
    a: MXFP8Rows,
    prepared,
    route_experts: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if str(getattr(prepared, "trellis_codebook", "")).lower() != _CODEBOOK:
        raise NotImplementedError("uniform W4A8 requires SQG-XOR-Cheb-T12")
    bits = int(prepared.trellis_bits)
    compiled = _compile_fc2(
        int(prepared.hidden_size),
        int(prepared.intermediate_size),
        bits,
        _device_index(route_experts.device),
    )
    rank_lut = sqg_xor_cheb_t12_lut(route_experts.device)
    compiled(
        _ptr(cutlass.Uint32, a.values.data_ptr()),
        _ptr(cutlass.Uint8, a.scale_rows.data_ptr()),
        _ptr(cutlass.Uint32, prepared.w2.data_ptr()),
        _ptr(cutlass.Uint8, rank_lut.data_ptr()),
        _ptr(cutlass.Int32, route_experts.data_ptr()),
        _ptr(cutlass.Float16, output.data_ptr()),
        int(route_experts.numel()),
        int(prepared.num_experts),
        current_cuda_stream(),
    )


__all__ = [
    "run_trellis_w4a8_uniform_fc1",
    "run_trellis_w4a8_uniform_fc2",
]
