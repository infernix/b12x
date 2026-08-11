"""QSRT trellis weight-source parity for the dynamic W4A8 MoE kernel.

Drives MoEDynamicKernelBackend(quant_recipe="w4a8_trellis") with a uniform
K2 payload and compares against the route-major trellis W4A8 scaffold with
identity Hadamard transforms. Both paths quantize activations to MXFP8 and
decode the identical payload through the T12 staircase, so outputs agree to
accumulation-order tolerance.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import make_ptr

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x.moe.fused_moe._impl import _DynamicMoEW4A8Launch, current_cuda_stream
from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend
from b12x.moe._shared.kernels.trellis_w4a8 import (
    make_trellis_w4a8_moe_scratch,
    run_trellis_w4a8_moe,
)
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)
from b12x._lib.quant.sqg_e4m3 import sqg_xor_cheb_t12_lut_cpu
from tests._reference.helpers import require_b12x
from tests.moe.test_w4a8_dynamic_kernel import (
    _OPT_LEVEL_2,
    _fake_f32,
    _fake_i32,
    _gptr,
)

require_b12x()

_TILE_M = 16
_TILE_N = 128
_BITS = 2


def _h128(device: torch.device) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < 128:
        h = torch.cat(
            (torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0
        )
    return (h / (128.0 ** 0.5)).to(device=device, dtype=torch.float32)


def _run_trellis_dynamic(
    *,
    activation: str,
    E: int,
    m: int,
    K: int,
    n: int,
    top_k: int,
    seed: int,
    keepalive: list | None = None,
    mac: int = 4,
    recipe: str = "w4a8_trellis",
):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    w1_n = 2 * n

    x = (torch.randn(m, K, device=device) * 1.0e-2).to(torch.bfloat16)
    w13_i16 = torch.randint(
        -32768, 32767, (2, E, K // 16, n // 16, 16 * _BITS),
        dtype=torch.int16, device=device,
    )
    w2_i16 = torch.randint(
        -32768, 32767, (E, n // 16, K // 16, 16 * _BITS),
        dtype=torch.int16, device=device,
    )
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(m)]
    ).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(m, top_k, device=device), dim=-1
    ).float()

    # ---- scaffold oracle (identity transforms) ----
    h_scale = torch.ones((1, K), dtype=torch.float16, device=device)
    prepared = prepare_trellis256_moe_weights(
        w13_i16,
        w2_i16,
        hidden_size=K,
        intermediate_size=n,
        num_experts=E,
        activation=activation,
        params_dtype=torch.float16,
        fc1_tile_n=128,
        fc2_tile_n=128,
        w13_layout="trellis3_t256_proj",
        trellis_bits=_BITS,
        codebook="sqg_xor_cheb_t12",
        gate_suh=h_scale,
        up_suh=h_scale,
        intermediate_rotations=torch.ones(
            (E, 3 * n), dtype=torch.float16, device=device
        ),
        down_svh=h_scale,
        tile_config=(128, 128, 128, 128),
    )
    scratch = make_trellis_w4a8_moe_scratch(
        m=m, topk=top_k, hidden_size=K, intermediate_size=n, device=device
    )
    oracle = run_trellis_w4a8_moe(
        x, prepared, topk_weights, topk_ids, scratch
    ).float()
    torch.cuda.synchronize()

    # ---- dynamic kernel, trellis source ----
    # The trellis codec encodes weights in the H128-rotated basis (suh/svh
    # identity here). Rotate the input on the host and un-rotate the output;
    # the activation-boundary rotation runs inside the kernel.
    had = _h128(device)
    x_dyn = (
        (x.float().reshape(m, K // 128, 128) @ had).reshape(m, K).to(torch.bfloat16)
    )
    # Expert-major flat payloads: w13 [E][proj][K16][N16], down [E][K16][N16].
    w13_flat = (
        w13_i16.permute(1, 0, 2, 3, 4).contiguous().view(torch.int32).reshape(-1)
    )
    down_flat = w2_i16.contiguous().view(torch.int32).reshape(-1)
    t12_lut = sqg_xor_cheb_t12_lut_cpu().to(device)
    rot_flat = (
        torch.ones((E, 3 * n), dtype=torch.float16, device=device)
        .reshape(-1)
        .contiguous()
    )
    sentinel_u32 = torch.zeros(1, dtype=torch.uint32, device=device)
    # Dummy packed-FP4 weight tensors: shape-only (TMA descriptors and tile
    # counts); never dereferenced by the trellis staging.
    w13_packed = torch.zeros(E, w1_n, K // 2, dtype=torch.uint8, device=device)
    w2_packed = torch.zeros(E, K, n // 2, dtype=torch.uint8, device=device)
    scale_flat = torch.zeros(
        (E + m * top_k + 1) * _TILE_M * (K // 8), dtype=torch.uint8, device=device
    )

    phys_tiles = E + (m * top_k + _TILE_M - 1) // _TILE_M
    rows_padded = phys_tiles * _TILE_M
    gate_tile_cnt = (w1_n // _TILE_N) // 2
    max_tasks = phys_tiles * max(gate_tile_cnt, 1)

    packed_a = torch.zeros(rows_padded * K, dtype=torch.uint8, device=device)
    intermediate_u32 = torch.zeros(
        rows_padded * (n + n // 32) // 4, dtype=torch.int32, device=device
    )

    def z1():
        return torch.zeros(1, dtype=torch.int32, device=device)

    barrier_count, barrier_epoch = z1(), z1()
    pair_head, producers_done, all_pub = z1(), z1(), z1()
    task_head, task_tail = z1(), z1()

    def zt():
        return torch.zeros(max_tasks, dtype=torch.int32, device=device)

    task_ready, task_expert, task_m_tile = zt(), zt(), zt()
    task_slice_begin, task_slice_count, task_valid_rows = zt(), zt(), zt()
    tile_write_count = torch.zeros(phys_tiles, dtype=torch.int32, device=device)
    row_counts = torch.zeros(E, dtype=torch.int32, device=device)
    expert_write_rows = torch.zeros(E, dtype=torch.int32, device=device)
    expert_tile_base = torch.zeros(E + 1, dtype=torch.int32, device=device)
    token_map = torch.zeros(rows_padded, dtype=torch.int32, device=device)
    token_weights = torch.zeros(rows_padded, dtype=torch.float32, device=device)
    scatter_output = torch.zeros(m, K, dtype=torch.bfloat16, device=device)
    ones = torch.ones(E, device=device)
    flat_ids = topk_ids.reshape(-1).contiguous()
    flat_weights = topk_weights.reshape(-1).contiguous()

    import os as _os
    _direct = _os.environ.get("BENCH_DIRECT", "0") == "1"
    if recipe == "w4a8_trellis":
        kernel = MoEDynamicKernelBackend(
            16,
            (_TILE_M, _TILE_N),
            activation=activation,
            quant_recipe="w4a8_trellis",
            w4a8_repacked=True,
            num_topk=top_k,
            trellis_bits=_BITS,
            direct_routing=_direct,
            materialize_intermediate=_direct,
            share_input_across_experts=_direct,
        )
    else:
        kernel = MoEDynamicKernelBackend(
            16,
            (_TILE_M, _TILE_N),
            activation=activation,
            quant_recipe=recipe,
            w4a8_repacked=True,
            num_topk=top_k,
            direct_routing=_direct,
            materialize_intermediate=_direct,
            share_input_across_experts=_direct,
        )
        w13_flat = torch.zeros(
            E * (w1_n // 256) * (K // 128) * 4096,
            dtype=torch.int32, device=device,
        )
        down_flat = torch.zeros(
            E * (K // 256) * (n // 128) * 4096,
            dtype=torch.int32, device=device,
        )
        sentinel_u32 = torch.zeros(
            max(E * (w1_n // 256) * (K // 128) * 256,
                E * (K // 256) * (n // 128) * 256),
            dtype=torch.uint32, device=device,
        )
    launch = _DynamicMoEW4A8Launch(kernel, k=K, n=n, w1_n=w1_n, num_topk=top_k)

    weight_dtype = cutlass.Float4E2M1FN
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype, (w1_n, K, E), stride_order=(1, 0, 2), assumed_align=16
    )
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype, (K, n, E), stride_order=(1, 0, 2), assumed_align=16
    )

    def fake_ptr_u8():
        return make_ptr(
            cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
        )

    def fake_ptr_i32():
        return make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)

    def fake_ptr_u32():
        return make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16)

    compiled = b12x_compile(
        launch,
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_i32(),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(weight_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_u8(),
        fake_ptr_u8(),
        fake_ptr_u32(),
        _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)),
        _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)), _fake_i32((1,)),
        fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(),
        fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(), fake_ptr_i32(),
        b_w13_fake,
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        b_down_fake,
        make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u8(), fake_ptr_u8(),
        fake_ptr_u32(), fake_ptr_u32(), fake_ptr_u32(), fake_ptr_u32(),
        _fake_i32((E,)), _fake_i32((E,)), _fake_i32((E + 1,)),
        _fake_f32((E,)), _fake_f32((E,)), _fake_f32((E,)), _fake_f32((E,)),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_ptr_i32(),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        1, 1, 1, 1, 1, 1, 1,
        current_cuda_stream(),
        fake_ptr_u8(),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        compile_spec=KernelCompileSpec.from_fields(
            "tests.w4a8_dynamic.trellis",
            1,
            ("activation", activation),
            ("bits", _BITS),
            ("experts", E),
            ("hidden", K),
            ("intermediate", n),
            ("top_k", top_k),
        ),
        dsl_compile_options=_OPT_LEVEL_2,
    )

    compiled(
        _gptr(cutlass.BFloat16, x_dyn),
        _gptr(cutlass.Int32, flat_ids, 4),
        _gptr(cutlass.Float32, flat_weights, 4),
        _gptr(weight_dtype, packed_a),
        _gptr(cutlass.Float8E4M3FN, scale_flat),
        _gptr(cutlass.Uint8, packed_a),
        _gptr(cutlass.Uint8, scale_flat),
        _gptr(cutlass.Uint32, intermediate_u32),
        barrier_count, barrier_epoch, pair_head, producers_done, all_pub,
        task_head, task_tail,
        _gptr(cutlass.Int32, task_ready, 4),
        _gptr(cutlass.Int32, task_expert, 4),
        _gptr(cutlass.Int32, task_m_tile, 4),
        _gptr(cutlass.Int32, task_slice_begin, 4),
        _gptr(cutlass.Int32, task_slice_count, 4),
        _gptr(cutlass.Int32, task_valid_rows, 4),
        _gptr(cutlass.Int32, tile_write_count, 4),
        w13_packed,
        _gptr(cutlass.Float8E4M3FN, scale_flat),
        w2_packed,
        _gptr(cutlass.Float8E4M3FN, scale_flat),
        _gptr(cutlass.Uint8, scale_flat),
        _gptr(cutlass.Uint8, scale_flat),
        _gptr(cutlass.Uint8, scale_flat),
        _gptr(cutlass.Uint8, scale_flat),
        _gptr(cutlass.Uint32, w13_flat),
        _gptr(cutlass.Uint32, sentinel_u32),
        _gptr(cutlass.Uint32, down_flat),
        _gptr(cutlass.Uint32, sentinel_u32),
        row_counts, expert_write_rows, expert_tile_base,
        ones, ones, ones, ones,
        _gptr(cutlass.BFloat16, scatter_output),
        _gptr(cutlass.Int32, token_map, 4),
        _gptr(cutlass.Float32, token_weights, 4),
        m,
        m * top_k,
        m,
        rows_padded,
        max_tasks,
        phys_tiles,
        mac,
        current_cuda_stream(),
        _gptr(cutlass.Uint8, t12_lut),
        _gptr(cutlass.Float16, rot_flat),
    )
    torch.cuda.synchronize()
    if keepalive is not None:
        keepalive.append(dict(locals()))
    out = (
        (scatter_output.float().reshape(m, K // 128, 128) @ had)
        .reshape(m, K)
    )
    return out, oracle


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("activation", ["situ"])
@pytest.mark.parametrize("m", [1, 3])
def test_dynamic_trellis_matches_scaffold(activation: str, m: int) -> None:
    got, want = _run_trellis_dynamic(
        activation=activation, E=8, m=m, K=512, n=256, top_k=4, seed=20260811
    )
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got) > 0
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(1, -1), want.reshape(1, -1)
    ).item()
    assert cosine > 0.998, cosine
    rel_l2 = ((got - want).norm() / want.norm().clamp_min(1e-9)).item()
    assert rel_l2 < 0.06, rel_l2
    for row in range(got.shape[0]):
        row_cos = torch.nn.functional.cosine_similarity(
            got[row : row + 1], want[row : row + 1]
        ).item()
        assert row_cos > 0.998, (row, row_cos)
