from __future__ import annotations

import pytest
import torch

from b12x.attention import glm_pooled_indexer, sparse_mla
from b12x.attention._shared.mla.reference import (
    pack_mla_kv_cache_reference,
    sparse_mla_reference,
    unpack_mla_kv_cache_reference,
)
from b12x.attention._shared.mla.kv_cache import (
    _glm_next_cache_byte_offset,
    _glm_next_cache_record_address,
    concat_and_cache_glm_next_mla,
)
from b12x.attention._shared.mla.traits import (
    ComputeMode,
    ModelType,
    ScaleFormat,
    infer_model_type,
    make_unified_traits,
)

from ..conftest import require_b12x as require_sm120


_GLM_NEXT_RECORD_BYTES = 528
_GLM_NEXT_PAGE_SIZE = 64
_GLM_NEXT_HEAD_DIM = 512
_GLM_NEXT_SM_SCALE = 256**-0.5


def _glm_next_plan_and_binding(
    *,
    device: torch.device,
    q: torch.Tensor,
    selected: torch.Tensor,
    active: torch.Tensor,
    cache_seqlens: torch.Tensor,
    mode: str = "decode",
    page_size: int = _GLM_NEXT_PAGE_SIZE,
):
    plan = sparse_mla.plan(
        sparse_mla.Caps(
            device=device,
            num_q_heads=int(q.shape[1]),
            max_q_rows=int(q.shape[0]),
            max_batch=int(cache_seqlens.shape[0]),
            max_width=int(selected.shape[1]),
            max_kv_rows=max(int(active.max().item()), 1),
            kv_dtype=torch.uint8,
            head_dim=_GLM_NEXT_HEAD_DIM,
            v_head_dim=_GLM_NEXT_HEAD_DIM,
            page_size=page_size,
            model_type=ModelType.GLM_NEXT,
            mode=mode,
        )
    )
    spec = plan.scratch_specs()[0]
    binding = sparse_mla.bind(
        plan,
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device),
        q=q,
        selected_indices=selected,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=active,
    )
    return plan, binding


def _assert_glm_next_attention_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.reshape(-1), expected_f.reshape(-1), dim=0
    ).item()
    max_abs = (actual_f - expected_f).abs().max().item()
    assert cosine > 0.995, f"cosine similarity {cosine:.8f} did not exceed 0.995"
    assert max_abs < 0.03, f"max absolute error {max_abs:.8f} exceeded 0.03"


def _glm_next_dense_physical_reference(
    q: torch.Tensor,
    latent: torch.Tensor,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
) -> torch.Tensor:
    keys = torch.einsum(
        "rl,hpl->rhp", latent.float(), w_uk_t.float()
    ).to(torch.bfloat16)
    values = torch.einsum(
        "rl,hlv->rhv", latent.float(), w_uv.float()
    ).to(torch.bfloat16)
    logits = torch.einsum("rhp,shp->hrs", q.float(), keys.float())
    logits.mul_(_GLM_NEXT_SM_SCALE)
    causal_mask = torch.ones(
        (q.shape[0], q.shape[0]), dtype=torch.bool, device=q.device
    ).triu_(1)
    logits.masked_fill_(causal_mask.unsqueeze(0), -torch.inf)
    probabilities = torch.softmax(logits, dim=-1)
    return torch.einsum(
        "hrs,shv->rhv", probabilities, values.float()
    ).to(torch.bfloat16)


def test_glm_next_traits_define_no_rope_record() -> None:
    traits = make_unified_traits(
        ModelType.GLM_NEXT,
        ComputeMode.FP8,
        ScaleFormat.ARBITRARY_FP32,
    )

    assert (traits.d_nope, traits.d_rope, traits.d_v) == (512, 0, 512)
    assert traits.kv_gmem_stride == 528
    assert traits.kv_smem_stride == 528
    assert traits.bulk_tx_bytes == 64 * 528
    assert traits.rope_payload_bytes == 0
    assert not traits.v_has_rope
    assert not traits.fp8_rope


def test_glm_next_requires_explicit_identity_for_ambiguous_head_width() -> None:
    assert infer_model_type(512, torch.uint8) == (
        ModelType.DSV4,
        ComputeMode.FP8,
        ScaleFormat.UE8M0_BYTE,
    )
    assert infer_model_type(
        512,
        torch.uint8,
        model_type=ModelType.GLM_NEXT,
    ) == (
        ModelType.GLM_NEXT,
        ComputeMode.FP8,
        ScaleFormat.ARBITRARY_FP32,
    )

    with pytest.raises(ValueError, match="requires q_head_dim=512"):
        infer_model_type(576, torch.uint8, model_type=ModelType.GLM_NEXT)


def test_glm_next_rejects_rope_and_nvfp4_cache_recipes() -> None:
    with pytest.raises(ValueError, match="ARBITRARY_FP32"):
        make_unified_traits(
            ModelType.GLM_NEXT,
            ComputeMode.BF16,
            ScaleFormat.NVFP4_E4M3,
        )
    with pytest.raises(ValueError, match="no RoPE cache payload"):
        make_unified_traits(
            ModelType.GLM_NEXT,
            ComputeMode.FP8,
            ScaleFormat.ARBITRARY_FP32,
            fp8_rope=True,
        )
    with pytest.raises(ValueError, match="ComputeMode.FP8"):
        make_unified_traits(
            ModelType.GLM_NEXT,
            ComputeMode.BF16,
            ScaleFormat.ARBITRARY_FP32,
        )


def test_glm_next_reference_record_is_528_bytes() -> None:
    torch.manual_seed(20260826)
    latent = torch.randn((7, 512), dtype=torch.bfloat16) / 4

    packed = pack_mla_kv_cache_reference(latent)
    unpacked = unpack_mla_kv_cache_reference(packed)

    assert tuple(packed.shape) == (7, 1, 528)
    assert tuple(unpacked.shape) == (7, 1, 512)
    cosine = torch.nn.functional.cosine_similarity(
        unpacked[:, 0].flatten(), latent.float().flatten(), dim=0
    )
    assert float(cosine) > 0.999


def test_glm_next_packed_page_view_preserves_padded_page_stride() -> None:
    from b12x.attention._shared.mla.api import _is_supported_packed_kv_cache_view
    from b12x.attention._shared.mla.kernel import _cache_block_stride_bytes

    page_size = 64
    record_bytes = 528
    semantic_page_bytes = page_size * record_bytes
    pooled_tail_bytes = (page_size // 4) * 128 * 2
    padded_page_bytes = semantic_page_bytes + pooled_tail_bytes
    storage = torch.empty((3, padded_page_bytes), dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(3, page_size, record_bytes),
        stride=(padded_page_bytes, record_bytes, 1),
    )

    assert tuple(cache.shape) == (3, 64, 528)
    assert not cache.is_contiguous()
    assert _is_supported_packed_kv_cache_view(cache, page_size=page_size)
    assert (
        _cache_block_stride_bytes(
            cache,
            page_size=page_size,
            model_type=ModelType.GLM_NEXT,
            record_bytes=record_bytes,
        )
        == padded_page_bytes
    )


def test_glm_next_cache_writer_uses_wide_padded_page_offsets() -> None:
    page_size = 64
    page_stride = page_size * 528 + (page_size // 4) * 128 * 2
    high_page = 2**31 // page_stride + 1
    slot = high_page * page_size + 7

    offset = _glm_next_cache_byte_offset(
        slot,
        block_size=page_size,
        block_stride=page_stride,
    )

    assert offset == high_page * page_stride + 7 * 528
    assert offset > 2**31
    annotations = _glm_next_cache_record_address.__annotations__
    assert annotations["slot"] == "Int64"
    assert annotations["block_stride"] == "Int64"
    assert annotations["entry_stride"] == "Int64"
    assert annotations["return"] == "Int64"


def _valid_glm_next_writer_args() -> list[torch.Tensor]:
    page_size = 64
    page_stride = page_size * 528 + (page_size // 4) * 128 * 2
    backing = torch.empty((2, page_stride), dtype=torch.uint8)
    cache = torch.as_strided(
        backing,
        size=(2, page_size, 528),
        stride=(page_stride, 528, 1),
    )
    return [
        torch.empty((2, 512), dtype=torch.bfloat16),
        cache,
        torch.arange(2, dtype=torch.int64),
    ]


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("source_shape", ValueError, r"kv_c must be \(num_tokens, 512\)"),
        ("source_dtype", TypeError, "kv_c must be BF16"),
        ("cache_shape", ValueError, "kv_cache must be"),
        ("cache_dtype", TypeError, "kv_cache must be uint8"),
        ("slot_dtype", TypeError, "1-D int32 or int64"),
        ("short_source", ValueError, "must cover slot_mapping"),
        ("record_stride", ValueError, "packed 528-byte semantic records"),
        ("overlapping_pages", ValueError, "page stride must cover"),
        ("cpu_device", ValueError, "all tensors must be on CUDA"),
        ("cpu_device_int32", ValueError, "all tensors must be on CUDA"),
    ],
)
def test_glm_next_cache_writer_rejects_invalid_contracts(
    case: str,
    error: type[Exception],
    match: str,
) -> None:
    kv_c, cache, slots = _valid_glm_next_writer_args()
    if case == "source_shape":
        kv_c = torch.empty((2, 511), dtype=torch.bfloat16)
    elif case == "source_dtype":
        kv_c = kv_c.float()
    elif case == "cache_shape":
        cache = torch.empty((2, 64, 527), dtype=torch.uint8)
    elif case == "cache_dtype":
        cache = torch.empty((2, 64, 528), dtype=torch.bfloat16)
    elif case == "slot_dtype":
        slots = slots.to(torch.int16)
    elif case == "short_source":
        kv_c = kv_c[:1]
    elif case == "record_stride":
        storage = torch.empty((2, 64, 544), dtype=torch.uint8)
        cache = torch.as_strided(
            storage,
            size=(2, 64, 528),
            stride=(64 * 544, 544, 1),
        )
    elif case == "overlapping_pages":
        semantic_page_bytes = 64 * 528
        storage = torch.empty(2 * semantic_page_bytes, dtype=torch.uint8)
        cache = torch.as_strided(
            storage,
            size=(2, 64, 528),
            stride=(semantic_page_bytes - 16, 528, 1),
        )
    elif case == "cpu_device_int32":
        slots = slots.to(torch.int32)
    elif case != "cpu_device":
        raise AssertionError(f"unknown case {case}")

    with pytest.raises(error, match=match):
        concat_and_cache_glm_next_mla(kv_c, cache, slots)


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_glm_next_cache_writer_preserves_padded_tail_and_record_abi(
    slot_dtype: torch.dtype,
) -> None:
    device = require_sm120()
    torch.manual_seed(20260826)
    page_size = 64
    num_pages = 3
    semantic_page_bytes = page_size * 528
    pooled_tail_bytes = (page_size // 4) * 128 * 2
    page_stride = semantic_page_bytes + pooled_tail_bytes
    sentinel = 0xA5
    backing = torch.full(
        (num_pages, page_stride),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )
    cache = torch.as_strided(
        backing,
        size=(num_pages, page_size, 528),
        stride=(page_stride, 528, 1),
    )
    kv_c = (torch.randn((6, 512), device=device) / 4).to(torch.bfloat16)
    capacity = num_pages * page_size
    huge_slot = 2**30 + 17 if slot_dtype == torch.int32 else 2**40 + 17
    slots = torch.tensor(
        [0, -1, capacity, 65, huge_slot, 130],
        dtype=slot_dtype,
        device=device,
    )

    sparse_mla.concat_and_cache_glm_next_mla(kv_c, cache, slots)
    torch.cuda.synchronize(device)

    assert torch.all(backing[:, semantic_page_bytes:] == sentinel)
    changed = (cache != sentinel).any(dim=-1).nonzero(as_tuple=False)
    expected_changed = torch.tensor(
        [[0, 0], [1, 1], [2, 2]], dtype=torch.int64, device=device
    )
    assert torch.equal(changed, expected_changed)

    records = torch.stack((cache[0, 0], cache[1, 1], cache[2, 2])).cpu()
    dequantized = unpack_mla_kv_cache_reference(records.unsqueeze(1))[:, 0]
    source = kv_c.index_select(
        0, torch.tensor([0, 3, 5], dtype=torch.int64, device=device)
    ).float().cpu()
    cosine = torch.nn.functional.cosine_similarity(
        dequantized.flatten(), source.flatten(), dim=0
    )
    assert float(cosine) > 0.999

    actual_scales = records[:, 512:].contiguous().view(torch.float32)
    expected_scales = source.reshape(3, 4, 128).abs().amax(dim=-1) / 448.0
    expected_scales = torch.where(
        expected_scales > 0,
        expected_scales,
        torch.ones_like(expected_scales),
    )
    torch.testing.assert_close(actual_scales, expected_scales, rtol=1e-6, atol=0.0)


def test_glm_next_public_cpu_reference_path_preserves_model_identity() -> None:
    torch.manual_seed(20260826)
    rows, heads, cache_tokens, width = 2, 8, 16, 8
    latent = torch.randn((cache_tokens, 512), dtype=torch.bfloat16) / 4
    cache = pack_mla_kv_cache_reference(latent)
    q = torch.randn((rows, heads, 512), dtype=torch.bfloat16) / 4
    selected = torch.stack(
        [torch.randperm(cache_tokens)[:width].sort().values for _ in range(rows)]
    ).to(torch.int32)
    cache_seqlens = torch.full((rows,), cache_tokens, dtype=torch.int32)
    active = torch.full((rows,), width, dtype=torch.int32)

    plan = sparse_mla.plan(
        sparse_mla.Caps(
            device="cpu",
            num_q_heads=heads,
            max_q_rows=rows,
            max_width=width,
            kv_dtype=torch.uint8,
            head_dim=512,
            v_head_dim=512,
            page_size=1,
            model_type=ModelType.GLM_NEXT,
        )
    )
    spec = plan.scratch_specs()[0]
    binding = sparse_mla.bind(
        plan,
        scratch=torch.empty(spec.shape, dtype=spec.dtype),
        q=q,
        selected_indices=selected,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=active,
    )

    actual = sparse_mla.run_decode(
        binding=binding,
        kv_cache=cache,
        sm_scale=256**-0.5,
    )
    expected = sparse_mla_reference(
        q_all=q,
        kv_cache=cache,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=256**-0.5,
        v_head_dim=512,
    )
    torch.testing.assert_close(actual, expected)

    assert sparse_mla.ModelType.GLM_NEXT == ModelType.GLM_NEXT
    assert callable(sparse_mla.compile_glm_next_mla_cache_writer)
    assert callable(sparse_mla.concat_and_cache_glm_next_mla)
    with pytest.raises(ValueError, match="does not match its plan"):
        sparse_mla.run_decode(
            binding=binding,
            kv_cache=cache,
            sm_scale=256**-0.5,
            model_type=ModelType.GLM_NSA,
        )
    with pytest.raises(ValueError, match="no RoPE cache payload"):
        sparse_mla.run_decode(
            binding=binding,
            kv_cache=cache,
            sm_scale=256**-0.5,
            fp8_rope=True,
        )


@pytest.mark.parametrize("container_width", [2051, 2112])
def test_glm_next_prefill_routes_exact_or_aligned_selector_width(
    monkeypatch: pytest.MonkeyPatch,
    container_width: int,
) -> None:
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.prefill import run_unified_prefill

    calls: list[dict] = []

    def fake_run_unified_prefill_mg(**kwargs):
        calls.append(kwargs)
        return kwargs["output"], kwargs["lse_out"]

    monkeypatch.setattr(
        prefill_mg,
        "run_unified_prefill_mg",
        fake_run_unified_prefill_mg,
    )

    q = torch.empty((2, 8, 512), dtype=torch.bfloat16)
    cache = torch.empty((4, 528), dtype=torch.uint8)
    selected = torch.full((2, container_width), -1, dtype=torch.int32)
    selected[:, :2051] = 0
    active = torch.full((2,), 2051, dtype=torch.int32)

    output, lse = run_unified_prefill(
        q=q,
        kv_cache=cache,
        topk_indices=selected,
        topk_length=active,
        sm_scale=256**-0.5,
        page_block_size=64,
        model_type=ModelType.GLM_NEXT,
    )

    assert output.shape == (2, 8, 512)
    assert lse.shape == (2, 8)
    assert len(calls) == 1
    assert calls[0]["model_type"] == ModelType.GLM_NEXT
    assert calls[0]["scale_format"] == ScaleFormat.ARBITRARY_FP32
    assert calls[0]["fp8_rope"] is False
    torch.testing.assert_close(calls[0]["topk_length"], active)


@torch.inference_mode()
def test_glm_next_production_decode_matches_packed_record_oracle() -> None:
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20260827)
    rows, heads, num_records, width = 2, 8, 3 * _GLM_NEXT_PAGE_SIZE, 129
    latent = (
        torch.randn(
            (num_records, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    cache = torch.empty(
        (
            num_records // _GLM_NEXT_PAGE_SIZE,
            _GLM_NEXT_PAGE_SIZE,
            _GLM_NEXT_RECORD_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(num_records, dtype=torch.int64, device=device)
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)

    q = (
        torch.randn(
            (rows, heads, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    selected = torch.stack(
        [
            torch.randperm(num_records, generator=generator, device=device)[:width]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    active = torch.tensor([width, 65], dtype=torch.int32, device=device)
    cache_seqlens = torch.full(
        (rows,), num_records, dtype=torch.int32, device=device
    )
    _, binding = _glm_next_plan_and_binding(
        device=device,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
    )

    actual, actual_lse = sparse_mla.run_decode(
        binding=binding,
        kv_cache=cache,
        sm_scale=_GLM_NEXT_SM_SCALE,
        return_lse=True,
        forced_num_splits=2,
    )
    expected, expected_lse = sparse_mla_reference(
        q_all=q,
        kv_cache=cache.view(num_records, 1, _GLM_NEXT_RECORD_BYTES),
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        return_lse=True,
    )
    torch.cuda.synchronize(device)

    _assert_glm_next_attention_close(actual, expected)
    assert bool(torch.isfinite(actual_lse).all().item())
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0.0, atol=0.05)


@torch.inference_mode()
def test_glm_next_hybrid_manager_page_replays_across_page_boundary() -> None:
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20260829)
    page_size = 2304
    num_pages = 2
    page_stride = (
        page_size * _GLM_NEXT_RECORD_BYTES
        + page_size // 4 * 128 * torch.bfloat16.itemsize
    )
    backing = torch.zeros(
        (num_pages, page_stride), dtype=torch.uint8, device=device
    )
    cache = torch.as_strided(
        backing,
        size=(num_pages, page_size, _GLM_NEXT_RECORD_BYTES),
        stride=(page_stride, _GLM_NEXT_RECORD_BYTES, 1),
    )
    slots = torch.tensor(
        [63, 64, 2302, 2303, 2304, 2305, 4607],
        dtype=torch.int64,
        device=device,
    )
    latent = (
        torch.randn(
            (slots.numel(), _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)

    q = (
        torch.randn(
            (1, 8, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    selected = slots.to(torch.int32).unsqueeze(0).contiguous()
    active = torch.full(
        (1,), slots.numel(), dtype=torch.int32, device=device
    )
    cache_seqlens = torch.full(
        (1,), num_pages * page_size, dtype=torch.int32, device=device
    )
    _, binding = _glm_next_plan_and_binding(
        device=device,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
        page_size=page_size,
    )

    actual, actual_lse = sparse_mla.run_decode(
        binding=binding,
        kv_cache=cache,
        sm_scale=_GLM_NEXT_SM_SCALE,
        return_lse=True,
        lse_scale="natural",
        forced_num_splits=1,
    )
    flat_cache = cache.contiguous().view(
        num_pages * page_size, 1, _GLM_NEXT_RECORD_BYTES
    )
    expected, expected_lse = sparse_mla_reference(
        q_all=q,
        kv_cache=flat_cache,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        return_lse=True,
    )
    expected_lse.mul_(0.6931471805599453)
    torch.cuda.synchronize(device)

    assert cache.stride(0) == page_stride
    _assert_glm_next_attention_close(actual, expected)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0.0, atol=0.05)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
        captured_output, captured_lse = sparse_mla.run_decode(
            binding=binding,
            kv_cache=cache,
            sm_scale=_GLM_NEXT_SM_SCALE,
            return_lse=True,
            lse_scale="natural",
            forced_num_splits=1,
        )
    assert captured_output.data_ptr() == actual.data_ptr()
    assert captured_lse.data_ptr() == actual_lse.data_ptr()

    for _ in range(2):
        q.copy_(
            (
                torch.randn(q.shape, generator=generator, device=device) / 4
            ).to(torch.bfloat16)
        )
        latent.copy_(
            (
                torch.randn(
                    latent.shape,
                    generator=generator,
                    device=device,
                )
                / 4
            ).to(torch.bfloat16)
        )
        sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
        flat_cache.copy_(
            cache.contiguous().view(
                num_pages * page_size, 1, _GLM_NEXT_RECORD_BYTES
            )
        )
        replay_expected, replay_expected_lse = sparse_mla_reference(
            q_all=q,
            kv_cache=flat_cache,
            page_table_1=selected,
            active_token_counts=active,
            sm_scale=_GLM_NEXT_SM_SCALE,
            v_head_dim=_GLM_NEXT_HEAD_DIM,
            return_lse=True,
        )
        replay_expected_lse.mul_(0.6931471805599453)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        graph.replay()
        torch.cuda.synchronize(device)

        assert torch.cuda.memory_allocated(device) == allocated_before
        assert torch.cuda.memory_reserved(device) == reserved_before
        _assert_glm_next_attention_close(captured_output, replay_expected)
        torch.testing.assert_close(
            captured_lse, replay_expected_lse, rtol=0.0, atol=0.05
        )


@torch.inference_mode()
def test_glm_next_production_prefill_2051_replays_without_allocation() -> None:
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20260828)
    rows, heads = 1, 8
    active_width = 2051
    container_width = 2112
    num_records = container_width
    latent = (
        torch.randn(
            (num_records, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    cache = torch.empty(
        (
            num_records // _GLM_NEXT_PAGE_SIZE,
            _GLM_NEXT_PAGE_SIZE,
            _GLM_NEXT_RECORD_BYTES,
        ),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(num_records, dtype=torch.int64, device=device)
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)

    q = (
        torch.randn(
            (rows, heads, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    selected = torch.full(
        (rows, container_width), -1, dtype=torch.int32, device=device
    )
    selected[0, :active_width] = torch.randperm(
        num_records, generator=generator, device=device
    )[:active_width].to(torch.int32)
    active = torch.full((rows,), active_width, dtype=torch.int32, device=device)
    cache_seqlens = torch.full(
        (rows,), num_records, dtype=torch.int32, device=device
    )
    _, binding = _glm_next_plan_and_binding(
        device=device,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
        mode="extend",
    )

    actual, actual_lse = sparse_mla.run_extend(
        binding=binding,
        kv_cache=cache,
        sm_scale=_GLM_NEXT_SM_SCALE,
        return_lse=True,
    )
    expected, expected_lse = sparse_mla_reference(
        q_all=q,
        kv_cache=cache.view(num_records, 1, _GLM_NEXT_RECORD_BYTES),
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        return_lse=True,
    )
    torch.cuda.synchronize(device)
    _assert_glm_next_attention_close(actual, expected)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0.0, atol=0.05)

    assert actual.data_ptr() == binding.scratch.output_buffer.data_ptr()
    assert binding.scratch.final_lse is not None
    assert actual_lse.data_ptr() == binding.scratch.final_lse.data_ptr()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
        captured_output, captured_lse = sparse_mla.run_extend(
            binding=binding,
            kv_cache=cache,
            sm_scale=_GLM_NEXT_SM_SCALE,
            return_lse=True,
        )
    assert captured_output.data_ptr() == actual.data_ptr()
    assert captured_lse.data_ptr() == actual_lse.data_ptr()

    q.copy_(
        (
            torch.randn(
                q.shape,
                generator=generator,
                device=device,
            )
            / 4
        ).to(torch.bfloat16)
    )
    replay_expected, replay_expected_lse = sparse_mla_reference(
        q_all=q,
        kv_cache=cache.view(num_records, 1, _GLM_NEXT_RECORD_BYTES),
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        return_lse=True,
    )
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    graph.replay()
    torch.cuda.synchronize(device)

    assert torch.cuda.memory_allocated(device) == allocated_before
    assert torch.cuda.memory_reserved(device) == reserved_before
    _assert_glm_next_attention_close(captured_output, replay_expected)
    torch.testing.assert_close(
        captured_lse, replay_expected_lse, rtol=0.0, atol=0.05
    )


@torch.inference_mode()
def test_glm_next_tp4_hybrid_page_prefill_matches_oracle_and_replays() -> None:
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20260830)
    rows, heads = 26, 16
    page_size = 2176
    num_pages = 3
    container_width = 2051
    semantic_page_bytes = page_size * _GLM_NEXT_RECORD_BYTES
    pooled_tail_bytes = page_size // 4 * 128 * torch.bfloat16.itemsize
    page_stride = semantic_page_bytes + pooled_tail_bytes
    sentinel = 0xA5
    backing = torch.full(
        (num_pages, page_stride), sentinel, dtype=torch.uint8, device=device
    )
    cache = torch.as_strided(
        backing,
        size=(num_pages, page_size, _GLM_NEXT_RECORD_BYTES),
        stride=(page_stride, _GLM_NEXT_RECORD_BYTES, 1),
    )
    latent = (
        torch.randn(
            (rows, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    slots = 2 * page_size + torch.arange(rows, dtype=torch.int64, device=device)
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)

    q = (
        torch.randn(
            (rows, heads, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    selected = torch.full(
        (rows, container_width), -1, dtype=torch.int32, device=device
    )
    for row in range(rows):
        selected[row, : row + 1] = slots[: row + 1].to(torch.int32)
    active = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    cache_seqlens = active.clone()
    _, binding = _glm_next_plan_and_binding(
        device=device,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
        mode="extend",
        page_size=page_size,
    )

    actual, actual_lse = sparse_mla.run_extend(
        binding=binding,
        kv_cache=cache,
        sm_scale=_GLM_NEXT_SM_SCALE,
        return_lse=True,
    )
    flat_cache = cache.contiguous().view(
        num_pages * page_size, 1, _GLM_NEXT_RECORD_BYTES
    )
    expected, expected_lse = sparse_mla_reference(
        q_all=q,
        kv_cache=flat_cache,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        return_lse=True,
    )
    torch.cuda.synchronize(device)

    assert cache.stride(0) == page_stride
    assert torch.all(backing[:, semantic_page_bytes:] == sentinel)
    _assert_glm_next_attention_close(actual, expected)
    per_row_cosine = torch.nn.functional.cosine_similarity(
        actual.float().reshape(rows, -1),
        expected.float().reshape(rows, -1),
        dim=1,
    )
    assert float(per_row_cosine.min()) > 0.995
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0.0, atol=0.05)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
        captured_output, captured_lse = sparse_mla.run_extend(
            binding=binding,
            kv_cache=cache,
            sm_scale=_GLM_NEXT_SM_SCALE,
            return_lse=True,
        )
    assert captured_output.data_ptr() == actual.data_ptr()
    assert captured_lse.data_ptr() == actual_lse.data_ptr()

    q.copy_(
        (
            torch.randn(q.shape, generator=generator, device=device) / 4
        ).to(torch.bfloat16)
    )
    latent.copy_(
        (
            torch.randn(latent.shape, generator=generator, device=device) / 4
        ).to(torch.bfloat16)
    )
    sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
    flat_cache.copy_(
        cache.contiguous().view(
            num_pages * page_size, 1, _GLM_NEXT_RECORD_BYTES
        )
    )
    replay_expected, replay_expected_lse = sparse_mla_reference(
        q_all=q,
        kv_cache=flat_cache,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_GLM_NEXT_SM_SCALE,
        v_head_dim=_GLM_NEXT_HEAD_DIM,
        return_lse=True,
    )
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    graph.replay()
    torch.cuda.synchronize(device)

    assert torch.cuda.memory_allocated(device) == allocated_before
    assert torch.cuda.memory_reserved(device) == reserved_before
    assert torch.all(backing[:, semantic_page_bytes:] == sentinel)
    _assert_glm_next_attention_close(captured_output, replay_expected)
    torch.testing.assert_close(
        captured_lse, replay_expected_lse, rtol=0.0, atol=0.05
    )


@torch.inference_mode()
def test_glm_next_composed_prefill_matches_physical_mha_and_replays() -> None:
    device = require_sm120()
    generator = torch.Generator(device=device).manual_seed(20260831)
    rows, heads = 26, 16
    physical_head_dim = 256
    page_size = 2176
    num_pages = 3
    selection_width = 2051
    physical_page = 2
    semantic_page_bytes = page_size * _GLM_NEXT_RECORD_BYTES
    pooled_tail_bytes = page_size // 4 * 128 * torch.bfloat16.itemsize
    page_stride_bytes = semantic_page_bytes + pooled_tail_bytes
    sentinel = 0xA5

    backing = torch.full(
        (num_pages, page_stride_bytes),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )
    cache = torch.as_strided(
        backing,
        size=(num_pages, page_size, _GLM_NEXT_RECORD_BYTES),
        stride=(page_stride_bytes, _GLM_NEXT_RECORD_BYTES, 1),
    )
    compressed_cache = torch.as_strided(
        backing.view(torch.bfloat16),
        size=(num_pages, page_size // 4, 128),
        stride=(page_stride_bytes // 2, 128, 1),
        storage_offset=semantic_page_bytes // 2,
    )
    block_table = torch.tensor(
        [[physical_page]], dtype=torch.int32, device=device
    )

    selector_plan = glm_pooled_indexer.plan(
        glm_pooled_indexer.Caps(
            device=device,
            max_batch=1,
            max_raw_state_slots=1,
            max_q_rows=rows,
            max_seq_len=page_size,
            num_compressed_cache_pages=num_pages,
            compressed_page_size=page_size // 4,
        )
    )
    (selector_spec,) = selector_plan.scratch_specs()
    selector_binding = glm_pooled_indexer.bind(
        selector_plan,
        scratch=torch.empty(
            selector_spec.shape, dtype=selector_spec.dtype, device=device
        ),
        compressed_k_cache=compressed_cache,
        compressed_block_table=block_table,
        raw_k_ring=torch.empty(
            (1, selector_plan.caps.raw_ring_capacity, 128),
            dtype=torch.bfloat16,
            device=device,
        ),
        raw_gate_ring=torch.empty(
            (1, selector_plan.caps.raw_ring_capacity, 128),
            dtype=torch.bfloat16,
            device=device,
        ),
        raw_logical_positions=torch.full(
            (1, selector_plan.caps.raw_ring_capacity),
            -1,
            dtype=torch.int64,
            device=device,
        ),
        raw_interval_start_positions=torch.full(
            (1,), -1, dtype=torch.int64, device=device
        ),
        raw_state_slot_ids=torch.zeros((1,), dtype=torch.int32, device=device),
        position_embedding=(
            torch.randn((4, 128), generator=generator, device=device) / 4
        ).to(torch.bfloat16),
        selected_positions=torch.full(
            (rows, selection_width), -1, dtype=torch.int32, device=device
        ),
    )

    index_query = (
        torch.randn((rows, 32, 128), generator=generator, device=device) / 4
    ).to(torch.bfloat16)
    normalized_index_key = (
        torch.randn((rows, 128), generator=generator, device=device) / 4
    ).to(torch.bfloat16)
    index_gate_logits = (
        torch.randn((rows, 128), generator=generator, device=device) / 4
    ).to(torch.bfloat16)
    index_head_weights = (
        torch.randn((rows, 32), generator=generator, device=device) / 4
    ).to(torch.bfloat16)
    request_ids = torch.zeros((rows,), dtype=torch.int32, device=device)
    query_positions = torch.arange(rows, dtype=torch.int64, device=device)
    sequence_lengths = torch.tensor([rows], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, rows], dtype=torch.int32, device=device)
    reset_mask = torch.ones((1,), dtype=torch.bool, device=device)
    prefix_lengths = torch.zeros((1,), dtype=torch.int32, device=device)

    latent = (
        torch.randn(
            (rows, _GLM_NEXT_HEAD_DIM), generator=generator, device=device
        )
        / 4
    ).to(torch.bfloat16)
    q_physical = (
        torch.randn(
            (rows, heads, physical_head_dim),
            generator=generator,
            device=device,
        )
        / 4
    ).to(torch.bfloat16)
    w_uk_t = (
        torch.randn(
            (heads, physical_head_dim, _GLM_NEXT_HEAD_DIM),
            generator=generator,
            device=device,
        )
        * (_GLM_NEXT_HEAD_DIM**-0.5)
    ).to(torch.bfloat16)
    w_uv = (
        torch.randn(
            (heads, _GLM_NEXT_HEAD_DIM, physical_head_dim),
            generator=generator,
            device=device,
        )
        * (_GLM_NEXT_HEAD_DIM**-0.5)
    ).to(torch.bfloat16)
    q_absorbed = torch.empty(
        (rows, heads, _GLM_NEXT_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    physical_output = torch.empty(
        (rows, heads, physical_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    selected_physical = torch.full(
        (rows, selection_width), -1, dtype=torch.int32, device=device
    )
    active_counts = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    cache_seqlens = active_counts.clone()
    slots = physical_page * page_size + query_positions
    _, attention_binding = _glm_next_plan_and_binding(
        device=device,
        q=q_absorbed,
        selected=selected_physical,
        active=active_counts,
        cache_seqlens=cache_seqlens,
        mode="extend",
        page_size=page_size,
    )

    def run_composed() -> torch.Tensor:
        glm_pooled_indexer.reset_state(
            selector_binding,
            reset_mask=reset_mask,
            prefix_lengths=prefix_lengths,
        )
        glm_pooled_indexer.run_prefill(
            selector_binding,
            index_query=index_query,
            normalized_index_key=normalized_index_key,
            index_gate_logits=index_gate_logits,
            index_head_weights=index_head_weights,
            request_ids=request_ids,
            query_positions=query_positions,
            sequence_lengths=sequence_lengths,
            query_start_loc=query_start_loc,
        )
        logical_selected = selector_binding.selected_positions
        active_counts.copy_(
            (logical_selected >= 0).sum(dim=1, dtype=torch.int32)
        )
        # All 26 rows occupy one 2176-token logical page, so this is the exact
        # request-relative block-table remap used by the serving path.
        selected_physical.copy_(logical_selected)
        selected_physical.add_(block_table[0, 0] * page_size)
        selected_physical.masked_fill_(logical_selected < 0, -1)

        sparse_mla.concat_and_cache_glm_next_mla(latent, cache, slots)
        torch.bmm(
            q_physical.transpose(0, 1),
            w_uk_t,
            out=q_absorbed.transpose(0, 1),
        )
        latent_output = sparse_mla.run_extend(
            binding=attention_binding,
            kv_cache=cache,
            sm_scale=_GLM_NEXT_SM_SCALE,
        )
        torch.bmm(
            latent_output.transpose(0, 1),
            w_uv,
            out=physical_output.transpose(0, 1),
        )
        return physical_output

    def assert_causal_selection() -> None:
        logical_selected = selector_binding.selected_positions
        for row in range(rows):
            logical_prefix = logical_selected[row, : row + 1]
            torch.testing.assert_close(
                logical_prefix.sort().values.cpu(),
                torch.arange(row + 1, dtype=torch.int32),
            )
            assert torch.all(logical_selected[row, row + 1 :] == -1)
            torch.testing.assert_close(
                selected_physical[row, : row + 1].cpu(),
                (physical_page * page_size + logical_prefix).cpu(),
            )
            assert torch.all(selected_physical[row, row + 1 :] == -1)
        torch.testing.assert_close(
            active_counts,
            torch.arange(1, rows + 1, dtype=torch.int32, device=device),
        )

    def assert_physical_output(
        actual_output: torch.Tensor, expected_output: torch.Tensor
    ) -> None:
        _assert_glm_next_attention_close(actual_output, expected_output)
        per_row_cosine = torch.nn.functional.cosine_similarity(
            actual_output.float().reshape(rows, -1),
            expected_output.float().reshape(rows, -1),
            dim=1,
        )
        assert float(per_row_cosine.min()) > 0.995

    expected = _glm_next_dense_physical_reference(
        q_physical, latent, w_uk_t, w_uv
    )
    actual = run_composed()
    torch.cuda.synchronize(device)
    assert cache.stride(0) == page_stride_bytes
    assert compressed_cache.stride(0) == page_stride_bytes // 2
    assert torch.all(backing[:physical_page] == sentinel)
    assert torch.all(cache[physical_page, rows:] == sentinel)
    assert_causal_selection()
    assert_physical_output(actual, expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = run_composed()
    assert captured_output.data_ptr() == physical_output.data_ptr()
    torch.cuda.synchronize(device)
    assert_causal_selection()
    assert_physical_output(captured_output, expected)

    q_physical.copy_(
        (
            torch.randn(q_physical.shape, generator=generator, device=device) / 4
        ).to(torch.bfloat16)
    )
    latent.copy_(
        (
            torch.randn(latent.shape, generator=generator, device=device) / 4
        ).to(torch.bfloat16)
    )
    index_query.copy_(
        (
            torch.randn(index_query.shape, generator=generator, device=device) / 4
        ).to(torch.bfloat16)
    )
    normalized_index_key.copy_(
        (
            torch.randn(
                normalized_index_key.shape, generator=generator, device=device
            )
            / 4
        ).to(torch.bfloat16)
    )
    index_gate_logits.copy_(
        (
            torch.randn(
                index_gate_logits.shape, generator=generator, device=device
            )
            / 4
        ).to(torch.bfloat16)
    )
    index_head_weights.copy_(
        (
            torch.randn(
                index_head_weights.shape, generator=generator, device=device
            )
            / 4
        ).to(torch.bfloat16)
    )
    replay_expected = _glm_next_dense_physical_reference(
        q_physical, latent, w_uk_t, w_uv
    )
    backing.fill_(sentinel)
    selector_binding.selected_positions.fill_(-777)
    selected_physical.fill_(-777)
    active_counts.zero_()
    physical_output.fill_(torch.nan)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    graph.replay()
    torch.cuda.synchronize(device)

    assert torch.cuda.memory_allocated(device) == allocated_before
    assert torch.cuda.memory_reserved(device) == reserved_before
    assert torch.all(backing[:physical_page] == sentinel)
    assert torch.all(cache[physical_page, rows:] == sentinel)
    assert_causal_selection()
    assert bool(torch.isfinite(captured_output).all())
    assert_physical_output(captured_output, replay_expected)


@torch.inference_mode()
def test_glm_next_writer_and_reader_use_int64_for_live_high_page() -> None:
    device = require_sm120()
    page_size = _GLM_NEXT_PAGE_SIZE
    semantic_page_bytes = page_size * _GLM_NEXT_RECORD_BYTES
    pooled_tail_bytes = (page_size // 4) * 128 * torch.bfloat16.itemsize
    page_stride_bytes = semantic_page_bytes + pooled_tail_bytes
    int32_max = torch.iinfo(torch.int32).max
    high_page = int32_max // page_stride_bytes + 2
    num_pages = high_page + 1
    required_bytes = num_pages * page_stride_bytes
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reserve_bytes = 2 * 1024**3
    if free_bytes < required_bytes + reserve_bytes:
        pytest.skip(
            "live GLM_NEXT high-page test requires "
            f"{required_bytes + reserve_bytes} bytes free, found {free_bytes}"
        )
    try:
        backing = torch.empty(
            (num_pages, page_stride_bytes), dtype=torch.uint8, device=device
        )
    except torch.OutOfMemoryError:
        pytest.skip(
            "CUDA allocator could not reserve the required mostly-uninitialized "
            f"{required_bytes}-byte GLM_NEXT cache"
        )
    cache = torch.as_strided(
        backing,
        size=(num_pages, page_size, _GLM_NEXT_RECORD_BYTES),
        stride=(page_stride_bytes, _GLM_NEXT_RECORD_BYTES, 1),
    )
    local_slot = 7
    high_slot = high_page * page_size + local_slot
    assert high_page * page_stride_bytes > int32_max
    assert high_slot < int32_max

    live_latent = (
        torch.linspace(
            -0.75,
            0.75,
            _GLM_NEXT_HEAD_DIM,
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        .to(torch.bfloat16)
    )
    sources = torch.cat((torch.zeros_like(live_latent), live_latent))
    slots = torch.tensor([0, high_slot], dtype=torch.int64, device=device)
    sparse_mla.concat_and_cache_glm_next_mla(sources, cache, slots)

    q = torch.randn(
        (1, 8, _GLM_NEXT_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    selected = torch.tensor([[high_slot]], dtype=torch.int32, device=device)
    active = torch.ones((1,), dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor(
        [high_slot + 1], dtype=torch.int32, device=device
    )
    _, binding = _glm_next_plan_and_binding(
        device=device,
        q=q,
        selected=selected,
        active=active,
        cache_seqlens=cache_seqlens,
    )
    actual = sparse_mla.run_decode(
        binding=binding,
        kv_cache=cache,
        sm_scale=_GLM_NEXT_SM_SCALE,
        forced_num_splits=1,
    )
    expected_value = unpack_mla_kv_cache_reference(
        cache[high_page, local_slot].reshape(1, 1, _GLM_NEXT_RECORD_BYTES)
    )[0, 0]
    expected = expected_value.view(1, 1, -1).expand_as(actual)
    torch.cuda.synchronize(device)

    assert _glm_next_cache_byte_offset(
        high_slot,
        block_size=page_size,
        block_stride=page_stride_bytes,
    ) > int32_max
    _assert_glm_next_attention_close(actual, expected)
