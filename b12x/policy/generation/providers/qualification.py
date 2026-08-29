"""Measured qualification providers for single-implementation components."""

from __future__ import annotations

import gc
import io
import json
import math
from contextlib import redirect_stdout

from b12x.policy.generation.contracts import GenerationContext
from b12x.policy.generation.measured import (
    GpuProbeMeasurement,
    MeasuredPolicyGenerator,
)

from .gpu_workers import (
    _cuda_event_samples_us,
    _l2_flush_fn,
    _median_of_group_medians,
)


def _timed_graph_measurement(
    *,
    context: GenerationContext,
    label: str,
    run,
    output,
    expected,
    flush,
) -> GpuProbeMeasurement:
    import torch
    import torch.nn.functional as torch_functional

    device = torch.device("cuda", context.device_ordinal)
    settings = context.settings
    for _ in range(settings.warmup):
        run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    if output.is_floating_point():
        output.fill_(float("nan"))
    else:
        output.zero_()
    graph.replay()
    torch.cuda.synchronize(device)
    finite = bool(torch.isfinite(output.float()).all().item())
    nonzero = bool(torch.count_nonzero(output).item())
    cosine = float(
        torch_functional.cosine_similarity(
            output.float().reshape(1, -1),
            expected.float().reshape(1, -1),
        ).item()
    )
    allocated_before = torch.cuda.memory_allocated(device)
    samples = _cuda_event_samples_us(
        graph.replay,
        count=settings.groups * settings.repetitions,
        device=device,
        flush=flush,
    )
    allocated_after = torch.cuda.memory_allocated(device)
    return GpuProbeMeasurement(
        label=label,
        latency_us=_median_of_group_medians(
            samples,
            groups=settings.groups,
            repetitions=settings.repetitions,
        ),
        correct=(
            finite
            and nonzero
            and cosine >= settings.minimum_cosine
            and allocated_after <= allocated_before
        ),
        metrics={
            "cosine": cosine,
            "finite": finite,
            "nonzero": nonzero,
            "replay_allocation_bytes": allocated_after - allocated_before,
        },
    )


class _DsaIndexerProbe:
    _CASES = (
        ("decode", 1, 4_096),
        ("decode", 16, 4_096),
        ("extend", 1, 4_096),
    )

    @property
    def case_count(self) -> int:
        return len(self._CASES)

    @property
    def description(self) -> str:
        return "production DSA paged-decode and contiguous-extend qualification"

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]:
        import torch

        from benchmarks.benchmark_dsa_indexer import (
            GLMNSAConfig,
            _run_decode_case,
            _run_extend_case,
        )

        device = torch.device("cuda", context.device_ordinal)
        settings = context.settings
        cfg = GLMNSAConfig(num_heads=16)
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        replays = settings.groups * settings.repetitions
        measurements = []
        for index, (mode, rows, cache_len) in enumerate(self._CASES):
            captured = io.StringIO()
            with redirect_stdout(captured):
                if mode == "decode":
                    _run_decode_case(
                        cfg=cfg,
                        q_rows=rows,
                        cache_len=cache_len,
                        width=cache_len,
                        topk=2_048,
                        warmup=settings.warmup,
                        replays=replays,
                        seed=settings.seed + 17 * index,
                        device=device,
                        pool_factor=2,
                        l2_flush=flush,
                    )
                else:
                    _run_extend_case(
                        cfg=cfg,
                        batch=rows,
                        q_len=128,
                        cache_len=cache_len,
                        width=cache_len,
                        topk=2_048,
                        warmup=settings.warmup,
                        replays=replays,
                        seed=settings.seed + 17 * index,
                        device=device,
                        pool_factor=2,
                        l2_flush=flush,
                    )
            records = [
                json.loads(line)
                for line in captured.getvalue().splitlines()
                if line.strip().startswith("{")
            ]
            if len(records) != 1:
                raise RuntimeError("DSA benchmark did not emit one timing record")
            record = records[0]
            latency = (
                record["replay_median_us"]
                if mode == "decode"
                else record["median_us"]
            )
            measurements.append(
                GpuProbeMeasurement(
                    label=f"{mode}-rows{rows}-ctx{cache_len}",
                    latency_us=float(latency),
                    correct=True,
                    metrics={"mode": mode, "rows": rows, "top_k": 2_048},
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
        return tuple(measurements)


class _SparseMlaProbe:
    _CASES = ((1, 512), (4, 2_048), (16, 2_048))

    @property
    def case_count(self) -> int:
        return len(self._CASES)

    @property
    def description(self) -> str:
        return "production sparse-MLA plan/bind/decode graph qualification"

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]:
        import torch

        from b12x.attention import sparse_mla
        from b12x.attention.sparse_mla.reference import sparse_mla_reference
        from b12x.policy import PolicyContext, PolicyMode
        from benchmarks.benchmark_unified_mla_sm120 import _make_glm_inputs

        device = torch.device("cuda", context.device_ordinal)
        flush = _l2_flush_fn(device, enabled=context.settings.cold_l2)
        policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        measurements = []
        for index, (rows, topk) in enumerate(self._CASES):
            q, kv_cache, selected, cache_seqlens, _kv_bytes = _make_glm_inputs(
                rows=rows,
                num_heads=16,
                topk=topk,
                device=device,
                seed=context.settings.seed + 17 * index,
            )
            plan = sparse_mla.plan(
                sparse_mla.Caps(
                    device=device,
                    num_q_heads=16,
                    max_q_rows=rows,
                    max_width=topk,
                    dtype=torch.bfloat16,
                    kv_dtype=torch.uint8,
                    head_dim=576,
                    v_head_dim=512,
                    mode="decode",
                    max_batch=rows,
                    max_kv_rows=int(kv_cache.shape[0]),
                    max_chunks_per_row=max(1, math.ceil(topk / 64)),
                    page_size=1,
                ),
                policy=policy,
            )
            (scratch_spec,) = plan.scratch_specs()
            scratch = torch.empty(
                scratch_spec.shape,
                dtype=scratch_spec.dtype,
                device=scratch_spec.device,
            )
            binding = sparse_mla.bind(
                plan,
                scratch=scratch,
                q=q,
                selected_indices=selected,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=cache_seqlens,
            )
            sm_scale = 1.0 / math.sqrt(576)
            expected = sparse_mla_reference(
                q_all=q,
                kv_cache=kv_cache,
                page_table_1=selected,
                active_token_counts=cache_seqlens,
                sm_scale=sm_scale,
                v_head_dim=512,
            )

            def run():
                return sparse_mla.run_decode(
                    kv_cache=kv_cache,
                    binding=binding,
                    sm_scale=sm_scale,
                    v_head_dim=512,
                )

            output = run()
            torch.cuda.synchronize(device)
            measurements.append(
                _timed_graph_measurement(
                    context=context,
                    label=f"decode-rows{rows}-topk{topk}",
                    run=run,
                    output=output,
                    expected=expected,
                    flush=flush,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
        return tuple(measurements)


class _MhcProbe:
    _CASES = ((4, 4_096, 64), (32, 4_096, 64), (4, 7_168, 112))

    @property
    def case_count(self) -> int:
        return len(self._CASES)

    @property
    def description(self) -> str:
        return "production mHC post/pre graph qualification"

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]:
        import torch

        from b12x.norm import mhc
        from b12x.policy import PolicyContext, PolicyMode
        from benchmarks.benchmark_residual import (
            _make_inputs,
            _mhc_pre_reference,
            _post_pre_reference,
        )

        device = torch.device("cuda", context.device_ordinal)
        flush = _l2_flush_fn(device, enabled=context.settings.cold_l2)
        policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        measurements = []
        for index, (tokens, hidden_size, split_k) in enumerate(self._CASES):
            residual, x, fn, scale, bias = _make_inputs(
                tokens=tokens,
                hidden_size=hidden_size,
                seed=context.settings.seed + 17 * index,
                device=device,
            )
            plan = mhc.plan(
                mhc.Caps(
                    device=device,
                    max_tokens=tokens,
                    hidden_size=hidden_size,
                    split_k=split_k,
                ),
                policy=policy,
            )
            scratch = tuple(
                torch.empty(shape, dtype=dtype, device=device)
                for shape, dtype in plan.shapes_and_dtypes()
            )
            output = torch.empty(
                (tokens, 4, hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            y = torch.empty(
                (tokens, hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            post = torch.empty((tokens, 4), dtype=torch.float32, device=device)
            comb = torch.empty((tokens, 4, 4), dtype=torch.float32, device=device)
            binding = mhc.bind(
                plan,
                scratch=scratch,
                tokens=tokens,
                y=y,
                post=post,
                comb=comb,
                out=output,
            )
            _, prev_post, prev_comb = _mhc_pre_reference(
                residual,
                fn,
                scale,
                bias,
                rms_eps=1.0e-6,
                hc_eps=1.0e-6,
                sinkhorn_iters=20,
            )
            prev_post = prev_post.contiguous()
            prev_comb = prev_comb.contiguous()
            expected, _expected_y, _expected_post, _expected_comb = (
                _post_pre_reference(
                    x,
                    residual,
                    prev_post,
                    prev_comb,
                    fn,
                    scale,
                    bias,
                    rms_eps=1.0e-6,
                    hc_eps=1.0e-6,
                    sinkhorn_iters=20,
                    norm_weight=None,
                    norm_eps=1.0e-6,
                )
            )

            def run() -> None:
                mhc.run_post_pre(
                    x,
                    residual,
                    prev_post,
                    prev_comb,
                    fn,
                    scale,
                    bias,
                    rms_eps=1.0e-6,
                    hc_eps=1.0e-6,
                    sinkhorn_iters=20,
                    binding=binding,
                )

            measurements.append(
                _timed_graph_measurement(
                    context=context,
                    label=f"post-pre-m{tokens}-h{hidden_size}",
                    run=run,
                    output=output,
                    expected=expected,
                    flush=flush,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
        return tuple(measurements)


class _EpMoeProbe:
    _CASES = ((4, 4), (16, 8))

    @property
    def case_count(self) -> int:
        return len(self._CASES)

    @property
    def description(self) -> str:
        return "production W4A16 EP-MoE graph qualification against fused MoE"

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]:
        import torch

        from b12x.moe import ep_moe, fused_moe
        from b12x.policy import PolicyContext, PolicyMode
        from b12x.policy.generation.moe_corpus import (
            MOE_RECIPES,
            MoePhysicalGeometry,
        )

        from .moe_gpu_worker import _packed_weights

        device = torch.device("cuda", context.device_ordinal)
        recipe = next(
            item for item in MOE_RECIPES if item.recipe_id == "modelopt-w4a16"
        )
        geometry = MoePhysicalGeometry(
            recipe=recipe,
            activation="silu",
            num_experts=8,
            hidden_size=2_560,
            intermediate_size=640,
            aliases=(),
        )
        experts = _packed_weights(geometry, device=device)
        expert_map_tensor = torch.full(
            (512,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        expert_map_tensor[:8] = torch.arange(8, dtype=torch.int32, device=device)
        expert_map = ep_moe.prepare_expert_map(
            expert_map_tensor,
            local_num_experts=8,
            global_num_experts=512,
            device=device,
        )
        policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        flush = _l2_flush_fn(device, enabled=context.settings.cold_l2)
        measurements = []
        for index, (tokens, top_k) in enumerate(self._CASES):
            generator = torch.Generator(device=device).manual_seed(
                context.settings.seed + 17 * index
            )
            activations = torch.randn(
                (tokens, 2_560),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            topk_ids = (
                torch.arange(top_k, dtype=torch.int32, device=device)
                .view(1, -1)
                .expand(tokens, -1)
                .contiguous()
            )
            topk_weights = torch.softmax(
                torch.randn(
                    (tokens, top_k),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                ),
                dim=-1,
            ).contiguous()
            fused_plan = fused_moe.plan_execution(
                experts=experts,
                capacity=fused_moe.ExecutionCapacity(
                    max_tokens=tokens,
                    top_k=top_k,
                ),
                policy=policy,
            )
            fused_moe.prewarm(fused_plan)
            fused_scratch = {
                spec.name: torch.empty(
                    spec.shape,
                    dtype=spec.dtype,
                    device=spec.device,
                )
                for spec in fused_plan.scratch_specs()
            }
            fused_binding = fused_moe.bind(
                fused_plan,
                scratch=fused_scratch,
                a=activations,
                experts=experts,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=torch.empty_like(activations),
            )
            expected = fused_moe.run(binding=fused_binding).clone()
            ep_plan = ep_moe.plan(
                ep_moe.Caps(
                    max_tokens=tokens,
                    num_topk=top_k,
                    global_num_experts=512,
                    device=device,
                    weight_plan=experts.plan._impl,
                ),
                policy=policy,
            )
            (scratch_spec,) = ep_plan.scratch_specs()
            scratch = torch.empty(
                scratch_spec.shape,
                dtype=scratch_spec.dtype,
                device=scratch_spec.device,
            )
            output = torch.empty_like(activations)
            binding = ep_moe.bind(
                ep_plan,
                scratch=scratch,
                a=activations,
                experts=experts._impl,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
                output=output,
            )

            def run() -> None:
                ep_moe.run(binding=binding)

            measurements.append(
                _timed_graph_measurement(
                    context=context,
                    label=f"m{tokens}-topk{top_k}",
                    run=run,
                    output=output,
                    expected=expected,
                    flush=flush,
                )
            )
        return tuple(measurements)


class DsaIndexerGenerator(MeasuredPolicyGenerator):
    """Generate a measured policy for DSA indexer production paths."""

    def __init__(self) -> None:
        from b12x.attention.dsa_indexer._policy import (
            DSA_INDEXER_POLICY,
            DsaIndexerQuery,
        )

        queries = (
            DsaIndexerQuery(
                source_layout="paged",
                mode="decode",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=128,
                num_idx_heads=1,
                max_q_rows=4,
                max_k_rows=0,
                top_k=2_048,
                page_size=64,
                score_mode="dsa",
                shared_page_table=False,
            ),
            DsaIndexerQuery(
                source_layout="paged",
                mode="prefill",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=16,
                num_idx_heads=1,
                max_q_rows=16,
                max_k_rows=0,
                top_k=2_048,
                page_size=64,
                score_mode="dsa",
                shared_page_table=True,
            ),
            DsaIndexerQuery(
                source_layout="contiguous",
                mode="decode",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=16,
                num_idx_heads=1,
                max_q_rows=4,
                max_k_rows=65_536,
                top_k=2_048,
                page_size=64,
                score_mode="dsa",
                shared_page_table=False,
            ),
        )
        super().__init__(
            policy=DSA_INDEXER_POLICY,
            queries=queries,
            encode_config=lambda config: config.to_dict(),
            probe=_DsaIndexerProbe(),
        )


class SparseMlaGenerator(MeasuredPolicyGenerator):
    """Generate a measured policy for sparse MLA."""

    def __init__(self) -> None:
        from b12x.attention.sparse_mla._policy import (
            SPARSE_MLA_POLICY,
            SparseMlaQuery,
        )

        queries = tuple(
            SparseMlaQuery(
                mode=mode,
                dtype="bfloat16",
                kv_dtype=kv_dtype,
                num_q_heads=16,
                qk_head_dim=576,
                v_head_dim=512,
                max_q_rows=rows,
                max_width=2_048,
                page_size=64,
                model_type=None,
                head_major_output=False,
            )
            for mode in ("decode", "extend")
            for kv_dtype in ("bfloat16", "float8_e4m3fn")
            for rows in (1, 16)
        )
        super().__init__(
            policy=SPARSE_MLA_POLICY,
            queries=queries,
            encode_config=lambda config: config.to_dict(),
            probe=_SparseMlaProbe(),
        )


class MhcGenerator(MeasuredPolicyGenerator):
    """Generate a measured policy for mHC residual fusion."""

    def __init__(self) -> None:
        from b12x.norm.mhc._policy import MHC_POLICY, MhcQuery

        queries = tuple(
            MhcQuery(
                dtype="bfloat16",
                max_tokens=tokens,
                hidden_size=hidden_size,
                split_k=64,
            )
            for hidden_size in (4_096, 7_168)
            for tokens in (4, 32)
        )
        super().__init__(
            policy=MHC_POLICY,
            queries=queries,
            encode_config=lambda config: config.to_dict(),
            probe=_MhcProbe(),
        )


class EpMoeGenerator(MeasuredPolicyGenerator):
    """Generate a measured policy for replicated-input EP MoE."""

    def __init__(self) -> None:
        from b12x.moe.ep_moe._policy import EP_MOE_POLICY, EpMoeQuery

        queries = tuple(
            EpMoeQuery(
                max_tokens=tokens,
                top_k=top_k,
                num_experts=512,
                hidden_size=2_560,
                intermediate_size=640,
                activation="silu",
            )
            for tokens in (4, 16)
            for top_k in (4, 8)
        )
        super().__init__(
            policy=EP_MOE_POLICY,
            queries=queries,
            encode_config=lambda config: config.to_dict(),
            probe=_EpMoeProbe(),
        )


__all__ = [
    "DsaIndexerGenerator",
    "EpMoeGenerator",
    "MhcGenerator",
    "SparseMlaGenerator",
]
