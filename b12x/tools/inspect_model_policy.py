"""Inspect model-level kernel selections without allocating model weights."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

from b12x.policy import (
    EMBEDDED_REGISTRY,
    ComponentPolicy,
    DeviceIdentity,
    PolicyContext,
    PolicyResolution,
    detect_device,
)


@dataclass(frozen=True, kw_only=True)
class KernelQuery:
    scenario: str
    kernel_family: str
    policy: ComponentPolicy[Any, Any]
    query: object


@dataclass(frozen=True, kw_only=True)
class DeviceSelection:
    identity: DeviceIdentity
    runtime_device: str


def _qwen_flash_next_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    from b12x.attention.paged._policy import GQA_POLICY, GqaQuery
    from b12x.attention.qsa._policy import QSA_POLICY, QsaQuery
    from b12x.gemm.block_fp8_linear._policy import (
        BLOCK_FP8_LINEAR_POLICY,
        BlockFp8LinearQuery,
    )
    from b12x.gemm.wo_projection._policy import (
        WO_PROJECTION_POLICY,
        WoProjectionQuery,
    )
    from b12x.moe.fused_moe._policy import MOE_DECODE_POLICY, MoeDecodeQuery
    from b12x.norm.hyperconnection._policy import (
        HYPERCONNECTION_POLICY,
        HyperConnectionQuery,
    )
    from b12x.sequence.gdn_decode._policy import GDN_POLICY, GdnQuery
    from b12x.sequence.mtp_feedback._policy import (
        MTP_FEEDBACK_POLICY,
        MtpFeedbackQuery,
    )
    from b12x.sequence.ple._policy import PLE_POLICY, PleQuery
    from b12x.sequence.ple_embedding._policy import (
        PLE_EMBEDDING_POLICY,
        PleEmbeddingQuery,
    )
    from b12x.sequence.ple_hash._policy import PLE_HASH_POLICY, PleHashQuery

    if tp_size not in (1, 2, 4):
        raise ValueError(
            "qwen3.8-flash-next-180b supports TP 1, 2, or 4 in the "
            "profiled QSA contract"
        )
    q_heads = 24 // tp_size
    kv_heads = max(1, 2 // tp_size)
    gdn_key_heads = 16 // tp_size
    gdn_value_heads = 48 // tp_size
    intermediate = ((640 + tp_size - 1) // tp_size + 15) // 16 * 16
    queries = [
        KernelQuery(
            scenario="full-attention-decode",
            kernel_family="paged-gqa",
            policy=GQA_POLICY,
            query=GqaQuery(
                device=runtime_device,
                mode="decode",
                q_dtype="bfloat16",
                kv_dtype="float8_e4m3fn",
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim_qk=256,
                head_dim_vo=256,
                page_size=128,
                kv_cache_layout="separate",
                batch_size=1,
                query_len=1,
                cache_tokens=65_536,
                window_left=-1,
                requested_graph_ctas_per_sm=None,
                requested_max_work_items=None,
                requested_max_partial_rows=None,
                force_split_kv=None,
            ),
        ),
        KernelQuery(
            scenario="qsa-spec4",
            kernel_family="qsa",
            policy=QSA_POLICY,
            query=QsaQuery(
                q_dtype="bfloat16",
                kv_dtype="float8_e4m3fn",
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=256,
                index_heads=4,
                index_kv_heads=1,
                index_head_dim=128,
                index_rotary_dim=64,
                main_page_size=16,
                max_batch=1,
                max_q_rows=4,
                max_seq_len=65_536,
                max_speculative_tokens=3,
                compress_ratio=4,
                budget=2_048,
                position_axes=3,
                mrope_interleaved=True,
            ),
        ),
        KernelQuery(
            scenario="gdn-spec4",
            kernel_family="gdn",
            policy=GDN_POLICY,
            query=GdnQuery(
                gate_activation="sigmoid",
                qk_l2norm=True,
                state_dtype="float32",
                key_heads=gdn_key_heads,
                value_heads=gdn_value_heads,
                max_seqs=1,
                max_tokens=4,
                state_index_columns=4,
            ),
        ),
        KernelQuery(
            scenario="attention-output",
            kernel_family="wo-projection",
            policy=WO_PROJECTION_POLICY,
            query=WoProjectionQuery(
                dtype="bfloat16",
                max_tokens=4,
                groups=q_heads,
                group_width=512,
                rank=512,
                hidden=2_560,
            ),
        ),
        KernelQuery(
            scenario="mxfp8-linear",
            kernel_family="block-fp8-linear",
            policy=BLOCK_FP8_LINEAR_POLICY,
            query=BlockFp8LinearQuery(
                max_tokens=4,
                in_features=2_560,
                out_features=2_560,
                output_dtype="bfloat16",
            ),
        ),
    ]
    for tokens in (1, 4, 7):
        queries.append(
            KernelQuery(
                scenario=f"moe-m{tokens}",
                kernel_family="fused-moe",
                policy=MOE_DECODE_POLICY,
                query=MoeDecodeQuery(
                    quant_mode="nvfp4",
                    source_format="modelopt_nvfp4",
                    activation="silu",
                    num_experts=512,
                    hidden_size=2_560,
                    intermediate_size=intermediate,
                    top_k=10,
                    num_tokens=tokens,
                    routed_rows=tokens * 10,
                ),
            )
        )
    queries.extend(
        (
            KernelQuery(
                scenario="residual-spec4",
                kernel_family="hyperconnection",
                policy=HYPERCONNECTION_POLICY,
                query=HyperConnectionQuery(
                    dtype="bfloat16",
                    max_tokens=4,
                    hidden_size=2_560,
                    streams=4,
                    lowrank=320,
                ),
            ),
            KernelQuery(
                scenario="mtp-feedback-spec4",
                kernel_family="mtp-feedback",
                policy=MTP_FEEDBACK_POLICY,
                query=MtpFeedbackQuery(
                    dtype="bfloat16",
                    max_tokens=4,
                    hidden_size=2_560,
                    streams=4,
                ),
            ),
            KernelQuery(
                scenario="ple-spec4",
                kernel_family="ple",
                policy=PLE_POLICY,
                query=PleQuery(
                    mode="decode",
                    dtype="bfloat16",
                    max_tokens=4,
                    max_seqs=1,
                    max_speculative_tokens=3,
                    streams=4,
                    hidden_size=2_560,
                    kernel_size=4,
                    dilation=3,
                ),
            ),
            KernelQuery(
                scenario="ple-hash-spec4",
                kernel_family="ple-hash",
                policy=PLE_HASH_POLICY,
                query=PleHashQuery(
                    max_tokens=4,
                    max_seqs=1,
                    vocab_size=248_320,
                    max_order=3,
                    heads_per_order=8,
                    base_table_size=20_000_000,
                ),
            ),
            KernelQuery(
                scenario="ple-embedding-spec4",
                kernel_family="ple-embedding",
                policy=PLE_EMBEDDING_POLICY,
                query=PleEmbeddingQuery(
                    quant_mode="nvfp4_group16",
                    table_memory="mapped_host",
                    output_dtype="bfloat16",
                    max_tokens=4,
                    max_seqs=1,
                    vocab_size=248_320,
                    max_order=3,
                    heads_per_order=8,
                    base_table_size=20_000_000,
                    embedding_dim=2_560,
                    tp_size=tp_size,
                ),
            ),
        )
    )
    return tuple(queries)


def _qwen_dense_queries(
    tp_size: int,
    *,
    runtime_device: str,
) -> tuple[KernelQuery, ...]:
    from b12x.attention.paged._policy import GQA_POLICY, GqaQuery
    from b12x.quantization.nvfp4._policy import (
        NVFP4_QUANTIZATION_POLICY,
        Nvfp4QuantizationQuery,
    )

    if tp_size not in (1, 2, 4, 8):
        raise ValueError("qwen3.8-27b supports profiled TP 1, 2, 4, or 8")
    q_heads = 24 // tp_size
    kv_heads = max(1, 4 // tp_size)
    return (
        KernelQuery(
            scenario="full-attention-decode",
            kernel_family="paged-gqa",
            policy=GQA_POLICY,
            query=GqaQuery(
                device=runtime_device,
                mode="decode",
                q_dtype="bfloat16",
                kv_dtype="float8_e4m3fn",
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim_qk=256,
                head_dim_vo=256,
                page_size=128,
                kv_cache_layout="separate",
                batch_size=1,
                query_len=1,
                cache_tokens=65_536,
                window_left=-1,
                requested_graph_ctas_per_sm=None,
                requested_max_work_items=None,
                requested_max_partial_rows=None,
                force_split_kv=None,
            ),
        ),
        KernelQuery(
            scenario="nvfp4-activation-block",
            kernel_family="nvfp4-quantization",
            policy=NVFP4_QUANTIZATION_POLICY,
            query=Nvfp4QuantizationQuery(
                dtype="bfloat16",
                rows=128,
                columns=5_120,
            ),
        ),
    )


_MODEL_FACTORIES = {
    "qwen3.8-flash-next-180b": _qwen_flash_next_queries,
    "qwen3.8-27b": _qwen_dense_queries,
}
_MODEL_ALIASES = {
    "qwen3.8-flash-next": "qwen3.8-flash-next-180b",
    "qwen38-flash-next": "qwen3.8-flash-next-180b",
    "qwen38-flash-next-180b": "qwen3.8-flash-next-180b",
    "qwen38-27b": "qwen3.8-27b",
}


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _canonical_model(value: str) -> str:
    key = value.casefold()
    canonical = _MODEL_ALIASES.get(key, key)
    if canonical not in _MODEL_FACTORIES:
        choices = ", ".join(sorted(_MODEL_FACTORIES))
        raise ValueError(f"unknown model preset {value!r}; choose one of {choices}")
    return canonical


def _device_selection(value: str) -> DeviceSelection:
    if value.casefold() == "auto":
        detected = detect_device()
        if detected.identity is None or detected.ordinal is None:
            raise ValueError("--device auto did not find a CUDA device")
        return DeviceSelection(
            identity=detected.identity,
            runtime_device=f"cuda:{detected.ordinal}",
        )

    needle = _normalize_name(value)
    exact: list[tuple[object, DeviceIdentity]] = []
    partial: list[tuple[object, DeviceIdentity]] = []
    for profile in EMBEDDED_REGISTRY.list_profiles():
        for target in profile.targets:
            aliases = (profile.profile_id, target.product_name)
            normalized = tuple(_normalize_name(alias) for alias in aliases)
            if needle in normalized:
                exact.append((profile, target))
            elif any(needle and needle in alias for alias in normalized):
                partial.append((profile, target))
    matches = exact or partial
    unique = {
        (profile.profile_id, target): (profile, target)
        for profile, target in matches
    }
    if len(unique) != 1:
        choices = ", ".join(
            profile.profile_id for profile in EMBEDDED_REGISTRY.list_profiles()
        )
        if not unique:
            raise ValueError(
                f"unknown embedded device/profile {value!r}; choose one of {choices}"
            )
        matched = ", ".join(sorted(profile_id for profile_id, _ in unique))
        raise ValueError(f"ambiguous device name {value!r}; matches {matched}")
    _key, (_profile, identity) = next(iter(unique.items()))
    detected = detect_device()
    ordinal = (
        detected.ordinal
        if detected.identity == identity and detected.ordinal is not None
        else 0
    )
    return DeviceSelection(identity=identity, runtime_device=f"cuda:{ordinal}")


def _config_dict(config: object) -> dict[str, object]:
    if is_dataclass(config):
        return asdict(config)
    for method_name in ("to_dict", "profile_dict"):
        method = getattr(config, method_name, None)
        if callable(method):
            payload = method()
            if isinstance(payload, dict):
                return payload
    raise TypeError(f"cannot serialize policy config {type(config).__name__}")


def _kernel_name(item: KernelQuery, config: dict[str, object]) -> str:
    backend = config.get("backend")
    if item.policy.component_id == "moe.decode":
        return f"{backend}/{config['route_planner']}"
    return str(backend) if backend is not None else item.kernel_family


def _record(
    item: KernelQuery,
    resolution: PolicyResolution[Any],
) -> dict[str, object]:
    config = _config_dict(resolution.config)
    return {
        "scenario": item.scenario,
        "component": item.policy.component_id,
        "kernel": _kernel_name(item, config),
        "source": resolution.source.value,
        "profile_id": resolution.profile_id,
        "rule": resolution.rule_name,
        "query": dict(item.policy.encode_query(item.query)),
        "config": config,
    }


def inspect_model_policy(
    model: str,
    *,
    tp_size: int,
    device: str,
) -> dict[str, object]:
    if tp_size <= 0:
        raise ValueError("TP size must be positive")
    canonical = _canonical_model(model)
    selected = _device_selection(device)
    context = PolicyContext.for_identity(selected.identity)
    queries = _MODEL_FACTORIES[canonical](
        tp_size,
        runtime_device=selected.runtime_device,
    )
    return {
        "model": canonical,
        "tp_size": tp_size,
        "device": {
            "product_name": selected.identity.product_name,
            "compute_capability": list(selected.identity.compute_capability),
            "sm_count": selected.identity.sm_count,
        },
        "profile_id": context.profile_id,
        "selections": [
            _record(item, context.resolve(item.policy, item.query))
            for item in queries
        ],
    }


def _render_table(console: Console, payload: dict[str, object]) -> None:
    table = Table(
        title=(
            f"{payload['model']} TP={payload['tp_size']} on "
            f"{payload['device']['product_name']}"
        )
    )
    table.add_column("Scenario")
    table.add_column("Component")
    table.add_column("Kernel")
    table.add_column("Source")
    table.add_column("Rule")
    table.add_column("Config")
    for selection in payload["selections"]:
        table.add_row(
            str(selection["scenario"]),
            str(selection["component"]),
            str(selection["kernel"]),
            str(selection["source"]),
            str(selection["rule"] or "-"),
            json.dumps(selection["config"], sort_keys=True, separators=(",", ":")),
        )
    console.print(table)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print the model-level kernels selected by embedded GPU profiles, "
            "including heuristic fallbacks."
        )
    )
    parser.add_argument("model", nargs="?", help="model preset name")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, an embedded profile ID, or a device-name fragment",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list_models:
        for model in sorted(_MODEL_FACTORIES):
            print(model)
        return 0
    if args.list_devices:
        for profile in EMBEDDED_REGISTRY.list_profiles():
            names = ", ".join(target.product_name for target in profile.targets)
            print(f"{profile.profile_id}: {names}")
        return 0
    if args.model is None:
        parser.error("model is required unless a --list option is used")
    try:
        payload = inspect_model_policy(
            args.model,
            tp_size=args.tp,
            device=args.device,
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_table(Console(), payload)
    return 0


__all__ = ["inspect_model_policy", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
