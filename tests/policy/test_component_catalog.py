from __future__ import annotations

import b12x
from b12x.attention.qsa._policy import QSA_POLICY, QsaQuery
from b12x.policy import (
    EMBEDDED_REGISTRY,
    PlanningPolicyMode,
    PolicyContext,
    PolicyMode,
    PolicySource,
    list_planning_components,
    list_profiled_components,
    validate_component_profile_contract,
)
from b12x.policy.generation import ComponentGeneratorRegistry
from b12x.policy.generation.providers import register_builtin_generators


def test_every_planned_op_has_exactly_one_policy_registration() -> None:
    planned_ops = {
        meta.qualname for meta in b12x.list_ops() if meta.api_style == "planned"
    }
    registrations = list_planning_components()

    assert len({item.op_qualname for item in registrations}) == len(registrations)
    assert {item.op_qualname for item in registrations} == planned_ops
    assert all(item.mode is PlanningPolicyMode.PROFILED for item in registrations)


def test_profiled_policies_generators_and_embedded_profile_stay_in_lockstep() -> None:
    registrations = list_profiled_components()
    expected_ids = tuple(str(item.component_id) for item in registrations)
    generator_registry = ComponentGeneratorRegistry()
    register_builtin_generators(generator_registry)
    profile = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm")

    assert generator_registry.component_ids() == expected_ids
    assert tuple(component.component_id for component in profile.components) == (
        expected_ids
    )
    for registration in registrations:
        policy = registration.load_policy()
        generator = registration.create_generator()
        component = profile.component(str(registration.component_id))
        assert component is not None
        assert generator.component_id == policy.component_id
        assert generator.query_schema_version == policy.query_schema_version
        assert generator.config_schema_version == policy.config_schema_version
        validate_component_profile_contract(policy, component)


def test_qwen_flash_next_qsa_serving_shape_resolves_from_gb10_profile() -> None:
    profile = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm")
    context = PolicyContext.for_identity(
        profile.targets[0],
        mode=PolicyMode.PREPLANNED_ONLY,
    )

    resolution = context.resolve(
        QSA_POLICY,
        QsaQuery(
            q_dtype="bfloat16",
            kv_dtype="float8_e4m3fn",
            q_heads=24,
            kv_heads=2,
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
            budget=2048,
            position_axes=3,
            mrope_interleaved=True,
        ),
    )

    assert resolution.source is PolicySource.PREPLANNED
    assert resolution.rule_name == "measured-production-implementation"
    assert resolution.config.backend == "cutedsl"


def test_qwen_flash_next_planners_are_profile_backed() -> None:
    registrations = {item.op_qualname: item for item in list_planning_components()}

    for op_qualname in (
        "norm.hyperconnection",
        "sequence.mtp_feedback",
        "sequence.ple",
        "sequence.ple_embedding",
        "sequence.ple_hash",
    ):
        registration = registrations[op_qualname]
        assert registration.mode is PlanningPolicyMode.PROFILED
        assert registration.component_id is not None


def test_unknown_gpu_uses_the_component_heuristic() -> None:
    from b12x.norm.hyperconnection._policy import (
        HYPERCONNECTION_POLICY,
        HyperConnectionQuery,
    )
    from b12x.policy import DeviceIdentity

    context = PolicyContext.for_identity(
        DeviceIdentity(
            vendor="nvidia",
            compute_capability=(12, 1),
            sm_count=47,
            product_name="unknown synthetic gpu",
        )
    )
    resolution = context.resolve(
        HYPERCONNECTION_POLICY,
        HyperConnectionQuery(
            dtype="bfloat16",
            max_tokens=4,
            hidden_size=2560,
            streams=4,
            lowrank=320,
        ),
    )

    assert resolution.source is PolicySource.HEURISTIC
    assert resolution.config.reduction_block_h == 4096


def test_every_profile_generator_has_real_gpu_measurement_work() -> None:
    for registration in list_profiled_components():
        generator = registration.create_generator()
        assert generator.component_id == registration.component_id
        assert "precomputed" not in type(generator).__module__
