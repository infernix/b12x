from __future__ import annotations

from benchmarks.benchmark_gdn_decode import QWEN38_GDN_CASES
from benchmarks.benchmark_paged_attention import BENCHMARK_PROFILES
from benchmarks.benchmark_qsa import PROFILES as QSA_PROFILES
from b12x.policy import list_profiled_components
from b12x.policy.generation.attention_corpus import (
    ATTENTION_BENCHMARK_PRESETS,
    GDN_GEOMETRIES,
    GQA_GEOMETRIES,
    MLA_GEOMETRIES,
    QSA_GEOMETRIES,
    SPARSE_MLA_GEOMETRIES,
    attention_corpus_manifest,
    gdn_cases,
    gqa_cases,
    mla_cases,
    qsa_cases,
    sparse_mla_cases,
)
from b12x.policy.generation.providers import register_builtin_generators
from b12x.policy.generation.providers.gpu_workers import GdnBenchmarkFactory
from b12x.policy.generation.providers.qualification import (
    DsaIndexerGenerator,
    MhcGenerator,
    SparseMlaGenerator,
)
from b12x.policy.generation.registry import ComponentGeneratorRegistry
from b12x.sequence.gdn_decode._policy import GDN_POLICY, GdnQuery


def test_builtin_registry_covers_every_top_level_component() -> None:
    registry = ComponentGeneratorRegistry()

    register_builtin_generators(registry)

    assert registry.component_ids() == tuple(
        str(item.component_id) for item in list_profiled_components()
    )


def test_attention_corpora_have_stable_reviewed_cross_products() -> None:
    assert len(GDN_GEOMETRIES) == 21
    assert len(GQA_GEOMETRIES) == 18
    assert len(MLA_GEOMETRIES) == 1
    assert len(QSA_GEOMETRIES) == 3
    assert len(SPARSE_MLA_GEOMETRIES) == 12
    assert len(gdn_cases()) == 38
    assert len(gqa_cases()) == 4_320
    assert len(mla_cases()) == 60
    assert len(qsa_cases()) == 972
    assert len(sparse_mla_cases()) == 72
    assert len({case.query for case in gqa_cases()}) == len(gqa_cases())

    all_cases = (
        *gdn_cases(),
        *gqa_cases(),
        *mla_cases(),
        *qsa_cases(),
        *sparse_mla_cases(),
    )
    assert len({case.case_id for case in all_cases}) == len(all_cases)


def test_gdn_corpus_includes_qwen_and_glm_decay_contracts() -> None:
    recipes = {case.metadata["decay_recipe"] for case in gdn_cases()}
    glm_cases = [
        case for case in gdn_cases() if case.metadata["decay_recipe"] == "kda"
    ]

    assert recipes == {"gdn", "kda"}
    assert len(glm_cases) == 20
    assert {case.query["key_heads"] for case in glm_cases} == {4, 8, 16, 32, 64}
    assert all(
        case.query["key_heads"] == case.query["value_heads"] for case in glm_cases
    )


def test_gdn_backend_identifies_decay_contract_from_head_geometry() -> None:
    common = {
        "gate_activation": "sigmoid",
        "qk_l2norm": True,
        "state_dtype": "float32",
        "max_seqs": 1,
        "max_tokens": 4,
        "state_index_columns": 4,
    }

    qwen = GdnQuery(key_heads=8, value_heads=24, **common)
    glm = GdnQuery(key_heads=8, value_heads=8, **common)

    assert GDN_POLICY.heuristic(qwen, None).backend == "cutedsl"
    assert GDN_POLICY.heuristic(glm, None).backend == "triton"


def test_gdn_benchmark_factory_accepts_grouped_capacity_cases() -> None:
    group_id = gdn_cases()[0].group_id
    cases = tuple(
        case for case in gdn_cases() if case.group_id == group_id
    )

    session = GdnBenchmarkFactory()(group_id, cases, object())

    assert len(cases) > 1
    assert session.candidates(cases[0])[0].config["backend"] == "cutedsl"


def test_attention_corpus_manifests_are_content_addressed() -> None:
    for component in ("gdn", "gqa", "mla", "qsa", "sparse_mla"):
        manifest = attention_corpus_manifest(component)

        assert manifest["schema_version"] == 1
        assert len(manifest["corpus_sha256"]) == 64


def test_glm_fixed_backend_qualification_envelope_matches_presets() -> None:
    dsa_queries = DsaIndexerGenerator().reviewed_queries()
    sparse_queries = SparseMlaGenerator().reviewed_queries()
    mhc_queries = MhcGenerator().reviewed_queries()

    assert any(
        query.num_q_heads == 32 and query.top_k == 2_048
        for query in dsa_queries
    )
    assert any(
        query.num_q_heads == 32 and query.top_k == 512 for query in dsa_queries
    )
    assert {
        (query.qk_head_dim, query.v_head_dim, query.model_type)
        for query in sparse_queries
    } == {(576, 512, None), (512, 512, 2)}
    assert {query.num_q_heads for query in sparse_queries} == {8, 16, 32, 64}
    assert any(
        query.max_tokens == 6
        and query.hidden_size == 4_096
        and query.split_k == 64
        for query in mhc_queries
    )
    assert {query.score_mode for query in dsa_queries} == {"dsa", "msa"}


def test_named_attention_benchmark_presets_are_in_the_reviewed_inventory() -> None:
    preset_ids = {preset.preset_id for preset in ATTENTION_BENCHMARK_PRESETS}
    assert {
        name.removeprefix("paged:")
        for name in preset_ids
        if name.startswith("paged:")
    } == set(BENCHMARK_PROFILES)
    assert {
        name.removeprefix("qsa:")
        for name in preset_ids
        if name.startswith("qsa:")
    } == set(QSA_PROFILES)
    assert {
        name.removeprefix("gdn:")
        for name in preset_ids
        if name.startswith("gdn:")
    } == {case.name for case in QWEN38_GDN_CASES}
    assert preset_ids == {
        "compressed-mla:deepseek-v4-flash-default",
        "compressed-mla:vllm-dsv4-trace",
        "dense-mla:kimi-k3",
        "dsa-indexer:glm-5.1-default",
        "gdn:qk16-v48-decode-bs1",
        "gdn:qk2-v6-decode-bs1",
        "gdn:qk4-v12-decode-bs1",
        "gdn:qk8-v24-decode-bs1",
        "gdn:qk8-v24-decode-bs4",
        "gdn:qk8-v24-spec2-bs4",
        "gdn:qk8-v24-spec4-bs1",
        "gdn:qk8-v24-spec4-bs4",
        "gdn:qk8-v24-spec4-uneven",
        "mla:target-dsv4-trace",
        "mla:target-glm52-prefill4k-ctx16k",
        "mla:target-prefill64k-bs1",
        "mla:glm-5.2-default",
        "msa-indexer:minimax-m3-default",
        "paged-msa:minimax-m3-default",
        "paged:minimax-m2.7",
        "paged:qwen-gqa",
        "paged:qwen3.8-27b",
        "paged-indexer:deepseek-v4-flash-default",
        "qsa:tp1",
        "qsa:tp2",
        "qsa:tp4",
        "unified-mla:deepseek-v4-flash-decode",
        "unified-mla:deepseek-v4-flash-prefill",
        "unified-mla:glm-5.1-decode",
        "vllm-paged:minimax-m2.7",
        "vllm-paged:qwen-gqa",
    }
