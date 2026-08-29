from __future__ import annotations

from b12x.policy import list_profiled_components
from b12x.policy.generation.attention_corpus import (
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
from b12x.policy.generation.registry import ComponentGeneratorRegistry


def test_builtin_registry_covers_every_top_level_component() -> None:
    registry = ComponentGeneratorRegistry()

    register_builtin_generators(registry)

    assert registry.component_ids() == tuple(
        str(item.component_id) for item in list_profiled_components()
    )


def test_attention_corpora_have_stable_reviewed_cross_products() -> None:
    assert len(GDN_GEOMETRIES) == 10
    assert len(GQA_GEOMETRIES) == 4
    assert len(MLA_GEOMETRIES) == 1
    assert len(QSA_GEOMETRIES) == 3
    assert len(SPARSE_MLA_GEOMETRIES) == 3
    assert len(gdn_cases()) == 10
    assert len(gqa_cases()) == 960
    assert len(mla_cases()) == 60
    assert len(qsa_cases()) == 432
    assert len(sparse_mla_cases()) == 18

    all_cases = (
        *gdn_cases(),
        *gqa_cases(),
        *mla_cases(),
        *qsa_cases(),
        *sparse_mla_cases(),
    )
    assert len({case.case_id for case in all_cases}) == len(all_cases)


def test_attention_corpus_manifests_are_content_addressed() -> None:
    for component in ("gdn", "gqa", "mla", "qsa", "sparse_mla"):
        manifest = attention_corpus_manifest(component)

        assert manifest["schema_version"] == 1
        assert len(manifest["corpus_sha256"]) == 64
