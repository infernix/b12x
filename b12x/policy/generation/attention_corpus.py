"""Reviewed serving-shape corpora for attention profile generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .sweep import SweepCase

COMMON_BATCHES = (1, 2, 4, 8, 12, 16)
COMMON_CONTEXT_TOKENS = (128, 16_384, 32_768, 65_536, 131_072)
COMMON_PAGE_SIZES = (64, 128)
COMMON_KV_DTYPES = ("bfloat16", "float8_e4m3fn")
QSA_BATCHES = (1, 4, 16)
QSA_CONTEXT_TOKENS = (65_536, 131_072)
QSA_PAGE_SIZES = (16, 64)
QSA_SPECULATIVE_TOKENS = (0, 3)
QSA_POSITION_LAYOUTS = ((1, False), (3, False), (3, True))


@dataclass(frozen=True, kw_only=True)
class GdnGeometry:
    model_id: str
    key_heads: int
    value_heads: int
    query_lengths: tuple[int, ...]
    source: str
    state_dtype: str = "float32"

    def __post_init__(self) -> None:
        if not self.model_id or not self.source or not self.query_lengths:
            raise ValueError("GDN geometry labels and query lengths are required")
        if self.key_heads <= 0 or self.value_heads <= 0:
            raise ValueError("GDN head counts must be positive")
        if any(length <= 0 for length in self.query_lengths):
            raise ValueError("GDN query lengths must be positive")


GDN_GEOMETRIES = (
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk16-v48-decode-bs1",
        query_lengths=(1,),
        key_heads=16,
        value_heads=48,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk16-v48-spec4-bs1",
        query_lengths=(4,),
        key_heads=16,
        value_heads=48,
        source="Qwen3.8 Flash Next TP1 MTP serving capacity",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk8-v24-decode-bs1",
        query_lengths=(1,),
        key_heads=8,
        value_heads=24,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk8-v24-decode-bs4",
        query_lengths=(1, 1, 1, 1),
        key_heads=8,
        value_heads=24,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk8-v24-spec2-bs4",
        query_lengths=(2, 2, 2, 2),
        key_heads=8,
        value_heads=24,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk8-v24-spec4-bs1",
        query_lengths=(4,),
        key_heads=8,
        value_heads=24,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk8-v24-spec4-uneven",
        query_lengths=(4, 2, 1, 3),
        key_heads=8,
        value_heads=24,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk8-v24-spec4-bs4",
        query_lengths=(4, 4, 4, 4),
        key_heads=8,
        value_heads=24,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk4-v12-decode-bs1",
        query_lengths=(1,),
        key_heads=4,
        value_heads=12,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk2-v6-decode-bs1",
        query_lengths=(1,),
        key_heads=2,
        value_heads=6,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
)


@dataclass(frozen=True, kw_only=True)
class GqaGeometry:
    model_id: str
    q_heads: int
    kv_heads: int
    head_dim: int
    source: str

    def __post_init__(self) -> None:
        if not self.model_id or not self.source:
            raise ValueError("GQA geometry labels are required")
        if min(self.q_heads, self.kv_heads, self.head_dim) <= 0:
            raise ValueError("GQA geometry values must be positive")
        if self.q_heads % self.kv_heads:
            raise ValueError("GQA query heads must be divisible by KV heads")


GQA_GEOMETRIES = (
    GqaGeometry(
        model_id="qwen3.8-flash-next-180b",
        q_heads=24,
        kv_heads=2,
        head_dim=256,
        source="Qwen3.8 Flash Next text_config",
    ),
    GqaGeometry(
        model_id="qwen3.8-27b",
        q_heads=24,
        kv_heads=4,
        head_dim=256,
        source="benchmark_paged_attention.BENCHMARK_PROFILES",
    ),
    GqaGeometry(
        model_id="qwen-gqa",
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        source="benchmark_paged_attention.BENCHMARK_PROFILES",
    ),
    GqaGeometry(
        model_id="minimax-m2.7",
        q_heads=24,
        kv_heads=4,
        head_dim=128,
        source="benchmark_paged_attention.BENCHMARK_PROFILES",
    ),
)


@dataclass(frozen=True, kw_only=True)
class QsaGeometry:
    model_id: str
    tensor_parallel_size: int
    q_heads: int
    kv_heads: int
    source: str


QSA_GEOMETRIES = (
    QsaGeometry(
        model_id="qwen3.8-flash-next-180b-tp1",
        tensor_parallel_size=1,
        q_heads=24,
        kv_heads=2,
        source="Qwen3.8 Flash Next text_config",
    ),
    QsaGeometry(
        model_id="qwen3.8-flash-next-180b-tp2",
        tensor_parallel_size=2,
        q_heads=12,
        kv_heads=1,
        source="Qwen3.8 Flash Next tensor-parallel slicing",
    ),
    QsaGeometry(
        model_id="qwen3.8-flash-next-180b-tp4",
        tensor_parallel_size=4,
        q_heads=6,
        kv_heads=1,
        source="Qwen3.8 Flash Next tensor-parallel slicing",
    ),
)


@dataclass(frozen=True, kw_only=True)
class MlaGeometry:
    model_id: str
    num_q_heads: int
    qk_head_dim: int
    v_head_dim: int
    page_size: int
    source: str


MLA_GEOMETRIES = (
    MlaGeometry(
        model_id="kimi-k3-dense-mla",
        num_q_heads=8,
        qk_head_dim=576,
        v_head_dim=512,
        page_size=944,
        source="benchmark_dense_mla.py production native K3 defaults",
    ),
)


@dataclass(frozen=True, kw_only=True)
class SparseMlaGeometry:
    model_id: str
    layout: str
    num_q_heads: int
    qk_head_dim: int
    v_head_dim: int
    swa_width: int
    swa_page_size: int
    indexed_width: int
    indexed_page_size: int
    source: str


SPARSE_MLA_GEOMETRIES = (
    SparseMlaGeometry(
        model_id="deepseek-v4-flash-swa",
        layout="compressed_dsv4",
        num_q_heads=32,
        qk_head_dim=512,
        v_head_dim=448,
        swa_width=128,
        swa_page_size=64,
        indexed_width=0,
        indexed_page_size=64,
        source="benchmark_compressed_sparse_mla.py",
    ),
    SparseMlaGeometry(
        model_id="deepseek-v4-flash-swa-c4",
        layout="compressed_dsv4",
        num_q_heads=32,
        qk_head_dim=512,
        v_head_dim=448,
        swa_width=128,
        swa_page_size=64,
        indexed_width=512,
        indexed_page_size=64,
        source="benchmark_compressed_sparse_mla.py",
    ),
    SparseMlaGeometry(
        model_id="deepseek-v4-flash-swa-c128",
        layout="compressed_dsv4",
        num_q_heads=32,
        qk_head_dim=512,
        v_head_dim=448,
        swa_width=128,
        swa_page_size=64,
        indexed_width=512,
        indexed_page_size=2,
        source="benchmark_compressed_sparse_mla.py",
    ),
)


def gdn_cases() -> tuple[SweepCase, ...]:
    cases = []
    for geometry in GDN_GEOMETRIES:
        lengths = geometry.query_lengths
        max_seqs = len(lengths)
        state_index_columns = max(lengths)
        query = {
            "gate_activation": "sigmoid",
            "qk_l2norm": True,
            "key_heads": geometry.key_heads,
            "value_heads": geometry.value_heads,
            "state_dtype": geometry.state_dtype,
            "max_seqs": max_seqs,
            "max_tokens": max_seqs * state_index_columns,
            "state_index_columns": state_index_columns,
        }
        cases.append(
            SweepCase.create(
                group_id=geometry.model_id,
                query=query,
                metadata={
                    "model_id": geometry.model_id,
                    "query_lengths": list(lengths),
                    "source": geometry.source,
                },
                label=geometry.model_id,
            )
        )
    return tuple(cases)


def gqa_cases() -> tuple[SweepCase, ...]:
    cases = []
    for geometry in GQA_GEOMETRIES:
        for kv_dtype in COMMON_KV_DTYPES:
            for page_size in COMMON_PAGE_SIZES:
                for batch_size in COMMON_BATCHES:
                    for cache_tokens in COMMON_CONTEXT_TOKENS:
                        group_id = (
                            f"{geometry.model_id}-{kv_dtype}-page{page_size}"
                            f"-ctx{cache_tokens}"
                        )
                        for layout in ("separate", "combined"):
                            query = {
                                "mode": "decode",
                                "q_dtype": "bfloat16",
                                "kv_dtype": kv_dtype,
                                "q_heads": geometry.q_heads,
                                "kv_heads": geometry.kv_heads,
                                "head_dim_qk": geometry.head_dim,
                                "head_dim_vo": geometry.head_dim,
                                "page_size": page_size,
                                "batch_size": batch_size,
                                "query_len": 1,
                                "cache_tokens": cache_tokens,
                                "window_left": -1,
                                "requested_graph_ctas_per_sm": None,
                                "force_split_kv": None,
                                "kv_cache_layout": layout,
                            }
                            cases.append(
                                SweepCase.create(
                                    group_id=group_id,
                                    query=query,
                                    metadata={
                                        "model_id": geometry.model_id,
                                        "source": geometry.source,
                                    },
                                    label=geometry.model_id,
                                )
                            )
    return tuple(cases)


def qsa_cases() -> tuple[SweepCase, ...]:
    cases = []
    for geometry in QSA_GEOMETRIES:
        for kv_dtype in COMMON_KV_DTYPES:
            for main_page_size in QSA_PAGE_SIZES:
                for max_batch in QSA_BATCHES:
                    for max_seq_len in QSA_CONTEXT_TOKENS:
                        for max_speculative_tokens in QSA_SPECULATIVE_TOKENS:
                            for (
                                position_axes,
                                mrope_interleaved,
                            ) in QSA_POSITION_LAYOUTS:
                                max_q_rows = max_batch * (1 + max_speculative_tokens)
                                group_id = (
                                    f"{geometry.model_id}-{kv_dtype}"
                                    f"-page{main_page_size}-ctx{max_seq_len}"
                                )
                                query = {
                                    "q_dtype": "bfloat16",
                                    "kv_dtype": kv_dtype,
                                    "q_heads": geometry.q_heads,
                                    "kv_heads": geometry.kv_heads,
                                    "head_dim": 256,
                                    "index_heads": 4,
                                    "index_kv_heads": 1,
                                    "index_head_dim": 128,
                                    "index_rotary_dim": 64,
                                    "main_page_size": main_page_size,
                                    "max_batch": max_batch,
                                    "max_q_rows": max_q_rows,
                                    "max_seq_len": max_seq_len,
                                    "max_speculative_tokens": (max_speculative_tokens),
                                    "compress_ratio": 4,
                                    "budget": 2048,
                                    "position_axes": position_axes,
                                    "mrope_interleaved": mrope_interleaved,
                                }
                                cases.append(
                                    SweepCase.create(
                                        group_id=group_id,
                                        query=query,
                                        metadata={
                                            "model_id": geometry.model_id,
                                            "source": geometry.source,
                                            "tensor_parallel_size": (
                                                geometry.tensor_parallel_size
                                            ),
                                        },
                                        label=geometry.model_id,
                                    )
                                )
    return tuple(cases)


def mla_cases() -> tuple[SweepCase, ...]:
    cases = []
    decode_rows = (1, 2, 4, 8, 16)
    extend_rows = (128, 2_048, 16_384)
    cache_tokens = (1_024, 32_768, 65_536, 131_072)
    for geometry in MLA_GEOMETRIES:
        for kv_dtype in COMMON_KV_DTYPES:
            group_id = f"{geometry.model_id}-{kv_dtype}"
            for mode, rows_values in (
                ("decode", decode_rows),
                ("extend", extend_rows),
            ):
                for query_rows in rows_values:
                    for width in cache_tokens:
                        if width < query_rows:
                            continue
                        query = {
                            "mode": mode,
                            "q_dtype": kv_dtype,
                            "kv_dtype": kv_dtype,
                            "num_q_heads": geometry.num_q_heads,
                            "qk_head_dim": geometry.qk_head_dim,
                            "v_head_dim": geometry.v_head_dim,
                            "page_size": geometry.page_size,
                            "query_rows": query_rows,
                            "max_batch": (query_rows if mode == "decode" else 1),
                            "cache_tokens": width,
                            "physical_record_width": geometry.qk_head_dim,
                            "window_size": None,
                            "use_cuda_graph": True,
                        }
                        cases.append(
                            SweepCase.create(
                                group_id=group_id,
                                query=query,
                                metadata={
                                    "model_id": geometry.model_id,
                                    "source": geometry.source,
                                },
                                label=geometry.model_id,
                            )
                        )
    return tuple(cases)


def sparse_mla_cases() -> tuple[SweepCase, ...]:
    cases = []
    for geometry in SPARSE_MLA_GEOMETRIES:
        for rows in (1, 4, 16, 64, 256, 4_096):
            query = {
                "layout": geometry.layout,
                "mode": "decode" if rows <= 256 else "extend",
                "q_dtype": "bfloat16",
                "kv_dtype": "float8_e4m3fn",
                "num_q_heads": geometry.num_q_heads,
                "qk_head_dim": geometry.qk_head_dim,
                "v_head_dim": geometry.v_head_dim,
                "query_rows": rows,
                "swa_width": geometry.swa_width,
                "swa_page_size": geometry.swa_page_size,
                "indexed_width": geometry.indexed_width,
                "indexed_page_size": geometry.indexed_page_size,
            }
            cases.append(
                SweepCase.create(
                    group_id=geometry.model_id,
                    query=query,
                    metadata={
                        "model_id": geometry.model_id,
                        "source": geometry.source,
                    },
                    label=geometry.model_id,
                )
            )
    return tuple(cases)


def _manifest_payload(component: str) -> dict[str, object]:
    shared = {
        "common_batches": list(COMMON_BATCHES),
        "common_context_tokens": list(COMMON_CONTEXT_TOKENS),
        "common_kv_dtypes": list(COMMON_KV_DTYPES),
        "common_page_sizes": list(COMMON_PAGE_SIZES),
    }
    if component == "gdn":
        shared["geometries"] = [asdict(item) for item in GDN_GEOMETRIES]
    elif component == "gqa":
        shared["geometries"] = [asdict(item) for item in GQA_GEOMETRIES]
    elif component == "qsa":
        shared["geometries"] = [asdict(item) for item in QSA_GEOMETRIES]
        shared["qsa_batches"] = list(QSA_BATCHES)
        shared["qsa_context_tokens"] = list(QSA_CONTEXT_TOKENS)
        shared["qsa_page_sizes"] = list(QSA_PAGE_SIZES)
        shared["qsa_position_layouts"] = [list(item) for item in QSA_POSITION_LAYOUTS]
        shared["qsa_speculative_tokens"] = list(QSA_SPECULATIVE_TOKENS)
    elif component == "mla":
        shared["geometries"] = [asdict(item) for item in MLA_GEOMETRIES]
    elif component == "sparse_mla":
        shared["geometries"] = [asdict(item) for item in SPARSE_MLA_GEOMETRIES]
    else:
        raise ValueError(f"unknown attention corpus {component!r}")
    return shared


def attention_corpus_manifest(component: str) -> dict[str, object]:
    payload = {"schema_version": 1, **_manifest_payload(component)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["corpus_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return payload


__all__ = [
    "COMMON_BATCHES",
    "COMMON_CONTEXT_TOKENS",
    "COMMON_KV_DTYPES",
    "COMMON_PAGE_SIZES",
    "GDN_GEOMETRIES",
    "GQA_GEOMETRIES",
    "MLA_GEOMETRIES",
    "QSA_GEOMETRIES",
    "QSA_BATCHES",
    "QSA_CONTEXT_TOKENS",
    "QSA_PAGE_SIZES",
    "QSA_POSITION_LAYOUTS",
    "QSA_SPECULATIVE_TOKENS",
    "SPARSE_MLA_GEOMETRIES",
    "attention_corpus_manifest",
    "gdn_cases",
    "gqa_cases",
    "mla_cases",
    "qsa_cases",
    "sparse_mla_cases",
]
