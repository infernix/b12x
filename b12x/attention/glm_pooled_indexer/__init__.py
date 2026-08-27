"""Pooled sparse-attention index selection for GLM-5.3-Flash.

The selector maintains a BF16 cache containing one learned representative per
complete four-token group.  Pool weights are a feature-wise softmax over the
sum of the per-token compression gate and the learned within-group positional
embedding.  There is no selector RoPE or post-pooling normalization.

Decode scoring applies 32 query heads to every cached representative, takes a
ReLU after the dimension-scaled dot product, and combines the heads with
query-dependent learned weights.  The highest-scoring 512 complete groups are
expanded back to original-token positions and followed by the causally visible
incomplete-group tail.  ``selected_positions`` therefore has fixed width 2051
and can be passed directly to ``attention.sparse_mla``.

The public lifecycle is ``Caps -> plan -> bind -> reset_state ->
run_prefill/run``.  Planning owns all scratch capacity, binding only validates
caller-owned tensors and creates views, and the execution paths are
allocation-free behind mutating dispatcher boundaries.  Persistent raw state
uses exact logical tags plus an accepted-interval anchor, so rejected
speculative rows may remain resident but cannot be consumed.  Page-,
state-slot-, and row-scaled addresses use signed 64-bit arithmetic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="glm_pooled_indexer",
    group="attention",
    api_style="planned",
    entry_points=(
        "CacheRequirements",
        "Caps",
        "Plan",
        "Binding",
        "cache_requirements",
        "plan",
        "bind",
        "reset_state",
        "update_prefill_cache",
        "run_prefill",
        "run",
        "is_supported",
    ),
    dtypes=("bf16",),
    recipes=("glm5_next_pooled_dsa",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="d78f48c2fde6eb29beaafc4a2248999b7540a5ae",
        paths=(
            "b12x/attention/qsa/",
            "b12x/attention/dsa_indexer/",
            "b12x/attention/sparse_mla/",
        ),
    ),
    test_path="tests/attention/test_glm_pooled_indexer.py",
    since="1.3.0",
    notes=(
        "The Triton decode path is a correctness implementation and is not "
        "throughput-qualified. Compressed-cache maintenance, packed "
        "prefill and speculative persistence, stable group selection, "
        "expansion, and the raw causal tail execute on device with "
        "caller-planned storage."
    ),
)

if TYPE_CHECKING:
    from .api import (
        Binding,
        CacheRequirements,
        Caps,
        Plan,
        bind,
        cache_requirements,
        is_supported,
        plan,
        reset_state,
        run,
        run_prefill,
        update_prefill_cache,
    )

install_lazy_api(globals(), META)
