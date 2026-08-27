"""Public planned API for :mod:`b12x.attention.glm_pooled_indexer`."""

from __future__ import annotations

from ._contract import (
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

__all__ = [
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
]
