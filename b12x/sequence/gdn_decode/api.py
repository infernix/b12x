"""Public surface for :mod:`b12x.sequence.gdn_decode`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from . import META

from . import reference
from ._impl import (
    Binding,
    Caps,
    KdaBinding,
    Plan,
    bind,
    bind_kda,
    plan,
    run,
    run_kda,
)


def is_supported(device=None) -> bool:
    """True when the registered b12x architecture can run this Triton op."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "KdaBinding",
    "plan",
    "bind",
    "bind_kda",
    "run",
    "run_kda",
    "reference",
    "is_supported",
]
