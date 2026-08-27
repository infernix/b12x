"""Public surface for :mod:`b12x.sequence.mtp_feedback`."""

from __future__ import annotations

import torch

from ..._lib.gating import default_is_supported
from . import META

from . import reference
from ._impl import Binding, Caps, Plan, bind, plan, run


def is_supported(device=None) -> bool:
    """True when mandatory CuTe projections and Triton auxiliaries are usable."""
    if not default_is_supported(device, requires=META.requires):
        return False
    target = torch.device(device) if device is not None else torch.device("cuda")
    if target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    return tuple(torch.cuda.get_device_capability(target)) == (12, 0)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "plan",
    "bind",
    "run",
    "reference",
    "is_supported",
]
