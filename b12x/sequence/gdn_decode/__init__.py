"""Stateful packed Gated Delta Network decode.

The op consumes already-projected and convolved packed Q/K/V plus the A, B,
and Z projections. It updates a caller-owned recurrent-state pool in place,
then applies per-value-head RMSNorm and either a SiLU or sigmoid output gate.
Projection GEMMs and causal-convolution state are intentionally outside this
package.

The recurrent-state pool uses the optimized physical layout
``[slot, value_head, value_dim, key_dim]``. This is the transpose of the
``[batch, head, key_dim, value_dim]`` state used by slow mathematical PyTorch
references; importing such a state requires transposing its final two axes.
The three inner dimensions must be contiguous. The outer slot stride may be
larger than one logical state to accommodate an aligned paged cache; binding
preserves that stride and never compacts or copies the caller-owned pool.
Pool-scaled slot offsets are computed with 64-bit arithmetic.

Packed requests use fixed-capacity device metadata. Request ``r`` consumes
``query_start_loc[r]:query_start_loc[r + 1]`` and reads its initial checkpoint
from state-index column ``num_accepted_tokens[r] - 1``. Tokens execute
sequentially per request and persist their post-token checkpoints to columns
starting at zero. A one-column plan with one token per request is ordinary
decode.

Planned lifecycle: ``plan(Caps(...))`` -> ``bind`` -> ``run``. Runtime launches
use caller-owned scratch, allocate no tensor storage, and are opaque to
``torch.compile``. Device-side validation is transactional: bit 0 reports a
duplicate active state slot, bit 1 malformed packed metadata, and bit 2 an
invalid active state slot. Any error poisons the complete output without
mutating recurrent state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="gdn_decode",
    group="sequence",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "plan",
        "bind",
        "run",
        "reference",
        "is_supported",
    ),
    dtypes=("bf16", "fp32", "int32", "int64"),
    recipes=("silu", "sigmoid"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="e8d02602f",
        paths=(
            "serve/kernels/fla/fused_recurrent.py",
            "serve/kernels/fla/fused_norm_gate.py",
        ),
    ),
    test_path="tests/sequence/test_gdn_decode.py",
    since="1.3.0",
    notes=(
        "Qualified for 128-wide K/V heads, value/key ratios 1,2,3,4,8, "
        "BF16 model tensors, BF16 or FP32 norm weights, and BF16 or FP32 "
        "recurrent state. The Triton "
        "implementation is a correctness reference and is not "
        "throughput-qualified."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        bind,
        is_supported,
        plan,
        reference,
        run,
    )

install_lazy_api(globals(), META)
