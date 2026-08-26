"""Prime-hashed PLE embedding lookup.

``plan`` owns the hash geometry, tensor-parallel table partition, and fixed
serving capacity. ``bind`` maps caller-owned model tensors, packed request
metadata, output, and scratch without allocation. ``run`` hashes tokens,
gathers selected rows from the local table shard, applies inline
dequantization when the table uses FP8 or NVFP4 storage, and writes a BF16
flattened embedding contribution.

The expressed operation is one hash, gather, and dequantization call. Its
binding exposes only caller-owned inputs, the output, and a device error code;
intermediate embedding IDs remain private so implementations may fuse the
operation without changing the integration API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="ple_embedding",
    group="sequence",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "plan",
        "bind",
        "run",
        "is_supported",
    ),
    dtypes=("bf16", "fp8_e4m3", "nvfp4", "int64"),
    recipes=(
        "packed_eos_bounded_bf16",
        "packed_eos_bounded_fp8_per_tensor",
        "packed_eos_bounded_nvfp4_group16",
    ),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="3a437ab51680",
        paths=("b12x/sequence/ple_embedding/",),
    ),
    test_path="tests/sequence/test_ple_embedding.py",
    since="1.3.0",
    notes=(
        "FP8 E4M3 and NVFP4 tables remain quantized in persistent storage; "
        "only selected local rows are dequantized. The expressed API is one "
        "opaque hash, local-shard gather, and inline-dequantization operation. "
        "Its Triton implementation is functional but not throughput-qualified."
    ),
)

if TYPE_CHECKING:
    from .api import Binding, Caps, Plan, bind, is_supported, plan, run  # noqa: F401

install_lazy_api(globals(), META)
