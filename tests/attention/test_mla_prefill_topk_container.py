"""Sub-container runtime top-k rows widen to the next MG index container."""

from __future__ import annotations

import pytest

from b12x.attention._shared.mla.prefill import _topk_container


@pytest.mark.parametrize(
    ("topk", "expected"),
    [
        (1, 512),
        (100, 512),
        (128, 128),
        (192, 512),
        (512, 512),
        (513, 1024),
        (1024, 1024),
        (1500, 2048),
        (2048, 2048),
        (2051, 2051),
        (2112, 2112),
        (4096, 4096),
    ],
)
def test_topk_container_rounds_up_to_supported_widths(topk: int, expected: int) -> None:
    """A 192-token prefill clamps top-k to 192; the kernel needs a 512 row."""
    assert _topk_container(topk) == expected
