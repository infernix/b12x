from __future__ import annotations

import pytest

from b12x.tools.inspect_model_policy import (
    _canonical_model,
    _device_selection,
    _qwen_flash_next_queries,
    inspect_model_policy,
    main,
)


def test_qwen_flash_next_preset_slices_rank_local_tp_geometry() -> None:
    selections = {
        item.scenario: item
        for item in _qwen_flash_next_queries(4, runtime_device="cuda:0")
    }

    qsa = selections["qsa-spec4"].query
    gdn = selections["gdn-spec4"].query
    moe = selections["moe-m4"].query
    assert (qsa.q_heads, qsa.kv_heads) == (6, 1)
    assert (gdn.key_heads, gdn.value_heads) == (4, 12)
    assert moe.intermediate_size == 160
    assert moe.routed_rows == 40


def test_qwen_flash_next_preset_rejects_unprofiled_qsa_tp() -> None:
    with pytest.raises(ValueError, match="TP 1, 2, or 4"):
        _qwen_flash_next_queries(8, runtime_device="cuda:0")


def test_model_aliases_are_canonicalized() -> None:
    assert _canonical_model("qwen38-flash-next") == "qwen3.8-flash-next-180b"
    assert _canonical_model("qwen38-27b") == "qwen3.8-27b"


def test_embedded_device_can_be_selected_by_product_fragment() -> None:
    selected = _device_selection("gb10")

    assert selected.identity.product_name == "nvidia gb10"
    assert selected.identity.sm_count == 48


def test_cli_lists_model_presets(capsys) -> None:
    assert main(["--list-models"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "qwen3.8-27b",
        "qwen3.8-flash-next-180b",
    ]


def test_qwen_flash_next_tp1_inspection_is_fully_preplanned_on_gb10() -> None:
    payload = inspect_model_policy(
        "qwen3.8-flash-next-180b",
        tp_size=1,
        device="gb10",
    )

    assert payload["profile_id"] == "nvidia.gb10.48sm"
    assert {selection["source"] for selection in payload["selections"]} == {
        "preplanned"
    }
