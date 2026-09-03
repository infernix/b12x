"""Oracle and contract tests for chunked lower-bounded KDA prefill.

The CPU tests pin the reference algebra and the mirror's rounding policy; the
GPU tests (added with the kernels) compare the CuTe DSL op against them.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.sequence._shared.kda_math import kda_beta, kda_log_decay, l2_normalize
from b12x.sequence.kda_prefill.reference import (
    MirrorPolicy,
    prefill_kda,
    prefill_kda_chunk_mirror,
    recurrent_kda,
)

HEAD_DIM = 128
CPU = torch.device("cpu")
PURE_FP32 = MirrorPolicy(shadow=False, inv_operand="fp32", u_operand="fp32", operands="fp32")
FLASHKDA_LIKE = MirrorPolicy(state_master="bf16", single_rounding=False, scale_dtype="bf16")


def rmse_ratio(reference: torch.Tensor, actual: torch.Tensor) -> float:
    delta = (reference.float() - actual.float()).flatten()
    base = reference.float().flatten()
    return (delta.square().mean().sqrt() / (base.square().mean().sqrt() + 1e-8)).item()


def assert_kda_close(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    ratio: float,
    peak_ratio: float = 4e-2,
    exact_atol: float = 1e-6,
) -> None:
    assert torch.isfinite(actual.float()).all(), f"{name}: non-finite values"
    delta = (reference.float() - actual.float()).abs()
    if delta.max().item() <= exact_atol:
        return
    observed = rmse_ratio(reference, actual)
    assert observed < ratio, f"{name}: rmse ratio {observed:.3e} >= {ratio}"
    rms = reference.float().square().mean().sqrt().item()
    peak = reference.float().abs().max().item()
    assert delta.max().item() <= peak_ratio * rms + 2**-6 * peak, f"{name}: peak error"


def make_inputs(
    *,
    lengths: list[int],
    heads: int = 2,
    device: torch.device = CPU,
    seed: int = 0,
    gate_profile: str = "random",
    key_profile: str = "random",
    lower_bound: float = -5.0,
    token_capacity: int | None = None,
    state_slots: int | None = None,
    initial: list[int] | None = None,
    final: list[int] | None = None,
    checkpoint: list[tuple[int, int]] | None = None,
    null_state_index: int | None = None,
) -> dict:
    """Build packed inputs; slot assignment defaults to distinct slots."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = sum(lengths)
    count = len(lengths)
    capacity = tokens if token_capacity is None else token_capacity

    def bf16(*shape, scale=0.25):
        return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16)

    q, k, v = bf16(capacity, heads, HEAD_DIM), bf16(capacity, heads, HEAD_DIM), bf16(capacity, heads, HEAD_DIM)
    raw_g = bf16(capacity, heads, HEAD_DIM, scale=1.0)
    raw_beta = bf16(capacity, heads, scale=1.0)
    if gate_profile == "long_memory":
        raw_g[:, :, :32] = -12.0
    elif gate_profile == "saturated":
        raw_g.fill_(12.0)
    elif gate_profile == "zero":
        raw_g.fill_(-12.0)
    if key_profile in ("repeated", "alternating"):
        unit = torch.randn(heads, HEAD_DIM, generator=generator)
        unit = unit / unit.norm(dim=-1, keepdim=True)
        pattern = unit[None].expand(capacity, heads, HEAD_DIM).clone()
        if key_profile == "alternating":
            pattern[1::2] *= -1.0
        k = pattern.to(torch.bfloat16)
        raw_beta.fill_(12.0)
    A_log = torch.randn(heads, generator=generator) * 0.1
    dt_bias = torch.randn(heads, HEAD_DIM, generator=generator) * 0.1
    slots = 3 * count + 2 if state_slots is None else state_slots
    pool = torch.randn(slots, heads, HEAD_DIM, HEAD_DIM, generator=generator) * 0.1
    initial = list(range(count)) if initial is None else initial
    final = list(range(count, 2 * count)) if final is None else final
    checkpoint = [(0, 0)] * count if checkpoint is None else checkpoint
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    to = lambda t: t.to(device)  # noqa: E731
    return {
        "q": to(q), "k": to(k), "v": to(v), "raw_g": to(raw_g), "raw_beta": to(raw_beta),
        "A_log": to(A_log), "dt_bias": to(dt_bias), "pool": to(pool),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device=device),
        "initial": torch.tensor(initial, dtype=torch.int32, device=device),
        "final": torch.tensor(final, dtype=torch.int32, device=device),
        "checkpoint_slots": torch.tensor([c[1] for c in checkpoint], dtype=torch.int32, device=device),
        "checkpoint_offsets": torch.tensor([c[0] for c in checkpoint], dtype=torch.int32, device=device),
        "num_seqs": count, "num_tokens": tokens, "lower_bound": lower_bound,
        "null_state_index": null_state_index,
    }


def run_oracle(inputs: dict, fn=prefill_kda, **extra):
    pool = inputs["pool"].clone()
    output = fn(
        inputs["q"], inputs["k"], inputs["v"], inputs["raw_g"], inputs["raw_beta"],
        inputs["A_log"], inputs["dt_bias"], pool, inputs["cu_seqlens"], inputs["initial"],
        inputs["final"], inputs["checkpoint_slots"], inputs["checkpoint_offsets"],
        inputs["num_seqs"], inputs["num_tokens"], lower_bound=inputs["lower_bound"],
        null_state_index=inputs["null_state_index"], **extra,
    )
    return output, pool


def test_shared_gate_helper_matches_decode_kda_expression() -> None:
    torch.manual_seed(3)
    raw_g = torch.randn(5, 2, HEAD_DIM).to(torch.bfloat16)
    dt_bias = torch.randn(2, HEAD_DIM) * 0.1
    A_log = torch.randn(2) * 0.1
    helper = kda_log_decay(raw_g, dt_bias, A_log, -5.0)
    for token in range(5):
        for head in range(2):
            rate = torch.exp(A_log[head].float())
            expected = -5.0 * torch.sigmoid(rate * (raw_g[token, head].float() + dt_bias[head].float()))
            torch.testing.assert_close(helper[token, head], expected, rtol=0, atol=0)
    beta = torch.randn(5, 2).to(torch.bfloat16)
    torch.testing.assert_close(kda_beta(beta), torch.sigmoid(beta.float()), rtol=0, atol=0)
    x = torch.randn(5, 2, HEAD_DIM).to(torch.bfloat16).float()
    torch.testing.assert_close(
        l2_normalize(x), x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6), rtol=0, atol=0
    )


def test_recurrent_oracle_matches_decode_kda_token_by_token() -> None:
    from b12x.sequence.gdn_decode.reference import decode_kda

    heads, tokens = 2, 8
    inputs = make_inputs(lengths=[tokens], heads=heads, seed=7, state_slots=tokens + 2)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    mixed = torch.cat([q.reshape(tokens, -1), k.reshape(tokens, -1), v.reshape(tokens, -1)], dim=1)
    pool = inputs["pool"].clone()
    state_indices = torch.arange(0, tokens, dtype=torch.int32)[None]
    z = torch.zeros(tokens, heads, HEAD_DIM, dtype=torch.bfloat16)
    norm_weight = torch.ones(HEAD_DIM)
    decode_out = decode_kda(
        mixed, inputs["raw_g"], inputs["raw_beta"], z, inputs["A_log"], inputs["dt_bias"],
        norm_weight, pool, torch.tensor([0, tokens], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32), state_indices, 1, tokens, heads=heads,
        lower_bound=-5.0,
    )
    del decode_out  # the decode epilogue applies a gated norm; compare states.
    out, final, _ = recurrent_kda(
        q, k, v, inputs["raw_g"], inputs["raw_beta"], inputs["A_log"], inputs["dt_bias"],
        lower_bound=-5.0, initial_state=inputs["pool"][0],
    )
    torch.testing.assert_close(final, pool[tokens - 1], rtol=1e-5, atol=1e-6)
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize(
    "lengths,checkpoint",
    [([0], None), ([0, 33], None), ([40], [(16, 5)]), ([40], [(0, 5)]), ([40], [(32, 5)])],
)
def test_prefill_oracle_contract_cases(lengths, checkpoint) -> None:
    inputs = make_inputs(lengths=lengths, seed=11, checkpoint=checkpoint, state_slots=8)
    tail = torch.full((3, 2, HEAD_DIM), float("nan"), dtype=torch.bfloat16)
    output = torch.cat([torch.zeros(inputs["num_tokens"], 2, HEAD_DIM, dtype=torch.bfloat16), tail])
    padded = dict(inputs)
    for name in ("q", "k", "v", "raw_g"):
        padded[name] = torch.cat([inputs[name], torch.zeros(3, 2, HEAD_DIM, dtype=torch.bfloat16)])
    padded["raw_beta"] = torch.cat([inputs["raw_beta"], torch.zeros(3, 2, dtype=torch.bfloat16)])
    out, pool = run_oracle(padded, output=output)
    assert torch.isnan(out[inputs["num_tokens"] :].float()).all()
    for request, length in enumerate(lengths):
        initial = int(inputs["initial"][request])
        final = int(inputs["final"][request])
        if length == 0:
            torch.testing.assert_close(pool[final], inputs["pool"][initial], rtol=0, atol=0)
    if checkpoint is not None:
        offset, slot = checkpoint[0]
        start = 0
        _, _, expected = recurrent_kda(
            inputs["q"][start : start + lengths[0]], inputs["k"][start : start + lengths[0]],
            inputs["v"][start : start + lengths[0]], inputs["raw_g"][start : start + lengths[0]],
            inputs["raw_beta"][start : start + lengths[0]], inputs["A_log"], inputs["dt_bias"],
            lower_bound=-5.0, initial_state=inputs["pool"][0], checkpoint_offset=offset,
        )
        if offset > 0:
            torch.testing.assert_close(pool[slot], expected, rtol=1e-6, atol=1e-6)
        else:
            torch.testing.assert_close(pool[slot], inputs["pool"][slot], rtol=0, atol=0)


def test_prefill_oracle_null_initial_never_reads_the_slot() -> None:
    inputs = make_inputs(lengths=[20], seed=5, initial=[0], final=[1], null_state_index=0)
    inputs["pool"][0].fill_(float("nan"))
    out, pool = run_oracle(inputs)
    assert torch.isfinite(out.float()).all()
    zero_start = make_inputs(lengths=[20], seed=5, initial=[2], final=[1])
    zero_start["pool"][2].zero_()
    expected_out, expected_pool = run_oracle(zero_start)
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(pool[1], expected_pool[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda i: i["cu_seqlens"].__setitem__(2, 30), "must equal num_tokens"),
        (lambda i: i.update(num_tokens=1_000_000), "exceeds capacity"),
        (lambda i: i["checkpoint_offsets"].__setitem__(0, 17), "not a multiple"),
        (lambda i: i["checkpoint_offsets"].__setitem__(0, 64), "exceeds the sequence"),
        (lambda i: i["final"].__setitem__(0, int(i["final"][1])), "duplicate write"),
        (lambda i: i["final"].__setitem__(1, int(i["initial"][0])), "written by another"),
    ],
)
def test_prefill_oracle_rejects_bad_metadata(mutate, match) -> None:
    inputs = make_inputs(lengths=[20, 20], seed=9, checkpoint=[(16, 6), (0, 0)])
    mutate(inputs)
    with pytest.raises((ValueError, IndexError), match=match):
        run_oracle(inputs)


@pytest.mark.parametrize(
    "lengths",
    [[1], [15], [16], [17], [64], [0, 64, 0, 15], [15, 100, 200, 900]],
    ids=lambda v: "-".join(map(str, v)),
)
@pytest.mark.parametrize("lower_bound", [-5.0, -3.0, -0.5])
@pytest.mark.parametrize("gate_profile", ["random", "long_memory", "saturated"])
def test_chunk_mirror_matches_recurrent_oracle(lengths, lower_bound, gate_profile) -> None:
    checkpoint = [(0, 0)] * len(lengths)
    if lengths[-1] >= 32:
        checkpoint[-1] = (32, 3 * len(lengths))
    inputs = make_inputs(
        lengths=lengths, seed=21, lower_bound=lower_bound, gate_profile=gate_profile,
        checkpoint=checkpoint,
    )
    expected_out, expected_pool = run_oracle(inputs)
    default_out, default_pool = run_oracle(inputs, prefill_kda_chunk_mirror)
    pure_out, pure_pool = run_oracle(inputs, prefill_kda_chunk_mirror, policy=PURE_FP32)
    tokens = inputs["num_tokens"]
    writes = [int(s) for s in inputs["final"]] + [slot for offset, slot in checkpoint if offset > 0]
    if tokens:
        assert_kda_close("out", expected_out[:tokens], default_out[:tokens], ratio=1e-2)
        assert_kda_close("out-fp32", expected_out[:tokens], pure_out[:tokens], ratio=2e-4)
    for slot in writes:
        assert_kda_close(f"state[{slot}]", expected_pool[slot], default_pool[slot], ratio=5e-3)
        assert_kda_close(f"state-fp32[{slot}]", expected_pool[slot], pure_pool[slot], ratio=2e-5)
    untouched = [s for s in range(inputs["pool"].shape[0]) if s not in writes]
    torch.testing.assert_close(default_pool[untouched], inputs["pool"][untouched], rtol=0, atol=0)


def test_mirror_policy_study_fp32_master_beats_bf16_state_on_long_memory() -> None:
    inputs = make_inputs(lengths=[16384], heads=1, seed=31, gate_profile="long_memory")
    _, expected_pool = run_oracle(inputs)
    slot = int(inputs["final"][0])
    _, fp32_pool = run_oracle(inputs, prefill_kda_chunk_mirror)
    _, bf16_pool = run_oracle(inputs, prefill_kda_chunk_mirror, policy=FLASHKDA_LIKE)
    err_fp32 = rmse_ratio(expected_pool[slot], fp32_pool[slot])
    err_bf16 = rmse_ratio(expected_pool[slot], bf16_pool[slot])
    assert err_fp32 <= 5e-3, err_fp32
    assert err_bf16 >= 1.5 * err_fp32, (err_fp32, err_bf16)


@pytest.mark.parametrize("key_profile", ["random", "repeated", "alternating"])
@pytest.mark.parametrize("gate_profile", ["random", "zero", "saturated"])
def test_mirror_inverse_growth_bound_on_adversarial_keys(key_profile, gate_profile) -> None:
    worst = 0.0
    for seed in range(8):
        inputs = make_inputs(
            lengths=[64], heads=2, seed=100 + seed, key_profile=key_profile, gate_profile=gate_profile
        )
        _, trace = prefill_kda_chunk_mirror(
            inputs["q"], inputs["k"], inputs["v"], inputs["raw_g"], inputs["raw_beta"],
            inputs["A_log"], inputs["dt_bias"], inputs["pool"].clone(), inputs["cu_seqlens"],
            inputs["initial"], inputs["final"], inputs["checkpoint_slots"],
            inputs["checkpoint_offsets"], inputs["num_seqs"], inputs["num_tokens"], trace=True,
        )
        for tile in trace.k1.values():
            assert torch.isfinite(tile["inv"]).all()
            worst = max(worst, tile["inv"].abs().max().item())
    assert worst <= 8.0, worst


def test_run_rejects_lower_bound_outside_range() -> None:
    inputs = make_inputs(lengths=[16], seed=1)
    for bad in (-5.5, 0.0, math.nan):
        inputs["lower_bound"] = bad
        with pytest.raises(ValueError, match="lower_bound"):
            run_oracle(inputs)
        with pytest.raises(ValueError, match="lower_bound"):
            run_oracle(inputs, prefill_kda_chunk_mirror)


# ---------------------------------------------------------------------------
# GPU: prologue and prepare kernels against the chunk mirror trace.
# ---------------------------------------------------------------------------


def make_binding(inputs: dict, *, max_tokens: int, max_seqs: int, metadata_validation: str = "transactional", **caps_extra):
    """Bind ``inputs`` (from make_inputs on a CUDA device) at planned capacity."""
    from b12x.policy import PolicyContext, PolicyMode
    from b12x.sequence.kda_prefill import _impl as impl

    device = inputs["q"].device
    heads = int(inputs["q"].shape[1])
    caps = impl.Caps(
        device=device, max_tokens=max_tokens, max_seqs=max_seqs,
        max_state_slots=int(inputs["pool"].shape[0]), heads=heads,
        null_state_index=inputs["null_state_index"], metadata_validation=metadata_validation,
        **caps_extra,
    )
    plan = impl.plan(caps, policy=PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY))
    scratch = torch.empty(plan.scratch_specs()[0].shape, dtype=torch.uint8, device=device)

    def pad_rows(t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((max_tokens,) + tuple(t.shape[1:]), dtype=t.dtype, device=device)
        out[: t.shape[0]] = t
        return out

    def pad_seqs(t: torch.Tensor, extra: int = 0) -> torch.Tensor:
        out = torch.zeros(max_seqs + extra, dtype=t.dtype, device=device)
        out[: t.shape[0]] = t
        return out

    tensors = {
        "q": pad_rows(inputs["q"]), "k": pad_rows(inputs["k"]), "v": pad_rows(inputs["v"]),
        "raw_g": pad_rows(inputs["raw_g"]), "raw_beta": pad_rows(inputs["raw_beta"]),
        "A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"],
        "recurrent_state": inputs["pool"].clone(),
        "cu_seqlens": pad_seqs(inputs["cu_seqlens"], extra=1),
        "initial_state_indices": pad_seqs(inputs["initial"]),
        "final_state_indices": pad_seqs(inputs["final"]),
        "checkpoint_state_indices": pad_seqs(inputs["checkpoint_slots"]),
        "checkpoint_offsets": pad_seqs(inputs["checkpoint_offsets"]),
        "num_seqs": torch.tensor([inputs["num_seqs"]], dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([inputs["num_tokens"]], dtype=torch.int32, device=device),
        "output": torch.zeros(max_tokens, heads, HEAD_DIM, dtype=torch.bfloat16, device=device),
    }
    return impl.bind(plan, scratch=scratch, **tensors), tensors


def _mirror_trace(inputs: dict):
    _, trace = prefill_kda_chunk_mirror(
        inputs["q"], inputs["k"], inputs["v"], inputs["raw_g"], inputs["raw_beta"],
        inputs["A_log"], inputs["dt_bias"], inputs["pool"].clone(), inputs["cu_seqlens"],
        inputs["initial"], inputs["final"], inputs["checkpoint_slots"], inputs["checkpoint_offsets"],
        inputs["num_seqs"], inputs["num_tokens"], lower_bound=inputs["lower_bound"],
        null_state_index=inputs["null_state_index"], trace=True,
    )
    return trace


@pytest.mark.parametrize(
    "lengths",
    [[1], [16], [17], [15, 100, 0, 300, 33], [64, 64]],
    ids=lambda v: "-".join(map(str, v)),
)
@pytest.mark.parametrize("lower_bound", [-5.0, -0.5])
def test_prepare_kernel_matches_chunk_mirror(lengths, lower_bound) -> None:
    from ..conftest import require_b12x
    from b12x.sequence.kda_prefill._cute_kernels import run_prepare, run_prologue, workspace_tiles

    device = require_b12x()
    checkpoint = [(0, 0)] * len(lengths)
    if lengths[-1] >= 32:
        checkpoint[-1] = (32, 3 * len(lengths))
    inputs = make_inputs(
        lengths=lengths, heads=2, seed=41, device=device, lower_bound=lower_bound,
        checkpoint=checkpoint, gate_profile="saturated" if lower_bound == -5.0 else "random",
    )
    binding, _ = make_binding(inputs, max_tokens=512, max_seqs=8)
    run_prologue(binding)
    run_prepare(binding, lower_bound=lower_bound, scale=HEAD_DIM**-0.5, eps=1e-6)
    torch.cuda.synchronize(device)
    assert binding.error_code.item() == 0
    trace = _mirror_trace(inputs)
    base = binding.seq_tile_base[: len(lengths) + 1].tolist()
    expected_base = [0]
    for length in lengths:
        expected_base.append(expected_base[-1] + (length + 15) // 16)
    assert base == expected_base
    tile_seq = binding.tile_seq.tolist()
    for seq in range(len(lengths)):
        for tile in range(expected_base[seq], expected_base[seq + 1]):
            assert tile_seq[tile] == seq
    assert all(t == -1 for t in tile_seq[expected_base[-1] :])
    tiles = workspace_tiles(binding)
    for (seq, local), record in trace.k1.items():
        tile = expected_base[seq] + local
        for name, key in (("q_tilde", "q_tilde"), ("k_tilde", "k_tilde"), ("k_r", "k_r")):
            got = tiles[name][tile].float()
            expected = record[key].float()
            assert torch.isfinite(got).all(), name
            # One bf16 ulp of slack covers the kernel's approximate exp2 and rsqrt.
            torch.testing.assert_close(got, expected, rtol=2**-7, atol=1e-6, msg=f"{name} {seq} {local}")
        for name, key, tol in (("inv", "inv_op", 2**-7), ("mqk", "mqk", 2**-9)):
            got = tiles[name][tile].float()
            expected = record[key].float()
            scale = max(1.0, expected.abs().max().item())
            assert torch.isfinite(got).all(), name
            assert (got - expected).abs().max().item() <= tol * scale, (name, seq, local)
        torch.testing.assert_close(tiles["lambda_c"][tile], record["lambda_c"], rtol=1e-4, atol=0)
        torch.testing.assert_close(tiles["beta"][tile], record["beta"], rtol=1e-5, atol=1e-6)
        rows = min(16, lengths[seq] - 16 * local)
        assert torch.count_nonzero(tiles["k_tilde"][tile, :, rows:]) == 0
        assert torch.count_nonzero(tiles["k_r"][tile, :, rows:]) == 0


@pytest.mark.parametrize(
    "mutate,bit",
    [
        (lambda t: t["final_state_indices"].__setitem__(1, int(t["final_state_indices"][0])), 1),
        (lambda t: t["final_state_indices"].__setitem__(1, int(t["initial_state_indices"][0])), 1),
        (lambda t: t["cu_seqlens"].__setitem__(2, 30), 2),
        (lambda t: t["cu_seqlens"].__setitem__(1, 45), 2),
        (lambda t: t["num_tokens"].fill_(10_000), 2),
        (lambda t: t["num_seqs"].fill_(9), 2),
        (lambda t: t["final_state_indices"].__setitem__(0, 99), 4),
        (lambda t: t["initial_state_indices"].__setitem__(0, -1), 4),
        (lambda t: t["checkpoint_offsets"].__setitem__(0, 17), 8),
        (lambda t: t["checkpoint_offsets"].__setitem__(0, 64), 8),
    ],
    ids=[
        "dup-final", "final-is-other-initial", "cu-end-mismatch", "cu-nonmonotonic",
        "num-tokens-over", "num-seqs-over", "final-out-of-range", "initial-negative",
        "ckpt-unaligned", "ckpt-past-length",
    ],
)
def test_prologue_reports_malformed_metadata(mutate, bit) -> None:
    from ..conftest import require_b12x
    from b12x.sequence.kda_prefill._cute_kernels import run_prepare, run_prologue

    device = require_b12x()
    inputs = make_inputs(lengths=[20, 20], heads=2, seed=43, device=device, checkpoint=[(16, 6), (0, 0)])
    binding, tensors = make_binding(inputs, max_tokens=64, max_seqs=8)
    mutate(tensors)
    binding.ws_inv.fill_(float("nan"))
    run_prologue(binding)
    run_prepare(binding, lower_bound=-5.0, scale=HEAD_DIM**-0.5, eps=1e-6)
    torch.cuda.synchronize(device)
    assert binding.error_code.item() & bit
    assert torch.isnan(binding.ws_inv.float()).all(), "prepare must not run after a metadata error"
    trusted, trusted_tensors = make_binding(inputs, max_tokens=64, max_seqs=8, metadata_validation="trusted")
    mutate(trusted_tensors)
    trusted.error_code.fill_(0)
    run_prologue(trusted)
    torch.cuda.synchronize(device)
    assert trusted.error_code.item() == 0
