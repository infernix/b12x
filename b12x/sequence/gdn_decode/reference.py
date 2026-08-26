"""Explicit PyTorch oracle for packed sequential GDN decode."""

from __future__ import annotations

import torch


def _scalar(value: torch.Tensor | int) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("device count tensors must contain one element")
        return int(value.item())
    return int(value)


def decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
    num_seqs: torch.Tensor | int,
    num_tokens: torch.Tensor | int,
    *,
    key_heads: int,
    value_heads: int,
    gate_activation: str,
    eps: float = 1e-6,
    scale: float | None = None,
    qk_l2norm: bool = True,
) -> torch.Tensor:
    """Return gated output capacity while updating state checkpoints in place."""
    key_head_dim = 128
    value_head_dim = 128
    ratio = value_heads // key_heads
    scale_value = key_head_dim**-0.5 if scale is None else float(scale)
    live_seqs = _scalar(num_seqs)
    live_tokens = _scalar(num_tokens)
    max_tokens = int(mixed_qkv.shape[0])
    max_seqs, columns = map(int, state_indices.shape)
    if live_seqs < 0 or live_seqs > max_seqs:
        raise ValueError(f"num_seqs={live_seqs} exceeds capacity {max_seqs}")
    if live_tokens < 0 or live_tokens > max_tokens:
        raise ValueError(f"num_tokens={live_tokens} exceeds capacity {max_tokens}")
    if int(query_start_loc[0]) != 0:
        raise ValueError("query_start_loc[0] must be zero")
    if int(query_start_loc[live_seqs]) != live_tokens:
        raise ValueError("query_start_loc[num_seqs] must equal num_tokens")

    request_spans: list[tuple[int, int, int]] = []
    active_cells: set[int] = set()
    for request in range(live_seqs):
        start = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        accepted = int(num_accepted_tokens[request])
        if start < 0 or end < start or end > live_tokens:
            raise ValueError(f"invalid query interval [{start}, {end})")
        if end - start > columns:
            raise ValueError("request token count exceeds state-index columns")
        if accepted < 1 or accepted > columns:
            raise ValueError("num_accepted_tokens must select a state-index column")
        request_spans.append((start, end, accepted))
        if end == start:
            continue
        for column in set(range(end - start)) | {accepted - 1}:
            slot = int(state_indices[request, column])
            if slot < 0 or slot >= recurrent_state.shape[0]:
                raise IndexError(f"state index {slot} is out of range")
            if slot in active_cells:
                raise ValueError(f"duplicate active state index {slot}")
            active_cells.add(slot)

    output = torch.zeros(
        (max_tokens, value_heads, value_head_dim),
        dtype=torch.bfloat16,
        device=mixed_qkv.device,
    )
    q_width = key_heads * key_head_dim
    q = mixed_qkv[:, :q_width].view(-1, key_heads, key_head_dim).float()
    k = mixed_qkv[:, q_width : 2 * q_width].view(-1, key_heads, key_head_dim).float()
    v = mixed_qkv[:, 2 * q_width :].view(-1, value_heads, value_head_dim).float()
    if qk_l2norm:
        q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    q *= scale_value

    for request, (start, end, accepted) in enumerate(request_spans):
        if end == start:
            continue
        source_slot = int(state_indices[request, accepted - 1])
        for value_head in range(value_heads):
            key_head = value_head // ratio
            state = recurrent_state[source_slot, value_head].float()
            for relative_token, token in enumerate(range(start, end)):
                softplus_input = (
                    a[token, value_head].float() + dt_bias[value_head].float()
                )
                softplus = torch.where(
                    softplus_input <= 20.0,
                    torch.log1p(torch.exp(softplus_input)),
                    softplus_input,
                )
                decay = torch.exp(-torch.exp(A_log[value_head].float()) * softplus)
                beta = (
                    torch.sigmoid(b[token, value_head].float())
                    .to(torch.bfloat16)
                    .float()
                )
                state = state * decay
                delta = v[token, value_head] - state.mv(k[token, key_head])
                state = state + (delta * beta).unsqueeze(1) * k[
                    token, key_head
                ].unsqueeze(0)
                output[token, value_head] = state.mv(q[token, key_head]).to(
                    torch.bfloat16
                )
                destination_slot = int(state_indices[request, relative_token])
                recurrent_state[destination_slot, value_head].copy_(
                    state.to(recurrent_state.dtype)
                )

    values = output[:live_tokens].float()
    values *= torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + float(eps))
    normalized = values.to(torch.bfloat16)
    if norm_weight.dtype == torch.bfloat16:
        weighted = (normalized * norm_weight).to(torch.bfloat16)
    elif norm_weight.dtype == torch.float32:
        weighted = normalized.float() * norm_weight
    else:
        raise TypeError("norm_weight must be bfloat16 or float32")
    gate = torch.sigmoid(z[:live_tokens].float())
    if gate_activation == "silu":
        gate *= z[:live_tokens].float()
    elif gate_activation != "sigmoid":
        raise ValueError(f"unsupported gate_activation={gate_activation!r}")
    output[:live_tokens].copy_((weighted * gate).to(torch.bfloat16))
    return output


__all__ = ["decode"]
