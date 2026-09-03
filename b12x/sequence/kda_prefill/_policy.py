"""Policy contract for chunked KDA prefill: query, config, heuristic."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import ComponentPolicy
from b12x.policy.components import KDA_PREFILL
from b12x.policy.types import FrozenMapping

BACKEND = "cutedsl"
V_SPLIT_CHOICES = (16, 32, 64, 128)


@dataclass(frozen=True, kw_only=True)
class KdaPrefillQuery:
    """Immutable geometry and planned capacity of one KDA prefill plan."""

    heads: int
    head_dim: int
    model_dtype: str
    state_dtype: str
    qk_l2norm: bool
    checkpoint_export: bool
    max_tokens: int
    max_seqs: int

    def profile_fields(self) -> dict[str, object]:
        return {
            "heads": int(self.heads),
            "head_dim": int(self.head_dim),
            "model_dtype": str(self.model_dtype),
            "state_dtype": str(self.state_dtype),
            "qk_l2norm": bool(self.qk_l2norm),
            "checkpoint_export": bool(self.checkpoint_export),
            "max_tokens": int(self.max_tokens),
            "max_seqs": int(self.max_seqs),
        }


@dataclass(frozen=True)
class KdaPrefillConfig:
    """Backend selection plus the recurrence kernel's value-row split.

    ``v_split`` is the number of value rows one recurrence CTA owns; smaller
    splits launch more CTAs per (sequence, head) at the cost of re-reading the
    prepared chunk tiles from L2.
    """

    backend: str = BACKEND
    v_split: int = 64

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "KdaPrefillConfig":
        keys = set(payload.keys())
        if "backend" not in keys or not keys <= {"backend", "v_split"}:
            raise ValueError(
                "KDA prefill profiles require backend and accept only v_split"
            )
        backend = payload["backend"]
        if not isinstance(backend, str):
            raise TypeError("backend must be a string")
        v_split = payload.get("v_split", 64)
        if isinstance(v_split, bool) or not isinstance(v_split, int):
            raise TypeError("v_split must be an integer")
        return cls(backend=backend, v_split=int(v_split))

    def to_dict(self) -> dict[str, object]:
        return {"backend": self.backend, "v_split": int(self.v_split)}


def _heuristic(query: KdaPrefillQuery, device) -> KdaPrefillConfig:
    del query, device
    return KdaPrefillConfig(backend=BACKEND, v_split=64)


def _validate(query: KdaPrefillQuery, config: KdaPrefillConfig, device) -> None:
    del device
    if config.backend != BACKEND:
        raise ValueError(f"unsupported {KDA_PREFILL} backend {config.backend!r}")
    if config.v_split not in V_SPLIT_CHOICES:
        raise ValueError(
            f"unsupported {KDA_PREFILL} v_split {config.v_split!r}; expected one "
            f"of {V_SPLIT_CHOICES}"
        )
    if query.head_dim != 128:
        raise ValueError(f"{KDA_PREFILL} requires head_dim 128, got {query.head_dim}")
    if query.model_dtype != "bfloat16" or query.state_dtype != "float32":
        raise ValueError(
            f"{KDA_PREFILL} requires bfloat16 activations and float32 state, got "
            f"{query.model_dtype}/{query.state_dtype}"
        )


KDA_PREFILL_POLICY = ComponentPolicy(
    component_id=KDA_PREFILL,
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(
        {
            "heads",
            "head_dim",
            "model_dtype",
            "state_dtype",
            "qk_l2norm",
            "checkpoint_export",
            "max_tokens",
            "max_seqs",
        }
    ),
    config_fields=frozenset({"backend", "v_split"}),
    encode_query=KdaPrefillQuery.profile_fields,
    decode_profile=KdaPrefillConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)

__all__ = [
    "BACKEND",
    "KDA_PREFILL_POLICY",
    "KdaPrefillConfig",
    "KdaPrefillQuery",
    "V_SPLIT_CHOICES",
]
