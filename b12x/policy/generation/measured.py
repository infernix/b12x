"""Measured generation for components with one production implementation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from b12x.policy.context import ComponentPolicy
from b12x.policy.types import FrozenMapping

from .contracts import (
    ComponentGenerationResult,
    GenerationContext,
    ProgressReporter,
    WorkEstimate,
)
from .store import CheckpointStore

QueryT = TypeVar("QueryT")
ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True, kw_only=True)
class GpuProbeMeasurement:
    """One correctness-gated production-path GPU timing."""

    label: str
    latency_us: float
    correct: bool
    metrics: FrozenMapping = FrozenMapping()

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("GPU probe labels must be non-empty")
        if not math.isfinite(self.latency_us) or self.latency_us <= 0:
            raise ValueError("GPU probe latency must be finite and positive")
        if not isinstance(self.correct, bool):
            raise TypeError("GPU probe correctness must be a boolean")
        if not isinstance(self.metrics, FrozenMapping):
            object.__setattr__(self, "metrics", FrozenMapping(self.metrics))

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "latency_us": self.latency_us,
            "correct": self.correct,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "GpuProbeMeasurement":
        metrics = value.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise TypeError("GPU probe metrics must be an object")
        correct = value.get("correct")
        if not isinstance(correct, bool):
            raise TypeError("GPU probe correct field must be a boolean")
        return cls(
            label=str(value["label"]),
            latency_us=float(value["latency_us"]),
            correct=correct,
            metrics=FrozenMapping(metrics),
        )


class GpuQualificationProbe(Protocol):
    """Run a production-path benchmark suite for one fixed implementation."""

    @property
    def case_count(self) -> int: ...

    @property
    def description(self) -> str: ...

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]: ...


class MeasuredPolicyGenerator(Generic[QueryT, ConfigT]):
    """Emit a fixed config only after real GPU qualification succeeds."""

    def __init__(
        self,
        *,
        policy: ComponentPolicy[QueryT, ConfigT],
        queries: Sequence[QueryT],
        encode_config: Callable[[ConfigT], Mapping[str, object]],
        probe: GpuQualificationProbe,
    ) -> None:
        self.component_id = policy.component_id
        self.query_schema_version = policy.query_schema_version
        self.config_schema_version = policy.config_schema_version
        self._policy = policy
        self._queries = tuple(queries)
        self._encode_config = encode_config
        self._probe = probe
        if not self._queries:
            raise ValueError(f"{self.component_id} requires validation queries")
        if self._probe.case_count <= 0:
            raise ValueError(f"{self.component_id} requires GPU probe cases")

    def estimate(self, context: GenerationContext) -> WorkEstimate:
        del context
        return WorkEstimate(
            component_id=self.component_id,
            work_units=self._probe.case_count + len(self._queries),
            case_count=self._probe.case_count,
            description=self._probe.description,
            dimensions={
                "gpu_measurement_cases": self._probe.case_count,
                "validated_runtime_queries": len(self._queries),
            },
        )

    def reviewed_queries(self) -> tuple[QueryT, ...]:
        return self._queries

    def _config(self, context: GenerationContext) -> tuple[ConfigT, FrozenMapping]:
        configs: dict[FrozenMapping, ConfigT] = {}
        for query in self._queries:
            config = self._policy.heuristic(query, context.device)
            self._policy.validate_config(query, config, context.device)
            encoded = FrozenMapping(self._encode_config(config))
            configs.setdefault(encoded, config)
        if len(configs) != 1:
            raise ValueError(
                f"{self.component_id} has multiple configurations and requires "
                "a candidate-race generator"
            )
        encoded, config = next(iter(configs.items()))
        return config, encoded

    def _measure(
        self,
        context: GenerationContext,
        checkpoints: CheckpointStore,
    ) -> tuple[GpuProbeMeasurement, ...]:
        checkpoint_id = "production-qualification"
        cached = checkpoints.load(self.component_id, checkpoint_id)
        if (
            cached is not None
            and cached.get("generation") == context.checkpoint_metadata()
            and cached.get("case_count") == self._probe.case_count
        ):
            raw = cached.get("measurements")
            if not isinstance(raw, list):
                raise TypeError("GPU qualification checkpoint must contain an array")
            return tuple(GpuProbeMeasurement.from_dict(item) for item in raw)

        measurements = self._probe(context)
        if len(measurements) != self._probe.case_count:
            raise ValueError(
                f"{self.component_id} GPU probe returned {len(measurements)} "
                f"measurements, expected {self._probe.case_count}"
            )
        checkpoints.save(
            self.component_id,
            checkpoint_id,
            {
                "schema_version": 1,
                "generation": context.checkpoint_metadata(),
                "case_count": self._probe.case_count,
                "measurements": [item.to_dict() for item in measurements],
            },
        )
        return measurements

    def generate(
        self,
        context: GenerationContext,
        *,
        progress: ProgressReporter,
        checkpoints: CheckpointStore,
    ) -> ComponentGenerationResult:
        config, encoded_config = self._config(context)
        del config
        progress.start_stage(
            self.component_id,
            stage="production-path GPU qualification",
            total=self._probe.case_count,
        )
        measurements = self._measure(context, checkpoints)
        for measurement in measurements:
            if not measurement.correct:
                raise RuntimeError(
                    f"{self.component_id} failed GPU qualification "
                    f"{measurement.label}"
                )
            progress.advance(self.component_id, detail=measurement.label)

        progress.start_stage(
            self.component_id,
            stage="validate profiled runtime envelope",
            total=len(self._queries),
        )
        for index, query in enumerate(self._queries):
            decoded = self._policy.decode_profile(encoded_config)
            self._policy.validate_config(query, decoded, context.device)
            progress.advance(self.component_id, detail=f"query-{index + 1}")

        latencies = [measurement.latency_us for measurement in measurements]
        estimate = self.estimate(context)
        return ComponentGenerationResult(
            component={
                "component_id": self.component_id,
                "query_schema_version": self.query_schema_version,
                "config_schema_version": self.config_schema_version,
                "planner": {
                    "kind": "leaf",
                    "name": "measured-production-implementation",
                    "config": encoded_config.to_dict(),
                },
            },
            evidence={
                "selection": "single_candidate_gpu_qualification",
                "gpu_measurement_cases": len(measurements),
                "latency_us": {
                    "minimum": min(latencies),
                    "median": statistics.median(latencies),
                    "maximum": max(latencies),
                },
            },
            completed_work_units=estimate.work_units,
        )


__all__ = [
    "GpuProbeMeasurement",
    "GpuQualificationProbe",
    "MeasuredPolicyGenerator",
]
