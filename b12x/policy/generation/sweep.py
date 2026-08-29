"""Reusable discrete-sweep generator for component-owned GPU races."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ContextManager, Protocol

from b12x.policy.types import FrozenMapping

from .contracts import (
    ComponentGenerationResult,
    GenerationContext,
    ProgressReporter,
    WorkEstimate,
)
from .reducer import DecisionRecord, build_axis_tree, decision_node_to_dict
from .store import CheckpointStore


def _stable_id(value: Mapping[str, object], *, length: int = 16) -> str:
    payload = value.to_dict() if isinstance(value, FrozenMapping) else dict(value)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True, kw_only=True)
class SweepCase:
    """One measured scenario for one runtime query point."""

    case_id: str
    group_id: str
    query: FrozenMapping
    scenario: str = "default"
    metadata: FrozenMapping = FrozenMapping()

    @classmethod
    def create(
        cls,
        *,
        group_id: str,
        query: Mapping[str, object],
        scenario: str = "default",
        metadata: Mapping[str, object] | None = None,
        label: str | None = None,
    ) -> "SweepCase":
        identity = {
            "group_id": group_id,
            "query": dict(query),
            "scenario": scenario,
            "metadata": dict(metadata or {}),
        }
        prefix = label or group_id
        case_id = f"{prefix}-{_stable_id(identity, length=12)}"
        return cls(
            case_id=case_id,
            group_id=group_id,
            query=FrozenMapping(query),
            scenario=scenario,
            metadata=FrozenMapping(metadata),
        )

    def __post_init__(self) -> None:
        if not self.case_id or not self.group_id or not self.scenario:
            raise ValueError("sweep case identifiers must be non-empty")
        if not self.query:
            raise ValueError("sweep cases require a non-empty runtime query")


@dataclass(frozen=True, kw_only=True)
class SweepCandidate:
    """One component config eligible for a measured case."""

    config: FrozenMapping
    candidate_id: str

    @classmethod
    def create(cls, config: Mapping[str, object]) -> "SweepCandidate":
        frozen = FrozenMapping(config)
        return cls(config=frozen, candidate_id=_stable_id(frozen))

    def __post_init__(self) -> None:
        if not self.config:
            raise ValueError("sweep candidates require a non-empty config")
        if self.candidate_id != _stable_id(self.config):
            raise ValueError("sweep candidate ID does not match its config")


@dataclass(frozen=True, kw_only=True)
class SweepMeasurement:
    """Correctness and timing result for one candidate."""

    candidate: SweepCandidate
    latency_us: float | None
    correct: bool
    metrics: FrozenMapping = FrozenMapping()
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, FrozenMapping):
            if not isinstance(self.metrics, Mapping):
                raise TypeError("metrics must be an object")
            object.__setattr__(self, "metrics", FrozenMapping(self.metrics))
        if self.latency_us is not None and (
            not math.isfinite(self.latency_us) or self.latency_us <= 0
        ):
            raise ValueError("latency_us must be finite and positive")
        if not isinstance(self.correct, bool):
            raise TypeError("correct must be a boolean")
        if self.error is not None and not self.error:
            raise ValueError("measurement errors must be non-empty")

    def passes(self) -> bool:
        return self.error is None and self.latency_us is not None and self.correct

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "config": self.candidate.config.to_dict(),
            "latency_us": self.latency_us,
            "correct": self.correct,
            "metrics": self.metrics.to_dict(),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SweepMeasurement":
        raw_config = value.get("config")
        if not isinstance(raw_config, Mapping):
            raise TypeError("measurement config must be an object")
        candidate = SweepCandidate.create(raw_config)
        if value.get("candidate_id") != candidate.candidate_id:
            raise ValueError("checkpoint candidate ID does not match its config")
        latency = value.get("latency_us")
        correct = value.get("correct")
        if not isinstance(correct, bool):
            raise TypeError("measurement correct field must be a boolean")
        raw_metrics = value.get("metrics", {})
        if not isinstance(raw_metrics, Mapping):
            raise TypeError("measurement metrics must be an object")
        error = value.get("error")
        return cls(
            candidate=candidate,
            latency_us=None if latency is None else float(latency),
            correct=correct,
            metrics=FrozenMapping(raw_metrics),
            error=None if error is None else str(error),
        )


class SweepSession(Protocol):
    """Stable-allocation measurement session for one case group."""

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]: ...

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]: ...


class SweepBenchmarkFactory(Protocol):
    """Create and fully release one geometry-scoped measurement session."""

    def __call__(
        self,
        group_id: str,
        cases: tuple[SweepCase, ...],
        context: GenerationContext,
    ) -> ContextManager[SweepSession]: ...


def _query_key(case: SweepCase, fields: tuple[str, ...]) -> tuple[object, ...]:
    return tuple(case.query[field] for field in fields)


class DiscreteSweepGenerator:
    """Generate one planner from correctness-gated component measurements."""

    def __init__(
        self,
        *,
        component_id: str,
        query_schema_version: int,
        config_schema_version: int,
        query_fields: tuple[str, ...],
        range_fields: frozenset[str],
        cases: Sequence[SweepCase],
        benchmark_factory: SweepBenchmarkFactory,
        coverage: Mapping[str, object],
    ) -> None:
        self.component_id = component_id
        self.query_schema_version = int(query_schema_version)
        self.config_schema_version = int(config_schema_version)
        self._query_fields = tuple(query_fields)
        self._range_fields = frozenset(range_fields)
        self._cases = tuple(cases)
        self._benchmark_factory = benchmark_factory
        self._coverage = FrozenMapping(coverage)
        if not self._cases:
            raise ValueError(f"{component_id} requires at least one sweep case")
        if not self._query_fields or len(self._query_fields) != len(
            set(self._query_fields)
        ):
            raise ValueError("query_fields must be non-empty and unique")
        if not self._range_fields <= frozenset(self._query_fields):
            raise ValueError("range_fields must be present in query_fields")
        expected = frozenset(self._query_fields)
        if any(frozenset(case.query) != expected for case in self._cases):
            raise ValueError("sweep case query fields differ from the component schema")
        if len({case.case_id for case in self._cases}) != len(self._cases):
            raise ValueError("sweep case IDs must be unique")

    def estimate(self, context: GenerationContext) -> WorkEstimate:
        del context
        query_count = len(
            {_query_key(case, self._query_fields) for case in self._cases}
        )
        group_count = len({case.group_id for case in self._cases})
        return WorkEstimate(
            component_id=self.component_id,
            work_units=len(self._cases) + query_count,
            case_count=len(self._cases),
            description=(
                f"{group_count} allocation groups; correctness-gated GPU race "
                "and decision-tree reduction"
            ),
            dimensions={
                "allocation_groups": group_count,
                "measurement_cases": len(self._cases),
                "runtime_queries": query_count,
            },
        )

    def _measure_case(
        self,
        *,
        case: SweepCase,
        session: SweepSession,
        context: GenerationContext,
        checkpoints: CheckpointStore,
    ) -> tuple[SweepMeasurement, ...]:
        candidates = session.candidates(case)
        if not candidates:
            raise RuntimeError(f"no candidates were produced for {case.case_id}")
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"candidate IDs are not unique for {case.case_id}")
        cached = checkpoints.load(self.component_id, case.case_id)
        if (
            cached is not None
            and cached.get("generation") == context.checkpoint_metadata()
            and cached.get("case_id") == case.case_id
            and cached.get("candidate_ids") == candidate_ids
        ):
            raw = cached.get("measurements")
            if not isinstance(raw, list):
                raise TypeError("sweep checkpoint measurements must be an array")
            return tuple(SweepMeasurement.from_dict(item) for item in raw)

        measurements = session.measure(case, candidates)
        measured_ids = [item.candidate.candidate_id for item in measurements]
        if measured_ids != candidate_ids:
            raise ValueError(
                "measurement sessions must preserve the requested candidate order"
            )
        checkpoints.save(
            self.component_id,
            case.case_id,
            {
                "schema_version": 1,
                "generation": context.checkpoint_metadata(),
                "case_id": case.case_id,
                "group_id": case.group_id,
                "query": case.query.to_dict(),
                "scenario": case.scenario,
                "metadata": case.metadata.to_dict(),
                "candidate_ids": candidate_ids,
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
        cases_by_group: dict[str, list[SweepCase]] = defaultdict(list)
        for case in self._cases:
            cases_by_group[case.group_id].append(case)
        measured: list[tuple[SweepCase, tuple[SweepMeasurement, ...]]] = []
        qualification_cases = 0
        race_cases = 0
        progress.start_stage(
            self.component_id,
            stage="correctness and candidate races",
            total=len(self._cases),
        )
        for group_id in sorted(cases_by_group):
            group_cases = tuple(cases_by_group[group_id])
            progress.advance(
                self.component_id,
                units=0,
                detail=f"prepare {group_id}",
            )
            with self._benchmark_factory(group_id, group_cases, context) as session:
                for case in group_cases:
                    progress.advance(
                        self.component_id,
                        units=0,
                        detail=f"race {case.case_id}",
                    )
                    measurements = self._measure_case(
                        case=case,
                        session=session,
                        context=context,
                        checkpoints=checkpoints,
                    )
                    if not any(item.passes() for item in measurements):
                        raise RuntimeError(
                            f"all candidates failed correctness for {case.case_id}"
                        )
                    measured.append((case, measurements))
                    if len(measurements) == 1:
                        qualification_cases += 1
                    else:
                        race_cases += 1
                    progress.advance(
                        self.component_id,
                        detail=f"race {case.case_id}",
                    )

        grouped_results: dict[
            tuple[object, ...],
            list[tuple[SweepCase, tuple[SweepMeasurement, ...]]],
        ] = defaultdict(list)
        for case, measurements in measured:
            grouped_results[_query_key(case, self._query_fields)].append(
                (case, measurements)
            )
        progress.start_stage(
            self.component_id,
            stage="scenario-robust reduction",
            total=len(grouped_results),
        )
        records: list[DecisionRecord] = []
        winner_counts: dict[str, int] = defaultdict(int)
        for grouped in grouped_results.values():
            by_candidate: dict[str, list[SweepMeasurement]] = defaultdict(list)
            for _case, measurements in grouped:
                for measurement in measurements:
                    if measurement.passes():
                        by_candidate[measurement.candidate.candidate_id].append(
                            measurement
                        )
            required_measurements = len(grouped)
            robust: list[tuple[float, SweepCandidate]] = []
            for candidate_measurements in by_candidate.values():
                if len(candidate_measurements) != required_measurements:
                    continue
                score = math.exp(
                    sum(
                        math.log(float(item.latency_us))
                        for item in candidate_measurements
                    )
                    / len(candidate_measurements)
                )
                robust.append((score, candidate_measurements[0].candidate))
            if not robust:
                raise RuntimeError(
                    "no candidate passed every scenario for query "
                    f"{grouped[0][0].query.to_dict()}"
                )
            _, winner = min(
                robust,
                key=lambda item: (item[0], item[1].candidate_id),
            )
            records.append(
                DecisionRecord(
                    query=grouped[0][0].query,
                    config=winner.config,
                )
            )
            winner_counts[winner.candidate_id] += 1
            progress.advance(
                self.component_id,
                detail=f"reduce {grouped[0][0].case_id}",
            )

        planner = build_axis_tree(
            records,
            field_order=self._query_fields,
            range_fields=self._range_fields,
        )
        coverage = self._coverage.to_dict()
        coverage.update(
            {
                "allocation_groups": len(cases_by_group),
                "measurement_cases": len(self._cases),
                "runtime_query_points": len(records),
            }
        )
        estimate = self.estimate(context)
        return ComponentGenerationResult(
            component={
                "component_id": self.component_id,
                "query_schema_version": self.query_schema_version,
                "config_schema_version": self.config_schema_version,
                "coverage": coverage,
                "planner": decision_node_to_dict(planner),
            },
            evidence={
                "winner_query_counts": dict(sorted(winner_counts.items())),
                "gpu_measurement_cases": len(measured),
                "profile_cases": len(measured),
                "candidate_race_cases": race_cases,
                "single_candidate_qualification_cases": qualification_cases,
            },
            completed_work_units=estimate.work_units,
        )


__all__ = [
    "DiscreteSweepGenerator",
    "SweepBenchmarkFactory",
    "SweepCandidate",
    "SweepCase",
    "SweepMeasurement",
    "SweepSession",
]
