from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, replace

from b12x.policy import DeviceIdentity
from b12x.policy.generation import (
    CheckpointStore,
    DiscreteSweepGenerator,
    GenerationContext,
    GenerationSettings,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.serialization import profile_from_dict

_DEVICE = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 1),
    sm_count=48,
    product_name="Synthetic GPU",
)


class _Session(AbstractContextManager["_Session"]):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self._candidates = (
            SweepCandidate.create({"backend": "left"}),
            SweepCandidate.create({"backend": "right"}),
        )

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def candidates(self, case):
        del case
        return self._candidates

    def measure(self, case, candidates):
        self._calls.append(case.case_id)
        measurements = []
        for candidate in candidates:
            backend = candidate.config["backend"]
            if case.query["rows"] == 1:
                latency = 10.0 if backend == "left" else 20.0
            else:
                latency = 30.0 if backend == "left" else 15.0
            if case.scenario == "strided":
                latency *= 1.1
            measurements.append(
                SweepMeasurement(
                    candidate=candidate,
                    latency_us=latency,
                    correct=True,
                    metrics={"cosine": 0.9995},
                )
            )
        return tuple(measurements)


@dataclass
class _Factory:
    calls: list[str]

    def __call__(self, group_id, cases, context):
        del group_id, cases, context
        return _Session(self.calls)


def _cases():
    return tuple(
        SweepCase.create(
            group_id="geometry",
            query={"family": "a", "rows": rows},
            scenario=scenario,
            label=f"m{rows}-{scenario}",
        )
        for rows in (1, 4)
        for scenario in ("contiguous", "strided")
    )


def test_discrete_sweep_reduces_scenarios_and_resumes(tmp_path) -> None:
    calls = []
    generator = DiscreteSweepGenerator(
        component_id="test.attention",
        query_schema_version=1,
        config_schema_version=1,
        query_fields=("family", "rows"),
        range_fields=frozenset({"rows"}),
        cases=_cases(),
        benchmark_factory=_Factory(calls),
        coverage={"corpus_sha256": "synthetic"},
    )
    context = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="abc123",
        settings=GenerationSettings(),
    )
    checkpoints = CheckpointStore(tmp_path / "checkpoints")

    result = generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    first_call_count = len(calls)
    resumed = generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )

    assert first_call_count == 4
    assert len(calls) == first_call_count
    assert result.component == resumed.component

    changed_context = replace(
        context,
        settings=GenerationSettings(repetitions=31),
    )
    generator.generate(
        changed_context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    assert len(calls) == 2 * first_call_count
    profile = profile_from_dict(
        {
            "profile_id": "nvidia.synthetic.48sm",
            "targets": [
                {
                    "vendor": "nvidia",
                    "compute_capability": [12, 1],
                    "sm_count": 48,
                    "product_name": "Synthetic GPU",
                }
            ],
            "components": [result.component],
        }
    )
    component = profile.component("test.attention")
    assert component is not None
    assert component.lookup({"family": "a", "rows": 1}).config["backend"] == "left"
    assert component.lookup({"family": "a", "rows": 4}).config["backend"] == "right"
