from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from b12x.policy import DeviceIdentity
from b12x.policy.generation import (
    CheckpointStore,
    ComponentGenerationResult,
    ComponentGeneratorRegistry,
    GenerationContext,
    GenerationSettings,
    ProgressReporter,
    WorkEstimate,
)
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.generation.runner import (
    generate_profile_artifact,
    write_artifact_atomic,
)
from b12x.tools.generate_gpu_profile import _is_generated_profile_data, _parser

_DEVICE = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 1),
    sm_count=48,
    product_name="Synthetic GPU",
)


def test_default_measurement_protocol_is_two_warmups_and_five_by_five() -> None:
    settings = GenerationSettings()
    args = _parser().parse_args([])

    assert (settings.warmup, settings.groups, settings.repetitions) == (2, 5, 5)
    assert (args.warmup, args.groups, args.repetitions) == (2, 5, 5)


@dataclass(frozen=True)
class _Generator:
    component_id: str
    query_schema_version: int = 1
    config_schema_version: int = 1

    def estimate(self, context: GenerationContext) -> WorkEstimate:
        del context
        return WorkEstimate(
            component_id=self.component_id,
            work_units=1,
            case_count=1,
            description="synthetic",
            dimensions={"cases": 1},
        )

    def generate(
        self,
        context: GenerationContext,
        *,
        progress: ProgressReporter,
        checkpoints: CheckpointStore,
    ) -> ComponentGenerationResult:
        del context
        progress.start_stage(self.component_id, stage="race", total=1)
        progress.advance(self.component_id, detail="synthetic-case")
        checkpoints.save(self.component_id, "synthetic-case", {"done": True})
        return ComponentGenerationResult(
            component={
                "component_id": self.component_id,
                "query_schema_version": 1,
                "config_schema_version": 1,
                "rules": [
                    {
                        "name": "synthetic",
                        "exact": {"rows": 1},
                        "ranges": {},
                        "config": {"backend": "synthetic"},
                    }
                ],
            },
            evidence={"gpu_measurement_cases": 1},
            completed_work_units=1,
        )


def test_registry_and_runner_assemble_all_components(tmp_path) -> None:
    registry = ComponentGeneratorRegistry()
    registry.register(_Generator("attention.gqa"))
    registry.register(_Generator("moe.decode"))
    context = GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="abc123",
        settings=GenerationSettings(),
    )

    artifact = generate_profile_artifact(
        profile_id="nvidia.synthetic.48sm",
        generators=registry.select(None),
        context=context,
        progress=NullProgressReporter(),
    )

    profile = artifact["profile"]
    assert [component["component_id"] for component in profile["components"]] == [
        "attention.gqa",
        "moe.decode",
    ]
    assert set(artifact["evidence"]["components"]) == {
        "attention.gqa",
        "moe.decode",
    }
    assert (tmp_path / "checkpoints" / "moe.decode" / "synthetic-case.json").is_file()


def test_compact_profile_writer_round_trips_runtime_payload(tmp_path) -> None:
    path = tmp_path / "profile.json"
    payload = {"profile_id": "synthetic", "components": []}

    write_artifact_atomic(path, payload, overwrite=False, compact=True)

    assert json.loads(path.read_text()) == payload
    assert path.read_text().count("\n") == 1


def test_source_fingerprint_excludes_generated_profile_payloads() -> None:
    assert _is_generated_profile_data(
        Path("b12x/policy/_profiles/data/nvidia.gb10.48sm.json")
    )
    assert not _is_generated_profile_data(
        Path("b12x/policy/_profiles/data/__init__.py")
    )
