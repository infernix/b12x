"""Offline generation contracts for embedded GPU component profiles."""

from .contracts import (
    ComponentGenerationResult,
    ComponentGenerator,
    GenerationContext,
    GenerationSettings,
    ProgressReporter,
    WorkEstimate,
)
from .reducer import (
    DecisionRecord,
    build_axis_tree,
    decision_node_to_dict,
    synthesize_integer_axis_coverage,
)
from .registry import ComponentGeneratorRegistry
from .store import CheckpointStore
from .sweep import (
    DiscreteSweepGenerator,
    SweepBenchmarkFactory,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
    SweepSession,
)

__all__ = [
    "CheckpointStore",
    "ComponentGenerationResult",
    "ComponentGenerator",
    "ComponentGeneratorRegistry",
    "DecisionRecord",
    "DiscreteSweepGenerator",
    "GenerationContext",
    "GenerationSettings",
    "ProgressReporter",
    "SweepBenchmarkFactory",
    "SweepCandidate",
    "SweepCase",
    "SweepMeasurement",
    "SweepSession",
    "WorkEstimate",
    "build_axis_tree",
    "decision_node_to_dict",
    "synthesize_integer_axis_coverage",
]
