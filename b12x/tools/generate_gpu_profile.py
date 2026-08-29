"""Generate one resumable, multi-component GPU profile artifact."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path

from rich.console import Console
from rich.table import Table

from b12x.policy import detect_device
from b12x.policy.generation import (
    ComponentGenerator,
    ComponentGeneratorRegistry,
    GenerationContext,
    GenerationSettings,
)
from b12x.policy.generation.progress import RichProgressReporter
from b12x.policy.generation.runner import (
    estimate_generators,
    generate_profile_artifact,
    merge_profile_artifacts,
    runtime_profile_payload,
    write_artifact_atomic,
)

_ENTRY_POINT_GROUP = "b12x.profile_generators"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATED_PROFILE_DATA = Path("b12x/policy/_profiles/data")
_GENERATED_PROFILE_PATHSPEC = ":(exclude)b12x/policy/_profiles/data/*.json*"


def _is_generated_profile_data(path: Path) -> bool:
    return path.parent == _GENERATED_PROFILE_DATA and (
        path.name.endswith(".json") or path.name.endswith(".json.gz")
    )


def _package_source_revision() -> str:
    try:
        version = metadata.version("b12x")
    except metadata.PackageNotFoundError:
        version = "uninstalled"
    fingerprint = hashlib.sha256()
    package_root = _REPO_ROOT / "b12x"
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        fingerprint.update(str(path.relative_to(package_root)).encode())
        fingerprint.update(path.read_bytes())
    return f"package.{version}.{fingerprint.hexdigest()[:16]}"


def _source_revision() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "HEAD",
                "--",
                "b12x",
                "benchmarks",
                "pyproject.toml",
                _GENERATED_PROFILE_PATHSPEC,
            ],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "b12x",
                "benchmarks",
            ],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError):
        return _package_source_revision()
    fingerprint = hashlib.sha256(diff)
    for raw_path in sorted(path for path in untracked if path):
        relative_path = Path(os.fsdecode(raw_path))
        if _is_generated_profile_data(relative_path):
            continue
        path = _REPO_ROOT / relative_path
        if not path.is_file():
            continue
        fingerprint.update(raw_path)
        fingerprint.update(path.read_bytes())
    digest = fingerprint.hexdigest()
    return (
        revision
        if not diff and not any(untracked)
        else f"{revision}-worktree.{digest[:16]}"
    )


def _profile_id(product_name: str, sm_count: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", ".", product_name.casefold()).strip(".")
    return f"{slug}.{sm_count}sm"


def _load_registry() -> ComponentGeneratorRegistry:
    from b12x.policy.generation.providers import register_builtin_generators

    registry = ComponentGeneratorRegistry()
    register_builtin_generators(registry)
    entry_points = metadata.entry_points()
    selected = entry_points.select(group=_ENTRY_POINT_GROUP)
    for entry_point in sorted(selected, key=lambda item: item.name):
        loaded = entry_point.load()
        generator = loaded() if isinstance(loaded, type) else loaded
        registry.register(generator)
    return registry


def _parse_components(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or raw.strip().casefold() == "all":
        return None
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("--components must select at least one component")
    return values


def _render_estimates(
    console: Console,
    generators: tuple[ComponentGenerator, ...],
    context: GenerationContext,
) -> None:
    estimates = estimate_generators(generators, context)
    table = Table(title="GPU profile generation plan")
    table.add_column("Component")
    table.add_column("Cases", justify="right")
    table.add_column("Work units", justify="right")
    table.add_column("Scope")
    for estimate in estimates:
        table.add_row(
            estimate.component_id,
            f"{estimate.case_count:,}",
            f"{estimate.work_units:,}",
            estimate.description,
        )
    table.add_section()
    table.add_row(
        "total",
        f"{sum(item.case_count for item in estimates):,}",
        f"{sum(item.work_units for item in estimates):,}",
        "",
    )
    console.print(table)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-id")
    parser.add_argument("--components", default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--merge-from",
        type=Path,
        help="base full profile whose unselected components are retained",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    parser.add_argument(
        "--max-candidate-seconds",
        type=float,
        default=2.0,
        help="cap timed replay work per candidate while retaining every group",
    )
    parser.add_argument(
        "--cold-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--embed",
        action="store_true",
        help="also install the generated artifact into b12x package data",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-components", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    console = Console()
    registry = _load_registry()
    if args.list_components:
        for component_id in registry.component_ids():
            console.print(component_id)
        return 0
    try:
        selected_ids = _parse_components(args.components)
        generators = registry.select(selected_ids)
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not generators:
        raise SystemExit(
            "no component profile generators are installed; "
            f"register providers through {_ENTRY_POINT_GROUP!r}"
        )

    detected = detect_device(args.device)
    if detected.identity is None or detected.ordinal is None:
        raise SystemExit(f"{args.device!r} did not resolve to an available CUDA GPU")
    profile_id = args.profile_id or _profile_id(
        detected.identity.product_name,
        detected.identity.sm_count,
    )
    work_dir = (args.work_dir or Path(".b12x-profile-work") / profile_id).resolve()
    output = (
        args.output
        or Path("validation/gpu_profiles/generated") / f"{profile_id}.json.gz"
    ).resolve()
    embedded_output = (
        _REPO_ROOT
        / "b12x"
        / "policy"
        / "_profiles"
        / "data"
        / f"{profile_id}.json.gz"
    ).resolve()
    merge_from = None if args.merge_from is None else args.merge_from.resolve()
    if selected_ids is not None and merge_from is None:
        if output.exists():
            merge_from = output
        elif args.embed and embedded_output.exists():
            merge_from = embedded_output
    if args.embed and selected_ids is not None and merge_from is None:
        raise SystemExit(
            "embedding a component subset requires an existing output profile "
            "or --merge-from"
        )
    if not args.dry_run and output.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite existing profile {output}; "
            "pass --overwrite or choose --output"
        )
    if (
        args.embed
        and not args.dry_run
        and embedded_output != output
        and embedded_output.exists()
        and not args.overwrite
    ):
        raise SystemExit(
            f"refusing to overwrite embedded profile {embedded_output}; "
            "pass --overwrite"
        )
    context = GenerationContext(
        device=detected.identity,
        device_ordinal=detected.ordinal,
        work_dir=work_dir,
        source_revision=_source_revision(),
        settings=GenerationSettings(
            warmup=args.warmup,
            repetitions=args.repetitions,
            groups=args.groups,
            seed=args.seed,
            minimum_cosine=args.minimum_cosine,
            cold_l2=args.cold_l2,
            max_candidate_seconds=args.max_candidate_seconds,
        ),
    )
    console.print(
        f"[bold]{profile_id}[/bold] on cuda:{detected.ordinal} "
        f"({detected.identity.product_name}, {detected.identity.sm_count} SMs)"
    )
    console.print(f"Checkpoint directory: {work_dir}")
    if merge_from is not None:
        console.print(f"Merge base: {merge_from}")
    _render_estimates(console, generators, context)
    if args.dry_run:
        return 0

    estimates = estimate_generators(generators, context)
    try:
        with RichProgressReporter(estimates) as progress:
            artifact = generate_profile_artifact(
                profile_id=profile_id,
                generators=generators,
                context=context,
                progress=progress,
            )
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted.[/yellow] Completed races are checkpointed; "
            f"rerun with the same --work-dir {work_dir} to resume."
        )
        return 130
    if merge_from is not None:
        raw = merge_from.read_bytes()
        if merge_from.suffix == ".gz":
            raw = gzip.decompress(raw)
        base_artifact = json.loads(raw)
        if not isinstance(base_artifact, Mapping):
            raise TypeError("merge base must contain a JSON object")
        artifact = merge_profile_artifacts(base_artifact, artifact)
    profile = artifact["profile"]
    if not isinstance(profile, Mapping):
        raise TypeError("generated artifact profile must be an object")
    output_is_embedded = args.embed and embedded_output == output
    embedded_profile = runtime_profile_payload(profile)
    write_artifact_atomic(
        output,
        embedded_profile if output_is_embedded else artifact,
        overwrite=args.overwrite,
        compact=output_is_embedded,
    )
    console.print(f"Wrote [bold]{output}[/bold]")
    if args.embed and embedded_output != output:
        write_artifact_atomic(
            embedded_output,
            embedded_profile,
            overwrite=args.overwrite,
            compact=True,
        )
        console.print(f"Embedded [bold]{embedded_output}[/bold]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
