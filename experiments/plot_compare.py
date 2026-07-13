#!/usr/bin/env python3
"""Overlay standard-FID curves, with optional provenance compatibility checks."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "runs", nargs="+", metavar="LABEL=EVAL_JSON",
        help="curve label and eval_curve.py result JSON",
    )
    parser.add_argument(
        "--title", default="MS-CLIP-GAN checkpoint comparison",
        help="plot title; no dataset size or causal claim is inferred from filenames",
    )
    parser.add_argument(
        "--require-comparable", action="store_true",
        help="require provenance sidecars and reject protocol/training confounders",
    )
    parser.add_argument(
        "--allow-training-difference", action="append", default=[],
        help="training_config key deliberately allowed to differ (repeatable)",
    )
    parser.add_argument(
        "--require-diffaugment-pair",
        metavar="POLICY",
        help=(
            "require exactly two runs ordered as no-augmentation baseline then "
            "DiffAugment treatment using POLICY"
        ),
    )
    args = parser.parse_args(argv)
    if args.require_diffaugment_pair is not None and not args.require_comparable:
        parser.error("--require-diffaugment-pair requires --require-comparable")
    return args


def load_run(spec: str) -> Tuple[str, Path, Dict[str, Any]]:
    if "=" not in spec:
        raise ValueError(f"run must be LABEL=EVAL_JSON, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    if not label or not raw_path:
        raise ValueError(f"run must contain a non-empty label and path: {spec!r}")
    path = Path(raw_path)
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict) or not result:
        raise ValueError(f"{path} contains no epoch results")
    for epoch, metrics in result.items():
        if not isinstance(metrics, dict) or "fid" not in metrics:
            raise ValueError(f"{path}: epoch {epoch!r} has no FID value")
    return label, path, result


def load_provenance(result_path: Path) -> Dict[str, Any]:
    path = Path(f"{result_path}.provenance.json")
    with path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    if not isinstance(provenance, dict):
        raise ValueError(f"invalid provenance object: {path}")
    artifact = provenance.get("result_artifact")
    if not isinstance(artifact, Mapping) or not artifact.get("sha256"):
        raise ValueError(f"provenance does not fingerprint its result JSON: {path}")
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if digest != artifact["sha256"]:
        raise ValueError(
            f"result JSON hash does not match its provenance sidecar: {result_path}"
        )
    return provenance


def _uniform_checkpoint_field(provenance: Mapping[str, Any], field: str) -> Any:
    checkpoints = provenance.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ValueError("provenance has no checkpoint details")
    values = [details.get(field) for details in checkpoints.values()]
    canonical = json.dumps(values[0], sort_keys=True)
    if any(json.dumps(value, sort_keys=True) != canonical for value in values[1:]):
        raise ValueError(f"one evaluation contains multiple {field} values")
    return values[0]


def validate_comparable(
    provenances: Sequence[Mapping[str, Any]], allowed_training_differences: Sequence[str]
) -> None:
    if len(provenances) < 2:
        raise ValueError("comparability validation requires at least two runs")

    def evaluation_contract(provenance: Mapping[str, Any]) -> Dict[str, Any]:
        config = provenance.get("config") or {}
        data = provenance.get("data") or {}
        return {
            "git_commit": provenance.get("git_commit"),
            "seed": provenance.get("seed"),
            "sample_count": provenance.get("sample_count"),
            "caption_selection": provenance.get("caption_selection"),
            "data_sha256": data.get("sha256"),
            "caption_embedding_dimension": config.get("caption_embedding_dimension"),
            "fid_feature_dimension": config.get("fid_feature_dimension"),
            "inception_input": config.get("inception_input"),
            "batch_size": config.get("batch_size"),
            "evaluated_epochs": config.get("evaluated_epochs"),
            "hardware": provenance.get("hardware"),
            "package_versions": provenance.get("package_versions"),
        }

    for index, provenance in enumerate(provenances, start=1):
        if provenance.get("git_dirty") is not False:
            raise ValueError(
                f"run {index} was evaluated from a dirty or unknown Git state; "
                "strict comparison cannot verify its code snapshot"
            )
        checkpoint_epochs = sorted(
            int(value) for value in (provenance.get("checkpoints") or {})
        )
        configured_epochs = sorted(
            int(value) for value in (provenance.get("config") or {}).get(
                "evaluated_epochs", []
            )
        )
        if checkpoint_epochs != configured_epochs:
            raise ValueError(
                f"run {index} provenance epoch list does not match checkpoint details"
            )

    reference_contract = evaluation_contract(provenances[0])
    reference_model = _uniform_checkpoint_field(provenances[0], "model_config")
    reference_training = _uniform_checkpoint_field(provenances[0], "training_config")
    reference_schedule = _uniform_checkpoint_field(provenances[0], "schedule_config")
    reference_provenance = _uniform_checkpoint_field(
        provenances[0], "training_provenance"
    )
    if not isinstance(reference_training, Mapping):
        raise ValueError("comparison requires v2 checkpoints with training_config provenance")
    if not isinstance(reference_schedule, Mapping):
        raise ValueError("comparison requires v2 checkpoints with schedule_config provenance")
    if not isinstance(reference_provenance, Mapping):
        raise ValueError("comparison requires v2 checkpoint training_provenance")

    def training_provenance_contract(provenance: Mapping[str, Any]) -> Dict[str, Any]:
        dataset = provenance.get("dataset") or {}
        source = provenance.get("source") or {}
        return {
            "dataset_sha256": dataset.get("sha256"),
            "dataset_size_bytes": dataset.get("size_bytes"),
            "source_sha256": source.get("sha256"),
            "runtime": provenance.get("runtime"),
            "hardware": provenance.get("hardware"),
        }

    reference_training_provenance = training_provenance_contract(reference_provenance)
    if any(value is None for value in reference_training_provenance.values()):
        raise ValueError("baseline training_provenance is incomplete")

    allowed = set(allowed_training_differences)
    reference_training = {
        key: value for key, value in reference_training.items() if key not in allowed
    }
    for index, provenance in enumerate(provenances[1:], start=2):
        if evaluation_contract(provenance) != reference_contract:
            raise ValueError(f"run {index} uses a different evaluation/data/runtime contract")
        if _uniform_checkpoint_field(provenance, "model_config") != reference_model:
            raise ValueError(f"run {index} uses different model semantics")
        if _uniform_checkpoint_field(provenance, "schedule_config") != reference_schedule:
            raise ValueError(f"run {index} uses a different scheduler phase/horizon")
        candidate_provenance = _uniform_checkpoint_field(
            provenance, "training_provenance"
        )
        if not isinstance(candidate_provenance, Mapping):
            raise ValueError(f"run {index} has no v2 training_provenance")
        if training_provenance_contract(candidate_provenance) != reference_training_provenance:
            raise ValueError(
                f"run {index} used different training data/source/runtime/hardware"
            )
        training = _uniform_checkpoint_field(provenance, "training_config")
        if not isinstance(training, Mapping):
            raise ValueError(f"run {index} has no v2 training_config provenance")
        training = {key: value for key, value in training.items() if key not in allowed}
        if training != reference_training:
            differing = sorted(
                key for key in set(reference_training).union(training)
                if reference_training.get(key) != training.get(key)
            )
            raise ValueError(
                f"run {index} differs in undeclared training settings: {', '.join(differing)}"
            )


def validate_diffaugment_pair(
    provenances: Sequence[Mapping[str, Any]], expected_policy: str
) -> None:
    """Validate the treatment direction behind a baseline/DiffAugment label."""
    if len(provenances) != 2:
        raise ValueError("DiffAugment pair validation requires exactly two runs")
    normalized_policy = ",".join(
        value.strip() for value in expected_policy.split(",") if value.strip()
    )
    if not normalized_policy:
        raise ValueError("expected DiffAugment policy must not be empty")

    baseline = _uniform_checkpoint_field(provenances[0], "training_config")
    treatment = _uniform_checkpoint_field(provenances[1], "training_config")
    if not isinstance(baseline, Mapping) or not isinstance(treatment, Mapping):
        raise ValueError("DiffAugment pair validation requires v2 training_config")
    if baseline.get("use_diffaugment") is not False:
        raise ValueError(
            "first run is labeled as the baseline but use_diffaugment is not false"
        )
    if treatment.get("use_diffaugment") is not True:
        raise ValueError(
            "second run is labeled as DiffAugment but use_diffaugment is not true"
        )
    actual_policy = ",".join(
        value.strip()
        for value in str(treatment.get("diffaugment_policy", "")).split(",")
        if value.strip()
    )
    if actual_policy != normalized_policy:
        raise ValueError(
            "DiffAugment treatment policy mismatch: "
            f"checkpoint={actual_policy!r}, requested={normalized_policy!r}"
        )


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    runs = [load_run(spec) for spec in args.runs]
    provenances: List[Dict[str, Any]] = []
    if args.require_comparable:
        provenances = [load_provenance(path) for _label, path, _result in runs]
        validate_comparable(provenances, args.allow_training_difference)
        if args.require_diffaugment_pair is not None:
            validate_diffaugment_pair(
                provenances, args.require_diffaugment_pair
            )

    colors = plt.cm.tab10.colors
    plt.figure(figsize=(9.5, 5.5))
    for index, (label, _path, result) in enumerate(runs):
        epochs = sorted(int(value) for value in result)
        fids = [result[str(epoch)]["fid"] for epoch in epochs]
        best = min(epochs, key=lambda epoch: result[str(epoch)]["fid"])
        plt.plot(
            epochs, fids, "o-", color=colors[index % len(colors)],
            label=f"{label}  (best {result[str(best)]['fid']:.0f} @ ep{best})",
        )

    title = args.title
    if provenances:
        title = f"{title} (N={provenances[0]['sample_count']})"
    plt.xlabel("epoch")
    plt.ylabel("standard FID (2048-d pool3, lower = better)")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=130)
    print("saved", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
