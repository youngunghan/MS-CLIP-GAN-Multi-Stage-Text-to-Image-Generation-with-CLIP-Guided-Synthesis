"""Resolve generator construction arguments from checkpoint metadata."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from utils.utils import peek_checkpoint_metadata


LEGACY_MODEL_CONFIG: Dict[str, Any] = {
    "g_in_chans": 1024,
    "g_out_chans": 3,
    "noise_dim": 100,
    "condition_dim": 128,
    "clip_embedding_dim": 512,
    "num_stage": 3,
    "conditioning_activation": "relu",
    "alignment_mode": "legacy_conditioned",
    "deterministic_cond": False,
}


def resolve_checkpoint_model_config(checkpoint_path, epoch) -> Tuple[dict, dict]:
    """Return canonical metadata and a complete, validated model config.

    Versioned checkpoints carry the complete config. Historical checkpoints are
    shape-inferred by ``peek_checkpoint_metadata``; any genuinely unavailable
    legacy field receives the architecture that produced this repository's old
    checkpoints, rather than an arbitrary current CLI value.
    """
    metadata = peek_checkpoint_metadata(checkpoint_path, epoch, device="cpu")
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    raw_config = metadata.get("model_config")
    if not isinstance(raw_config, Mapping):
        raw_config = {}

    config = dict(LEGACY_MODEL_CONFIG)
    for key in config:
        if raw_config.get(key) is not None:
            config[key] = raw_config[key]

    for key in (
        "g_in_chans",
        "g_out_chans",
        "noise_dim",
        "condition_dim",
        "clip_embedding_dim",
        "num_stage",
    ):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid checkpoint model_config.{key}: {value!r}")
    if config["g_out_chans"] != 3:
        raise ValueError("evaluation/inference requires checkpoint g_out_chans=3")
    if config["clip_embedding_dim"] != 512:
        raise ValueError(
            "this pipeline uses CLIP ViT-B/32 and requires checkpoint clip_embedding_dim=512"
        )
    if config["conditioning_activation"] not in {"linear", "relu"}:
        raise ValueError(
            "invalid checkpoint conditioning_activation: "
            f"{config['conditioning_activation']!r}"
        )
    if config["alignment_mode"] not in {"image_only", "legacy_conditioned"}:
        raise ValueError(f"invalid checkpoint alignment_mode: {config['alignment_mode']!r}")
    if not isinstance(config["deterministic_cond"], bool):
        raise ValueError(
            f"invalid checkpoint deterministic_cond: {config['deterministic_cond']!r}"
        )
    return dict(metadata), config


def apply_checkpoint_model_config(args) -> Tuple[dict, dict]:
    metadata, config = resolve_checkpoint_model_config(args.checkpoint_path, args.load_epoch)
    for key, value in config.items():
        setattr(args, key, value)
    compatibility = "legacy" if metadata.get("legacy") else f"format v{metadata.get('format_version')}"
    print(
        f"Checkpoint config ({compatibility}): "
        f"G={config['g_in_chans']}ch/{config['num_stage']} stages, "
        f"noise={config['noise_dim']}, condition={config['condition_dim']}, "
        f"conditioning={config['conditioning_activation']}, "
        f"deterministic_cond={config['deterministic_cond']}"
    )
    return metadata, config
