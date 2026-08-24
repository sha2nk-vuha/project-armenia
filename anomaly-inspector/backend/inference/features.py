"""Active-Feature registry: which inspection Feature is live, and its pipeline.

A thin coordinator over `engine` (which owns the single loaded model session).
It records the active Feature, loads the RF-DETR sidecar when Presence/Absence
is active, and builds the matching pipeline on demand. `/api/infer` asks for
`current_pipeline()` and stays agnostic to which Feature is running.

Exactly one Feature is active at a time; switching reloads that Feature's
default model (see docs/adr/0001).
"""
import logging
from pathlib import Path

from config import FEATURES, PRESENCE_FEATURE, SEGMENTATION_FEATURE
from inference import decision, engine
from inference.engine import ModelSession
from inference.pipeline import (
    AnomalyPipeline,
    PresenceAbsencePipeline,
    SegmentationPipeline,
)
from inference.model_config import ModelConfig, load_config, sidecar_path

logger = logging.getLogger(__name__)

# Features whose model carries a sidecar (Class Catalog, preprocessing, rule
# param defaults). Anomaly Detection needs none.
_SIDECAR_FEATURES = (PRESENCE_FEATURE, SEGMENTATION_FEATURE)

_active_feature: str | None = None
_model_config: ModelConfig | None = None


def get_active_feature() -> str | None:
    return _active_feature


def get_model_config() -> ModelConfig | None:
    return _model_config


def activate(feature: str) -> ModelSession:
    """Make `feature` active by loading its default model (swaps the session).

    For sidecar-backed Features, also loads the model's sidecar so the Class
    Catalog, preprocessing, and rule param defaults are ready. Raises ValueError for an unknown
    Feature and FileNotFoundError when the default model file is missing.
    """
    global _active_feature, _model_config
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature: {feature!r}")

    spec = FEATURES[feature]
    model_path = Path(spec["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Default model for {feature!r} not found at {model_path}."
        )

    # Load the sidecar first so a malformed one fails before we swap the session
    # or flip active state (no half-applied activation).
    model_config = (
        load_config(sidecar_path(str(model_path)))
        if feature in _SIDECAR_FEATURES
        else None
    )
    engine.load_model_from_path(str(model_path), spec["model_version"])
    _active_feature = feature
    _model_config = model_config
    logger.info("Feature activated: %s (model=%s)", feature, model_path.name)
    return engine.get_session()


def register_upload(feature: str) -> None:
    """Record that the currently-loaded (uploaded) session belongs to `feature`.

    Used by /api/load-model after `engine` has loaded the uploaded bytes. Unlike
    `activate`, it does not reload a default model. An uploaded model carries no
    sidecar, so the existing Presence config (if any) is kept, else defaults.
    """
    global _active_feature, _model_config
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature: {feature!r}")
    _active_feature = feature
    if feature in _SIDECAR_FEATURES and _model_config is None:
        _model_config = ModelConfig()


def current_pipeline():
    """Build the pipeline for the active Feature over the loaded session.

    Returns None when no Feature is active or no model is loaded.
    """
    sess = engine.get_session()
    if sess is None or _active_feature is None:
        return None
    if _active_feature == PRESENCE_FEATURE:
        cfg = _model_config or ModelConfig()
        return PresenceAbsencePipeline(sess, cfg, cfg.expected_classes)
    if _active_feature == SEGMENTATION_FEATURE:
        return SegmentationPipeline(sess, _model_config or ModelConfig())
    return AnomalyPipeline(sess)


def current_decision_rules() -> list[dict]:
    """UI metadata for the Decision Rules the active pipeline can run.

    Filtered by the output kinds the active Feature's decode advertises, so the
    GUI never offers a rule that cannot consume this model's output. Empty when
    no Feature is active.
    """
    pipeline = current_pipeline()
    if pipeline is None:
        return []
    return [decision.describe(r) for r in pipeline.compatible_rules()]


def reset() -> None:
    """Clear active-Feature state (test hook; does not unload the engine session)."""
    global _active_feature, _model_config
    _active_feature = None
    _model_config = None
