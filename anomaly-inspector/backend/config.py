import os
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
import logging

logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"


class _FeatureSpec(BaseModel):
    """Validated feature catalog entry (boundary: env-config)."""

    label: str = Field(min_length=1)
    threshold_label: str = Field(min_length=1)
    model_config = {"extra": "allow"}

# Repo layout: <repo_root>/anomaly-inspector/backend/config.py and the bundled
# model lives at <repo_root>/model/. Resolve from this file so it works no
# matter what the process working directory is.
_BACKEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BACKEND_DIR.parent.parent

DEFAULT_MODEL_PATH = Path(
    os.environ.get("DEFAULT_MODEL_PATH", _REPO_ROOT / "model" / "dinomaly_s_mvtec.onnx")
)
DEFAULT_MODEL_VERSION = os.environ.get("DEFAULT_MODEL_VERSION", "dinomaly_s_mvtec")

# ── Features ────────────────────────────────────────────────────────────────
# A Feature is a selectable inspection capability backed by one model + one
# inference pipeline. Exactly one Feature is active at a time; switching swaps
# the loaded model. See docs/adr/0001 and 0003.
ANOMALY_FEATURE = "anomaly_detection"
PRESENCE_FEATURE = "presence_absence"
SEGMENTATION_FEATURE = "segmentation"
CASCADE_FEATURE = "cascade"
DEFAULT_FEATURE = os.environ.get("DEFAULT_FEATURE", ANOMALY_FEATURE)

# Per-Feature default model. `DEFAULT_MODEL_PATH`/`_VERSION` stay as the Anomaly
# default for backward compatibility and feed the entry below.
FEATURES: dict[str, dict] = {
    ANOMALY_FEATURE: {
        "label": "Anomaly Detection",
        "threshold_label": "Anomaly Threshold",
        "model_path": DEFAULT_MODEL_PATH,
        "model_version": DEFAULT_MODEL_VERSION,
    },
    PRESENCE_FEATURE: {
        "label": "Presence / Absence",
        "threshold_label": "Detection Confidence",
        "model_path": Path(
            os.environ.get(
                "PRESENCE_MODEL_PATH", _REPO_ROOT / "model" / "rfdetr-nano.onnx"
            )
        ),
        "model_version": os.environ.get("PRESENCE_MODEL_VERSION", "rfdetr-nano"),
    },
    # The bundled segmentation export is customer-specific, so the path is an
    # env override rather than a fixed generic default.
    SEGMENTATION_FEATURE: {
        "label": "Segmentation",
        "threshold_label": "Detection Confidence",
        "model_path": Path(
            os.environ.get(
                "SEGMENTATION_MODEL_PATH",
                _REPO_ROOT / "model" / "three_cee_caps_rfdetr-seg-nano_v0.0.1.onnx",
            )
        ),
        "model_version": os.environ.get(
            "SEGMENTATION_MODEL_VERSION", "rfdetr-seg-nano-v0.0.1"
        ),
    },
    # Cascade is a composite Feature: it runs several of the above as stages and
    # combines their Verdicts. It has no model of its own (see docs/adr/0006),
    # so it carries no model_path and is special-cased in activation.
    CASCADE_FEATURE: {
        "label": "Cascade",
        "threshold_label": "Per-stage",
        "composite": True,
    },
}

# ── Validate at import boundary (env is a trust boundary per validation-pydantic) ──
for _name, _spec in list(FEATURES.items()):
    try:
        _FeatureSpec.model_validate(_spec)
    except ValidationError as _exc:
        logger.warning("Feature %r has invalid catalog entry: %s — using as-is", _name, _exc)

if DEFAULT_FEATURE not in FEATURES:
    logger.warning("DEFAULT_FEATURE %r not in FEATURES; falling back to %r", DEFAULT_FEATURE, ANOMALY_FEATURE)
    DEFAULT_FEATURE = ANOMALY_FEATURE
