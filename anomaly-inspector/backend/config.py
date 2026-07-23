import os
from pathlib import Path

APP_VERSION = "1.0.0"

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
}

# Root of the on-disk image dataset browsable from the UI. Each immediate
# subdirectory that contains a `test/` folder is treated as a SKU (e.g. the
# MVTec AD object categories: pill, bottle, cable, ...).
DATA_ROOT = Path(os.environ.get("DATA_ROOT", _REPO_ROOT / "data" / "MVTecAD"))
