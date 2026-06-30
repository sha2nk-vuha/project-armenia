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

# Root of the on-disk image dataset browsable from the UI. Each immediate
# subdirectory that contains a `test/` folder is treated as a SKU (e.g. the
# MVTec AD object categories: pill, bottle, cable, ...).
DATA_ROOT = Path(os.environ.get("DATA_ROOT", _REPO_ROOT / "data" / "MVTecAD"))
