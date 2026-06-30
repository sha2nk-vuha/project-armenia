# Anomaly Inspector — Backend

FastAPI service that runs ONNX anomaly-detection inference over product images,
stores inspection results in SQLite, and generates PDF reports. It also serves
the dataset images browsed by the frontend.

## Prerequisites

- **Python 3.11** (the pinned dependencies are tested against 3.11)
- **Node.js 18+ and npm** (only needed to run the frontend)
- The **model file** and the **image dataset** — these are *not* in the git
  repository (they are large and gitignored). You must place them yourself:

  | Asset | Default location | Override env var |
  |-------|------------------|------------------|
  | ONNX model (~144 MB) | `<repo>/model/dinomaly_s_mvtec.onnx` | `DEFAULT_MODEL_PATH` |
  | Image dataset (MVTec AD) | `<repo>/data/MVTecAD/` | `DATA_ROOT` |

  The dataset root must contain one subdirectory per SKU, each with a `test/`
  folder (e.g. `data/MVTecAD/bottle/test/...`).

## Quick start (recommended)

From the `anomaly-inspector/` directory, the start script sets up both the
backend and frontend on first run, then launches them together:

```bash
cd anomaly-inspector
./start.sh
```

Then open <http://localhost:5173>. Press `Ctrl-C` to stop both servers.

To only install dependencies without starting the servers:

```bash
./start.sh --setup
```

## Manual setup (backend only)

If you prefer to run the backend by hand:

```bash
cd anomaly-inspector/backend

# 1. Create and activate a virtualenv
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the API server
uvicorn main:app --host 127.0.0.1 --port 8000
```

The server prints `Application startup complete` when ready. On startup it:

- creates `inspections.db` (SQLite) in the backend directory if it does not exist, and
- auto-loads the default model if `DEFAULT_MODEL_PATH` exists.

> The frontend dev server (Vite) proxies `/api` to `http://localhost:8000` and
> the backend allows CORS from `http://localhost:5173`, so use those default
> ports for local development.

## Configuration

All settings have sensible defaults and can be overridden with environment
variables (see [`config.py`](config.py)):

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_MODEL_PATH` | `<repo>/model/dinomaly_s_mvtec.onnx` | Model auto-loaded on startup |
| `DEFAULT_MODEL_VERSION` | `dinomaly_s_mvtec` | Version label recorded with inspections |
| `DATA_ROOT` | `<repo>/data/MVTecAD` | Root of the browsable image dataset |

Example with custom paths:

```bash
DATA_ROOT=/srv/images \
DEFAULT_MODEL_PATH=/srv/models/model.onnx \
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Running the tests

```bash
cd anomaly-inspector/backend
source venv/bin/activate
python -m pytest
```

## API overview

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/status` | Whether a model is loaded and its details |
| `POST` | `/api/load-model` | Upload and load an ONNX model |
| `GET`  | `/api/skus` | List SKUs (dataset folders with a `test/`) |
| `GET`  | `/api/skus/{sku}/images` | List a SKU's test images |
| `GET`  | `/api/images?path=...` | Serve a dataset image (thumbnails) |
| `POST` | `/api/infer` | Run inference on an uploaded image or dataset path |
| `GET`  | `/api/stats?sku_name=...` | Inspection statistics (optionally per SKU) |
| `POST` | `/api/reset` | Erase all recorded inspections |
| `POST` | `/api/report` | Generate a PDF report for a date range / customer |

## Database

Inspections are stored in `inspections.db` (SQLite) next to the backend code.
This file is gitignored — it is per-machine user data and is never committed.
Use the **Reset database** button in the UI (or `POST /api/reset`) to clear it.

## Notes

- `onnxruntime` runs inference on CPU by default. The `openvino` dependency is
  optional acceleration; if its native library fails to load the service logs a
  warning and falls back to CPU automatically.
