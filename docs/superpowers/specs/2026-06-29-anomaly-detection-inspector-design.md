# Anomaly Detection Inspector — Design Spec

**Date:** 2026-06-29
**Author:** Shashank Shivakumar
**Status:** Approved

---

## 1. Overview

A prototype end-to-end anomaly detection inspection application. The user loads an Anomalib-exported ONNX model, uploads an image, runs inference, and sees the defect location visualised as a heatmap and segmentation map alongside an OK / NOT OK verdict. Every inspection result is persisted to a local database. A PDF report can be generated for any date range.

---

## 2. Architecture

### 2.1 High-Level

```
React Frontend (Vite, port 5173)
        │  HTTP REST  /api/*
FastAPI Backend (port 8000)
   ├── Inference Engine (ONNX / OpenVINO / CPU)
   ├── SQLite Database (SQLAlchemy)
   └── PDF Generator (ReportLab)
```

In **development**: Vite dev server proxies `/api/*` to FastAPI. Hot-reload on both sides.
In **production / demo**: `npm run build` produces a static bundle that FastAPI mounts and serves from `/`. Single process, single port.
**Dockerisation** (future): one multi-stage Dockerfile — Node stage builds frontend, Python stage runs FastAPI serving the bundle.

### 2.2 Inference Runtime Selection

At model load time the backend auto-selects the runtime in priority order:

1. **ONNX Runtime + CUDAExecutionProvider** — if a CUDA-capable GPU is present
2. **OpenVINO Runtime** — if no GPU but Intel CPU is available (leverages Intel hardware optimisation)
3. **ONNX Runtime CPUExecutionProvider** — universal fallback

The selected runtime is logged and exposed via `/api/status`.

### 2.3 API Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/load-model` | Load `.onnx` or OpenVINO `.xml` + set model version string |
| POST | `/api/infer` | Run inference on uploaded image; persist result |
| GET | `/api/stats` | Return aggregate counts (total, ok, not_ok, pass_rate) |
| POST | `/api/report` | Generate and return PDF for a date range |
| GET | `/api/status` | Active model name, version, runtime backend |

---

## 3. Inference Pipeline

### 3.1 Input → Output Flow

```
POST /api/infer
  multipart body: { image: File, sku_name: str, threshold: float }

1. Preprocess
   - Decode image (PIL / OpenCV)
   - Resize to model input shape (read from ONNX metadata)
   - Normalise: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] (ImageNet)
   - Add batch dim → float32 tensor

2. Run model
   Anomalib ONNX outputs:
     - anomaly_map  : float32 [1, 1, H, W]  — pixel-level anomaly scores
     - pred_score   : float32 [1]            — image-level anomaly score

3. Apply threshold
   - verdict = "ok" if pred_score < threshold else "not_ok"

4. Visualise
   - Heatmap: resize anomaly_map to original image size,
              apply jet colormap, alpha-blend onto original (α=0.5)
   - Segmentation: threshold anomaly_map at same value,
                   produce binary mask, draw red contour overlay

5. Persist to SQLite (see Section 4)

6. Return JSON
   {
     verdict: "ok" | "not_ok",
     anomaly_score: float,
     heatmap_image: "<base64 PNG>",
     segmentation_image: "<base64 PNG>"
   }
```

### 3.2 Model Loading

```
POST /api/load-model
  multipart body: { model_file: File, model_version: str }

- Detect format by extension: .onnx → ONNX Runtime; .xml → OpenVINO IR
- Select execution provider (Section 2.2)
- Store session in backend application state (singleton, replaced on re-load)
- Extract input shape from model metadata for preprocessing
- Return { status: "loaded", runtime: str, input_shape: [H, W] }
```

---

## 4. Database

### 4.1 Schema — `inspections` table

| Column | Type | Notes |
|---|---|---|
| id | INTEGER | Primary key, autoincrement |
| timestamp | DATETIME | UTC, set at inference time |
| sku_name | TEXT | User-supplied SKU identifier |
| anomaly_score | REAL | Raw pred_score from model |
| threshold | REAL | Threshold value used for this inference |
| verdict | TEXT | `"ok"` or `"not_ok"` |
| model_version | TEXT | Version string set at model load time |
| heatmap_image | BLOB | PNG bytes |
| segmentation_image | BLOB | PNG bytes |

### 4.2 Queries Used

- **Stats** (`GET /api/stats`): `SELECT COUNT(*), SUM(verdict='ok'), SUM(verdict='not_ok') FROM inspections`
- **Report** (`POST /api/report`): all columns WHERE `timestamp BETWEEN :start AND :end`

---

## 5. PDF Report

### 5.1 Trigger

`POST /api/report` body: `{ start_date: ISO8601, end_date: ISO8601 }`
Response: `application/pdf` file download.

### 5.2 Report Sections

| Section | Content |
|---|---|
| Header | Application name, software version (`APP_VERSION`), model version, report generated timestamp |
| Date Range | `from: YYYY-MM-DD  to: YYYY-MM-DD` |
| Summary Statistics | Total inspected, OK count, NOT OK count, pass rate (%) |
| SKU | SKU name(s) present in the period |
| Threshold | Threshold value(s) used during the period (min / max if varied) |

### 5.3 Implementation

Built with **ReportLab**. No per-image thumbnails in v1 — summary stats only, keeping the PDF lightweight.

---

## 6. Frontend

### 6.1 Layout

```
┌─────────────────────────────────────────────────────────────┐
│  ANOMALY DETECTION INSPECTOR              [Report]  [ ⚙ ]  │
├──────────────────────┬──────────────────────────────────────┤
│  SETUP               │  RESULTS                             │
│  SKU Name [______]   │  [Original] [Heatmap] [Segmentation] │
│  Load Image [ ▲ ]    │                                      │
│                      │  ┌──────────────────────────────┐   │
│  Threshold           │  │  ✓ OK  /  ✗ NOT OK           │   │
│  0 ────●──── 1       │  │  Anomaly Score: 0.73          │   │
│       0.5            │  └──────────────────────────────┘   │
│                      ├──────────────────────────────────────┤
│  [ INFER ]           │  STATISTICS                          │
│                      │  [Total 142] [OK 128] [NOT OK 14]    │
│                      │  Pass Rate: 90.1%                    │
└──────────────────────┴──────────────────────────────────────┘
```

### 6.2 Settings Modal (⚙)

Opens on gear icon click. Contains:
- Model file picker (`.onnx` or `.xml`)
- Model version text input
- Load button + status indicator (● Loaded / ○ Not loaded)
- Active runtime display (e.g. "OpenVINO")

Dismissed after successful load. Not visible in main workflow.

### 6.3 Report Modal

Opens on `[Report]` button click. Contains:
- Start date picker
- End date picker
- Generate PDF button → triggers download

### 6.4 Component Tree

```
App
├── Navbar (title, ReportModal trigger, SettingsModal trigger)
├── SettingsModal
├── ReportModal
├── SetupPanel
│   ├── SkuInput
│   ├── ImageUploader
│   ├── ThresholdControl
│   └── InferButton
└── ResultsPanel
    ├── ImageTriple (original / heatmap / segmentation)
    ├── VerdictBadge
    └── StatsPanel
```

### 6.5 Tech Stack

| Package | Purpose |
|---|---|
| React 18 + Vite | UI framework + dev server |
| Tailwind CSS + shadcn/ui | Styling + accessible components |
| react-datepicker | Date range picker for report |
| `src/api/client.ts` | Typed fetch wrappers for all `/api/*` endpoints |

---

## 7. Project Structure

```
anomaly-inspector/
├── backend/
│   ├── main.py                    # FastAPI app + route registration
│   ├── config.py                  # APP_VERSION constant, settings
│   ├── inference/
│   │   ├── engine.py              # Runtime factory + model session singleton
│   │   ├── preprocessor.py        # Anomalib-compatible image preprocessing
│   │   └── visualizer.py          # Heatmap + segmentation image generation
│   ├── database/
│   │   ├── db.py                  # SQLite engine + session factory
│   │   ├── models.py              # SQLAlchemy Inspection ORM model
│   │   └── crud.py                # create_inspection(), get_stats(), get_range()
│   ├── reports/
│   │   └── pdf_generator.py       # ReportLab PDF builder
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── SettingsModal.tsx
│   │   │   ├── ReportModal.tsx
│   │   │   ├── ImageUploader.tsx
│   │   │   ├── ThresholdControl.tsx
│   │   │   ├── InferencePanel.tsx
│   │   │   ├── ResultsDisplay.tsx
│   │   │   ├── VerdictBadge.tsx
│   │   │   └── StatsPanel.tsx
│   │   ├── api/
│   │   │   └── client.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts             # /api proxy → localhost:8000
└── README.md
```

---

## 8. Out of Scope (v1)

- Live camera feed (planned for v2 — architecture supports it via browser `getUserMedia` + WebSocket)
- User authentication
- Per-image thumbnails in PDF report
- Batch inference (multiple images at once)
- Model performance metrics (ROC, AUC)
