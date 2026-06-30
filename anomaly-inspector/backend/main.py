import base64
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from config import (
    APP_VERSION,
    DATA_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_VERSION,
)
from database import crud
from database.db import get_db, init_db
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from inference import engine
from inference.preprocessor import preprocess
from inference.visualizer import generate_heatmap, generate_segmentation
from reports.pdf_generator import generate_report
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Anomaly Inspector", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()
    _load_default_model()


def _load_default_model() -> None:
    """Auto-load the bundled model so the app is usable without a manual upload."""
    if engine.get_session() is not None:
        return
    if not DEFAULT_MODEL_PATH.exists():
        logging.warning(
            "Default model not found at %s; load one via /api/load-model.", DEFAULT_MODEL_PATH
        )
        return
    try:
        sess = engine.load_model_from_path(str(DEFAULT_MODEL_PATH), DEFAULT_MODEL_VERSION)
        logging.info(
            "Default model loaded: version=%s runtime=%s input_shape=%s",
            sess.model_version,
            sess.runtime,
            sess.input_shape,
        )
    except Exception as e:
        logging.error("Failed to load default model from %s: %s", DEFAULT_MODEL_PATH, e)


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _resolve_data_path(rel_path: str) -> Path:
    """Resolve a dataset-relative path, rejecting anything outside DATA_ROOT."""
    root = DATA_ROOT.resolve()
    target = (root / rel_path).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(status_code=400, detail="Path is outside the dataset root.")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return target


@app.get("/api/skus")
def list_skus():
    """SKUs are dataset subdirectories that contain a `test/` folder."""
    if not DATA_ROOT.exists():
        return {"skus": []}
    skus = sorted(
        p.name for p in DATA_ROOT.iterdir() if p.is_dir() and (p / "test").is_dir()
    )
    return {"skus": skus}


@app.get("/api/skus/{sku}/images")
def list_sku_images(sku: str):
    """List the test images for a SKU, grouped implicitly by defect category."""
    test_dir = (DATA_ROOT / sku / "test").resolve()
    root = DATA_ROOT.resolve()
    if root not in test_dir.parents or not test_dir.is_dir():
        raise HTTPException(status_code=404, detail="SKU not found.")
    images = []
    for f in sorted(test_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
            images.append(
                {
                    "path": str(f.relative_to(root)),
                    "category": f.parent.name,
                    "name": f.name,
                }
            )
    return {"sku": sku, "images": images}


@app.get("/api/images")
def get_image(path: str):
    """Serve a dataset image by its DATA_ROOT-relative path (for thumbnails)."""
    return FileResponse(_resolve_data_path(path))


@app.get("/api/status")
def get_status():
    sess = engine.get_session()
    if sess is None:
        return {"model_loaded": False}
    return {
        "model_loaded": True,
        "model_version": sess.model_version,
        "runtime": sess.runtime,
        "input_shape": list(sess.input_shape),
    }


@app.post("/api/load-model")
async def load_model(
    model_file: UploadFile = File(...),
    model_version: str = Form(...),
):
    model_bytes = await model_file.read()
    try:
        sess = engine.load_model(model_bytes, model_file.filename or "model.onnx", model_version)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "status": "loaded",
        "runtime": sess.runtime,
        "input_shape": list(sess.input_shape),
        "model_version": sess.model_version,
    }


@app.post("/api/infer")
async def infer(
    sku_name: str = Form(...),
    threshold: float = Form(...),
    customer_name: str = Form(""),
    image: UploadFile | None = File(None),
    image_path: str | None = Form(None),
    db: Session = Depends(get_db),
):
    sess = engine.get_session()
    if sess is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")

    # Image source: an uploaded file, or a path into the on-disk dataset.
    if image is not None:
        image_bytes = await image.read()
    elif image_path:
        image_bytes = _resolve_data_path(image_path).read_bytes()
    else:
        raise HTTPException(status_code=400, detail="Provide an image file or image_path.")
    try:
        tensor, original_rgb = preprocess(image_bytes, sess.input_shape)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file.")

    try:
        anomaly_map, pred_score = engine.run_inference(tensor)
    except (IndexError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="Model output format unexpected. Expected anomaly_map at index 0 and pred_score at index 1.",
        )

    verdict = "ok" if pred_score < threshold else "not_ok"

    heatmap_bytes = generate_heatmap(anomaly_map, original_rgb)
    segmentation_bytes = generate_segmentation(anomaly_map, original_rgb, threshold)

    crud.create_inspection(
        db,
        sku_name=sku_name,
        anomaly_score=pred_score,
        threshold=threshold,
        verdict=verdict,
        model_version=sess.model_version,
        customer_name=customer_name,
        heatmap_image=heatmap_bytes,
        segmentation_image=segmentation_bytes,
    )

    return {
        "verdict": verdict,
        "anomaly_score": round(pred_score, 4),
        "heatmap_image": base64.b64encode(heatmap_bytes).decode(),
        "segmentation_image": base64.b64encode(segmentation_bytes).decode(),
    }


@app.get("/api/stats")
def get_stats(sku_name: str | None = None, db: Session = Depends(get_db)):
    return crud.get_stats(db, sku_name=sku_name or None)


@app.post("/api/reset")
def reset_database(db: Session = Depends(get_db)):
    """Erase all recorded inspections so a fresh demo can start from a clean slate."""
    deleted = crud.delete_all_inspections(db)
    return {"status": "reset", "deleted": deleted}


@app.post("/api/report")
def get_report(
    start_date: str = Form(...),
    end_date: str = Form(...),
    customer_name: str = Form(""),
    db: Session = Depends(get_db),
):
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date) + timedelta(days=1) - timedelta(microseconds=1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use ISO 8601.")

    inspections = crud.get_inspections_in_range(db, start, end, customer_name=customer_name or None)
    if not inspections:
        detail = (
            f"No inspections found for customer '{customer_name}' in the selected date range."
            if customer_name
            else "No inspections found in the selected date range."
        )
        raise HTTPException(status_code=404, detail=detail)

    total = len(inspections)
    ok_count = sum(1 for i in inspections if i.verdict == "ok")
    not_ok_count = total - ok_count
    pass_rate = round(ok_count / total * 100, 1)
    threshold_values = [i.threshold for i in inspections]

    # Per-SKU breakdown: each SKU is reported independently, then aggregated.
    sku_groups: dict[str, list] = defaultdict(list)
    for i in inspections:
        sku_groups[i.sku_name].append(i)

    sku_stats = []
    for sku in sorted(sku_groups):
        items = sku_groups[sku]
        s_total = len(items)
        s_ok = sum(1 for x in items if x.verdict == "ok")
        s_not_ok = s_total - s_ok
        s_thresholds = [x.threshold for x in items]
        sku_stats.append(
            {
                "sku_name": sku,
                "total": s_total,
                "ok": s_ok,
                "not_ok": s_not_ok,
                "pass_rate": round(s_ok / s_total * 100, 1) if s_total else 0.0,
                "threshold_min": min(s_thresholds),
                "threshold_max": max(s_thresholds),
            }
        )

    current_sess = engine.get_session()
    model_version = current_sess.model_version if current_sess else "unknown"

    pdf_bytes = generate_report(
        start_date=start,
        end_date=end,
        total=total,
        ok_count=ok_count,
        not_ok_count=not_ok_count,
        pass_rate=pass_rate,
        sku_stats=sku_stats,
        threshold_min=min(threshold_values),
        threshold_max=max(threshold_values),
        customer_name=customer_name,
        model_version=model_version,
        app_version=APP_VERSION,
    )

    filename = f"inspection-report-{start_date[:10]}-to-{end_date[:10]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
