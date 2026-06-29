import base64
import logging
from datetime import datetime

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.orm import Session

from config import APP_VERSION
from database import crud
from database.db import get_db, init_db
from inference import engine
from inference.preprocessor import preprocess
from inference.visualizer import generate_heatmap, generate_segmentation
from reports.pdf_generator import generate_report

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
    image: UploadFile = File(...),
    sku_name: str = Form(...),
    threshold: float = Form(...),
    db: Session = Depends(get_db),
):
    sess = engine.get_session()
    if sess is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")

    image_bytes = await image.read()
    tensor, original_rgb = preprocess(image_bytes, sess.input_shape)
    anomaly_map, pred_score = engine.run_inference(tensor)

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
def get_stats(db: Session = Depends(get_db)):
    return crud.get_stats(db)


@app.post("/api/report")
def get_report(
    start_date: str = Form(...),
    end_date: str = Form(...),
    db: Session = Depends(get_db),
):
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use ISO 8601.")

    inspections = crud.get_inspections_in_range(db, start, end)
    if not inspections:
        raise HTTPException(status_code=404, detail="No inspections found in the selected date range.")

    total = len(inspections)
    ok_count = sum(1 for i in inspections if i.verdict == "ok")
    not_ok_count = total - ok_count
    pass_rate = round(ok_count / total * 100, 1)
    sku_names = [i.sku_name for i in inspections]
    threshold_values = [i.threshold for i in inspections]

    current_sess = engine.get_session()
    model_version = current_sess.model_version if current_sess else "unknown"

    pdf_bytes = generate_report(
        start_date=start,
        end_date=end,
        total=total,
        ok_count=ok_count,
        not_ok_count=not_ok_count,
        pass_rate=pass_rate,
        sku_names=sku_names,
        threshold_min=min(threshold_values),
        threshold_max=max(threshold_values),
        model_version=model_version,
        app_version=APP_VERSION,
    )

    filename = f"inspection-report-{start_date[:10]}-to-{end_date[:10]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
