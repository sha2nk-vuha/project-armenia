import base64
import io
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from config import (
    APP_VERSION,
    DATA_ROOT,
    DEFAULT_FEATURE,
    FEATURES,
)
from database import crud
from database.db import get_db, init_db
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from inference import engine, features
from inference.decision import ClassNotFound
from PIL import Image
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
    """Activate the startup default Feature so the app is usable immediately.

    Loads that Feature's bundled model (Anomaly Detection by default). Missing
    files are logged, not fatal — the operator can switch Features / upload a
    model from the UI.
    """
    if engine.get_session() is not None:
        return
    try:
        sess = features.activate(DEFAULT_FEATURE)
        logging.info(
            "Default Feature '%s' active: version=%s runtime=%s input_shape=%s",
            DEFAULT_FEATURE,
            sess.model_version,
            sess.runtime,
            sess.input_shape,
        )
    except (FileNotFoundError, ValueError) as e:
        logging.warning("Could not activate default Feature '%s': %s", DEFAULT_FEATURE, e)


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


def _feature_catalog() -> list[dict]:
    """The selectable Features and their UI metadata (labels, threshold naming)."""
    return [
        {
            "name": name,
            "label": spec["label"],
            "threshold_label": spec["threshold_label"],
        }
        for name, spec in FEATURES.items()
    ]


@app.get("/api/status")
def get_status():
    catalog = _feature_catalog()
    sess = engine.get_session()
    if sess is None:
        return {
            "model_loaded": False,
            "active_feature": features.get_active_feature(),
            "features": catalog,
        }
    return {
        "model_loaded": True,
        "model_version": sess.model_version,
        "runtime": sess.runtime,
        "input_shape": list(sess.input_shape),
        "active_feature": features.get_active_feature(),
        "features": catalog,
    }


@app.post("/api/feature")
def set_feature(feature: str = Form(...)):
    """Activate an inspection Feature, loading its default model (swaps session)."""
    if feature not in FEATURES:
        raise HTTPException(status_code=400, detail=f"Unknown feature: {feature!r}")
    try:
        sess = features.activate(feature)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid model sidecar: {e}")
    return {
        "status": "activated",
        "active_feature": feature,
        "model_version": sess.model_version,
        "runtime": sess.runtime,
        "input_shape": list(sess.input_shape),
    }


@app.post("/api/load-model")
async def load_model(
    model_file: UploadFile = File(...),
    model_version: str = Form(...),
    feature: str = Form(""),
):
    target_feature = feature or features.get_active_feature() or DEFAULT_FEATURE
    if target_feature not in FEATURES:
        raise HTTPException(status_code=400, detail=f"Unknown feature: {target_feature!r}")
    model_bytes = await model_file.read()
    try:
        sess = engine.load_model(model_bytes, model_file.filename or "model.onnx", model_version)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    features.register_upload(target_feature)
    return {
        "status": "loaded",
        "runtime": sess.runtime,
        "input_shape": list(sess.input_shape),
        "model_version": sess.model_version,
        "active_feature": target_feature,
    }


@app.get("/api/decision-rules")
def list_decision_rules():
    """Decision Rules the active Feature can run, with their param schemas.

    The GUI renders controls generically from `params`, so a new rule needs no
    frontend code. Empty list when no Feature is active.
    """
    config = features.get_model_config()
    pipeline = features.current_pipeline()
    return {
        "active_feature": features.get_active_feature(),
        # The Class Catalog, so `class` params render as real dropdowns rather
        # than free-text class ids the operator has to guess.
        "labels": {str(k): v for k, v in (config.labels if config else {}).items()},
        "default_rule": (config.default_rule if config else None)
        or (pipeline.default_rule if pipeline else None),
        "rules": features.current_decision_rules(),
    }


def _active_rule_or_400(decision_rule: str):
    """Resolve the rule to calibrate, defaulting to the active pipeline's."""
    pipeline = features.current_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")
    try:
        return pipeline, pipeline._resolve_rule(decision_rule or None)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/skus/{sku}/calibration")
def get_sku_calibration(
    sku: str,
    decision_rule: str = "",
    db: Session = Depends(get_db),
):
    """A SKU's taught baseline for a Decision Rule, if it has one."""
    pipeline, rule = _active_rule_or_400(decision_rule)
    record = crud.get_calibration(db, sku, pipeline.feature, rule.name)
    if record is None:
        return {"sku_name": sku, "decision_rule": rule.name, "calibrated": False, "params": {}}
    return {
        "sku_name": sku,
        "decision_rule": rule.name,
        "calibrated": True,
        "params": json.loads(record.params),
        "sample_count": record.sample_count,
        "spread": record.spread,
        "updated_at": record.updated_at.isoformat(),
    }


@app.delete("/api/skus/{sku}/calibration")
def clear_sku_calibration(
    sku: str,
    decision_rule: str = "",
    db: Session = Depends(get_db),
):
    """Forget a baseline; the rule falls back to its sidecar defaults."""
    pipeline, rule = _active_rule_or_400(decision_rule)
    return {"cleared": crud.delete_calibration(db, sku, pipeline.feature, rule.name)}


@app.post("/api/calibrate")
async def calibrate(
    sku_name: str = Form(...),
    image_paths: str = Form(...),
    threshold: float = Form(...),
    decision_rule: str = Form(""),
    rule_params: str = Form(""),
    db: Session = Depends(get_db),
):
    """Teach a SKU's baseline from images known to be good.

    Artwork that is not symmetric about its own centre measures non-zero even
    when correctly placed, by an amount that differs per SKU. Teaching the
    median of that measurement lets one tolerance serve every SKU.

    The median is used rather than the mean so a single mislabelled sample
    cannot drag the baseline, and the spread is recorded because a tolerance
    below it would be measuring sample noise.
    """
    pipeline, rule = _active_rule_or_400(decision_rule)
    spec = getattr(rule, "calibration", None)
    if spec is None:
        raise HTTPException(
            status_code=422,
            detail=f"Decision Rule {rule.name!r} has no per-SKU baseline to teach.",
        )

    try:
        paths = json.loads(image_paths)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="image_paths must be valid JSON.")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(status_code=400, detail="Provide a non-empty list of image_paths.")

    try:
        overrides = json.loads(rule_params) if rule_params else None
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="rule_params must be valid JSON.")

    measured: list[float] = []
    skipped: list[str] = []
    for rel in paths:
        image_bytes = _resolve_data_path(rel).read_bytes()
        try:
            result = pipeline.infer(image_bytes, threshold, rule.name, overrides)
        except ClassNotFound as e:
            raise HTTPException(status_code=422, detail=str(e))
        except (ValueError, IndexError):
            skipped.append(rel)
            continue
        value = result.metrics.get(spec.metric)
        # A sample the rule could not measure (a part it failed to find) carries
        # no information about the baseline, so it is reported, not averaged in.
        if isinstance(value, (int, float)):
            measured.append(float(value))
        else:
            skipped.append(rel)

    if not measured:
        raise HTTPException(
            status_code=422,
            detail=(
                f"None of the {len(paths)} images produced a {spec.metric!r} "
                "measurement; check that the expected parts are detected."
            ),
        )

    values = sorted(measured)
    median = values[len(values) // 2] if len(values) % 2 else (
        values[len(values) // 2 - 1] + values[len(values) // 2]
    ) / 2
    spread = max(values) - min(values)

    record = crud.upsert_calibration(
        db,
        sku_name=sku_name,
        feature=pipeline.feature,
        decision_rule=rule.name,
        params={spec.param: round(median, 4)},
        sample_count=len(measured),
        spread=round(spread, 4),
    )
    return {
        "sku_name": sku_name,
        "decision_rule": rule.name,
        "params": json.loads(record.params),
        "sample_count": record.sample_count,
        "spread": record.spread,
        "skipped": skipped,
        "measured": [round(v, 4) for v in values],
    }


@app.post("/api/infer")
async def infer(
    sku_name: str = Form(...),
    threshold: float = Form(...),
    customer_name: str = Form(""),
    decision_rule: str = Form(""),
    rule_params: str = Form(""),
    image: UploadFile | None = File(None),
    image_path: str | None = Form(None),
    db: Session = Depends(get_db),
):
    pipeline = features.current_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")

    # Image source: an uploaded file, or a path into the on-disk dataset.
    if image is not None:
        image_bytes = await image.read()
    elif image_path:
        image_bytes = _resolve_data_path(image_path).read_bytes()
    else:
        raise HTTPException(status_code=400, detail="Provide an image file or image_path.")

    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file.")

    try:
        parsed_params = json.loads(rule_params) if rule_params else None
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="rule_params must be valid JSON.")
    if parsed_params is not None and not isinstance(parsed_params, dict):
        raise HTTPException(status_code=400, detail="rule_params must be a JSON object.")

    # Layering: sidecar defaults < SKU calibration < request. The GUI normally
    # sends the calibrated value already; this keeps a bare API call correct too.
    try:
        rule_name = pipeline._resolve_rule(decision_rule or None).name
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    taught = crud.calibration_params(db, sku_name, pipeline.feature, rule_name)
    effective_params = {**taught, **(parsed_params or {})} or None

    try:
        result = pipeline.infer(
            image_bytes, threshold, decision_rule or None, effective_params
        )
    except ClassNotFound as e:
        # A rule param names a class this model's Class Catalog lacks — a
        # misconfiguration, and one that would otherwise judge the wrong object.
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        # Unknown/incompatible Decision Rule, or an unexpected model output shape.
        raise HTTPException(status_code=422, detail=str(e))
    except IndexError:
        raise HTTPException(
            status_code=422,
            detail="Model output format unexpected for the active Feature.",
        )

    heatmap_bytes = result.images.get("heatmap")
    segmentation_bytes = result.images.get("segmentation")
    annotated_bytes = result.images.get("annotated")

    crud.create_inspection(
        db,
        sku_name=sku_name,
        feature=pipeline.feature,
        score=result.score,
        decision_rule=result.decision_rule,
        params=effective_params,
        metrics=result.metrics,
        threshold=threshold,
        verdict=result.verdict,
        model_version=pipeline.model_version,
        customer_name=customer_name,
        heatmap_image=heatmap_bytes,
        segmentation_image=segmentation_bytes,
    )

    def _b64(b: bytes | None) -> str | None:
        return base64.b64encode(b).decode() if b is not None else None

    rounded_score = round(result.score, 4) if result.score is not None else None
    return {
        "feature": pipeline.feature,
        "decision_rule": result.decision_rule,
        "verdict": result.verdict,
        "score": rounded_score,
        "score_label": result.score_label,
        "metrics": result.metrics,
        "reason": result.reason,
        "heatmap_image": _b64(heatmap_bytes),
        "segmentation_image": _b64(segmentation_bytes),
        "annotated_image": _b64(annotated_bytes),
        "overlay_image": _b64(result.images.get("overlay")),
        "detections": result.detections,
    }


@app.get("/api/stats")
def get_stats(
    sku_name: str | None = None,
    feature: str | None = None,
    decision_rule: str | None = None,
    db: Session = Depends(get_db),
):
    return crud.get_stats(
        db,
        sku_name=sku_name or None,
        feature=feature or None,
        decision_rule=decision_rule or None,
    )


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
