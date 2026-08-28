import asyncio
import base64
import io
import json
import logging
from datetime import datetime, timedelta
from typing import Annotated

from config import (
    APP_VERSION,
    CASCADE_FEATURE,
    DEFAULT_FEATURE,
    FEATURES,
)
from database import crud
from database.db import get_db, init_db
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from inference import features
from inference.decision import ClassNotFound
from PIL import Image
from pydantic import ValidationError
from reports.pdf_generator import generate_report
from schemas import CascadeSpec
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
    """Initialise the database and select the default Feature."""
    init_db()
    # No model is auto-loaded (docs/adr/0007): the operator uploads one per
    # Feature. Only the default *active* Feature is set so the UI has a starting
    # selection; inference before an upload returns a clear "no model" error.
    try:
        features.activate(DEFAULT_FEATURE)
    except ValueError as e:
        logging.warning("Could not set default Feature '%s': %s", DEFAULT_FEATURE, e)


def _feature_catalog() -> list[dict]:
    """The selectable Features and their UI metadata.

    Returns:
        One dict per Feature with name, operator-facing label, and the
        Threshold slider's label for that Feature.
    """
    return [
        {
            "name": name,
            "label": spec["label"],
            "threshold_label": spec["threshold_label"],
        }
        for name, spec in FEATURES.items()
    ]


@app.get("/api/status")
def get_status() -> dict:
    """Active Feature, per-Feature model state, and the selectable catalog.

    Returns:
        Dict with active_feature, features catalog, loaded_models, and
        model_loaded plus active-model details when one is loaded.
    """
    catalog = _feature_catalog()
    # Per-Feature: which Features have an uploaded model. Drives both the single
    # "Model" indicator (active Feature) and the per-stage cascade indicators.
    loaded = features.loaded_models()
    sess = features.active_model()
    base = {
        "active_feature": features.get_active_feature(),
        "features": catalog,
        "loaded_models": loaded,
    }
    if sess is None:
        return {**base, "model_loaded": False}
    return {
        **base,
        "model_loaded": True,
        "model_version": sess.model_version,
        "runtime": sess.runtime,
        "input_shape": list(sess.input_shape),
    }


@app.post("/api/feature")
def set_feature(feature: str = Form(...)) -> dict:
    """Activate an inspection Feature.

    Activation loads no model (docs/adr/0007); the operator uploads one per
    Feature. The response reports whether a model is already uploaded so the
    UI shows the right state immediately.

    Args:
        feature: Feature name to activate (may be the Cascade composite).

    Returns:
        Confirmation with the new active_feature and model_loaded state.

    Raises:
        HTTPException: 400 for an unknown Feature; 422 if activation is
            rejected by the Feature registry.
    """
    if feature not in FEATURES:
        raise HTTPException(status_code=400, detail=f"Unknown feature: {feature!r}")
    try:
        features.activate(feature)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    # Activation loads no model (docs/adr/0007). Report whether one is already
    # uploaded for this Feature so the UI shows the right state immediately.
    model = features.loaded_models().get(feature)
    return {
        "status": "activated",
        "active_feature": feature,
        "model_loaded": model is not None,
        "model": model,
    }


@app.post("/api/load-model")
async def load_model(
    model_file: UploadFile = File(...),
    model_version: str = Form(...),
    feature: str = Form(""),
    sidecar_file: UploadFile | None = File(None),
) -> dict:
    """Upload a model for a specific Feature (its own store slot).

    `feature` names which Feature the model is for -- required in cascade mode,
    where the active Feature is the composite. An optional `sidecar_file` (the
    model's JSON) supplies its Class Catalog and rule defaults; without it the
    bundled sidecar template is used and the input size is read from the model.

    Args:
        model_file: Uploaded .onnx file.
        model_version: Version string to record for reporting.
        feature: Target Feature; defaults to the active one.
        sidecar_file: Optional sidecar JSON with Class Catalog and defaults.

    Returns:
        Confirmation with runtime, input_shape, model_version, and which
        Feature now holds this model.

    Raises:
        HTTPException: 400 when the target is not a concrete Feature; 422
            for invalid sidecar JSON or an unloadable/non-.onnx payload.
    """
    target_feature = feature or features.get_active_feature() or DEFAULT_FEATURE
    if target_feature not in FEATURES or target_feature == CASCADE_FEATURE:
        raise HTTPException(
            status_code=400,
            detail=f"Upload a model for a concrete Feature, not {target_feature!r}.",
        )
    if sidecar_file is not None:
        model_bytes, sidecar_bytes = await asyncio.gather(
            model_file.read(), sidecar_file.read()
        )
        try:
            sidecar = json.loads(sidecar_bytes)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="Sidecar file is not valid JSON.")
    else:
        model_bytes = await model_file.read()
        sidecar = None

    try:
        sess = features.upload_model(
            target_feature, model_bytes, model_file.filename or "model.onnx",
            model_version, sidecar,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "status": "loaded",
        "runtime": sess.runtime,
        "input_shape": list(sess.input_shape),
        "model_version": sess.model_version,
        "active_feature": features.get_active_feature(),
        "feature": target_feature,
    }


@app.get("/api/decision-rules")
def list_decision_rules() -> dict:
    """Decision Rules the active Feature can run, with their param schemas.

    The GUI renders controls generically from `params`, so a new rule needs no
    frontend code.

    Returns:
        Dict with active_feature, the Class Catalog as label strings, the
        default Decision Rule name, and `rules` (empty when no Feature is
        active).
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


def _json_object_form(raw: str, name: str) -> dict | None:
    """Parse a JSON-object form field; None when the field is absent/empty.

    The one boundary check for JSON-in-a-form-field payloads: invalid JSON or
    a non-object payload is a client error naming the offending field.

    Args:
        raw: The form field's raw string ("" counts as absent).
        name: Field name used in error details.

    Returns:
        Parsed dict, or None when the field was absent/empty.

    Raises:
        HTTPException: 400 for invalid JSON or a non-object payload.
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"{name} must be valid JSON.")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail=f"{name} must be a JSON object.")
    return parsed


def _active_rule_or_400(decision_rule: str):
    """Resolve the rule to calibrate against.

    Args:
        decision_rule: Requested Decision Rule name; empty selects the
            active pipeline's default.

    Returns:
        Tuple of (pipeline, resolved DecisionRule).

    Raises:
        HTTPException: 400 when no model is loaded; 422 for an unknown or
            incompatible rule name.
    """
    pipeline = features.current_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")
    try:
        return pipeline, pipeline.resolve_rule(decision_rule or None)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/skus/{sku}/calibration")
def get_sku_calibration(
    sku: str,
    decision_rule: str = "",
    db: Session = Depends(get_db),
) -> dict:
    """A SKU's taught baseline for a Decision Rule, if it has one.

    Args:
        sku: SKU whose baseline to read.
        decision_rule: Rule name; empty selects the active default.

    Returns:
        Dict with "calibrated" plus params and teaching metadata when a
        baseline exists.

    Raises:
        HTTPException: Via _active_rule_or_400 when no model is loaded or
            the rule cannot run on the active Feature.
    """
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
) -> dict:
    """Forget a baseline; the rule falls back to its sidecar defaults.

    Args:
        sku: SKU whose baseline to forget.
        decision_rule: Rule name; empty selects the active default.

    Returns:
        Dict with "cleared": whether a baseline existed.

    Raises:
        HTTPException: Via _active_rule_or_400 when no model is loaded or
            the rule cannot run on the active Feature.
    """
    pipeline, rule = _active_rule_or_400(decision_rule)
    return {"cleared": crud.delete_calibration(db, sku, pipeline.feature, rule.name)}


@app.post("/api/calibrate")
async def calibrate(
    sku_name: Annotated[str, Form(min_length=1, max_length=128)],
    threshold: Annotated[float, Form(ge=0.0, le=1.0)],
    images: list[UploadFile] = File(default=[]),
    decision_rule: str = Form(""),
    rule_params: str = Form(""),
    db: Session = Depends(get_db),
) -> dict:
    """Teach a SKU's baseline from images known to be good.

    Artwork that is not symmetric about its own centre measures non-zero even
    when correctly placed, by an amount that differs per SKU. Teaching the
    median of that measurement lets one tolerance serve every SKU.

    The median is used rather than the mean so a single mislabelled sample
    cannot drag the baseline, and the spread is recorded because a tolerance
    below it would be measuring sample noise.

    Args:
        sku_name: SKU to teach.
        threshold: Threshold in effect while measuring samples.
        images: Known-good sample images whose metric is the baseline.
        decision_rule: Rule to teach; empty selects the active default.
        rule_params: Optional JSON-object overrides for the run (e.g. class
            references that must be resolved).

    Returns:
        Dict with sku_name, decision_rule, taught params, sample_count,
        spread, skipped filenames, and the sorted per-sample measurements.

    Raises:
        HTTPException: 400 when no images are provided; 422 when the rule
            has no baseline, the rule_params are malformed, a class cannot
            be resolved, or no samples produced a usable measurement.
    """
    pipeline, rule = _active_rule_or_400(decision_rule)
    spec = getattr(rule, "calibration", None)
    if spec is None:
        raise HTTPException(
            status_code=422,
            detail=f"Decision Rule {rule.name!r} has no per-SKU baseline to teach.",
        )

    # Samples arrive as uploaded files (a folder browsed in the GUI lives in
    # the browser, not on disk). Read concurrently with bounded parallelism
    # into a (label, bytes) list.
    semaphore = asyncio.Semaphore(10)

    async def _read_limited(upload: UploadFile) -> bytes:
        async with semaphore:
            return await upload.read()

    samples: list[tuple[str, bytes]] = [
        (upload.filename or "uploaded", data)
        for upload, data in zip(
            images, await asyncio.gather(*(_read_limited(u) for u in images))
        )
    ]

    if not samples:
        raise HTTPException(status_code=400, detail="Provide at least one calibration image.")

    overrides = _json_object_form(rule_params, "rule_params")

    measured: list[float] = []
    skipped: list[str] = []
    for label, image_bytes in samples:
        try:
            result = pipeline.infer(image_bytes, threshold, rule.name, overrides)
        except ClassNotFound as e:
            raise HTTPException(status_code=422, detail=str(e))
        except (ValueError, IndexError):
            skipped.append(label)
            continue
        value = result.metrics.get(spec.metric)
        # A sample the rule could not measure (a part it failed to find) carries
        # no information about the baseline, so it is reported, not averaged in.
        if not isinstance(value, (int, float)):
            skipped.append(label)
            continue
        measured.append(float(value))

    if not measured:
        raise HTTPException(
            status_code=422,
            detail=(
                f"None of the {len(samples)} images produced a {spec.metric!r} "
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


@app.get("/api/cascade/options")
def cascade_options() -> dict:
    """Everything the cascade stage builder needs, without loading any ONNX.

    Per cascadable Feature: its compatible Decision Rules (schema + sidecar
    defaults) and its Class Catalog; plus the available Combinators.

    Returns:
        Dict with "features" (compatible rules + catalog per Feature) and
        "combinators" (each Combinator's UI metadata).
    """
    return features.cascade_options()


def _build_cascade_or_400(spec: dict, sku_name: str, db: Session):
    """Assemble a CascadePipeline, injecting each stage's SKU calibration.

    Layering per stage matches single-Feature mode: sidecar < SKU calibration <
    request params. Configuration problems (bad spec, unknown feature/rule,
    incompatible rule, missing model) surface as 4xx rather than silent NOKs.

    Args:
        spec: Parsed cascade spec with "stages" already validated; each
            stage's "params" is mutated in place to inject any taught
            calibration.
        sku_name: SKU whose calibration to inject per stage.
        db: Database session for calibration lookups.

    Returns:
        A CascadePipeline ready to run.

    Raises:
        HTTPException: 503 for a missing model file; 422 for an invalid
            cascade configuration.
    """
    for stage in spec["stages"]:
        feature, rule = stage.get("feature"), stage.get("rule")
        if not (feature and rule):
            continue
        taught = crud.calibration_params(db, sku_name, feature, rule)
        if not taught:
            continue
        stage["params"] = {**taught, **(stage.get("params") or {})}
    try:
        return features.build_cascade_pipeline(spec)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid cascade: {e}")


def _acquire_cascade_pipeline(cascade_spec: str, sku_name: str, db: Session):
    """Validate the cascade spec and return (pipeline, spec-dict-to-record).

    Args:
        cascade_spec: JSON object string describing the cascade stages.
        sku_name: SKU whose calibration to inject per stage.
        db: Database session.

    Returns:
        Tuple of (CascadePipeline, raw spec dict as stored for auditing).

    Raises:
        HTTPException: 400 when cascade_spec is absent/invalid; 422 for a
            structurally invalid cascade (layered through _build_cascade_or_400).
    """
    raw_spec = _json_object_form(cascade_spec, "cascade_spec")
    if raw_spec is None:
        raise HTTPException(status_code=400, detail="Cascade mode needs a cascade_spec.")
    try:
        # schemas.CascadeSpec is the wire contract for the spec's shape;
        # semantic checks (known Feature, loaded model, compatible rules)
        # stay in features.build_cascade_pipeline.
        spec = CascadeSpec.model_validate(raw_spec).model_dump()
    except ValidationError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid cascade: {e.errors()[0]['msg']}"
        )
    pipeline = _build_cascade_or_400(spec, sku_name, db)
    return pipeline, raw_spec


def _acquire_single_pipeline(
    db: Session, sku_name: str, decision_rule: str, rule_params: str
):
    """Return (pipeline, params-to-record) for single-Feature mode, or 4xx.

    Args:
        db: Database session for calibration lookups and for surfacing
            HTTP errors.
        sku_name: SKU whose calibration to layer onto defaults.
        decision_rule: Requested Decision Rule name; empty selects default.
        rule_params: JSON-object string with caller overrides; empty means
            none.

    Returns:
        Tuple of (pipeline, params dict to persist as JSON, or None).

    Raises:
        HTTPException: 400 for malformed rule_params; 422 for an unknown or
            incompatible Decision Rule, or when no model is loaded.
    """
    pipeline = features.current_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")
    parsed_params = _json_object_form(rule_params, "rule_params")
    # Layering: sidecar defaults < SKU calibration < request. The GUI normally
    # sends the calibrated value already; this keeps a bare API call correct too.
    try:
        rule_name = pipeline.resolve_rule(decision_rule or None).name
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    taught = crud.calibration_params(db, sku_name, pipeline.feature, rule_name)
    record_params = {**taught, **(parsed_params or {})} or None
    return pipeline, record_params


def _run_pipeline(pipeline, image_bytes: bytes, threshold: float,
                  decision_rule: str | None, record_params: dict | None):
    """Execute one inspection, mapping domain errors onto HTTP semantics.

    Args:
        pipeline: Active pipeline to run (single-Feature or Cascade).
        image_bytes: Encoded image to inspect.
        threshold: Threshold forwarded to the decision logic.
        decision_rule: Decision Rule name, or None for the pipeline default.
        record_params: Resolved rule params to forward; None skips overrides.

    Returns:
        The pipeline's InferenceResult.

    Raises:
        HTTPException: 422 when a rule param names an unknown class, when
            params are invalid, or when the model output format does not
            match the active Feature.
    """
    try:
        return pipeline.infer(image_bytes, threshold, decision_rule, record_params)
    except ClassNotFound as e:
        # A rule param names a class the model's Class Catalog lacks — a
        # misconfiguration, and one that would otherwise judge the wrong object.
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except IndexError:
        raise HTTPException(
            status_code=422,
            detail="Model output format unexpected for the active Feature.",
        )


def _verify_image(image_bytes: bytes) -> None:
    """Reject bytes PIL cannot open as an image.

    Args:
        image_bytes: Raw file bytes; must decode as a supported image.

    Raises:
        HTTPException: 400 when the bytes cannot be opened as an image.
    """
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except (OSError, ValueError, SyntaxError):
        # PIL signals a bad image via OSError (UnidentifiedImageError is an
        # OSError subclass); truncated data surfaces as ValueError/SyntaxError.
        # Anything else is a real bug and should surface as a 500, not hide
        # behind a 400.
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file.")


def _b64(b: bytes | None) -> str | None:
    """Base64-encode image bytes for the JSON response.

    Args:
        b: JPEG bytes, or None (e.g. a Feature that did not emit one).

    Returns:
        Base64 string, or None when the input was None.
    """
    return base64.b64encode(b).decode() if b is not None else None


def _infer_payload(pipeline, result) -> dict:
    """The /api/infer response body: Verdict, images, and per-stage breakdown.

    Args:
        pipeline: Pipeline (single-Feature or Cascade) that produced the
            result; its `feature` names the response.
        result: InferenceResult from the run, already decided and
            annotated.

    Returns:
        JSON-serialisable dict with verdict, score/reason/metrics, base64
        images, and the ordered per-stage breakdown for cascades.
    """
    return {
        "feature": pipeline.feature,
        "decision_rule": result.decision_rule,
        "verdict": result.verdict,
        "score": round(result.score, 4) if result.score is not None else None,
        "score_label": result.score_label,
        "metrics": result.metrics,
        "reason": result.reason,
        "heatmap_image": _b64(result.images.get("heatmap")),
        "segmentation_image": _b64(result.images.get("segmentation")),
        "annotated_image": _b64(result.images.get("annotated")),
        "overlay_image": _b64(result.images.get("overlay")),
        "detections": result.detections,
        # Cascade: the ordered per-stage breakdown, each with its own image.
        "stages": [
            {
                "feature": st.feature,
                "decision_rule": st.decision_rule,
                "verdict": st.verdict,
                "evaluated": st.evaluated,
                "score": round(st.score, 4) if st.score is not None else None,
                "score_label": st.score_label,
                "reason": st.reason,
                "images": [
                    {"label": label, "image": _b64(img)} for label, img in st.images
                ],
                "detections": st.detections,
            }
            for st in result.stages
        ],
    }


@app.post("/api/infer")
async def infer(
    sku_name: Annotated[str, Form(min_length=1, max_length=128)],
    threshold: Annotated[float, Form(ge=0.0, le=1.0)],
    customer_name: str = Form(""),
    decision_rule: str = Form(""),
    rule_params: str = Form(""),
    cascade_spec: str = Form(""),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    """Run one inspection with the active Feature and record its Verdict.

    In single-Feature mode the active pipeline runs the requested Decision
    Rule; when the Cascade Feature is active, `cascade_spec` names the ordered
    stages and Combinator instead and each stage runs on its own model.

    Args:
        sku_name: SKU under inspection (scopes calibration + stats).
        threshold: Threshold cut-off for this run.
        customer_name: Customer the inspection was run for.
        decision_rule: Requested Decision Rule; empty selects the default
            (ignored in cascade mode).
        rule_params: JSON-object overrides for single-Feature mode; empty
            means none.
        cascade_spec: JSON-object describing the cascade stages; required
            when the Cascade Feature is active.
        image: Uploaded image file.

    Returns:
        Verdict payload (single-Feature or cascade) and persisted via
        create_inspection.

    Raises:
        HTTPException: 400 for a bad image; 4xx from the acquire helpers for
            missing/invalid specs, rules, or an unloaded model.
    """
    is_cascade = features.get_active_feature() == CASCADE_FEATURE

    # Image source: an uploaded file (folder images live in the browser).
    image_bytes = await image.read()
    _verify_image(image_bytes)

    if is_cascade:
        pipeline, record_params = _acquire_cascade_pipeline(cascade_spec, sku_name, db)
        result = _run_pipeline(pipeline, image_bytes, threshold, None, None)
    else:
        pipeline, record_params = _acquire_single_pipeline(
            db, sku_name, decision_rule, rule_params
        )
        result = _run_pipeline(pipeline, image_bytes, threshold, decision_rule or None, record_params)

    crud.create_inspection(
        db,
        sku_name=sku_name,
        feature=pipeline.feature,
        score=result.score,
        decision_rule=result.decision_rule,
        params=record_params,
        metrics=result.metrics,
        threshold=threshold,
        verdict=result.verdict,
        model_version=pipeline.model_version,
        customer_name=customer_name,
        heatmap_image=result.images.get("heatmap"),
        segmentation_image=result.images.get("segmentation"),
    )

    return _infer_payload(pipeline, result)


@app.get("/api/stats")
def get_stats(
    sku_name: str | None = None,
    feature: str | None = None,
    decision_rule: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    """Pass-rate statistics, optionally scoped by SKU / Feature / Decision Rule.

    Args:
        sku_name: Restrict to this SKU when given.
        feature: Restrict to this Feature when given.
        decision_rule: Restrict to this Decision Rule when given.

    Returns:
        Dict with "total", "ok", "not_ok", and "pass_rate".
    """
    return crud.get_stats(
        db,
        sku_name=sku_name or None,
        feature=feature or None,
        decision_rule=decision_rule or None,
    )


@app.post("/api/reset")
def reset_database(db: Session = Depends(get_db)) -> dict:
    """Erase all recorded inspections so a fresh demo can start from a clean slate.

    Args:
        db: Database session (provided by FastAPI).

    Returns:
        Dict with status "reset" and the number of deleted rows.
    """
    deleted = crud.delete_all_inspections(db)
    return {"status": "reset", "deleted": deleted}



def _report_row(i) -> dict:
    """Flat row-dict for the report layer.

    Args:
        i: Inspection ORM row from the database.

    Returns:
        Dict with the columns the PDF layer needs; metrics JSON is left
        unparsed there (best-effort decode with fallbacks).
    """
    return {
        "timestamp": i.timestamp,
        "sku_name": i.sku_name,
        "feature": i.feature,
        "score": i.score,
        "threshold": i.threshold,
        "verdict": i.verdict,
        "model_version": i.model_version,
        "decision_rule": i.decision_rule,
        "metrics": i.metrics,
    }


@app.post("/api/report")
def get_report(
    start_date: str = Form(...),
    end_date: str = Form(...),
    customer_name: str = Form(""),
    db: Session = Depends(get_db),
) -> Response:
    """Build the customer-facing PDF report for a date range.

    Args:
        start_date: ISO 8601 date string (inclusive start of range).
        end_date: ISO 8601 date string (inclusive end of range).
        customer_name: Optional customer to filter by.

    Returns:
        PDF bytes with Content-Disposition header.

    Raises:
        HTTPException: 400 for an invalid date; 404 when the filtered range
            has no inspections.
    """
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

    report_rows = [_report_row(i) for i in inspections]

    pdf_bytes = generate_report(
        start_date=start,
        end_date=end,
        inspections=report_rows,
        customer_name=customer_name,
        app_version=APP_VERSION,
    )

    filename = f"inspection-report-{start_date[:10]}-to-{end_date[:10]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
