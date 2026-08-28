import json
from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from .models import Inspection, SkuCalibration
from inference.verdict import NOT_OK, OK


def create_inspection(
    db: Session,
    sku_name: str,
    threshold: float,
    verdict: str,
    model_version: str,
    feature: str = "anomaly_detection",
    score: float | None = None,
    decision_rule: str | None = None,
    params: dict | None = None,
    metrics: dict | None = None,
    heatmap_image: bytes | None = None,
    segmentation_image: bytes | None = None,
    customer_name: str = "",
) -> Inspection:
    """Record one inspection.

    `params` and `metrics` are JSON-encoded: a customer-facing report has to be
    able to state which Decision Rule ran, under what tolerances, and what it
    measured (see docs/adr/0005).

    Args:
        db: Database session (committed by this function).
        sku_name: SKU under inspection.
        threshold: Threshold in effect for this run.
        verdict: Verdict string ("ok" | "not_ok") from inference.verdict.
        model_version: Version of the model that produced the Verdict.
        feature: Feature that produced the record.
        score: The rule's primary scalar; None for rules without one.
        decision_rule: Name of the Decision Rule that judged the image.
        params: Resolved rule params to persist as JSON.
        metrics: Rule metrics to persist as JSON.
        heatmap_image: JPEG bytes, or None when the Feature emits none.
        segmentation_image: JPEG bytes, or None when the Feature emits none.
        customer_name: Customer the inspection was run for.

    Returns:
        The persisted Inspection row (refreshed).
    """
    record = Inspection(
        sku_name=sku_name,
        feature=feature,
        score=score,
        decision_rule=decision_rule,
        params=json.dumps(params) if params else None,
        metrics=json.dumps(metrics) if metrics else None,
        threshold=threshold,
        verdict=verdict,
        model_version=model_version,
        customer_name=customer_name,
        heatmap_image=heatmap_image,
        segmentation_image=segmentation_image,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_stats(
    db: Session,
    sku_name: str | None = None,
    feature: str | None = None,
    decision_rule: str | None = None,
) -> dict:
    """Pass-rate stats, scoped by SKU / Feature / Decision Rule.

    Scoping by Decision Rule matters for the same reason ADR 0004 scoped by
    Feature: two rules on the *same* Feature ("logo is centred" vs "logo is
    present") produce OKs that mean different things, so pooling them yields a
    pass rate nobody can act on.

    Args:
        db: Database session.
        sku_name: Restrict to this SKU when given.
        feature: Restrict to this Feature when given.
        decision_rule: Restrict to this Decision Rule when given.

    Returns:
        Dict with "total", "ok", "not_ok" counts and "pass_rate".
    """

    def _scoped(q):
        if sku_name:
            q = q.filter(Inspection.sku_name == sku_name)
        if feature:
            q = q.filter(Inspection.feature == feature)
        if decision_rule:
            q = q.filter(Inspection.decision_rule == decision_rule)
        return q

    total = _scoped(db.query(func.count(Inspection.id))).scalar() or 0
    ok_count = _scoped(
        db.query(func.count(Inspection.id)).filter(Inspection.verdict == OK)
    ).scalar() or 0
    not_ok_count = _scoped(
        db.query(func.count(Inspection.id)).filter(Inspection.verdict == NOT_OK)
    ).scalar() or 0
    pass_rate = round(ok_count / total * 100, 1) if total > 0 else 0.0
    return {"total": total, "ok": ok_count, "not_ok": not_ok_count, "pass_rate": pass_rate}


def get_inspections_in_range(
    db: Session,
    start: datetime,
    end: datetime,
    customer_name: str | None = None,
) -> list[Inspection]:
    """Inspections in [start, end], oldest first.

    Args:
        db: Database session.
        start: Range start (inclusive).
        end: Range end (inclusive).
        customer_name: Restrict to this customer when given.

    Returns:
        Inspection rows ordered by timestamp ascending.
    """
    q = db.query(Inspection).filter(
        Inspection.timestamp >= start, Inspection.timestamp <= end
    )
    if customer_name:
        q = q.filter(Inspection.customer_name == customer_name)
    return q.order_by(Inspection.timestamp).all()


def delete_all_inspections(db: Session) -> int:
    """Erase every inspection record.

    Args:
        db: Database session (committed by this function).

    Returns:
        Number of rows deleted.
    """
    deleted = db.query(Inspection).delete()
    db.commit()
    return deleted


# ── Per-SKU calibration ─────────────────────────────────────────────────────


def get_calibration(
    db: Session, sku_name: str, feature: str, decision_rule: str
) -> SkuCalibration | None:
    """The SKU's taught baseline for one Decision Rule.

    Args:
        db: Database session.
        sku_name: SKU the baseline belongs to.
        feature: Feature it was taught under.
        decision_rule: Decision Rule it was taught for.

    Returns:
        The SkuCalibration row, or None if never taught.
    """
    return (
        db.query(SkuCalibration)
        .filter(
            SkuCalibration.sku_name == sku_name,
            SkuCalibration.feature == feature,
            SkuCalibration.decision_rule == decision_rule,
        )
        .one_or_none()
    )


def calibration_params(
    db: Session, sku_name: str, feature: str, decision_rule: str
) -> dict:
    """The taught params for a SKU.

    Args:
        db: Database session.
        sku_name: SKU the baseline belongs to.
        feature: Feature it was taught under.
        decision_rule: Decision Rule it was taught for.

    Returns:
        Parsed params dict; empty dict when never taught.
    """
    record = get_calibration(db, sku_name, feature, decision_rule)
    return json.loads(record.params) if record else {}


def upsert_calibration(
    db: Session,
    sku_name: str,
    feature: str,
    decision_rule: str,
    params: dict,
    sample_count: int,
    spread: float | None = None,
) -> SkuCalibration:
    """Store (or replace) a SKU's taught baseline; teaching again overwrites.

    Args:
        db: Database session (committed by this function).
        sku_name: SKU to teach.
        feature: Feature the baseline applies to.
        decision_rule: Decision Rule the baseline applies to.
        params: Taught param values to store as JSON.
        sample_count: Number of samples that produced the baseline.
        spread: Sample spread, recorded so tolerances can be sanity-checked.

    Returns:
        The upserted SkuCalibration row (refreshed).
    """
    record = get_calibration(db, sku_name, feature, decision_rule)
    if record is None:
        record = SkuCalibration(
            sku_name=sku_name, feature=feature, decision_rule=decision_rule
        )
        db.add(record)
    record.params = json.dumps(params)
    record.sample_count = sample_count
    record.spread = spread
    record.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(record)
    return record


def delete_calibration(
    db: Session, sku_name: str, feature: str, decision_rule: str
) -> bool:
    """Forget a SKU's baseline; the rule falls back to its sidecar defaults.

    Args:
        db: Database session (committed by this function).
        sku_name: SKU whose baseline to forget.
        feature: Feature the baseline applies to.
        decision_rule: Decision Rule the baseline applies to.

    Returns:
        True when a baseline existed and was deleted, False otherwise.
    """
    record = get_calibration(db, sku_name, feature, decision_rule)
    if record is None:
        return False
    db.delete(record)
    db.commit()
    return True
