from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from .models import Inspection


def create_inspection(
    db: Session,
    sku_name: str,
    threshold: float,
    verdict: str,
    model_version: str,
    feature: str = "anomaly_detection",
    anomaly_score: float | None = None,
    heatmap_image: bytes | None = None,
    segmentation_image: bytes | None = None,
    customer_name: str = "",
) -> Inspection:
    record = Inspection(
        sku_name=sku_name,
        feature=feature,
        anomaly_score=anomaly_score,
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


def get_stats(db: Session, sku_name: str | None = None, feature: str | None = None) -> dict:
    def _scoped(q):
        if sku_name:
            q = q.filter(Inspection.sku_name == sku_name)
        if feature:
            q = q.filter(Inspection.feature == feature)
        return q

    total = _scoped(db.query(func.count(Inspection.id))).scalar() or 0
    ok_count = _scoped(
        db.query(func.count(Inspection.id)).filter(Inspection.verdict == "ok")
    ).scalar() or 0
    not_ok_count = _scoped(
        db.query(func.count(Inspection.id)).filter(Inspection.verdict == "not_ok")
    ).scalar() or 0
    pass_rate = round(ok_count / total * 100, 1) if total > 0 else 0.0
    return {"total": total, "ok": ok_count, "not_ok": not_ok_count, "pass_rate": pass_rate}


def get_inspections_in_range(
    db: Session,
    start: datetime,
    end: datetime,
    customer_name: str | None = None,
) -> list[Inspection]:
    q = db.query(Inspection).filter(
        Inspection.timestamp >= start, Inspection.timestamp <= end
    )
    if customer_name:
        q = q.filter(Inspection.customer_name == customer_name)
    return q.order_by(Inspection.timestamp).all()


def delete_all_inspections(db: Session) -> int:
    """Erase every inspection record. Returns the number of rows deleted."""
    deleted = db.query(Inspection).delete()
    db.commit()
    return deleted
