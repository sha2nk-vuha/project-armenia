from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Session
from .models import Inspection


def create_inspection(
    db: Session,
    sku_name: str,
    anomaly_score: float,
    threshold: float,
    verdict: str,
    model_version: str,
    heatmap_image: bytes,
    segmentation_image: bytes,
    customer_name: str = "",
) -> Inspection:
    record = Inspection(
        sku_name=sku_name,
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


def get_stats(db: Session) -> dict:
    total = db.query(func.count(Inspection.id)).scalar() or 0
    ok_count = (
        db.query(func.count(Inspection.id))
        .filter(Inspection.verdict == "ok")
        .scalar() or 0
    )
    not_ok_count = (
        db.query(func.count(Inspection.id))
        .filter(Inspection.verdict == "not_ok")
        .scalar() or 0
    )
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
