from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary, Text
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime, timezone


class Base(DeclarativeBase):
    pass


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    sku_name = Column(String, nullable=False)
    # The active Feature that produced this record (see docs/adr/0004).
    feature = Column(
        String, nullable=False, server_default="anomaly_detection", default="anomaly_detection"
    )
    # The Decision Rule's primary scalar (anomaly score, offset ratio, ...);
    # NULL for rules without a single number. Named generically because the
    # rule that produced it decides what it means (see docs/adr/0005).
    score = Column(Float, nullable=True)
    threshold = Column(Float, nullable=False)
    verdict = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    customer_name = Column(String, nullable=False, server_default="", default="")
    # Which Decision Rule produced the Verdict, the params it ran with, and what
    # it measured. Stored as JSON text so a customer-facing report can explain
    # why a part failed and under what tolerances (see docs/adr/0005).
    decision_rule = Column(String, nullable=True)
    params = Column(Text, nullable=True)
    metrics = Column(Text, nullable=True)
    # Feature-specific visualizations; NULL when a Feature does not produce them.
    heatmap_image = Column(LargeBinary, nullable=True)
    segmentation_image = Column(LargeBinary, nullable=True)
