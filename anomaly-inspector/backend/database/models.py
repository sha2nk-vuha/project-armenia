from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime, timezone


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for the inspector's schema."""


class Inspection(Base):
    """One recorded inspection: which Feature ran, what it decided, and why.

    A row is immutable evidence for stats and customer reports; per-SKU
    configuration lives in SkuCalibration instead.
    """

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


class SkuCalibration(Base):
    """A per-SKU baseline taught from known-good samples, per Decision Rule.

    Kept separate from `inspections` because it is configuration, not a record
    of an inspection: it is read on every run and rewritten on every teach.
    Scoped by (sku, feature, rule) because the same SKU inspected under a
    different rule needs a different baseline.
    """

    __tablename__ = "sku_calibrations"
    __table_args__ = (
        UniqueConstraint("sku_name", "feature", "decision_rule", name="uq_sku_calibration"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku_name = Column(String, nullable=False)
    feature = Column(String, nullable=False)
    decision_rule = Column(String, nullable=False)
    # Taught param values as JSON, so a rule can grow its baseline without a
    # migration; the rule's `calibration` declaration says what belongs here.
    params = Column(Text, nullable=False)
    sample_count = Column(Integer, nullable=False, default=0)
    # Spread of the taught samples. A tolerance must sit above this or it is
    # measuring sample noise, so it is recorded rather than discarded.
    spread = Column(Float, nullable=True)
    updated_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
