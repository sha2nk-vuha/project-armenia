from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime, timezone


class Base(DeclarativeBase):
    pass


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    sku_name = Column(String, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    verdict = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    customer_name = Column(String, nullable=False, server_default="", default="")
    heatmap_image = Column(LargeBinary, nullable=False)
    segmentation_image = Column(LargeBinary, nullable=False)
