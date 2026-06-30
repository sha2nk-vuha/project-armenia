from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from .models import Base

_DB_PATH = Path(__file__).parent.parent / "inspections.db"
DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    # Lightweight migration: add customer_name if it was not present in the
    # existing DB (create_all will not ALTER a table that already exists).
    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(inspections)"))}
        if "customer_name" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE inspections"
                    " ADD COLUMN customer_name VARCHAR NOT NULL DEFAULT ''"
                )
            )
            conn.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
