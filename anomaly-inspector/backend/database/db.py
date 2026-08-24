from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from .models import Base

_DB_PATH = Path(__file__).parent.parent / "inspections.db"
DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    # Upgrade a legacy table (NOT NULL score/images) before create_all so
    # Presence/Absence records with NULL score/images can be stored. The rebuild
    # is split by SQLite's DDL limits, so make it crash-safe: rename aside, let
    # create_all build the new schema, then copy rows. Recovery is idempotent —
    # a leftover `inspections_legacy` from an interrupted run is resumed here.
    with engine.begin() as conn:
        _rename_legacy_if_needed(conn)
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        _copy_from_legacy_if_present(conn)
        _add_missing_columns(conn)


def _table_columns(conn, table: str) -> list[str]:
    return [row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))]


def _rename_legacy_if_needed(conn) -> None:
    """Rename a pre-Feature `inspections` (NOT NULL score/images) out of the way.

    No-op when there is nothing to migrate, or when a prior interrupted run
    already left an `inspections_legacy` to be resumed by the copy step.
    """
    if _table_columns(conn, "inspections_legacy"):
        return  # resume in progress — leave it for _copy_from_legacy_if_present
    info = list(conn.execute(text("PRAGMA table_info(inspections)")))
    if not info:
        return
    notnull = {row[1]: row[3] for row in info}
    if not any(
        notnull.get(col) for col in ("anomaly_score", "heatmap_image", "segmentation_image")
    ):
        return
    conn.execute(text("ALTER TABLE inspections RENAME TO inspections_legacy"))


def _copy_from_legacy_if_present(conn) -> None:
    """Copy rows from a renamed legacy table into the fresh table, then drop it.

    Copies only columns common to both tables (the new `feature` column is
    back-filled by its default). `INSERT OR IGNORE` keeps this safe to re-run if
    a previous copy was interrupted before the drop.
    """
    legacy_cols = _table_columns(conn, "inspections_legacy")
    if not legacy_cols:
        return
    new_cols = set(_table_columns(conn, "inspections"))
    common = [c for c in legacy_cols if c in new_cols]
    cols = ", ".join(common)
    conn.execute(text(
        f"INSERT OR IGNORE INTO inspections ({cols}) SELECT {cols} FROM inspections_legacy"
    ))
    conn.execute(text("DROP TABLE inspections_legacy"))


def _add_missing_columns(conn) -> None:
    """Add columns to an already-nullable table that lacks them (forward-compat).

    create_all will not ALTER an existing table, so add `customer_name` and
    `feature` when a prior-version DB is missing them.
    """
    cols = {row[1] for row in conn.execute(text("PRAGMA table_info(inspections)"))}
    if "customer_name" not in cols:
        conn.execute(text(
            "ALTER TABLE inspections ADD COLUMN customer_name VARCHAR NOT NULL DEFAULT ''"
        ))
    if "feature" not in cols:
        conn.execute(text(
            "ALTER TABLE inspections"
            " ADD COLUMN feature VARCHAR NOT NULL DEFAULT 'anomaly_detection'"
        ))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
