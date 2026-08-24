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


# Columns renamed across schema versions: {old name: new name}. The legacy
# copy consults this so a rename does not silently drop the column's data.
_RENAMED_COLUMNS = {"anomaly_score": "score"}


def _copy_from_legacy_if_present(conn) -> None:
    """Copy rows from a renamed legacy table into the fresh table, then drop it.

    Copies columns common to both tables, mapping any renamed column to its new
    name (new columns like `feature` are back-filled by their defaults).
    `INSERT OR IGNORE` keeps this safe to re-run if a previous copy was
    interrupted before the drop.
    """
    legacy_cols = _table_columns(conn, "inspections_legacy")
    if not legacy_cols:
        return
    new_cols = set(_table_columns(conn, "inspections"))
    pairs = [
        (old, _RENAMED_COLUMNS.get(old, old))
        for old in legacy_cols
        if _RENAMED_COLUMNS.get(old, old) in new_cols
    ]
    select_cols = ", ".join(old for old, _ in pairs)
    insert_cols = ", ".join(new for _, new in pairs)
    conn.execute(text(
        f"INSERT OR IGNORE INTO inspections ({insert_cols})"
        f" SELECT {select_cols} FROM inspections_legacy"
    ))
    conn.execute(text("DROP TABLE inspections_legacy"))


def _add_missing_columns(conn) -> None:
    """Bring an already-nullable table up to the current schema (forward-compat).

    create_all will not ALTER an existing table, so add columns a prior-version
    DB is missing and apply in-place renames. Every step is guarded on the
    current PRAGMA, so this is idempotent and safe to re-run.
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
    # `anomaly_score` became the Decision Rule's generic primary scalar. Rename
    # in place rather than adding a second numeric column, so there is exactly
    # one meaning per column.
    if "anomaly_score" in cols and "score" not in cols:
        conn.execute(text("ALTER TABLE inspections RENAME COLUMN anomaly_score TO score"))
        cols.add("score")
    for name in ("decision_rule", "params", "metrics"):
        if name not in cols:
            conn.execute(text(f"ALTER TABLE inspections ADD COLUMN {name} TEXT"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
