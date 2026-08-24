"""Schema migration: a legacy inspections table (NOT NULL score/images, no
`feature` column) must be upgraded in place so Presence/Absence records with
NULL score/images can be stored, without losing existing rows.
"""
from sqlalchemy import create_engine, text

import database.db as db_module


def _legacy_engine(url):
    """Create the pre-Feature schema: score + images NOT NULL, no `feature`."""
    engine = create_engine(url, connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                sku_name VARCHAR NOT NULL,
                anomaly_score FLOAT NOT NULL,
                threshold FLOAT NOT NULL,
                verdict VARCHAR NOT NULL,
                model_version VARCHAR NOT NULL,
                customer_name VARCHAR NOT NULL DEFAULT '',
                heatmap_image BLOB NOT NULL,
                segmentation_image BLOB NOT NULL
            )
            """
        ))
        conn.execute(text(
            "INSERT INTO inspections (timestamp, sku_name, anomaly_score, threshold,"
            " verdict, model_version, customer_name, heatmap_image, segmentation_image)"
            " VALUES ('2024-01-01 00:00:00', 'SKU-A', 0.8, 0.5, 'not_ok', 'v1',"
            " 'Acme', x'0102', x'0304')"
        ))
    return engine


def test_migration_rebuilds_legacy_table(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    engine = _legacy_engine(url)

    # Point the module's engine at the legacy DB, then run init_db.
    monkeypatch.setattr(db_module, "engine", engine)
    db_module.init_db()

    with engine.connect() as conn:
        info = {row[1]: row for row in conn.execute(text("PRAGMA table_info(inspections)"))}
        # New column present.
        assert "feature" in info
        # score/images relaxed to nullable (notnull flag == 0).
        assert info["anomaly_score"][3] == 0
        assert info["heatmap_image"][3] == 0
        assert info["segmentation_image"][3] == 0
        # Existing row preserved and back-filled with the default Feature.
        rows = list(conn.execute(text(
            "SELECT sku_name, customer_name, feature FROM inspections"
        )))
    assert rows == [("SKU-A", "Acme", "anomaly_detection")]


def test_migration_resumes_after_interrupted_rename(tmp_path, monkeypatch):
    """A crash between the legacy rename and the row copy must not lose data.

    Simulate the interrupted state (only `inspections_legacy` exists) and verify
    the next init_db resumes: it recreates `inspections` and copies the rows.
    """
    url = f"sqlite:///{tmp_path / 'interrupted.db'}"
    engine = _legacy_engine(url)
    # Emulate the crash point: the rename committed, nothing else ran.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE inspections RENAME TO inspections_legacy"))

    monkeypatch.setattr(db_module, "engine", engine)
    db_module.init_db()

    with engine.connect() as conn:
        tables = {row[0] for row in conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ))}
        rows = list(conn.execute(text("SELECT sku_name, feature FROM inspections")))
    # Legacy table consumed, data preserved and back-filled with the default Feature.
    assert "inspections_legacy" not in tables
    assert rows == [("SKU-A", "anomaly_detection")]
