from datetime import datetime, timezone, timedelta
from database.crud import create_inspection, get_stats, get_inspections_in_range


FAKE_IMAGE = b"\x89PNG\r\n"  # minimal stub bytes


def _make(db, verdict="ok", threshold=0.5, sku="SKU-A", ts=None):
    record = create_inspection(
        db,
        sku_name=sku,
        anomaly_score=0.3 if verdict == "ok" else 0.8,
        threshold=threshold,
        verdict=verdict,
        model_version="v1.0",
        heatmap_image=FAKE_IMAGE,
        segmentation_image=FAKE_IMAGE,
    )
    if ts is not None:
        record.timestamp = ts
        db.commit()
    return record


def test_create_inspection_returns_id(db):
    record = _make(db)
    assert record.id is not None
    assert record.id >= 1


def test_create_inspection_stores_all_fields(db):
    record = _make(db, verdict="not_ok", threshold=0.6, sku="WIDGET-7")
    assert record.sku_name == "WIDGET-7"
    assert record.verdict == "not_ok"
    assert record.threshold == 0.6
    assert record.heatmap_image == FAKE_IMAGE


def test_get_stats_empty_db(db):
    stats = get_stats(db)
    assert stats == {"total": 0, "ok": 0, "not_ok": 0, "pass_rate": 0.0}


def test_get_stats_counts_correctly(db):
    _make(db, verdict="ok")
    _make(db, verdict="ok")
    _make(db, verdict="not_ok")
    stats = get_stats(db)
    assert stats["total"] == 3
    assert stats["ok"] == 2
    assert stats["not_ok"] == 1
    assert stats["pass_rate"] == 66.7


def test_get_inspections_in_range_filters_by_date(db):
    now = datetime.now(timezone.utc)
    _make(db, ts=now - timedelta(days=5))
    in_range = _make(db, ts=now - timedelta(days=2))
    _make(db, ts=now + timedelta(days=1))

    start = now - timedelta(days=3)
    end = now
    results = get_inspections_in_range(db, start, end)
    assert len(results) == 1
    assert results[0].id == in_range.id
