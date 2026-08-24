from datetime import datetime, timezone, timedelta
from database.crud import create_inspection, get_stats, get_inspections_in_range


FAKE_IMAGE = b"\x89PNG\r\n"  # minimal stub bytes


def _make(db, verdict="ok", threshold=0.5, sku="SKU-A", ts=None, customer=""):
    record = create_inspection(
        db,
        sku_name=sku,
        score=0.3 if verdict == "ok" else 0.8,
        threshold=threshold,
        verdict=verdict,
        model_version="v1.0",
        heatmap_image=FAKE_IMAGE,
        segmentation_image=FAKE_IMAGE,
        customer_name=customer,
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


def test_create_inspection_persists_customer_name(db):
    record = create_inspection(
        db,
        sku_name="SKU-C",
        score=0.4,
        threshold=0.5,
        verdict="ok",
        model_version="v1.0",
        heatmap_image=FAKE_IMAGE,
        segmentation_image=FAKE_IMAGE,
        customer_name="Acme Corp",
    )
    assert record.customer_name == "Acme Corp"


def test_create_inspection_customer_name_defaults_empty(db):
    record = _make(db)
    assert record.customer_name == ""


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


def test_get_inspections_in_range_filters_by_customer(db):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=1)
    end = now + timedelta(days=1)

    acme = _make(db, customer="Acme Corp")
    _make(db, customer="Beta Inc")
    _make(db, customer="")

    results = get_inspections_in_range(db, start, end, customer_name="Acme Corp")
    assert len(results) == 1
    assert results[0].id == acme.id


def test_get_inspections_in_range_no_customer_filter_returns_all(db):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=1)
    end = now + timedelta(days=1)

    _make(db, customer="Acme Corp")
    _make(db, customer="Beta Inc")
    _make(db, customer="")

    results = get_inspections_in_range(db, start, end)
    assert len(results) == 3


def test_create_inspection_defaults_to_anomaly_feature(db):
    record = _make(db)
    assert record.feature == "anomaly_detection"


def test_create_inspection_presence_allows_null_score_and_images(db):
    # Presence/Absence records carry no anomaly score or heatmap/segmentation.
    record = create_inspection(
        db,
        sku_name="SKU-P",
        threshold=0.5,
        verdict="ok",
        model_version="rfdetr-nano",
        feature="presence_absence",
    )
    assert record.feature == "presence_absence"
    assert record.score is None
    assert record.heatmap_image is None
    assert record.segmentation_image is None


def test_get_stats_scoped_by_feature(db):
    _make(db, verdict="ok")  # anomaly
    _make(db, verdict="not_ok")  # anomaly
    create_inspection(
        db, sku_name="SKU-P", threshold=0.5, verdict="ok",
        model_version="rfdetr-nano", feature="presence_absence",
    )

    anomaly_stats = get_stats(db, feature="anomaly_detection")
    presence_stats = get_stats(db, feature="presence_absence")

    assert anomaly_stats["total"] == 2
    assert presence_stats["total"] == 1
    assert presence_stats["ok"] == 1



def test_create_inspection_persists_rule_params_and_metrics(db):
    # Report traceability: a record must be able to say which rule ran, under
    # what tolerances, and what it measured.
    import json

    record = create_inspection(
        db,
        sku_name="SKU-S",
        threshold=0.5,
        verdict="not_ok",
        model_version="seg-v1",
        feature="segmentation",
        score=0.21,
        decision_rule="concentricity",
        params={"max_offset_ratio": 0.10},
        metrics={"offset_ratio": 0.21, "offset_px": 18.4},
    )

    assert record.decision_rule == "concentricity"
    assert json.loads(record.params) == {"max_offset_ratio": 0.10}
    assert json.loads(record.metrics)["offset_px"] == 18.4


def test_create_inspection_leaves_rule_fields_null_when_unset(db):
    record = _make(db)
    assert record.params is None and record.metrics is None


def test_get_stats_scoped_by_decision_rule(db):
    # Two rules on the same Feature produce OKs that mean different things;
    # pooling them would report a pass rate nobody can act on.
    for rule, verdict in (("concentricity", "ok"), ("concentricity", "not_ok"),
                          ("expected_classes", "ok")):
        create_inspection(
            db, sku_name="SKU-S", threshold=0.5, verdict=verdict,
            model_version="seg-v1", feature="segmentation", decision_rule=rule,
        )

    conc = get_stats(db, feature="segmentation", decision_rule="concentricity")
    pres = get_stats(db, feature="segmentation", decision_rule="expected_classes")
    pooled = get_stats(db, feature="segmentation")

    assert (conc["total"], conc["pass_rate"]) == (2, 50.0)
    assert (pres["total"], pres["pass_rate"]) == (1, 100.0)
    # Pooling the two rules gives a third, meaningless number.
    assert pooled["total"] == 3
