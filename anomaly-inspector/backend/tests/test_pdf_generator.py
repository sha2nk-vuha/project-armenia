import io
import json
from datetime import datetime, timezone

from pypdf import PdfReader
from reports.pdf_generator import generate_report


def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return " ".join(page.extract_text() or "" for page in reader.pages)


def _insp(**overrides):
    """One inspection row-dict, as main.py builds from the ORM model."""
    base = dict(
        timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        sku_name="WIDGET-A",
        feature="anomaly_detection",
        score=0.73,
        threshold=0.50,
        verdict="ok",
        model_version="patchcore-v1.2",
        decision_rule="anomaly_threshold",
        metrics=None,
    )
    base.update(overrides)
    return base


def _make_report(inspections=None, **overrides):
    defaults = dict(
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 1, 31, tzinfo=timezone.utc),
        inspections=inspections if inspections is not None else [_insp()],
        app_version="1.0.0",
        customer_name="Acme Corp",
    )
    defaults.update(overrides)
    return generate_report(**defaults)


def test_returns_bytes():
    result = _make_report()
    assert isinstance(result, bytes)


def test_is_pdf_magic_bytes():
    result = _make_report()
    assert result[:4] == b"%PDF"


def test_non_empty_output():
    result = _make_report()
    assert len(result) > 1000


def test_customer_name_appears_in_pdf():
    text = _pdf_text(_make_report(customer_name="Acme Corp"))
    assert "Acme Corp" in text


def test_customer_name_na_when_empty():
    text = _pdf_text(_make_report(customer_name=""))
    assert "N/A" in text


def test_models_used_table_lists_feature_and_version():
    insps = [
        _insp(feature="anomaly_detection", model_version="patchcore-v1.2"),
        _insp(feature="segmentation", model_version="rfdetr-seg-nano-v0.0.1",
              verdict="not_ok", score=0.30, threshold=0.15),
    ]
    text = _pdf_text(_make_report(inspections=insps))
    assert "Models Used" in text
    assert "patchcore-v1.2" in text
    assert "rfdetr-seg-nano-v0.0.1" in text


def test_threshold_settings_per_feature_rule():
    insps = [
        _insp(feature="anomaly_detection", decision_rule="anomaly_threshold", threshold=0.50),
        _insp(feature="segmentation", decision_rule="concentricity", threshold=0.15,
              verdict="not_ok", score=0.30),
    ]
    text = _pdf_text(_make_report(inspections=insps))
    assert "Threshold Settings" in text
    assert "anomaly_threshold" in text
    assert "concentricity" in text


def test_cascade_threshold_shown_as_per_stage():
    insp = _insp(feature="cascade", decision_rule="and", threshold=0.50,
                 verdict="not_ok", score=None,
                 metrics=json.dumps({
                     "combinator": "and", "short_circuit": True, "decisive_stage": 1,
                     "stages": [
                         {"feature": "anomaly_detection", "rule": "anomaly_threshold",
                          "verdict": "ok", "evaluated": True, "score": 0.20,
                          "reason": "0.20 < 0.50", "threshold": 0.50,
                          "model_version": "patchcore-v1.2"},
                         {"feature": "segmentation", "rule": "concentricity",
                          "verdict": "not_ok", "evaluated": True, "score": 0.30,
                          "reason": "offset 0.30 > 0.06", "threshold": 0.06,
                          "model_version": "rfdetr-seg-nano-v0.0.1"},
                     ],
                 }))
    text = _pdf_text(_make_report(inspections=[insp]))
    assert "Per-stage" in text


def test_per_sku_stats_split_by_feature():
    insps = [
        _insp(sku_name="WIDGET-A", feature="anomaly_detection", verdict="ok"),
        _insp(sku_name="WIDGET-A", feature="segmentation", verdict="not_ok",
              score=0.30, threshold=0.15),
    ]
    text = _pdf_text(_make_report(inspections=insps))
    assert "Per-SKU Statistics" in text
    assert "Anomaly Detection" in text
    assert "Segmentation" in text


def test_nok_section_shows_timestamp_score_threshold():
    insp = _insp(
        verdict="not_ok", score=0.73, threshold=0.50,
        feature="anomaly_detection", decision_rule="anomaly_threshold",
    )
    text = _pdf_text(_make_report(inspections=[insp]))
    assert "NOK Cases" in text
    assert "2026-01-15" in text
    assert "0.73" in text
    assert "0.50" in text


def test_nok_cascade_shows_per_stage_breakdown():
    insp = _insp(
        feature="cascade", decision_rule="and", threshold=0.50,
        verdict="not_ok", score=None,
        metrics=json.dumps({
            "combinator": "and", "short_circuit": True, "decisive_stage": 1,
            "stages": [
                {"feature": "anomaly_detection", "rule": "anomaly_threshold",
                 "verdict": "ok", "evaluated": True, "score": 0.20,
                 "reason": "0.20 below threshold", "threshold": 0.50,
                 "model_version": "patchcore-v1.2"},
                {"feature": "segmentation", "rule": "concentricity",
                 "verdict": "not_ok", "evaluated": True, "score": 0.30,
                 "reason": "offset 0.30 > 0.06", "threshold": 0.06,
                 "model_version": "rfdetr-seg-nano-v0.0.1"},
            ],
        }),
    )
    text = _pdf_text(_make_report(inspections=[insp]))
    assert "NOK Cases" in text
    assert "segmentation" in text.lower()
    assert "rfdetr-seg-nano-v0.0.1" in text
    assert "offset 0.30" in text


def test_null_score_nok_shows_dash():
    insp = _insp(
        verdict="not_ok", score=None, threshold=0.50,
        feature="presence_absence", decision_rule="class_present",
        metrics=json.dumps({"reason": "expected class 'gasket' not found"}),
    )
    text = _pdf_text(_make_report(inspections=[insp]))
    assert "—" in text
    assert "NOK Cases" in text


def test_legacy_nok_without_metrics_shows_placeholder():
    insp = _insp(
        verdict="not_ok", score=0.60, threshold=0.50,
        feature="anomaly_detection", decision_rule=None, metrics=None,
    )
    text = _pdf_text(_make_report(inspections=[insp]))
    assert "details not recorded" in text


def test_no_nok_cases_message():
    text = _pdf_text(_make_report(inspections=[_insp(verdict="ok")]))
    assert "No NOK cases" in text


def test_aggregate_stats_only_when_multiple_groups():
    single = _make_report(inspections=[_insp()])
    assert "Aggregate Statistics" not in _pdf_text(single)
    multi = _make_report(inspections=[
        _insp(sku_name="A", feature="anomaly_detection"),
        _insp(sku_name="B", feature="segmentation", verdict="not_ok", score=0.3, threshold=0.15),
    ])
    assert "Aggregate Statistics" in _pdf_text(multi)
