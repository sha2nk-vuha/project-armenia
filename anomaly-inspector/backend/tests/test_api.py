import numpy as np
import pytest
import cv2
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from database.models import Base
from database.db import get_db
import inference.engine as _engine
import inference.features as _features


def _make_png_bytes(size=64) -> bytes:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[10:30, 10:30] = [180, 90, 45]
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@pytest.fixture
def client():
    from main import app

    # Reset model + active-Feature state before each test to avoid leakage
    _engine._current_session = None
    _features.reset()

    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    TestSession = sessionmaker(bind=test_engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    # Suppress startup auto-load of the bundled default model so each test
    # controls model state explicitly (via the loaded_model fixture or not).
    with patch("main._load_default_model"):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()
    # Reset model session after each test
    _engine._current_session = None
    _features.reset()


@pytest.fixture
def loaded_model(client):
    mock_session_obj = MagicMock()
    meta = MagicMock()
    meta.name = "input"
    meta.shape = [1, 3, 64, 64]
    mock_session_obj.get_inputs.return_value = [meta]
    amap = np.zeros((1, 1, 64, 64), dtype=np.float32)
    amap[0, 0, 10:30, 10:30] = 0.9
    mock_session_obj.run.return_value = [amap, np.array([0.8], dtype=np.float32)]

    with patch("inference.engine._load_onnx_session", return_value=mock_session_obj):
        resp = client.post(
            "/api/load-model",
            data={"model_version": "v1.0-test"},
            files={"model_file": ("model.onnx", b"fake_onnx", "application/octet-stream")},
        )
    assert resp.status_code == 200
    return mock_session_obj


def test_status_no_model(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False


def test_load_model_success(client, loaded_model):
    resp = client.get("/api/status")
    assert resp.json()["model_loaded"] is True
    assert resp.json()["model_version"] == "v1.0-test"
    assert resp.json()["runtime"] == "cpu"


def test_load_model_bad_extension_returns_422(client):
    resp = client.post(
        "/api/load-model",
        data={"model_version": "v1.0-test"},
        files={"model_file": ("model.pt", b"fake", "application/octet-stream")},
    )
    assert resp.status_code == 422


def test_infer_without_model_returns_400(client):
    resp = client.post(
        "/api/infer",
        data={"sku_name": "TEST-SKU", "threshold": "0.5"},
        files={"image": ("test.png", _make_png_bytes(), "image/png")},
    )
    assert resp.status_code == 400


def test_infer_success(client, loaded_model):
    resp = client.post(
        "/api/infer",
        data={"sku_name": "SKU-X", "threshold": "0.5"},
        files={"image": ("test.png", _make_png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "not_ok"
    assert 0.0 <= body["anomaly_score"] <= 1.0
    assert body["heatmap_image"].startswith("/9j/")
    assert body["segmentation_image"].startswith("/9j/")


def test_stats_after_infer(client, loaded_model):
    client.post(
        "/api/infer",
        data={"sku_name": "SKU-A", "threshold": "0.5"},
        files={"image": ("test.png", _make_png_bytes(), "image/png")},
    )
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["not_ok"] == 1


def test_report_no_records_returns_404(client, loaded_model):
    resp = client.post(
        "/api/report",
        data={"start_date": "2020-01-01T00:00:00", "end_date": "2020-01-02T00:00:00"},
    )
    assert resp.status_code == 404


def test_report_returns_pdf(client, loaded_model):
    client.post(
        "/api/infer",
        data={"sku_name": "SKU-B", "threshold": "0.5"},
        files={"image": ("test.png", _make_png_bytes(), "image/png")},
    )
    resp = client.post(
        "/api/report",
        data={"start_date": "2020-01-01T00:00:00", "end_date": "2099-12-31T00:00:00"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:4] == b"%PDF"
    assert resp.headers["content-disposition"].startswith("attachment; filename=")


def test_infer_with_customer_name_returns_200(client, loaded_model):
    resp = client.post(
        "/api/infer",
        data={"sku_name": "SKU-X", "threshold": "0.5", "customer_name": "Acme Corp"},
        files={"image": ("test.png", _make_png_bytes(), "image/png")},
    )
    assert resp.status_code == 200


def test_report_scoped_to_customer(client, loaded_model):
    # Two infers: one for Acme, one for Beta
    client.post(
        "/api/infer",
        data={"sku_name": "SKU-A", "threshold": "0.5", "customer_name": "Acme Corp"},
        files={"image": ("test.png", _make_png_bytes(), "image/png")},
    )
    client.post(
        "/api/infer",
        data={"sku_name": "SKU-A", "threshold": "0.5", "customer_name": "Beta Inc"},
        files={"image": ("test.png", _make_png_bytes(), "image/png")},
    )
    # Report scoped to Acme should succeed (has records)
    resp = client.post(
        "/api/report",
        data={
            "start_date": "2020-01-01T00:00:00",
            "end_date": "2099-12-31T00:00:00",
            "customer_name": "Acme Corp",
        },
    )
    assert resp.status_code == 200
    assert resp.content[:4] == b"%PDF"

    # Report scoped to an unknown customer should return 404
    resp = client.post(
        "/api/report",
        data={
            "start_date": "2020-01-01T00:00:00",
            "end_date": "2099-12-31T00:00:00",
            "customer_name": "Unknown Corp",
        },
    )
    assert resp.status_code == 404


# ── Feature switching + Presence/Absence ─────────────────────────────────────

def _mock_detector_session(favored_class=0, num_classes=3):
    """Mock RF-DETR session: one confident query on `favored_class`."""
    session = MagicMock()
    logits = np.full((1, 2, num_classes), -5.0, dtype=np.float32)
    logits[0, 0, favored_class] = 5.0
    boxes = np.array([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]]], dtype=np.float32)
    session.run.return_value = [logits, boxes]
    return session


@pytest.fixture
def presence_active(client):
    """Activate Presence/Absence with a mock detector, via the real endpoint."""
    from inference.engine import ModelSession
    from inference.rfdetr import ModelConfig

    sess = ModelSession(
        session=_mock_detector_session(favored_class=0),
        runtime="cpu",
        input_name="input",
        input_shape=(20, 20),
        model_version="rfdetr-nano",
    )
    cfg = ModelConfig(
        labels={0: "gasket", 1: "no-gasket", 2: "background"},
        input_size=(20, 20),
        rule_params={"expected_classes": {"expected_classes": [0]}},
    )

    def fake_activate(feature):
        _engine._current_session = sess
        _features._active_feature = feature
        _features._model_config = cfg
        return sess

    with patch("main.features.activate", side_effect=fake_activate):
        resp = client.post("/api/feature", data={"feature": "presence_absence"})
    assert resp.status_code == 200
    return sess


def test_status_reports_active_feature_and_catalog(client, loaded_model):
    body = client.get("/api/status").json()
    assert body["active_feature"] == "anomaly_detection"
    # The Feature catalog drives the UI selector.
    names = {f["name"] for f in body["features"]}
    assert {"anomaly_detection", "presence_absence"} <= names


def test_feature_switch_activates_presence(client, presence_active):
    body = client.get("/api/status").json()
    assert body["active_feature"] == "presence_absence"


def test_feature_unknown_returns_400(client):
    resp = client.post("/api/feature", data={"feature": "bogus"})
    assert resp.status_code == 400


def test_presence_infer_returns_feature_shaped_payload(client, presence_active):
    resp = client.post(
        "/api/infer",
        data={"sku_name": "SKU-P", "threshold": "0.5"},
        files={"image": ("test.png", _make_png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["feature"] == "presence_absence"
    assert body["verdict"] == "ok"  # gasket present
    assert body["annotated_image"].startswith("/9j/")
    assert body["detections"] and body["detections"][0]["label"] == "gasket"
    # No anomaly-specific fields for this Feature.
    assert body["anomaly_score"] is None


def test_presence_infer_nok_when_gasket_absent(client):
    from inference.engine import ModelSession
    from inference.rfdetr import ModelConfig

    sess = ModelSession(
        session=_mock_detector_session(favored_class=1),  # no-gasket
        runtime="cpu",
        input_name="input",
        input_shape=(20, 20),
        model_version="rfdetr-nano",
    )
    cfg = ModelConfig(
        labels={0: "gasket", 1: "no-gasket", 2: "background"},
        input_size=(20, 20),
        rule_params={"expected_classes": {"expected_classes": [0]}},
    )

    def fake_activate(feature):
        _engine._current_session = sess
        _features._active_feature = feature
        _features._model_config = cfg
        return sess

    with patch("main.features.activate", side_effect=fake_activate):
        client.post("/api/feature", data={"feature": "presence_absence"})

    resp = client.post(
        "/api/infer",
        data={"sku_name": "SKU-P", "threshold": "0.5"},
        files={"image": ("test.png", _make_png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["verdict"] == "not_ok"


def test_presence_stats_scoped_to_feature(client, presence_active):
    client.post(
        "/api/infer",
        data={"sku_name": "SKU-P", "threshold": "0.5"},
        files={"image": ("test.png", _make_png_bytes(), "image/png")},
    )
    body = client.get("/api/stats?feature=presence_absence").json()
    assert body["total"] == 1
    assert body["ok"] == 1
    # Anomaly feature has no records.
    assert client.get("/api/stats?feature=anomaly_detection").json()["total"] == 0


def test_decision_rules_empty_without_active_feature(client):
    body = client.get("/api/decision-rules").json()
    assert body["active_feature"] is None
    assert body["rules"] == []


def test_decision_rules_lists_compatible_rules_for_active_feature(client, loaded_model):
    body = client.get("/api/decision-rules").json()
    names = {r["name"] for r in body["rules"]}
    # Anomaly Detection produces an anomaly map, so only that rule can consume it.
    assert names == {"anomaly_threshold"}


def test_infer_reports_which_rule_decided_and_why(client, loaded_model):
    body = client.post(
        "/api/infer",
        data={"sku_name": "SKU-A", "threshold": "0.5"},
        files={"image": ("t.png", _make_png_bytes(), "image/png")},
    ).json()

    assert body["decision_rule"] == "anomaly_threshold"
    assert body["score_label"] == "Anomaly Score"
    assert "anomaly score" in body["reason"]
    assert body["metrics"]["threshold"] == 0.5
    # Deprecated alias kept until the GUI moves to `score`.
    assert body["anomaly_score"] == body["score"]


def test_infer_rejects_malformed_rule_params(client, loaded_model):
    resp = client.post(
        "/api/infer",
        data={"sku_name": "SKU-A", "threshold": "0.5", "rule_params": "{not json"},
        files={"image": ("t.png", _make_png_bytes(), "image/png")},
    )
    assert resp.status_code == 400


def test_infer_rejects_rule_incompatible_with_active_feature(client, loaded_model):
    # expected_classes consumes detections; Anomaly Detection produces a map.
    resp = client.post(
        "/api/infer",
        data={
            "sku_name": "SKU-A",
            "threshold": "0.5",
            "decision_rule": "expected_classes",
        },
        files={"image": ("t.png", _make_png_bytes(), "image/png")},
    )
    assert resp.status_code == 422
    assert "expected_classes" in resp.json()["detail"]


def test_infer_persists_rule_for_scoped_stats(client, loaded_model):
    # Proves the rule landed in its column, through the public surface that
    # reporting actually uses.
    client.post(
        "/api/infer",
        data={"sku_name": "SKU-A", "threshold": "0.5"},
        files={"image": ("t.png", _make_png_bytes(), "image/png")},
    )

    scoped = client.get("/api/stats?decision_rule=anomaly_threshold").json()
    other = client.get("/api/stats?decision_rule=expected_classes").json()

    assert scoped["total"] == 1
    assert other["total"] == 0


def test_presence_infer_overrides_expected_classes_from_request(client, presence_active):
    # Params travel per-request like `threshold`; the sidecar only seeds defaults.
    body = client.post(
        "/api/infer",
        data={
            "sku_name": "SKU-P",
            "threshold": "0.5",
            "rule_params": '{"expected_classes": ["no-gasket"]}',
        },
        files={"image": ("t.png", _make_png_bytes(), "image/png")},
    ).json()

    assert body["decision_rule"] == "expected_classes"
    assert body["verdict"] == "not_ok"
    assert body["metrics"]["missing"] == ["no-gasket"]


def test_presence_infer_unknown_class_name_is_a_clear_422(client, presence_active):
    resp = client.post(
        "/api/infer",
        data={
            "sku_name": "SKU-P",
            "threshold": "0.5",
            "rule_params": '{"expected_classes": ["bottle_cap"]}',
        },
        files={"image": ("t.png", _make_png_bytes(), "image/png")},
    )
    assert resp.status_code == 422
    assert "bottle_cap" in resp.json()["detail"]
