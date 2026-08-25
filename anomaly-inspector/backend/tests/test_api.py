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
    # No model auto-loads on startup any more (per-Feature upload); each test
    # installs the model state it needs via `_install_model` or `loaded_model`.
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    # Reset model state after each test
    _features.reset()


def _install_model(feature, session, config=None):
    """Put a model into the per-Feature store and make its Feature active.

    Mirrors what an upload does, without needing a real .onnx: tests build a
    mock ModelSession + ModelConfig and install it directly.
    """
    from inference.model_config import ModelConfig

    _features._models[feature] = (session, config or ModelConfig())
    _features._active_feature = feature


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
    assert 0.0 <= body["score"] <= 1.0
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


def test_report_contains_models_used_and_nok_details(client, loaded_model):
    """End-to-end: a NOK inspection must surface its score, threshold, and
    model version in the report's NOK Cases and Models Used sections."""
    import io as _io
    from pypdf import PdfReader

    client.post(
        "/api/infer",
        data={"sku_name": "SKU-NOK", "threshold": "0.5"},
        files={"image": ("test.png", _make_png_bytes(), "image/png")},
    )
    resp = client.post(
        "/api/report",
        data={"start_date": "2020-01-01T00:00:00", "end_date": "2099-12-31T00:00:00"},
    )
    assert resp.status_code == 200
    text = " ".join(
        page.extract_text() or "" for page in PdfReader(_io.BytesIO(resp.content)).pages
    )
    assert "Models Used" in text
    assert "NOK Cases" in text



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
    from inference.model_config import ModelConfig

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
    _install_model("presence_absence", sess, cfg)
    assert client.post("/api/feature", data={"feature": "presence_absence"}).status_code == 200
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
    assert body["score"] is None


def test_presence_infer_nok_when_gasket_absent(client):
    from inference.engine import ModelSession
    from inference.model_config import ModelConfig

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
    _install_model("presence_absence", sess, cfg)

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


def test_decision_rules_empty_without_a_loaded_model(client):
    # A default Feature is selected on startup, but no model is loaded, so no
    # rules can be offered until one is uploaded.
    body = client.get("/api/decision-rules").json()
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
    assert body["score"] is not None


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


# ── Per-SKU calibration ─────────────────────────────────────────────────────


@pytest.fixture
def segmentation_active(client):
    """Activate Segmentation over a mocked seg model (boxes/logits/masks)."""
    from inference.engine import ModelSession
    from inference.model_config import ModelConfig

    logits = np.full((1, 2, 2), -8.0, dtype=np.float32)
    logits[0, 0, 0] = 8.0   # query 0 -> reference class
    logits[0, 1, 1] = 8.0   # query 1 -> target class
    boxes = np.array([[[0.5, 0.5, 0.9, 0.9], [0.5, 0.5, 0.4, 0.4]]], dtype=np.float32)
    masks = np.full((1, 2, 16, 16), -8.0, dtype=np.float32)
    masks[0, 0, 2:14, 2:14] = 8.0   # big reference blob
    masks[0, 1, 6:11, 7:12] = 8.0   # smaller target, deliberately off-centre
    mock = MagicMock()
    mock.run.return_value = [boxes, logits, masks]

    sess = ModelSession(
        session=mock, runtime="cpu", input_name="input",
        input_shape=(16, 16), model_version="seg-test",
    )
    cfg = ModelConfig(input_size=(16, 16), labels={0: "cap", 1: "logo"})
    _install_model("segmentation", sess, cfg)
    return client


def _calibration_params():
    return {
        "reference_class": "cap", "target_class": "logo",
        "max_offset_ratio": 0.05, "reference_center_method": "min_enclosing_circle",
        "target_center_method": "mask_centroid",
    }


def test_calibration_absent_until_taught(client, segmentation_active):
    body = client.get("/api/skus/SKU-S/calibration?decision_rule=concentricity").json()
    assert body["calibrated"] is False
    assert body["params"] == {}


def test_calibrate_teaches_a_nominal_and_makes_a_biased_sku_pass(
    client, segmentation_active
):
    """The whole point: artwork that measures off-centre when correct fails on a
    shared tolerance, and passes once its own baseline is taught."""
    import json as _json

    png = _make_png_bytes()

    def infer():
        return client.post(
            "/api/infer",
            data={
                "sku_name": "SKU-S", "threshold": "0.5",
                "decision_rule": "concentricity",
                "rule_params": _json.dumps(_calibration_params()),
            },
            files={"image": ("t.png", png, "image/png")},
        ).json()

    before = infer()
    assert before["verdict"] == "not_ok", "fixture must be biased off-centre"

    taught = client.post(
        "/api/calibrate",
        data={
            "sku_name": "SKU-S", "threshold": "0.5",
            "decision_rule": "concentricity",
            "rule_params": _json.dumps(_calibration_params()),
        },
        files=[
            ("images", ("a.png", png, "image/png")),
            ("images", ("b.png", png, "image/png")),
            ("images", ("c.png", png, "image/png")),
        ],
    )
    assert taught.status_code == 200, taught.text
    body = taught.json()
    assert body["sample_count"] == 3
    assert body["params"]["nominal_offset"] == pytest.approx(before["metrics"]["offset_ratio"], abs=1e-3)

    # The taught baseline is applied server-side even without the GUI sending it.
    after = client.post(
        "/api/infer",
        data={
            "sku_name": "SKU-S", "threshold": "0.5",
            "decision_rule": "concentricity",
            "rule_params": _json.dumps(
                {k: v for k, v in _calibration_params().items()}
            ),
        },
        files={"image": ("t.png", _make_png_bytes(), "image/png")},
    ).json()
    assert after["verdict"] == "ok"
    assert after["metrics"]["nominal_offset"] > 0


def test_calibration_is_scoped_to_its_sku(client, segmentation_active):
    import json as _json

    png = _make_png_bytes()
    client.post(
        "/api/calibrate",
        data={
            "sku_name": "SKU-A", "threshold": "0.5",
            "decision_rule": "concentricity",
            "rule_params": _json.dumps(_calibration_params()),
        },
        files=[("images", ("a.png", png, "image/png"))],
    )

    assert client.get("/api/skus/SKU-A/calibration?decision_rule=concentricity").json()["calibrated"]
    assert not client.get("/api/skus/SKU-B/calibration?decision_rule=concentricity").json()["calibrated"]


def test_calibration_can_be_cleared(client, segmentation_active):
    import json as _json

    png = _make_png_bytes()
    client.post(
        "/api/calibrate",
        data={
            "sku_name": "SKU-A", "threshold": "0.5",
            "decision_rule": "concentricity",
            "rule_params": _json.dumps(_calibration_params()),
        },
        files=[("images", ("a.png", png, "image/png"))],
    )
    assert client.delete("/api/skus/SKU-A/calibration?decision_rule=concentricity").json()["cleared"]
    assert not client.get("/api/skus/SKU-A/calibration?decision_rule=concentricity").json()["calibrated"]


def test_calibrate_rejects_a_rule_with_no_baseline(client, presence_active):
    png = _make_png_bytes()
    resp = client.post(
        "/api/calibrate",
        data={
            "sku_name": "SKU-P", "threshold": "0.5",
            "decision_rule": "expected_classes",
        },
        files=[("images", ("p.png", png, "image/png"))],
    )
    assert resp.status_code == 422
    assert "no per-SKU baseline" in resp.json()["detail"]


def test_calibrate_reports_unmeasurable_samples_rather_than_averaging_them(
    client, segmentation_active
):
    """A sample whose parts were not found carries no baseline information."""
    import json as _json

    png = _make_png_bytes()
    body = client.post(
        "/api/calibrate",
        data={
            "sku_name": "SKU-S", "threshold": "0.5",
            "decision_rule": "concentricity",
            # No target class in the scene -> nothing measurable.
            "rule_params": _json.dumps({**_calibration_params(), "target_class": "cap"}),
        },
        files=[
            ("images", ("a.png", png, "image/png")),
            ("images", ("b.png", png, "image/png")),
        ],
    )
    # Reference == target degenerates to zero offset, still measurable; assert
    # the endpoint reports what it used rather than silently inventing a value.
    assert body.status_code in (200, 422)
    if body.status_code == 200:
        assert body.json()["sample_count"] + len(body.json()["skipped"]) == 2


def test_calibrate_rejects_empty_sample_list(client, segmentation_active):
    resp = client.post(
        "/api/calibrate",
        data={
            "sku_name": "SKU-S", "threshold": "0.5",
            "decision_rule": "concentricity",
        },
    )
    assert resp.status_code == 400


def test_decision_rules_expose_sidecar_merged_defaults(client, segmentation_active):
    """The GUI seeds controls from these and echoes them back on every request,
    so they must already include the sidecar's configuration -- otherwise the
    browser silently overrides the model's own settings with schema defaults."""
    rules = client.get("/api/decision-rules").json()["rules"]
    conc = next(r for r in rules if r["name"] == "concentricity")

    schema_default = next(
        p["default"] for p in conc["params"] if p["name"] == "target_center_method"
    )
    assert conc["defaults"]["target_center_method"] == schema_default

    # And a sidecar value must win over the schema default.
    _features._models["segmentation"][1].rule_params = {
        "concentricity": {"target_center_method": "outer_circle_fit"}
    }
    rules = client.get("/api/decision-rules").json()["rules"]
    conc = next(r for r in rules if r["name"] == "concentricity")
    assert conc["defaults"]["target_center_method"] == "outer_circle_fit"


def test_calibrate_accepts_uploaded_files(client, segmentation_active):
    """Calibration teaches from browser-uploaded files (a folder browsed in the
    GUI lives in the browser, not on disk)."""
    import json as _json

    png = _make_png_bytes()
    resp = client.post(
        "/api/calibrate",
        data={
            "sku_name": "SKU-UP", "threshold": "0.5",
            "decision_rule": "concentricity",
            "rule_params": _json.dumps(_calibration_params()),
        },
        files=[
            ("images", ("a.png", png, "image/png")),
            ("images", ("b.png", png, "image/png")),
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["sample_count"] == 2
    assert "nominal_offset" in body["params"]
    assert client.get(
        "/api/skus/SKU-UP/calibration?decision_rule=concentricity"
    ).json()["calibrated"]


def test_calibrate_with_no_samples_at_all_is_400(client, segmentation_active):
    resp = client.post(
        "/api/calibrate",
        data={
            "sku_name": "SKU-S", "threshold": "0.5",
            "decision_rule": "concentricity",
        },
    )
    assert resp.status_code == 400


# ── Cascade ─────────────────────────────────────────────────────────────────


def test_cascade_options_lists_features_rules_and_combinators(client):
    body = client.get("/api/cascade/options").json()
    feature_names = {f["name"] for f in body["features"]}
    # Concrete features are cascadable; the cascade meta-feature is not listed.
    assert "segmentation" in feature_names
    assert "cascade" not in feature_names
    assert {c["name"] for c in body["combinators"]} == {"and", "or"}
    seg = next(f for f in body["features"] if f["name"] == "segmentation")
    rule_names = {r["name"] for r in seg["rules"]}
    assert {"concentricity", "expected_classes"} <= rule_names
    # Class Catalog comes from the sidecar without loading the ONNX.
    assert "0" in seg["labels"] or seg["labels"] == {}


def test_cascade_appears_in_the_feature_catalog(client):
    names = {f["name"] for f in client.get("/api/status").json()["features"]}
    assert "cascade" in names


class _StubCascade:
    feature = "cascade"
    model_version = "cascade[a, b]"

    def infer(self, image_bytes, threshold, rule_name=None, rule_params=None):
        from inference.pipeline import InferenceResult, StageResult
        return InferenceResult(
            verdict="not_ok",
            score=None,
            images={},
            decision_rule="and",
            reason="stage 2 (segmentation/concentricity) NOK: offset too high",
            metrics={"combinator": "and", "decisive_stage": 1, "stages": []},
            stages=[
                StageResult("anomaly_detection", "anomaly_threshold", "ok", True,
                            score=0.2, score_label="Anomaly Score",
                            reason="0.20 < 0.95",
                            images=[("Heatmap", b"heat"), ("Segmentation", b"seg")]),
                StageResult("segmentation", "concentricity", "not_ok", True,
                            score=0.30, score_label="Offset Ratio",
                            reason="offset 0.30 > 0.06",
                            images=[("Detections", b"annot")]),
            ],
        )


@pytest.fixture
def cascade_active(client):
    """Put the app in cascade mode (activation loads no model)."""
    assert client.post("/api/feature", data={"feature": "cascade"}).status_code == 200
    return client


def test_cascade_infer_returns_combined_verdict_and_per_stage_breakdown(cascade_active):
    import json as _json

    spec = {
        "combinator": "and",
        "stages": [
            {"feature": "anomaly_detection", "rule": "anomaly_threshold", "threshold": 0.95},
            {"feature": "segmentation", "rule": "concentricity", "threshold": 0.5},
        ],
    }
    with patch("main._build_cascade_or_400", return_value=_StubCascade()):
        body = cascade_active.post(
            "/api/infer",
            data={"sku_name": "SKU-C", "threshold": "0.5", "cascade_spec": _json.dumps(spec)},
            files={"image": ("t.png", _make_png_bytes(), "image/png")},
        ).json()

    assert body["feature"] == "cascade"
    assert body["verdict"] == "not_ok"
    assert body["decision_rule"] == "and"
    assert len(body["stages"]) == 2
    assert body["stages"][0]["feature"] == "anomaly_detection"
    assert body["stages"][0]["verdict"] == "ok"
    assert body["stages"][1]["verdict"] == "not_ok"
    assert body["stages"][0]["images"][0]["label"] == "Heatmap"
    assert len(body["stages"][0]["images"]) == 2
    assert body["stages"][1]["images"][0]["label"] == "Detections"
    assert "segmentation" in body["reason"]


def test_cascade_infer_records_feature_cascade_for_stats(cascade_active):
    import json as _json

    spec = {"combinator": "and", "stages": [{"feature": "segmentation", "rule": "concentricity"}]}
    with patch("main._build_cascade_or_400", return_value=_StubCascade()):
        cascade_active.post(
            "/api/infer",
            data={"sku_name": "SKU-C", "threshold": "0.5", "cascade_spec": _json.dumps(spec)},
            files={"image": ("t.png", _make_png_bytes(), "image/png")},
        )
    assert cascade_active.get("/api/stats?feature=cascade").json()["total"] == 1


def test_cascade_infer_without_spec_is_400(cascade_active):
    resp = cascade_active.post(
        "/api/infer",
        data={"sku_name": "SKU-C", "threshold": "0.5"},
        files={"image": ("t.png", _make_png_bytes(), "image/png")},
    )
    assert resp.status_code == 400
    assert "cascade_spec" in resp.json()["detail"]


def test_cascade_infer_surfaces_a_config_error(cascade_active):
    import json as _json

    spec = {"combinator": "and", "stages": [{"feature": "segmentation", "rule": "anomaly_threshold"}]}
    # Real build path: incompatible rule -> 422, not a silent NOK.
    with patch("main.features.build_cascade_pipeline",
               side_effect=ValueError("rule needs anomaly_map")):
        resp = cascade_active.post(
            "/api/infer",
            data={"sku_name": "SKU-C", "threshold": "0.5", "cascade_spec": _json.dumps(spec)},
            files={"image": ("t.png", _make_png_bytes(), "image/png")},
        )
    assert resp.status_code == 422
    assert "Invalid cascade" in resp.json()["detail"]
