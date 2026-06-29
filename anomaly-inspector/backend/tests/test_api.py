import io
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


def _make_png_bytes(size=64) -> bytes:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[10:30, 10:30] = [180, 90, 45]
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@pytest.fixture
def client():
    from main import app

    # Reset model session before each test to avoid state leakage
    _engine._current_session = None

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
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    # Reset model session after each test
    _engine._current_session = None


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
