import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock

from inference.engine import ModelSession
from inference.pipeline import AnomalyPipeline, InferenceResult


def _png_bytes() -> bytes:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[20:44, 20:44] = [200, 100, 50]
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _mock_onnx_session(input_shape=(32, 32), pred_score=0.75):
    session = MagicMock()
    amap = np.zeros((1, 1, input_shape[0], input_shape[1]), dtype=np.float32)
    amap[0, 0, 10:22, 10:22] = 0.9
    session.run.return_value = [amap, np.array([pred_score], dtype=np.float32)]
    return session


def _anomaly_pipeline(input_shape=(32, 32), pred_score=0.75) -> AnomalyPipeline:
    model = ModelSession(
        session=_mock_onnx_session(input_shape, pred_score),
        runtime="cpu",
        input_name="input",
        input_shape=input_shape,
        model_version="v1.0-test",
    )
    return AnomalyPipeline(model)


def test_anomaly_pipeline_returns_verdict_and_visualizations():
    pipe = _anomaly_pipeline(pred_score=0.75)
    result = pipe.infer(_png_bytes(), threshold=0.5)

    assert isinstance(result, InferenceResult)
    assert result.verdict == "not_ok"
    assert set(result.images) == {"heatmap", "segmentation"}
    assert isinstance(result.images["heatmap"], bytes)
    assert isinstance(result.images["segmentation"], bytes)


def test_anomaly_pipeline_verdict_ok_below_threshold():
    pipe = _anomaly_pipeline(pred_score=0.2)
    result = pipe.infer(_png_bytes(), threshold=0.5)

    assert result.verdict == "ok"
    assert result.score == pytest.approx(0.2, abs=1e-6)



def _mock_detector_session(favored_class=1, num_classes=2):
    """A mock ONNX detector: one confident query on `favored_class`, one dead query."""
    session = MagicMock()
    logits = np.full((1, 2, num_classes), -5.0, dtype=np.float32)
    logits[0, 0, favored_class] = 5.0  # query 0 strongly predicts favored_class
    boxes = np.array(
        [[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]]], dtype=np.float32
    )
    session.run.return_value = [logits, boxes]
    return session


def _presence_pipeline(expected_classes, favored_class=1):
    from inference.pipeline import PresenceAbsencePipeline
    from inference.rfdetr import PresenceConfig

    model = ModelSession(
        session=_mock_detector_session(favored_class),
        runtime="cpu",
        input_name="input",
        input_shape=(20, 20),
        model_version="rf-detr-test",
    )
    config = PresenceConfig(input_size=(20, 20), labels={0: "bg", 1: "cap"})
    return PresenceAbsencePipeline(model, config, expected_classes)


def test_presence_pipeline_ok_when_expected_class_present():
    pipe = _presence_pipeline(expected_classes=[1], favored_class=1)
    result = pipe.infer(_png_bytes(), threshold=0.5)

    assert isinstance(result, InferenceResult)
    assert result.verdict == "ok"
    assert "annotated" in result.images
    assert isinstance(result.images["annotated"], bytes)
    assert result.detections and result.detections[0]["label"] == "cap"


def test_presence_pipeline_nok_when_expected_class_absent():
    pipe = _presence_pipeline(expected_classes=[1], favored_class=0)
    result = pipe.infer(_png_bytes(), threshold=0.5)

    assert result.verdict == "not_ok"
