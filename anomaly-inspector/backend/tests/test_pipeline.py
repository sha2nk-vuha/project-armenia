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


def _presence_pipeline(favored_class=1):
    from inference.pipeline import PresenceAbsencePipeline
    from inference.model_config import ModelConfig

    model = ModelSession(
        session=_mock_detector_session(favored_class),
        runtime="cpu",
        input_name="input",
        input_shape=(20, 20),
        model_version="rf-detr-test",
    )
    config = ModelConfig(input_size=(20, 20), labels={0: "bg", 1: "cap"})
    return PresenceAbsencePipeline(model, config)


def test_presence_pipeline_ok_when_expected_class_present():
    # Expected Classes now travel per request (operator/GUI), not the sidecar.
    pipe = _presence_pipeline(favored_class=1)
    result = pipe.infer(_png_bytes(), threshold=0.5, rule_params={"expected_classes": [1]})

    assert isinstance(result, InferenceResult)
    assert result.verdict == "ok"
    assert "annotated" in result.images
    assert isinstance(result.images["annotated"], bytes)
    assert result.detections and result.detections[0]["label"] == "cap"


def test_presence_pipeline_nok_when_expected_class_absent():
    pipe = _presence_pipeline(favored_class=0)
    result = pipe.infer(_png_bytes(), threshold=0.5, rule_params={"expected_classes": [1]})

    assert result.verdict == "not_ok"


def _mock_seg_session(num_queries=2, num_classes=3, mask_hw=(8, 8)):
    """A mock segmentation ONNX: query 0 is a confident class-1 instance."""
    session = MagicMock()
    logits = np.full((1, num_queries, num_classes), -8.0, dtype=np.float32)
    logits[0, 0, 1] = 8.0
    boxes = np.tile(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (1, num_queries, 1))
    masks = np.full((1, num_queries, *mask_hw), -8.0, dtype=np.float32)
    masks[0, 0, 2:6, 2:6] = 8.0
    session.run.return_value = [boxes, logits, masks]
    return session


def _segmentation_pipeline(labels=None, rule_params=None):
    from inference.model_config import ModelConfig
    from inference.pipeline import SegmentationPipeline

    model = ModelSession(
        session=_mock_seg_session(),
        runtime="cpu",
        input_name="input",
        input_shape=(32, 32),
        model_version="seg-test",
    )
    config = ModelConfig(
        input_size=(32, 32),
        labels=labels if labels is not None else {0: "bottle_cap", 1: "logo"},
        rule_params=rule_params or {},
    )
    return SegmentationPipeline(model, config)


def test_segmentation_pipeline_advertises_both_output_kinds():
    # The payoff of typing rules on output kind: masks *and* detections.
    pipe = _segmentation_pipeline()
    assert pipe.kinds == frozenset({"detections", "masks"})
    assert "expected_classes" in {r.name for r in pipe.compatible_rules()}


def test_segmentation_pipeline_runs_a_detection_rule_unchanged():
    pipe = _segmentation_pipeline()
    result = pipe.infer(
        _png_bytes(), threshold=0.5, rule_params={"expected_classes": ["logo"]}
    )

    assert result.verdict == "ok"
    assert result.decision_rule == "expected_classes"
    assert isinstance(result.images["annotated"], bytes)


def test_segmentation_pipeline_nok_when_expected_class_absent():
    pipe = _segmentation_pipeline()
    result = pipe.infer(
        _png_bytes(), threshold=0.5, rule_params={"expected_classes": ["bottle_cap"]}
    )
    assert result.verdict == "not_ok"


def test_segmentation_detections_carry_mask_area():
    pipe = _segmentation_pipeline()
    result = pipe.infer(_png_bytes(), threshold=0.5)

    assert len(result.detections) == 1
    det = result.detections[0]
    assert det["label"] == "logo"
    assert det["mask_area_px"] > 0


def test_rule_params_default_from_schema_not_sidecar():
    # The sidecar no longer seeds rule params: with none supplied, the rule's
    # own schema default applies (Expected Classes defaults to empty -> every
    # image passes), and the request still overrides it.
    pipe = _segmentation_pipeline(
        rule_params={"expected_classes": {"expected_classes": ["bottle_cap"]}}
    )
    # Sidecar rule_params are ignored now, so the empty schema default passes.
    assert pipe.infer(_png_bytes(), threshold=0.5).verdict == "ok"
    # A request naming an absent class still fails.
    assert pipe.infer(
        _png_bytes(), threshold=0.5, rule_params={"expected_classes": ["bottle_cap"]}
    ).verdict == "not_ok"


def test_segmentation_rejects_an_incompatible_rule():
    pipe = _segmentation_pipeline()
    with pytest.raises(ValueError, match="anomaly_threshold"):
        pipe.infer(_png_bytes(), threshold=0.5, rule_name="anomaly_threshold")
