"""Per-Feature model store: activation, upload, and pipeline construction.

No model auto-loads (docs/adr/0007): activating a Feature only selects it, and a
model must be uploaded before inference. Both single-Feature and Cascade modes
read the same store.
"""
from unittest.mock import MagicMock, patch

import pytest

import inference.features as features
from config import ANOMALY_FEATURE, CASCADE_FEATURE, PRESENCE_FEATURE, SEGMENTATION_FEATURE
from inference.engine import ModelSession
from inference.model_config import ModelConfig
from inference.pipeline import AnomalyPipeline, PresenceAbsencePipeline, SegmentationPipeline


@pytest.fixture(autouse=True)
def _clean():
    features.reset()
    yield
    features.reset()


def _session(version="v-test", input_shape=(32, 32)):
    return ModelSession(
        session=MagicMock(), runtime="cpu", input_name="input",
        input_shape=input_shape, model_version=version,
    )


def _install(feature, session=None, config=None):
    features._models[feature] = (session or _session(), config or ModelConfig())


def test_no_active_feature_or_model_by_default():
    assert features.get_active_feature() is None
    assert features.current_pipeline() is None
    assert features.loaded_models() == {}


def test_activate_selects_without_loading_a_model():
    features.activate(ANOMALY_FEATURE)
    assert features.get_active_feature() == ANOMALY_FEATURE
    # No model uploaded yet -> no pipeline.
    assert features.current_pipeline() is None


def test_activate_unknown_feature_raises():
    with pytest.raises(ValueError):
        features.activate("not_a_feature")


def test_upload_builds_the_anomaly_pipeline():
    sess = _session()
    with patch("inference.features.engine.build_session", return_value=sess) as build:
        features.upload_model(ANOMALY_FEATURE, b"onnx", "m.onnx", "v1")
    build.assert_called_once()
    features.activate(ANOMALY_FEATURE)
    assert isinstance(features.current_pipeline(), AnomalyPipeline)


def test_upload_segmentation_takes_input_size_from_the_model_and_sidecar_labels():
    sess = _session(input_shape=(312, 312))
    with patch("inference.features.engine.build_session", return_value=sess):
        features.upload_model(
            SEGMENTATION_FEATURE, b"onnx", "m.onnx", "seg-v1",
            sidecar={"labels": {"0": "bottle_cap", "1": "logo"}, "input_size": [999, 999]},
        )
    _, cfg = features.get_model(SEGMENTATION_FEATURE)
    assert cfg.labels == {0: "bottle_cap", 1: "logo"}
    # Preprocess size comes from the model, not the sidecar's stale value.
    assert cfg.input_size == (312, 312)
    features.activate(SEGMENTATION_FEATURE)
    assert isinstance(features.current_pipeline(), SegmentationPipeline)


def test_upload_presence_pipeline():
    sess = _session(input_shape=(20, 20))
    with patch("inference.features.engine.build_session", return_value=sess):
        features.upload_model(PRESENCE_FEATURE, b"onnx", "m.onnx", "v1")
    features.activate(PRESENCE_FEATURE)
    assert isinstance(features.current_pipeline(), PresenceAbsencePipeline)


def test_upload_replaces_the_previous_model_for_that_feature():
    with patch("inference.features.engine.build_session", return_value=_session("v1")):
        features.upload_model(ANOMALY_FEATURE, b"a", "m.onnx", "v1")
    with patch("inference.features.engine.build_session", return_value=_session("v2")):
        features.upload_model(ANOMALY_FEATURE, b"b", "m.onnx", "v2")
    assert features.loaded_models()[ANOMALY_FEATURE]["model_version"] == "v2"


def test_cannot_upload_a_model_for_the_cascade_feature():
    with pytest.raises(ValueError):
        features.upload_model(CASCADE_FEATURE, b"onnx", "m.onnx", "v1")


def test_loaded_models_reports_each_upload():
    _install(ANOMALY_FEATURE, _session("a"))
    _install(SEGMENTATION_FEATURE, _session("s"))
    loaded = features.loaded_models()
    assert set(loaded) == {ANOMALY_FEATURE, SEGMENTATION_FEATURE}
    assert loaded[ANOMALY_FEATURE]["model_version"] == "a"


def test_active_model_tracks_the_active_feature():
    _install(SEGMENTATION_FEATURE, _session("s"))
    features.activate(SEGMENTATION_FEATURE)
    assert features.active_model().model_version == "s"
    # The cascade Feature has no single model of its own.
    features.activate(CASCADE_FEATURE)
    assert features.active_model() is None


def test_clear_model_unloads():
    _install(ANOMALY_FEATURE)
    assert features.clear_model(ANOMALY_FEATURE) is True
    assert features.clear_model(ANOMALY_FEATURE) is False
    assert features.loaded_models() == {}
