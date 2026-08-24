from unittest.mock import MagicMock, patch

import pytest

import inference.engine as engine
import inference.features as features
from config import ANOMALY_FEATURE, PRESENCE_FEATURE
from inference.pipeline import AnomalyPipeline, PresenceAbsencePipeline
from inference.rfdetr import ModelConfig


@pytest.fixture(autouse=True)
def _clean_state():
    engine._current_session = None
    features.reset()
    yield
    engine._current_session = None
    features.reset()


def _session(version="v-test"):
    return engine.ModelSession(
        session=MagicMock(),
        runtime="cpu",
        input_name="input",
        input_shape=(32, 32),
        model_version=version,
    )


def test_no_active_feature_by_default():
    assert features.get_active_feature() is None
    assert features.current_pipeline() is None


def test_activate_anomaly_loads_default_model_and_builds_pipeline():
    with patch("inference.features.engine.load_model_from_path", return_value=_session()) as load:
        features.activate(ANOMALY_FEATURE)

    load.assert_called_once()
    assert features.get_active_feature() == ANOMALY_FEATURE
    engine._current_session = load.return_value  # emulate engine holding it
    pipe = features.current_pipeline()
    assert isinstance(pipe, AnomalyPipeline)


def test_activate_presence_loads_sidecar_and_builds_presence_pipeline():
    sess = _session("rfdetr")
    cfg = ModelConfig(
        labels={0: "gasket"},
        rule_params={"expected_classes": {"expected_classes": [0]}},
    )
    with patch("inference.features.engine.load_model_from_path", return_value=sess), \
         patch("inference.features.load_config", return_value=cfg):
        features.activate(PRESENCE_FEATURE)

    assert features.get_active_feature() == PRESENCE_FEATURE
    engine._current_session = sess
    pipe = features.current_pipeline()
    assert isinstance(pipe, PresenceAbsencePipeline)
    # Expected Class policy flows from the sidecar into the pipeline.
    assert pipe.expected_classes == [0]


def test_activate_unknown_feature_raises():
    with pytest.raises(ValueError):
        features.activate("not_a_feature")


def test_activate_missing_model_file_raises():
    with patch("inference.features.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            features.activate(ANOMALY_FEATURE)


def test_register_upload_sets_active_feature_without_reloading_default():
    # Upload path: engine already holds the uploaded session; the manager just
    # tracks which Feature it belongs to (no default-model reload).
    sess = _session("uploaded")
    engine._current_session = sess
    with patch("inference.features.engine.load_model_from_path") as load:
        features.register_upload(ANOMALY_FEATURE)

    load.assert_not_called()
    assert features.get_active_feature() == ANOMALY_FEATURE
    assert isinstance(features.current_pipeline(), AnomalyPipeline)
