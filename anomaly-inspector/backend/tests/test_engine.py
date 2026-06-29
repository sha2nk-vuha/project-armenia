import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import inference.engine as engine_module
from inference.engine import ModelSession, get_session, run_inference


@pytest.fixture(autouse=True)
def reset_session():
    engine_module._current_session = None
    yield
    engine_module._current_session = None


def _make_mock_onnx_session(input_shape=(256, 256)):
    session = MagicMock()
    input_meta = MagicMock()
    input_meta.name = "input"
    input_meta.shape = [1, 3, input_shape[0], input_shape[1]]
    session.get_inputs.return_value = [input_meta]
    anomaly_map = np.random.rand(1, 1, input_shape[0], input_shape[1]).astype(np.float32)
    pred_score = np.array([0.75], dtype=np.float32)
    session.run.return_value = [anomaly_map, pred_score]
    return session


def test_get_session_returns_none_before_load():
    assert get_session() is None


def test_load_model_sets_session(tmp_path):
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_bytes(b"fake")

    mock_session = _make_mock_onnx_session()

    with patch("inference.engine._load_onnx", return_value=(mock_session, "cpu")):
        result = engine_module.load_model(b"fake", "model.onnx", "v1.0-test")

    assert get_session() is not None
    assert result.model_version == "v1.0-test"
    assert result.runtime == "cpu"
    assert result.input_shape == (256, 256)


def test_run_inference_returns_correct_shapes():
    mock_session = _make_mock_onnx_session(input_shape=(256, 256))

    engine_module._current_session = ModelSession(
        session=mock_session,
        runtime="cpu",
        input_name="input",
        input_shape=(256, 256),
        model_version="v1.0",
    )

    tensor = np.random.rand(1, 3, 256, 256).astype(np.float32)
    anomaly_map, pred_score = run_inference(tensor)

    assert anomaly_map.shape == (1, 1, 256, 256)
    assert isinstance(pred_score, float)


def test_run_inference_raises_when_no_model_loaded():
    with pytest.raises(RuntimeError, match="No model loaded"):
        run_inference(np.zeros((1, 3, 256, 256), dtype=np.float32))


def test_load_model_replaces_existing_session():
    mock_session = _make_mock_onnx_session()
    with patch("inference.engine._load_onnx", return_value=(mock_session, "cpu")):
        engine_module.load_model(b"fake", "model.onnx", "v1.0")
        engine_module.load_model(b"fake", "model.onnx", "v2.0")
    assert get_session().model_version == "v2.0"
