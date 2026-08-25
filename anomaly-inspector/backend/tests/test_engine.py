import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from inference.engine import ModelSession, build_session, run_inference_on


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


def _build_cpu_session(mock_session, version="v1.0-test"):
    """Run build_session with CUDA/OpenVINO patched out, forcing the CPU path."""
    with patch("inference.engine._load_onnx_session", return_value=mock_session):
        with patch("inference.engine._get_onnx_meta", return_value=("input", (256, 256))):
            with patch("inference.engine._load_openvino", side_effect=ImportError):
                import onnxruntime as ort
                with patch.object(
                    ort, "get_available_providers", return_value=["CPUExecutionProvider"]
                ):
                    return build_session(b"fake", "model.onnx", version)


def test_build_session_loads_cpu_session():
    result = _build_cpu_session(_make_mock_onnx_session())
    assert result.model_version == "v1.0-test"
    assert result.runtime == "cpu"
    assert result.input_shape == (256, 256)


def test_build_session_rejects_non_onnx():
    with pytest.raises(ValueError, match="Only .onnx files"):
        build_session(b"fake", "model.xml", "v1.0")


def test_run_inference_on_returns_correct_shapes():
    sess = ModelSession(
        session=_make_mock_onnx_session(input_shape=(256, 256)),
        runtime="cpu",
        input_name="input",
        input_shape=(256, 256),
        model_version="v1.0",
    )

    tensor = np.random.rand(1, 3, 256, 256).astype(np.float32)
    anomaly_map, pred_score = run_inference_on(sess, tensor)

    assert anomaly_map.shape == (1, 1, 256, 256)
    assert isinstance(pred_score, float)
    assert pred_score == pytest.approx(0.75)


def test_build_session_falls_back_to_cpu_when_openvino_unavailable():
    mock_session = _make_mock_onnx_session()
    with patch("inference.engine._load_openvino", side_effect=ImportError("no openvino")):
        with patch("inference.engine._load_onnx_session", return_value=mock_session) as mock_load:
            assert mock_load is not None
            result = _build_cpu_session(mock_session, version="v1.0")
    assert result.runtime == "cpu"


def test_build_session_uses_openvino_when_no_cuda():
    mock_ov_session = MagicMock()
    with patch("inference.engine._load_openvino", return_value=mock_ov_session):
        with patch("inference.engine._get_openvino_meta", return_value=("input", (256, 256))):
            import onnxruntime as ort
            with patch.object(
                ort, "get_available_providers", return_value=["CPUExecutionProvider"]
            ):
                result = build_session(b"fake", "model.onnx", "v1.0-ov")
    assert result.runtime == "openvino"
    assert result.model_version == "v1.0-ov"


def test_build_session_defaults_output_plan_for_inspectable_outputs():
    """A real-looking session gets its map/score outputs classified by name."""
    mock_session = _make_mock_onnx_session()
    map_out, score_out = MagicMock(), MagicMock()
    map_out.name, score_out.name = "anomaly_map", "pred_score"
    map_out.shape, score_out.shape = (1, 1, 256, 256), (1,)
    mock_session.get_outputs.return_value = [map_out, score_out]

    sess = _build_cpu_session(mock_session)
    assert (sess.map_idx, sess.score_idx) == (0, 1)


def test_build_session_falls_back_when_outputs_uninspectable():
    """A mocked session whose outputs cannot be inspected still builds."""
    sess = _build_cpu_session(_make_mock_onnx_session())
    assert (sess.map_idx, sess.score_idx) == (0, 1)
