import logging
import os
import tempfile
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelSession:
    session: object
    runtime: str
    input_name: str
    input_shape: tuple[int, int]
    model_version: str


_current_session: ModelSession | None = None


def get_session() -> ModelSession | None:
    return _current_session


def load_model(model_bytes: bytes, filename: str, model_version: str) -> ModelSession:
    global _current_session

    if filename.endswith(".xml"):
        session = _load_openvino(model_bytes)
        runtime = "openvino"
        input_name, input_shape = _get_openvino_meta(session)
    else:
        session, runtime = _load_onnx(model_bytes)
        input_name, input_shape = _get_onnx_meta(session)

    _current_session = ModelSession(
        session=session,
        runtime=runtime,
        input_name=input_name,
        input_shape=input_shape,
        model_version=model_version,
    )
    logger.info("Model loaded: runtime=%s input_shape=%s", runtime, input_shape)
    return _current_session


def run_inference(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns (anomaly_map [1,1,H,W] float32, pred_score float)."""
    sess = _current_session
    if sess is None:
        raise RuntimeError("No model loaded. Call load_model() first.")

    if sess.runtime == "openvino":
        return _run_openvino(sess.session, tensor)
    return _run_onnx(sess.session, sess.input_name, tensor)


# ── Private helpers ──────────────────────────────────────────────────────────

def _load_onnx(model_bytes: bytes) -> tuple[object, str]:
    import onnxruntime as ort

    providers: list[str]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        runtime = "cuda"
    else:
        providers = ["CPUExecutionProvider"]
        runtime = "cpu"

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(model_bytes)
        tmp_path = f.name
    try:
        session = ort.InferenceSession(tmp_path, providers=providers)
    finally:
        os.unlink(tmp_path)

    return session, runtime


def _load_openvino(model_bytes: bytes) -> object:
    from openvino.runtime import Core

    core = Core()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(model_bytes)
        tmp_path = f.name
    try:
        model = core.read_model(tmp_path)
        compiled = core.compile_model(model, "CPU")
    finally:
        os.unlink(tmp_path)

    return compiled


def _get_onnx_meta(session) -> tuple[str, tuple[int, int]]:
    meta = session.get_inputs()[0]
    shape = meta.shape
    return meta.name, (int(shape[2]), int(shape[3]))


def _get_openvino_meta(compiled_model) -> tuple[str, tuple[int, int]]:
    inp = compiled_model.input(0)
    shape = inp.shape
    return inp.any_name, (int(shape[2]), int(shape[3]))


def _run_onnx(session, input_name: str, tensor: np.ndarray) -> tuple[np.ndarray, float]:
    outputs = session.run(None, {input_name: tensor})
    anomaly_map: np.ndarray = outputs[0]
    pred_score = float(outputs[1].squeeze())
    return anomaly_map, pred_score


def _run_openvino(compiled_model, tensor: np.ndarray) -> tuple[np.ndarray, float]:
    results = compiled_model([tensor])
    out0 = results[compiled_model.output(0)]
    out1 = results[compiled_model.output(1)]
    return out0, float(out1.squeeze())
