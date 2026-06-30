import logging
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelSession:
    session: object
    runtime: str
    input_name: str
    input_shape: tuple[int, int]
    model_version: str
    output_names: list[str] = field(default_factory=list)
    map_idx: int = 0
    # Index of the scalar image-score output. None → derive score from the
    # anomaly map's maximum (Anomalib's image score == max pixel score).
    score_idx: int | None = 1


_current_session: ModelSession | None = None


def get_session() -> ModelSession | None:
    return _current_session


def load_model_from_path(path: str, model_version: str) -> ModelSession:
    """Load an .onnx model from a filesystem path (e.g. the bundled default)."""
    with open(path, "rb") as f:
        model_bytes = f.read()
    return load_model(model_bytes, str(path), model_version)


def load_model(model_bytes: bytes, filename: str, model_version: str) -> ModelSession:
    global _current_session

    if not filename.endswith(".onnx"):
        raise ValueError(f"Only .onnx files are accepted; got: {filename!r}")

    session, runtime = _select_runtime(model_bytes)
    input_name, input_shape = _get_onnx_meta(session) if runtime != "openvino" else _get_openvino_meta(session)
    output_names, map_idx, score_idx = _get_output_plan(session, runtime)

    _current_session = ModelSession(
        session=session,
        runtime=runtime,
        input_name=input_name,
        input_shape=input_shape,
        model_version=model_version,
        output_names=output_names,
        map_idx=map_idx,
        score_idx=score_idx,
    )
    logger.info("Model loaded: runtime=%s input_shape=%s", runtime, input_shape)
    return _current_session


def _select_runtime(model_bytes: bytes) -> tuple[object, str]:
    """Try CUDA → OpenVINO → CPU in priority order."""
    import onnxruntime as ort

    # 1. CUDA
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return _load_onnx_session(model_bytes, ["CUDAExecutionProvider", "CPUExecutionProvider"]), "cuda"

    # 2. OpenVINO
    try:
        session = _load_openvino(model_bytes)
        return session, "openvino"
    except Exception as e:
        logger.warning("OpenVINO runtime unavailable: %s — falling back to CPU", e)

    # 3. CPU
    return _load_onnx_session(model_bytes, ["CPUExecutionProvider"]), "cpu"


def run_inference(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns (anomaly_map [1,1,H,W] float32, pred_score float)."""
    sess = _current_session
    if sess is None:
        raise RuntimeError("No model loaded. Call load_model() first.")

    if sess.runtime == "openvino":
        outputs = _run_openvino_raw(sess.session, tensor)
    else:
        outputs = sess.session.run(None, {sess.input_name: tensor})
    return _extract_outputs(outputs, sess.map_idx, sess.score_idx)


# ── Private helpers ──────────────────────────────────────────────────────────

def _load_onnx_session(model_bytes: bytes, providers: list[str]) -> object:
    import onnxruntime as ort

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(model_bytes)
        tmp_path = f.name
    try:
        return ort.InferenceSession(tmp_path, providers=providers)
    finally:
        os.unlink(tmp_path)


def _load_openvino(model_bytes: bytes) -> object:
    from openvino.runtime import Core  # raises ImportError/ModuleNotFoundError if unavailable

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


def _parse_spatial_dim(d, fallback: int = 392) -> int:
    """Return a concrete spatial dimension; use fallback for symbolic/dynamic dims.

    The default 392 (= 28 × 14) matches the DINOv2-backbone (patch size 14)
    Dinomaly export, whose anomaly map is emitted at 392×392. Feeding the model
    its native resolution keeps inference closest to training.
    """
    try:
        v = int(d)
        return v if v > 0 else fallback
    except (TypeError, ValueError):
        return fallback


def _get_onnx_meta(session) -> tuple[str, tuple[int, int]]:
    meta = session.get_inputs()[0]
    shape = meta.shape
    h, w = _parse_spatial_dim(shape[2]), _parse_spatial_dim(shape[3])
    if shape[2] != h or shape[3] != w:
        logger.info("Model has dynamic spatial dims (%s, %s); using %dx%d fallback", shape[2], shape[3], h, w)
    return meta.name, (h, w)


def _get_openvino_meta(compiled_model) -> tuple[str, tuple[int, int]]:
    inp = compiled_model.input(0)
    shape = inp.shape
    h, w = _parse_spatial_dim(shape[2]), _parse_spatial_dim(shape[3])
    return inp.any_name, (h, w)


def _run_openvino_raw(compiled_model, tensor: np.ndarray) -> list:
    """Run the OpenVINO model and return outputs as a list in port order."""
    results = compiled_model([tensor])
    return [results[compiled_model.output(i)] for i in range(len(compiled_model.outputs))]


def _classify_outputs(specs: list[tuple[str, tuple]]) -> tuple[int, int | None]:
    """Identify which output is the anomaly map and which is the image score.

    Anomalib ONNX exports vary by version and may emit anomaly_map, pred_score,
    pred_label, and pred_mask in any order. Select by name hint, falling back to
    shape: the anomaly map is the highest-dimensional output; the score is a
    scalar output that is neither a label nor a mask. Returns (map_idx,
    score_idx); score_idx is None when no scalar score exists (derive from the
    map maximum).
    """
    # anomaly map — prefer an explicit name, else the most-dimensional output
    map_idx = next(
        (i for i, (name, _) in enumerate(specs)
         if "anomaly" in name.lower() or "map" in name.lower()),
        None,
    )
    if map_idx is None:
        map_idx = max(range(len(specs)), key=lambda i: len(specs[i][1]))

    # score — prefer a name containing "score"
    score_idx = next(
        (i for i, (name, _) in enumerate(specs)
         if i != map_idx and "score" in name.lower()),
        None,
    )
    # else the first scalar-ish output that is not a label or mask
    if score_idx is None:
        score_idx = next(
            (i for i, (name, shape) in enumerate(specs)
             if i != map_idx
             and "label" not in name.lower()
             and "mask" not in name.lower()
             and len(shape) <= 1),
            None,
        )
    return map_idx, score_idx


def _get_output_plan(session, runtime: str) -> tuple[list[str], int, int | None]:
    """Inspect model outputs and decide which are the anomaly map and score.

    Defensive: if the outputs cannot be inspected (e.g. a mocked session in
    tests), fall back to the conventional map=#0, score=#1 layout.
    """
    try:
        if runtime == "openvino":
            specs = [(o.any_name, tuple(o.shape)) for o in session.outputs]
        else:
            specs = [(o.name, tuple(o.shape)) for o in session.get_outputs()]
        if not specs:
            raise ValueError("model exposes no outputs")

        names = [s[0] for s in specs]
        map_idx, score_idx = _classify_outputs(specs)
        logger.info("Model outputs: %s", specs)
        logger.info(
            "Output plan: anomaly_map=#%d (%s), pred_score=%s",
            map_idx,
            names[map_idx],
            "max(anomaly_map)" if score_idx is None else f"#{score_idx} ({names[score_idx]})",
        )
        return names, map_idx, score_idx
    except Exception as e:
        logger.warning(
            "Could not classify model outputs (%s); defaulting to map=#0, score=#1", e
        )
        return [], 0, 1


def _extract_outputs(outputs: list, map_idx: int, score_idx: int | None) -> tuple[np.ndarray, float]:
    """Pull the anomaly map and a scalar image score from the raw model outputs."""
    anomaly_map = np.asarray(outputs[map_idx], dtype=np.float32)
    if score_idx is not None and score_idx < len(outputs):
        raw = np.asarray(outputs[score_idx]).squeeze()
        try:
            return anomaly_map, float(raw)
        except (TypeError, ValueError):
            pass  # not a scalar — fall through to map maximum
    return anomaly_map, float(anomaly_map.max())
