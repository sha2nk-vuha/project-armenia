import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelSession:
    """One loaded model plus everything needed to run it.

    `session` is whatever the chosen runtime returns (onnxruntime or OpenVINO);
    `input_shape`/`input_name` describe the expected input tensor, and
    map_idx/score_idx point at the anomaly-map and scalar-score outputs within
    the raw output list (see `_get_output_plan`).
    """

    session: object
    runtime: str
    input_name: str
    input_shape: tuple[int, int]
    model_version: str
    map_idx: int = 0
    # Index of the scalar image-score output. None → derive score from the
    # anomaly map's maximum (Anomalib's image score == max pixel score).
    score_idx: int | None = 1


def build_session(model_bytes: bytes, filename: str, model_version: str) -> ModelSession:
    """Load an .onnx payload into a ModelSession.

    The single way a model enters the app: callers hand over raw .onnx bytes
    (an upload or a bundled file path) and receive an inert session. Nothing
    here is global — single-Feature mode keeps its session in the per-Feature
    store (inference/features.py); a Cascade builds one per member model and
    holds them in its own store (see docs/adr/0006, 0007).

    Args:
        model_bytes: Raw contents of an .onnx export.
        filename: Original filename; used only to reject non-.onnx uploads.
        model_version: Version string recorded on the session for reporting.

    Returns:
        A ModelSession ready to run via run_inference_on / run_raw.

    Raises:
        ValueError: If `filename` does not end in ".onnx".
    """
    if not filename.endswith(".onnx"):
        raise ValueError(f"Only .onnx files are accepted; got: {filename!r}")

    session, runtime = _select_runtime(model_bytes)
    input_name, input_shape = _get_onnx_meta(session) if runtime != "openvino" else _get_openvino_meta(session)
    map_idx, score_idx = _get_output_plan(session, runtime)

    logger.info("Model loaded: runtime=%s input_shape=%s", runtime, input_shape)
    return ModelSession(
        session=session,
        runtime=runtime,
        input_name=input_name,
        input_shape=input_shape,
        model_version=model_version,
        map_idx=map_idx,
        score_idx=score_idx,
    )


def _select_runtime(model_bytes: bytes) -> tuple[object, str]:
    """Pick the best available runtime for these bytes.

    Tries CUDA, then OpenVINO, then plain CPU onnxruntime.

    Args:
        model_bytes: Raw .onnx bytes to load.

    Returns:
        Tuple of (opaque runtime session object, runtime name).
    """
    import onnxruntime as ort

    # 1. CUDA
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return _load_onnx_session(model_bytes, ["CUDAExecutionProvider", "CPUExecutionProvider"]), "cuda"

    # 2. OpenVINO
    try:
        session = _load_openvino(model_bytes)
        return session, "openvino"
    except (ImportError, ModuleNotFoundError, RuntimeError, OSError) as e:
        logger.warning("OpenVINO runtime unavailable: %s — falling back to CPU", e)

    # 3. CPU
    return _load_onnx_session(model_bytes, ["CPUExecutionProvider"]), "cpu"


def run_inference_on(sess: "ModelSession", tensor: np.ndarray) -> tuple[np.ndarray, float]:
    """Run one anomaly-model forward pass.

    Args:
        sess: The session whose model to run.
        tensor: Preprocessed input tensor [1,3,H,W] float32.

    Returns:
        Tuple of (anomaly_map [1,1,H,W] float32, scalar pred_score).
    """
    outputs = run_raw(sess, tensor)
    return _extract_outputs(outputs, sess.map_idx, sess.score_idx)


def run_raw(sess: "ModelSession", tensor: np.ndarray) -> list:
    """Run a session's model and return its raw outputs as a list, in port order.

    Feature pipelines that interpret outputs themselves (e.g. detectors) use this
    instead of run_inference_on, which is specific to the anomaly-map layout.

    Args:
        sess: The session whose model to run.
        tensor: Preprocessed input tensor [1,3,H,W] float32.

    Returns:
        Raw output tensors as a list, in the model's port order.
    """
    if sess.runtime == "openvino":
        return _run_openvino_raw(sess.session, tensor)
    return sess.session.run(None, {sess.input_name: tensor})


# ── Private helpers ──────────────────────────────────────────────────────────

@contextmanager
def _temp_onnx_file(model_bytes: bytes) -> Iterator[str]:
    """Yield a temp .onnx path for model_bytes, cleaned up afterwards.

    Centralizes the NamedTemporaryFile boilerplate shared by the ONNX and
    OpenVINO loaders so the DRY principle holds and cleanup is consistent.

    Args:
        model_bytes: Raw .onnx bytes to spill to disk for runtime loaders.

    Yields:
        Filesystem path to the temp file (caller must not unlink it).
    """
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(model_bytes)
        tmp_path = f.name
    try:
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _load_onnx_session(model_bytes: bytes, providers: list[str]) -> object:
    import onnxruntime as ort

    with _temp_onnx_file(model_bytes) as tmp_path:
        return ort.InferenceSession(tmp_path, providers=providers)


def _load_openvino(model_bytes: bytes) -> object:
    from openvino.runtime import Core  # raises ImportError/ModuleNotFoundError if unavailable

    core = Core()
    with _temp_onnx_file(model_bytes) as tmp_path:
        model = core.read_model(tmp_path)
        return core.compile_model(model, "CPU")


def _parse_spatial_dim(d, fallback: int = 392) -> int:
    """Return a concrete spatial dimension; use fallback for symbolic/dynamic dims.

    The default 392 (= 28 × 14) matches the DINOv2-backbone (patch size 14)
    Dinomaly export, whose anomaly map is emitted at 392×392. Feeding the model
    its native resolution keeps inference closest to training.

    Args:
        d: A raw dimension entry from the runtime's metadata, possibly a
            string or symbolic marker.
        fallback: Dimension to use when `d` is symbolic, dynamic, or unparsable.

    Returns:
        A concrete positive spatial dimension.
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


def _ov_dims(partial_shape) -> tuple:
    """Convert an OpenVINO PartialShape to a tuple of ints.

    Reading `.shape` directly raises on dynamic-shaped models, so we inspect
    the partial shape and let callers apply fallbacks for -1.

    Args:
        partial_shape: OpenVINO PartialShape to read.

    Returns:
        One int per dimension; -1 marks a dynamic dimension.
    """
    dims = []
    for d in partial_shape:
        try:
            dims.append(int(d.get_length()) if d.is_static else -1)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            dims.append(-1)
    return tuple(dims)


def _get_openvino_meta(compiled_model) -> tuple[str, tuple[int, int]]:
    inp = compiled_model.input(0)
    shape = _ov_dims(inp.partial_shape())
    h, w = _parse_spatial_dim(shape[2]), _parse_spatial_dim(shape[3])
    return inp.any_name, (h, w)


def _run_openvino_raw(compiled_model, tensor: np.ndarray) -> list:
    """Run the compiled OpenVINO model.

    Args:
        compiled_model: Compiled OpenVINO model returned by Core.compile_model.
        tensor: Preprocessed input tensor [1,3,H,W] float32.

    Returns:
        Raw output tensors as a list, in port order.
    """
    results = compiled_model([tensor])
    return [results[compiled_model.output(i)] for i in range(len(compiled_model.outputs))]


def _classify_outputs(specs: list[tuple[str, tuple]]) -> tuple[int, int | None]:
    """Identify which output is the anomaly map and which is the image score.

    Anomalib ONNX exports vary by version and may emit anomaly_map, pred_score,
    pred_label, and pred_mask in any order. Select by name hint, falling back to
    shape: the anomaly map is the highest-dimensional output; the score is a
    scalar output that is neither a label nor a mask.

    Args:
        specs: (name, shape-tuple) pairs for every model output.

    Returns:
        Tuple of (map_idx, score_idx). score_idx is None when no scalar score
        exists; callers then derive the score from the anomaly map maximum.
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


def _get_output_plan(session, runtime: str) -> tuple[int, int | None]:
    """Inspect model outputs and decide which are the anomaly map and score.

    Defensive: if the outputs cannot be inspected (e.g. a mocked session in
    tests), fall back to the conventional map=#0, score=#1 layout.

    Args:
        session: Runtime session whose outputs are inspected by name and shape.
        runtime: Which runtime produced the session ("openvino" or other).

    Returns:
        Tuple of (map_idx, score_idx); see _classify_outputs.
    """
    try:
        if runtime == "openvino":
            specs = [(o.any_name, tuple(o.shape)) for o in session.outputs]
        else:
            specs = [(o.name, tuple(o.shape)) for o in session.get_outputs()]
        if not specs:
            raise ValueError("model exposes no outputs")

        map_idx, score_idx = _classify_outputs(specs)
        logger.info("Model outputs: %s", specs)
        logger.info(
            "Output plan: anomaly_map=#%d (%s), pred_score=%s",
            map_idx,
            specs[map_idx][0],
            "max(anomaly_map)" if score_idx is None else f"#{score_idx} ({specs[score_idx][0]})",
        )
        return map_idx, score_idx
    except (AttributeError, ValueError, IndexError, KeyError, TypeError, RuntimeError) as e:
        logger.warning(
            "Could not classify model outputs (%s); defaulting to map=#0, score=#1", e
        )
        return 0, 1


def _extract_outputs(outputs: list, map_idx: int, score_idx: int | None) -> tuple[np.ndarray, float]:
    """Pull the anomaly map and a scalar image score from raw model outputs.

    Args:
        outputs: Raw output tensors in the model's port order.
        map_idx: Index of the anomaly-map output.
        score_idx: Index of the scalar score output, or None to derive the
            score from the anomaly map maximum.

    Returns:
        Tuple of (anomaly_map [1,1,H,W] float32, pred_score float).
    """
    anomaly_map = np.asarray(outputs[map_idx], dtype=np.float32)
    if score_idx is not None and score_idx < len(outputs):
        raw = np.asarray(outputs[score_idx]).squeeze()
        try:
            return anomaly_map, float(raw)
        except (TypeError, ValueError):
            pass  # not a scalar — fall through to map maximum
    return anomaly_map, float(anomaly_map.max())
