"""RF-DETR–specific knowledge for the Presence/Absence Feature.

This isolates everything that depends on the exact RF-DETR ONNX export — the
preprocessing parameters, the raw-output decode, and box drawing — from the
Feature-agnostic pipeline orchestration in `pipeline.py`.

Confirmed against the rfdetr-nano export: input `[1,3,384,384]`, outputs
`dets [1,300,4]` (normalised cxcywh boxes) and `labels [1,300,C]` (per-class
logits; probabilities via sigmoid). The decode identifies boxes vs logits by
shape, so output order does not matter.
"""
from dataclasses import dataclass, field
import io
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Standard ImageNet normalisation, the RF-DETR default. Overridable via sidecar.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DEFAULT_INPUT_SIZE = (384, 384)


@dataclass
class Detection:
    """One detected object in original-image pixel coordinates."""

    class_id: int
    confidence: float
    box: tuple[float, float, float, float]  # xyxy


@dataclass
class PresenceConfig:
    """Model-scoped configuration, sourced from the model sidecar JSON.

    Carries the Class Catalog plus the RF-DETR preprocessing parameters. Fields
    left unset fall back to RF-DETR defaults so the common case is zero-config.
    """

    labels: dict[int, str] = field(default_factory=dict)
    input_size: tuple[int, int] = _DEFAULT_INPUT_SIZE  # (height, width)
    mean: tuple[float, float, float] = _IMAGENET_MEAN
    std: tuple[float, float, float] = _IMAGENET_STD
    normalize: bool = True  # False when normalisation is baked into the graph


def sidecar_path(model_path: str) -> str:
    """The sidecar JSON path for a model: same directory and stem, `.json`.

    e.g. `model/rfdetr-nano.onnx` -> `model/rfdetr-nano.json`.
    """
    return str(Path(model_path).with_suffix(".json"))


def load_config(path: str) -> PresenceConfig:
    """Load a `PresenceConfig` from a sidecar JSON, defaulting when absent.

    The sidecar carries the Class Catalog (`labels`, index->name) and optional
    preprocessing overrides. Missing files or fields fall back to RF-DETR
    defaults so the Feature stays usable (with raw class indices) even without a
    sidecar. JSON object keys are strings, so `labels` keys are coerced to int.
    """
    p = Path(path)
    if not p.is_file():
        logger.info("No RF-DETR sidecar at %s; using defaults (raw class indices).", path)
        return PresenceConfig()

    data = json.loads(p.read_text())
    defaults = PresenceConfig()
    labels = {int(k): str(v) for k, v in data.get("labels", {}).items()}
    input_size = data.get("input_size")
    mean = data.get("mean")
    std = data.get("std")
    return PresenceConfig(
        labels=labels,
        input_size=tuple(input_size) if input_size else defaults.input_size,
        mean=tuple(mean) if mean else defaults.mean,
        std=tuple(std) if std else defaults.std,
        normalize=data.get("normalize", defaults.normalize),
    )


def evaluate_presence(
    detections: list[Detection],
    expected_classes: list[int],
    threshold: float,
) -> str:
    """Expected-object rule: OK only if every Expected Class is present.

    A class counts as present when at least one detection of that class scores
    at or above `threshold` (the detection-confidence floor). See docs/adr and
    CONTEXT.md ("Expected Class").
    """
    present = {d.class_id for d in detections if d.confidence >= threshold}
    all_present = all(cls in present for cls in expected_classes)
    return "ok" if all_present else "not_ok"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def preprocess_image(
    image_bytes: bytes, config: PresenceConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Resize, scale, and (optionally) normalise an image for RF-DETR.

    Returns (tensor [1,3,H,W] float32, original_rgb [H,W,3] uint8). Unlike the
    Anomalib preprocessor, RF-DETR normalisation is applied here (mean/std) when
    `config.normalize` is set; disable it if the export bakes normalisation in.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_rgb = np.array(image, dtype=np.uint8)

    h, w = config.input_size
    resized = cv2.resize(original_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    scaled = resized.astype(np.float32) / 255.0

    if config.normalize:
        mean = np.array(config.mean, dtype=np.float32)
        std = np.array(config.std, dtype=np.float32)
        scaled = (scaled - mean) / std

    tensor = scaled.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]
    return tensor, original_rgb


def _split_logits_and_boxes(outputs: list) -> tuple[np.ndarray, np.ndarray]:
    """Identify the logits and boxes tensors among the raw model outputs.

    The boxes tensor is the one whose last dimension is 4 (cxcywh); the other is
    the class logits. Selecting by shape rather than position makes the decode
    robust to output ordering across exports.
    """
    arrays = [np.asarray(o) for o in outputs]
    box_idx = next((i for i, a in enumerate(arrays) if a.shape[-1] == 4), None)
    if box_idx is None:
        raise ValueError("No box output (last dim == 4) found among model outputs.")
    logits_idx = next(i for i in range(len(arrays)) if i != box_idx)
    return arrays[logits_idx], arrays[box_idx]


def decode_detections(
    outputs: list,
    orig_hw: tuple[int, int],
    conf_threshold: float,
    config: PresenceConfig,
) -> list[Detection]:
    """Turn raw RF-DETR outputs into Detections in original-image pixels.

    Assumes DETR-style outputs: per-query class logits and normalised cxcywh
    boxes. Each query is reduced to its top-scoring class (confidence via
    sigmoid); queries below `conf_threshold` are dropped.
    """
    logits, boxes = _split_logits_and_boxes(outputs)
    logits = logits.reshape(-1, logits.shape[-1])  # [Q, C]
    boxes = boxes.reshape(-1, 4)  # [Q, 4]

    probs = _sigmoid(logits)
    class_ids = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    orig_h, orig_w = orig_hw
    detections: list[Detection] = []
    for cls, conf, (cx, cy, bw, bh) in zip(class_ids, confidences, boxes):
        if conf < conf_threshold:
            continue
        x1 = (cx - bw / 2) * orig_w
        y1 = (cy - bh / 2) * orig_h
        x2 = (cx + bw / 2) * orig_w
        y2 = (cy + bh / 2) * orig_h
        detections.append(
            Detection(class_id=int(cls), confidence=float(conf), box=(x1, y1, x2, y2))
        )
    return detections


def draw_detections(
    original_rgb: np.ndarray,
    detections: list[Detection],
    labels: dict[int, str],
) -> bytes:
    """Draw detection boxes + class/confidence labels on the original image.

    Returns JPEG bytes. Boxes are drawn in the image's own pixel coordinates
    (the decode already scaled them to the original size).
    """
    result_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.box)
        cv2.rectangle(result_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        name = labels.get(det.class_id, str(det.class_id))
        caption = f"{name} {det.confidence:.2f}"
        cv2.putText(
            result_bgr,
            caption,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )

    _, buf = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()
