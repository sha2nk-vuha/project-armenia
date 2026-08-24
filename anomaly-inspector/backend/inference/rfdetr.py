"""RF-DETR–specific knowledge for the Presence/Absence Feature.

This isolates everything that depends on the exact RF-DETR ONNX export — the
preprocessing parameters, the raw-output decode, and box drawing — from the
Feature-agnostic pipeline orchestration in `pipeline.py`.

Confirmed against the rfdetr-nano export: input `[1,3,384,384]`, outputs
`dets [1,300,4]` (normalised cxcywh boxes) and `labels [1,300,C]` (per-class
logits; probabilities via sigmoid). The decode identifies boxes vs logits by
shape, so output order does not matter.

This two-output decode is detection-only. A segmentation export emits a third
`masks` tensor and is decoded by `rfdetr_seg.py`; do not route it here — the
shape-based split would take `masks` as the logits tensor.
"""
from dataclasses import dataclass
import io
import logging

import cv2
import numpy as np
from PIL import Image

from inference.model_config import ModelConfig, load_config, sidecar_path  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """One detected object in original-image pixel coordinates."""

    class_id: int
    confidence: float
    box: tuple[float, float, float, float]  # xyxy


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def preprocess_image(
    image_bytes: bytes, config: ModelConfig
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
    config: ModelConfig,
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


def draw_detections_rgb(
    original_rgb: np.ndarray,
    detections: list[Detection],
    labels: dict[int, str],
) -> np.ndarray:
    """Draw detection boxes + class/confidence labels; returns an RGB array.

    Boxes are drawn in the image's own pixel coordinates (the decode already
    scaled them to the original size). Returning an array rather than encoded
    bytes lets a pipeline composite a Decision Rule's annotations on top before
    a single final encode.
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
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def draw_detections(
    original_rgb: np.ndarray,
    detections: list[Detection],
    labels: dict[int, str],
) -> bytes:
    """Draw detection boxes + class/confidence labels. Returns JPEG bytes."""
    annotated = draw_detections_rgb(original_rgb, detections, labels)
    _, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    return buf.tobytes()
