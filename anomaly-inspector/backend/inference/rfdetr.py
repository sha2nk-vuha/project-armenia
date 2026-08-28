"""RF-DETR–specific knowledge for the detector Features.

This isolates everything that depends on the exact RF-DETR ONNX export — the
preprocessing parameters, the raw-output decode, and box drawing — from the
Feature-agnostic pipeline orchestration in `pipeline.py`. The decode
primitives shared with the segmentation sibling (`sigmoid`, `query_probs`,
`unnormalize_cxcywh`) live here too; `rfdetr_seg.py` imports them rather than
re-deriving the arithmetic.

Confirmed against the rfdetr-nano export: input `[1,3,384,384]`, outputs
`dets [1,300,4]` (normalised cxcywh boxes) and `labels [1,300,C]` (per-class
logits; probabilities via sigmoid). The decode identifies boxes vs logits by
shape, so output order does not matter.

This two-output decode is detection-only. A segmentation export emits a third
`masks` tensor and is decoded by `rfdetr_seg.py`; do not route it here — the
shape-based split would take `masks` as the logits tensor.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from inference.model_config import ModelConfig
from inference.preprocessor import chw_tensor, decode_rgb


@dataclass
class Detection:
    """One detected object in original-image pixel coordinates."""

    class_id: int
    confidence: float
    box: tuple[float, float, float, float]  # xyxy


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise logistic sigmoid.

    RF-DETR logits are per-class, so a sigmoid (not softmax) yields each
    class's probability independently.

    Args:
        x: Array of raw class logits.

    Returns:
        Probabilities in [0, 1], same shape as `x`.
    """
    return 1.0 / (1.0 + np.exp(-x))


def query_probs(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-query class logits to (class_ids, confidences).

    Each DETR query's [C] logit vector becomes its best class id and that
    class's sigmoid probability.

    Args:
        logits: [Q, C] array of per-query class logits.

    Returns:
        Tuple of (class_ids [Q] int, confidences [Q] float).
    """
    probs = sigmoid(logits)
    return probs.argmax(axis=1), probs.max(axis=1)


def unnormalize_cxcywh(
    box: tuple[float, float, float, float], orig_hw: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Convert one normalised box to original-image pixel corners.

    Args:
        box: (cx, cy, w, h) centre-format box in normalised coordinates.
        orig_hw: Original image size as (height, width).

    Returns:
        xyxy corner coordinates (x1, y1, x2, y2) in pixels.
    """
    cx, cy, bw, bh = box
    orig_h, orig_w = orig_hw
    return (
        (cx - bw / 2) * orig_w,
        (cy - bh / 2) * orig_h,
        (cx + bw / 2) * orig_w,
        (cy + bh / 2) * orig_h,
    )


def preprocess_image(
    image_bytes: bytes, config: ModelConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Resize, scale, and (optionally) normalise an image for RF-DETR.

    Unlike the Anomalib preprocessor, RF-DETR normalisation is applied here
    (mean/std) when `config.normalize` is set; disable it if the export bakes
    normalisation in.

    When `config.preprocess` is False the whole transform is bypassed: the raw
    image is fed as a [1,H,W,3] float32 tensor of pixel values, because the
    export bakes resize + normalisation into the graph and takes the raw image
    on its input port. The channels are fed in BGR order — such exports are
    calibrated on `cv2.imread` output, so feeding RGB silently swaps red/blue
    and changes detections on colour-dependent parts. `input_size`, `normalize`,
    `mean`, and `std` are then unused.

    Args:
        image_bytes: Encoded input image.
        config: ModelConfig carrying input_size and normalisation parameters.

    Returns:
        Tuple of (tensor float32, original_rgb [H,W,3] uint8). The tensor is
        [1,3,H,W] normalised RGB for the normal path and [1,H,W,3] BGR pixel
        values when preprocessing is baked into the model. `original_rgb` is
        always RGB, for drawing.
    """
    original_rgb = decode_rgb(image_bytes)

    if config.preprocess is False:
        # Feed the raw image untouched (BGR, the export's training convention);
        # the graph resizes and normalises it.
        bgr = original_rgb[:, :, ::-1]
        return np.ascontiguousarray(bgr, dtype=np.float32)[np.newaxis], original_rgb

    h, w = config.input_size
    resized = cv2.resize(original_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    scaled = resized.astype(np.float32) / 255.0

    if config.normalize:
        mean = np.array(config.mean, dtype=np.float32)
        std = np.array(config.std, dtype=np.float32)
        scaled = (scaled - mean) / std

    return chw_tensor(scaled), original_rgb


def _split_logits_and_boxes(outputs: list) -> tuple[np.ndarray, np.ndarray]:
    """Identify the logits and boxes tensors among the raw model outputs.

    The boxes tensor is the one whose last dimension is 4 (cxcywh); the other is
    the class logits. Selecting by shape rather than position makes the decode
    robust to output ordering across exports.

    Args:
        outputs: Raw model output tensors.

    Returns:
        Tuple of (logits, boxes) arrays.

    Raises:
        ValueError: If no output with last dim == 4 exists.
    """
    arrays = [np.asarray(o) for o in outputs]
    box_idx = next((i for i, a in enumerate(arrays) if a.shape[-1] == 4), None)
    if box_idx is None:
        raise ValueError("No box output (last dim == 4) found among model outputs.")
    logits_idx = next(i for i in range(len(arrays)) if i != box_idx)
    return arrays[logits_idx], arrays[box_idx]


def _split_boxes_and_scores_baked(
    outputs: list, output_names: list[str] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the xyxy-boxes and scores tensors from a baked RF-DETR export.

    Both outputs are shaped [.,Q,4] here, so shape alone cannot tell them apart.
    Prefer the port names (`boxes_xyxy` vs `scores`); when names are absent,
    fall back to value range — pixel boxes exceed the [0,1] band that sigmoid
    scores live in.

    Args:
        outputs: Raw model output tensors.
        output_names: Port-ordered output names, or None/empty when unknown.

    Returns:
        Tuple of (boxes [.,Q,4] xyxy pixels, scores [.,Q,C] probabilities).
    """
    arrays = [np.asarray(o) for o in outputs]
    names = [n.lower() for n in (output_names or [])]
    box_idx = next((i for i, n in enumerate(names) if "box" in n), None)
    score_idx = next((i for i, n in enumerate(names) if "score" in n), None)
    if box_idx is None or score_idx is None or box_idx == score_idx:
        # No usable names: the boxes tensor is the one with out-of-[0,1] values.
        box_idx = max(range(len(arrays)), key=lambda i: float(np.max(np.abs(arrays[i]))))
        score_idx = next(i for i in range(len(arrays)) if i != box_idx)
    return arrays[box_idx], arrays[score_idx]


def decode_detections(
    outputs: list,
    orig_hw: tuple[int, int],
    conf_threshold: float,
    config: ModelConfig,
    output_names: list[str] | None = None,
) -> list[Detection]:
    """Turn raw RF-DETR outputs into Detections in original-image pixels.

    Assumes DETR-style outputs: per-query class logits and normalised cxcywh
    boxes. When `config.postprocess` is False the export has already decoded its
    outputs — xyxy pixel boxes and per-class sigmoid scores — so this reads them
    directly, without re-applying sigmoid or un-normalising boxes. Because the
    raw image is fed at its own resolution (see `preprocess_image`), those boxes
    are already in original-image pixels.

    Args:
        outputs: Raw model output tensors.
        orig_hw: Original image size as (height, width).
        conf_threshold: Minimum confidence to keep a detection.
        config: ModelConfig; `postprocess` selects the decode path.
        output_names: Port-ordered output names, used only on the baked path to
            tell the boxes tensor from the scores tensor.

    Returns:
        Detections in original-image pixel coordinates, highest-confidence
        queries first preserved in query order.
    """
    if config.postprocess is False:
        boxes, scores = _split_boxes_and_scores_baked(outputs, output_names)
        boxes = boxes.reshape(-1, 4)  # [Q, 4] xyxy pixels
        scores = scores.reshape(boxes.shape[0], -1)  # [Q, C] probabilities
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)
        # The graph leaves boxes unclamped (they can fall outside the frame);
        # clip to image bounds so they draw cleanly and report real pixels.
        orig_h, orig_w = orig_hw
        return [
            Detection(
                class_id=int(cls),
                confidence=float(conf),
                box=(
                    float(np.clip(x1, 0, orig_w)),
                    float(np.clip(y1, 0, orig_h)),
                    float(np.clip(x2, 0, orig_w)),
                    float(np.clip(y2, 0, orig_h)),
                ),
            )
            for cls, conf, (x1, y1, x2, y2) in zip(class_ids, confidences, boxes)
            if conf >= conf_threshold
        ]

    logits, boxes = _split_logits_and_boxes(outputs)
    logits = logits.reshape(-1, logits.shape[-1])  # [Q, C]
    boxes = boxes.reshape(-1, 4)  # [Q, 4]

    class_ids, confidences = query_probs(logits)

    return [
        Detection(
            class_id=int(cls),
            confidence=float(conf),
            box=unnormalize_cxcywh((cx, cy, bw, bh), orig_hw),
        )
        for cls, conf, (cx, cy, bw, bh) in zip(class_ids, confidences, boxes)
        if conf >= conf_threshold
    ]


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

    Args:
        original_rgb: Frame to draw onto ([H,W,3] uint8).
        detections: Detections in original-image pixel coordinates.
        labels: Class Catalog for label captions.

    Returns:
        A new RGB array with boxes and captions drawn.
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
