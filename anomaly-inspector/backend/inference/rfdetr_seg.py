"""RF-DETR *segmentation* decode: raw ONNX outputs -> per-instance masks.

Separate from `rfdetr.py` because the two exports are not interchangeable. The
detection decode splits outputs by "the tensor whose last dim is 4 is boxes, the
other is logits" — with a third `masks` tensor that heuristic takes `masks` as
the logits and produces confident nonsense. Outputs are therefore identified by
rank here, which is unambiguous for this signature.

Verified against three_cee_caps_rfdetr-seg-nano_v0.0.1.onnx:
    input  [1, 3, 312, 312]
    dets   [1, 100, 4]        normalised cxcywh
    labels [1, 100, 3]        per-class logits (sigmoid -> probability)
    masks  [1, 100, 78, 78]   per-query mask logits

The masks are **full-frame at stride 4** (312 / 4 = 78), not ROI-cropped mask
heads: measured on real images, each query's binarised mask bbox agrees with its
own predicted box to within ~0.005 in normalised coordinates. So a mask is
decoded by sigmoid -> threshold -> resize to the original image, with no
box-relative paste step.
"""
import cv2
import numpy as np

from inference.decision.base import Instance
from inference.model_config import ModelConfig
from inference.rfdetr import query_probs, sigmoid, unnormalize_cxcywh


def split_outputs(outputs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify (logits, boxes, masks) among the raw outputs, by rank and shape.

    Masks are the only rank-4 tensor; of the two rank-3 tensors, boxes are the
    one whose last dim is 4. Selecting structurally rather than positionally
    keeps the decode robust to output ordering across exports.

    Args:
        outputs: Raw model output tensors.

    Returns:
        Tuple of (logits, boxes, masks) arrays.

    Raises:
        ValueError: If no rank-4 mask output or no last-dim-4 box output is
            found (the latter usually means a detection-only export was routed
            here).
    """
    arrays = [np.asarray(o) for o in outputs]
    masks_idx = next((i for i, a in enumerate(arrays) if a.ndim == 4), None)
    if masks_idx is None:
        raise ValueError(
            "No mask output (rank 4) found; this looks like a detection-only "
            "export — use the Presence/Absence Feature for it."
        )
    rest = [i for i in range(len(arrays)) if i != masks_idx]
    box_idx = next((i for i in rest if arrays[i].shape[-1] == 4), None)
    if box_idx is None:
        raise ValueError("No box output (last dim == 4) found among model outputs.")
    logits_idx = next(i for i in rest if i != box_idx)
    return arrays[logits_idx], arrays[box_idx], arrays[masks_idx]


def decode_instances(
    outputs: list,
    orig_hw: tuple[int, int],
    conf_threshold: float,
    config: ModelConfig,
) -> list[Instance]:
    """Turn raw segmentation outputs into Instances in original-image pixels.

    Each query is reduced to its top-scoring class (confidence via sigmoid);
    queries below `conf_threshold` are dropped. Surviving masks are binarised at
    `config.mask_threshold` and resized to the original image with nearest
    neighbour, so the boolean mask is never softened into intermediate values.

    Queries whose mask is empty after binarisation are dropped: a detection with
    no pixels cannot yield a centre, and carrying it forward would force every
    geometry rule to re-handle the same degenerate case.

    Args:
        outputs: Raw model output tensors.
        orig_hw: Original image size as (height, width).
        conf_threshold: Minimum confidence to keep an instance.
        config: ModelConfig supplying mask_threshold and input geometry.

    Returns:
        Instances in original-image pixels with boolean full-resolution masks.
    """
    logits, boxes, masks = split_outputs(outputs)
    logits = logits.reshape(-1, logits.shape[-1])  # [Q, C]
    boxes = boxes.reshape(-1, 4)  # [Q, 4]
    masks = masks.reshape(-1, *masks.shape[-2:])  # [Q, h, w]

    class_ids, confidences = query_probs(logits)

    orig_h, orig_w = orig_hw
    instances: list[Instance] = []
    for q, (cls, conf) in enumerate(zip(class_ids, confidences)):
        if conf < conf_threshold:
            continue
        binary = (sigmoid(masks[q]) > config.mask_threshold).astype(np.uint8)
        if not binary.any():
            continue
        full = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask = full.astype(bool)
        if not mask.any():
            continue
        instances.append(
            Instance(
                class_id=int(cls),
                confidence=float(conf),
                box=unnormalize_cxcywh(boxes[q], orig_hw),
                mask=mask,
            )
        )
    return instances


# Distinct hues per class id, so cap and logo are separable at a glance.
_CLASS_COLORS = [
    (0, 140, 255), (255, 60, 60), (60, 220, 120),
    (255, 200, 0), (200, 80, 255), (0, 220, 220),
]


def draw_instances_rgb(
    original_rgb: np.ndarray,
    instances: list[Instance],
    labels: dict[int, str],
    alpha: float = 0.45,
) -> np.ndarray:
    """Tint each instance's mask and outline it; returns an RGB array.

    Returns an array rather than encoded bytes so a pipeline can composite a
    Decision Rule's annotations on top before a single final encode.

    Args:
        original_rgb: Frame to draw onto ([H,W,3] uint8).
        instances: Decoded instances with boolean masks at image resolution.
        labels: Class Catalog for caption text.
        alpha: Mask tint opacity per instance.

    Returns:
        A new RGB array with tinted masks, outlines, and captions.
    """
    canvas = original_rgb.copy()
    for inst in instances:
        color = np.array(_CLASS_COLORS[inst.class_id % len(_CLASS_COLORS)], dtype=np.float32)
        m = inst.mask
        canvas[m] = ((1 - alpha) * canvas[m] + alpha * color).astype(np.uint8)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    for inst in instances:
        r, g, b = _CLASS_COLORS[inst.class_id % len(_CLASS_COLORS)]
        contours, _ = cv2.findContours(
            inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas_bgr, contours, -1, (b, g, r), 2, cv2.LINE_AA)
        name = labels.get(inst.class_id, str(inst.class_id))
        x1, y1 = int(round(inst.box[0])), int(round(inst.box[1]))
        cv2.putText(
            canvas_bgr, f"{name} {inst.confidence:.2f}", (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1, cv2.LINE_AA,
        )
    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
