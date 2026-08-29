"""Image decoding shared by every Feature's preprocessor.

Both preprocessors (Anomaly Detection's `preprocess` and the RF-DETR family's
`preprocess_image`) start from the same two steps — raw bytes to an RGB uint8
array, and a scaled HWC array to a [1,3,H,W] tensor — so those primitives live
here once. What each preprocessor adds on top (normalisation or its deliberate
absence) is Feature knowledge and stays with the Feature.
"""
from dataclasses import dataclass
import io

import cv2
import numpy as np
from PIL import Image

from inference.graph_probe import GraphPreprocessing

# Standard ImageNet statistics, the normalisation both Anomalib and RF-DETR
# train against. Single source of truth for both preprocessors.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessTransform:
    """Maps coordinates between the original image and model-input space.

    Letterbox mode scales uniformly and pads to `model_hw`; stretch mode fills
    `model_hw` directly. The visualizer inverts this transform when resizing an
    anomaly map back onto `original_rgb`.
    """

    orig_hw: tuple[int, int]
    model_hw: tuple[int, int]
    scaled_hw: tuple[int, int]
    pad_top: int
    pad_left: int

    @property
    def letterboxed(self) -> bool:
        return self.pad_top > 0 or self.pad_left > 0


def decode_rgb(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to an RGB uint8 array.

    Args:
        image_bytes: Encoded image in any PIL-decodable format.

    Returns:
        RGB array [H,W,3] uint8.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image, dtype=np.uint8)


def chw_tensor(scaled_rgb: np.ndarray) -> np.ndarray:
    """Reshape a scaled HWC array into a batched channel-first tensor.

    Args:
        scaled_rgb: [H,W,3] float32 array, already scaled/normalised.

    Returns:
        Tensor [1,3,H,W] ready for the model.
    """
    return scaled_rgb.transpose(2, 0, 1)[np.newaxis]


def _fit_to_model(
    original_rgb: np.ndarray,
    input_size: tuple[int, int],
    *,
    letterbox: bool,
) -> tuple[np.ndarray, PreprocessTransform]:
    """Resize (and optionally letterbox-pad) an image to model input size."""
    orig_h, orig_w = original_rgb.shape[:2]
    model_h, model_w = input_size

    if letterbox:
        scale = min(model_w / orig_w, model_h / orig_h)
        scaled_w = max(1, int(round(orig_w * scale)))
        scaled_h = max(1, int(round(orig_h * scale)))
        resized = cv2.resize(
            original_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
        )
        pad_left = (model_w - scaled_w) // 2
        pad_top = (model_h - scaled_h) // 2
        pad_right = model_w - scaled_w - pad_left
        pad_bottom = model_h - scaled_h - pad_top
        fitted = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        transform = PreprocessTransform(
            orig_hw=(orig_h, orig_w),
            model_hw=(model_h, model_w),
            scaled_hw=(scaled_h, scaled_w),
            pad_top=pad_top,
            pad_left=pad_left,
        )
        return fitted, transform

    resized = cv2.resize(
        original_rgb, (model_w, model_h), interpolation=cv2.INTER_LINEAR
    )
    transform = PreprocessTransform(
        orig_hw=(orig_h, orig_w),
        model_hw=(model_h, model_w),
        scaled_hw=(model_h, model_w),
        pad_top=0,
        pad_left=0,
    )
    return resized, transform


def preprocess(
    image_bytes: bytes,
    input_size: tuple[int, int],
    graph: GraphPreprocessing | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    *,
    letterbox: bool = False,
) -> tuple[np.ndarray, np.ndarray, PreprocessTransform]:
    """
    Returns (tensor [1,3,H,W] float32, original_rgb [H,W,3] uint8, transform).
    input_size: (height, width)

    The tensor is scaled to [0, 1] only. ImageNet mean/std normalization is NOT
    applied here: Anomalib's exported ONNX/OpenVINO models embed the
    normalization transform inside the graph, so normalizing again would
    double-normalize the input and saturate every anomaly score to 1.0.

    Args:
        image_bytes: Encoded image in any PIL-decodable format.
        input_size: Model input size as (height, width).
        letterbox: When True, scale uniformly and pad to `input_size` instead of
            stretching. Use when overlays on non-square images look shifted.

    Returns:
        Tuple of (tensor [1,3,H,W] float32 in [0,1], original_rgb [H,W,3]
        uint8, PreprocessTransform for inverting the resize on anomaly maps).
    """
    original_rgb = decode_rgb(image_bytes)
    fitted, transform = _fit_to_model(
        original_rgb, input_size, letterbox=letterbox
    )
    scaled = fitted.astype(np.float32) / 255.0
    return chw_tensor(scaled), original_rgb, transform
