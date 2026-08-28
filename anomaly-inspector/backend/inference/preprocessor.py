"""Image decoding shared by every Feature's preprocessor.

Both preprocessors (Anomaly Detection's `preprocess` and the RF-DETR family's
`preprocess_image`) start from the same two steps — raw bytes to an RGB uint8
array, and a scaled HWC array to a [1,3,H,W] tensor — so those primitives live
here once. What each preprocessor adds on top (normalisation or its deliberate
absence) is Feature knowledge and stays with the Feature.
"""
import io

import cv2
import numpy as np
from PIL import Image

from inference.graph_probe import GraphPreprocessing

# Standard ImageNet statistics, the normalisation both Anomalib and RF-DETR
# train against. Single source of truth for both preprocessors.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def preprocess(
    image_bytes: bytes,
    input_size: tuple[int, int],
    graph: GraphPreprocessing | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (tensor [1,3,H,W] float32, original_rgb [H,W,3] uint8).
    input_size: (height, width)

    The tensor is scaled to [0, 1] only. ImageNet mean/std normalization is NOT
    applied here: Anomalib's exported ONNX/OpenVINO models embed the
    normalization transform inside the graph, so normalizing again would
    double-normalize the input and saturate every anomaly score to 1.0.

    Args:
        image_bytes: Encoded image in any PIL-decodable format.
        input_size: Model input size as (height, width).

    Returns:
        Tuple of (tensor [1,3,H,W] float32 in [0,1], original_rgb [H,W,3]
        uint8).
    """
    original_rgb = decode_rgb(image_bytes)

    h, w = input_size
    resized = cv2.resize(original_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    scaled = resized.astype(np.float32) / 255.0

    return chw_tensor(scaled), original_rgb
