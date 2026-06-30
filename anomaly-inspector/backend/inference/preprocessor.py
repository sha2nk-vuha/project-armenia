import io
import cv2
import numpy as np
from PIL import Image


def preprocess(
    image_bytes: bytes, input_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (tensor [1,3,H,W] float32 in [0,1], original_rgb [H,W,3] uint8).
    input_size: (height, width)

    The tensor is scaled to [0, 1] only. ImageNet mean/std normalization is NOT
    applied here: Anomalib's exported ONNX/OpenVINO models embed the
    normalization transform inside the graph, so normalizing again would
    double-normalize the input and saturate every anomaly score to 1.0.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_rgb = np.array(image, dtype=np.uint8)

    h, w = input_size
    resized = cv2.resize(original_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    scaled = resized.astype(np.float32) / 255.0
    tensor = scaled.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

    return tensor, original_rgb
