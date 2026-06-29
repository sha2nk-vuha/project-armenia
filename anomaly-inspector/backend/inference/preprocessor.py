import io
import cv2
import numpy as np
from PIL import Image

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(
    image_bytes: bytes, input_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (tensor [1,3,H,W] float32, original_rgb [H,W,3] uint8).
    input_size: (height, width)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_rgb = np.array(image, dtype=np.uint8)

    h, w = input_size
    resized = cv2.resize(original_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    normalised = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    tensor = normalised.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

    return tensor, original_rgb
