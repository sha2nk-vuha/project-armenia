import io

import cv2
import numpy as np
from PIL import Image

from inference.graph_probe import GraphPreprocessing

# Standard ImageNet statistics, the normalisation both Anomalib and RF-DETR
# train against. Single source of truth for both preprocessors.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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

    Which steps run is decided by `graph` — what the model's own ONNX graph was
    found to do (see inference.graph_probe). Anomalib's exports embed the
    normalisation transform, so normalising here as well would double-normalise
    and saturate every anomaly score to 1.0; a model that does not embed it
    needs the mean/std applied here instead. Passing `graph=None` means "not
    probed" and yields the conservative default: scale to [0,1], skip mean/std.
    """
    graph = graph or GraphPreprocessing()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_rgb = np.array(image, dtype=np.uint8)

    h, w = input_size
    resized = cv2.resize(original_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    tensor = resized.astype(np.float32)
    if graph.needs_scaling:
        tensor /= 255.0
    if graph.needs_normalization:
        tensor = (tensor - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)

    tensor = tensor.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]
    return tensor, original_rgb
