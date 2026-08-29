import io

import cv2
import numpy as np
import pytest
from PIL import Image
from inference.visualizer import (
    generate_heatmap,
    generate_segmentation,
    resize_anomaly_map_to_original,
)


def _load_image(img_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(img_bytes)))


def test_heatmap_returns_png_bytes(anomaly_map, png_bytes):
    from inference.preprocessor import preprocess
    _, original, _ = preprocess(png_bytes, (64, 64))
    result = generate_heatmap(anomaly_map, original)
    assert isinstance(result, bytes)
    assert result[:3] == b"\xff\xd8\xff"


def test_heatmap_output_matches_original_size(anomaly_map, png_bytes):
    from inference.preprocessor import preprocess
    _, original, _ = preprocess(png_bytes, (64, 64))
    result = generate_heatmap(anomaly_map, original)
    img = _load_image(result)
    assert img.shape[:2] == original.shape[:2]


def test_segmentation_returns_png_bytes(anomaly_map, png_bytes):
    from inference.preprocessor import preprocess
    _, original, _ = preprocess(png_bytes, (64, 64))
    result = generate_segmentation(anomaly_map, original, threshold=0.5)
    assert isinstance(result, bytes)
    assert result[:3] == b"\xff\xd8\xff"


def test_segmentation_output_matches_original_size(anomaly_map, png_bytes):
    from inference.preprocessor import preprocess
    _, original, _ = preprocess(png_bytes, (64, 64))
    result = generate_segmentation(anomaly_map, original, threshold=0.5)
    img = _load_image(result)
    assert img.shape[:2] == original.shape[:2]


def test_heatmap_with_uniform_anomaly_map(png_bytes):
    from inference.preprocessor import preprocess
    _, original, _ = preprocess(png_bytes, (64, 64))
    uniform_map = np.ones((1, 1, 32, 32), dtype=np.float32) * 0.5
    result = generate_heatmap(uniform_map, original)
    assert result[:3] == b"\xff\xd8\xff"


def test_heatmap_preserves_rgb_channel_order(png_bytes):
    """Red in the original must stay red after heatmap encode/decode."""
    from inference.preprocessor import preprocess
    img = np.full((64, 64, 3), [255, 0, 0], dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    _, original, _ = preprocess(buf.getvalue(), (64, 64))
    uniform_map = np.zeros((1, 1, 32, 32), dtype=np.float32)
    result = generate_heatmap(uniform_map, original)
    decoded = _load_image(result)
    assert decoded[0, 0, 0] > decoded[0, 0, 2], "red channel should dominate blue on a red frame"


def test_segmentation_preserves_rgb_channel_order(png_bytes):
    """Background colours must not have red/blue swapped after encode/decode."""
    from inference.preprocessor import preprocess
    img = np.full((64, 64, 3), [200, 100, 50], dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    _, original, _ = preprocess(buf.getvalue(), (64, 64))
    uniform_map = np.zeros((1, 1, 32, 32), dtype=np.float32)
    result = generate_segmentation(uniform_map, original, threshold=0.5)
    decoded = _load_image(result)
    np.testing.assert_array_almost_equal(decoded[0, 0], original[0, 0], decimal=0)


def test_resize_anomaly_map_through_model_space():
    """A low-res map must pass through model resolution before mapping to original."""
    from inference.preprocessor import preprocess

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[30:50, 50:70] = 255
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")

    _, original, transform = preprocess(buf.getvalue(), (40, 40))
    model_map = np.zeros((40, 40), dtype=np.float32)
    model_map[15:25, 17:23] = 1.0
    low_res_map = cv2.resize(model_map, (10, 10), interpolation=cv2.INTER_LINEAR)

    aligned = resize_anomaly_map_to_original(
        low_res_map, transform, original, interpolation=cv2.INTER_NEAREST
    )

    assert aligned.shape == (80, 120)
    peak = np.unravel_index(np.argmax(aligned), aligned.shape)
    assert 28 <= peak[0] <= 52
    assert 48 <= peak[1] <= 72


def test_letterbox_transform_recentres_peak_to_original_pixels():
    from inference.preprocessor import preprocess

    image = np.zeros((60, 100, 3), dtype=np.uint8)
    image[20:40, 60:80] = 255
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")

    _, original, transform = preprocess(buf.getvalue(), (50, 50), letterbox=True)
    model_map = np.zeros((50, 50), dtype=np.float32)
    model_map[20:30, 30:40] = 1.0

    aligned = resize_anomaly_map_to_original(
        model_map, transform, original, interpolation=cv2.INTER_NEAREST
    )

    assert aligned.shape == original.shape[:2]
    peak = np.unravel_index(np.argmax(aligned), aligned.shape)
    assert 18 <= peak[0] <= 42
    assert 58 <= peak[1] <= 82
