import numpy as np
import pytest
import io
from PIL import Image
from inference.visualizer import generate_heatmap, generate_segmentation


def _load_image(img_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(img_bytes)))


def test_heatmap_returns_png_bytes(anomaly_map, png_bytes):
    from inference.preprocessor import preprocess
    _, original = preprocess(png_bytes, (64, 64))
    result = generate_heatmap(anomaly_map, original)
    assert isinstance(result, bytes)
    assert result[:3] == b"\xff\xd8\xff"


def test_heatmap_output_matches_original_size(anomaly_map, png_bytes):
    from inference.preprocessor import preprocess
    _, original = preprocess(png_bytes, (64, 64))
    result = generate_heatmap(anomaly_map, original)
    img = _load_image(result)
    assert img.shape[:2] == original.shape[:2]


def test_segmentation_returns_png_bytes(anomaly_map, png_bytes):
    from inference.preprocessor import preprocess
    _, original = preprocess(png_bytes, (64, 64))
    result = generate_segmentation(anomaly_map, original, threshold=0.5)
    assert isinstance(result, bytes)
    assert result[:3] == b"\xff\xd8\xff"


def test_segmentation_output_matches_original_size(anomaly_map, png_bytes):
    from inference.preprocessor import preprocess
    _, original = preprocess(png_bytes, (64, 64))
    result = generate_segmentation(anomaly_map, original, threshold=0.5)
    img = _load_image(result)
    assert img.shape[:2] == original.shape[:2]


def test_heatmap_with_uniform_anomaly_map(png_bytes):
    from inference.preprocessor import preprocess
    _, original = preprocess(png_bytes, (64, 64))
    uniform_map = np.ones((1, 1, 32, 32), dtype=np.float32) * 0.5
    result = generate_heatmap(uniform_map, original)
    assert result[:3] == b"\xff\xd8\xff"


def test_heatmap_preserves_rgb_channel_order(png_bytes):
    """Red in the original must stay red after heatmap encode/decode."""
    from inference.preprocessor import preprocess
    img = np.full((64, 64, 3), [255, 0, 0], dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    _, original = preprocess(buf.getvalue(), (64, 64))
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
    _, original = preprocess(buf.getvalue(), (64, 64))
    uniform_map = np.zeros((1, 1, 32, 32), dtype=np.float32)
    result = generate_segmentation(uniform_map, original, threshold=0.5)
    decoded = _load_image(result)
    np.testing.assert_array_almost_equal(decoded[0, 0], original[0, 0], decimal=0)
