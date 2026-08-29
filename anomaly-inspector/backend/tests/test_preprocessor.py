import numpy as np
import pytest
from inference.preprocessor import preprocess


def test_output_tensor_shape(png_bytes):
    tensor, original, transform = preprocess(png_bytes, (224, 224))
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert transform.model_hw == (224, 224)


def test_original_rgb_shape_matches_source(png_bytes):
    _, original, transform = preprocess(png_bytes, (224, 224))
    assert original.ndim == 3
    assert original.shape[2] == 3
    assert original.dtype == np.uint8
    assert transform.orig_hw == original.shape[:2]


def test_tensor_is_normalised(png_bytes):
    tensor, _, _ = preprocess(png_bytes, (224, 224))
    # Normalised values should not be in [0, 255] range
    assert tensor.max() < 10.0
    assert tensor.min() > -10.0


def test_different_input_sizes(png_bytes):
    tensor_256, _, t256 = preprocess(png_bytes, (256, 256))
    assert tensor_256.shape == (1, 3, 256, 256)
    assert t256.model_hw == (256, 256)
    tensor_128, _, t128 = preprocess(png_bytes, (128, 128))
    assert tensor_128.shape == (1, 3, 128, 128)
    assert t128.model_hw == (128, 128)


def test_unprobed_graph_keeps_scale_only_behaviour():
    """No probe result: scale to [0,1] and leave normalisation to the graph."""
    tensor, _, _ = preprocess(_white_png(), (8, 8))

    assert tensor.max() == pytest.approx(1.0, abs=1e-6)
    assert tensor.min() == pytest.approx(1.0, abs=1e-6)


def test_letterbox_pads_non_square_images():
    tensor, original, transform = preprocess(
        _rect_png(120, 80), (100, 100), letterbox=True
    )
    assert tensor.shape == (1, 3, 100, 100)
    assert original.shape[:2] == (80, 120)
    assert transform.letterboxed is True
    assert transform.pad_top > 0
    assert transform.pad_left == 0
    assert transform.scaled_hw == (67, 100)


def test_stretch_has_no_padding():
    _, _, transform = preprocess(_rect_png(120, 80), (100, 100), letterbox=False)
    assert transform.letterboxed is False
    assert transform.pad_top == 0
    assert transform.pad_left == 0
    assert transform.scaled_hw == (100, 100)


def _white_png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _rect_png(width: int, height: int) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (128, 64, 32)).save(buf, format="PNG")
    return buf.getvalue()
