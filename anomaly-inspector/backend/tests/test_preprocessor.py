import numpy as np
import pytest
from inference.preprocessor import preprocess


def test_output_tensor_shape(png_bytes):
    tensor, original = preprocess(png_bytes, (224, 224))
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32


def test_original_rgb_shape_matches_source(png_bytes):
    _, original = preprocess(png_bytes, (224, 224))
    assert original.ndim == 3
    assert original.shape[2] == 3
    assert original.dtype == np.uint8


def test_tensor_is_normalised(png_bytes):
    tensor, _ = preprocess(png_bytes, (224, 224))
    # Normalised values should not be in [0, 255] range
    assert tensor.max() < 10.0
    assert tensor.min() > -10.0


def test_different_input_sizes(png_bytes):
    tensor_256, _ = preprocess(png_bytes, (256, 256))
    assert tensor_256.shape == (1, 3, 256, 256)
    tensor_128, _ = preprocess(png_bytes, (128, 128))
    assert tensor_128.shape == (1, 3, 128, 128)


def test_normalizes_when_graph_does_not():
    from inference.graph_probe import GraphPreprocessing
    from inference.preprocessor import IMAGENET_MEAN, IMAGENET_STD

    bare = GraphPreprocessing(scales=False, normalizes=False, detail="test")
    tensor, _ = preprocess(_white_png(), (8, 8), bare)

    expected_r = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert tensor[0, 0].mean() == pytest.approx(expected_r, abs=1e-3)


def test_skips_scaling_when_graph_already_scales():
    from inference.graph_probe import GraphPreprocessing

    scaling = GraphPreprocessing(scales=True, normalizes=True, detail="test")
    tensor, _ = preprocess(_white_png(), (8, 8), scaling)

    assert tensor.max() == pytest.approx(255.0, abs=1e-6)


def test_unprobed_graph_keeps_scale_only_behaviour():
    """No probe result: scale to [0,1] and leave normalisation to the graph."""
    tensor, _ = preprocess(_white_png(), (8, 8))

    assert tensor.max() == pytest.approx(1.0, abs=1e-6)
    assert tensor.min() == pytest.approx(1.0, abs=1e-6)


def _white_png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()
