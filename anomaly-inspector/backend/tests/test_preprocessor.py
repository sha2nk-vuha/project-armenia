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
