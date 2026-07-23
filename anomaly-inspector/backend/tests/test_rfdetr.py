import numpy as np
import pytest

from inference.rfdetr import (
    Detection,
    PresenceConfig,
    decode_detections,
    evaluate_presence,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _det(class_id, confidence, box=(0.0, 0.0, 10.0, 10.0)):
    return Detection(class_id=class_id, confidence=confidence, box=box)


def test_presence_ok_when_expected_class_present():
    dets = [_det(1, 0.9)]
    assert evaluate_presence(dets, expected_classes=[1], threshold=0.5) == "ok"


def test_presence_nok_when_expected_class_missing():
    dets = [_det(2, 0.9)]
    assert evaluate_presence(dets, expected_classes=[1], threshold=0.5) == "not_ok"


def test_presence_nok_when_expected_below_threshold():
    dets = [_det(1, 0.3)]
    assert evaluate_presence(dets, expected_classes=[1], threshold=0.5) == "not_ok"


def test_presence_requires_all_expected_classes():
    dets = [_det(1, 0.9)]
    assert evaluate_presence(dets, expected_classes=[1, 2], threshold=0.5) == "not_ok"


def test_decode_filters_by_confidence_and_scales_boxes():
    # 2 queries, 3 classes. Query 0 is a confident class-1 detection;
    # query 1 is below-confidence on every class and should be dropped.
    logits = np.array(
        [[[-5.0, 5.0, -5.0], [-5.0, -5.0, -5.0]]], dtype=np.float32
    )  # [1, 2, 3]
    # Normalised cxcywh: centre (0.5, 0.5), size (0.5, 0.5) -> xyxy (0.25,0.25,0.75,0.75)
    boxes = np.array(
        [[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]], dtype=np.float32
    )  # [1, 2, 4]

    dets = decode_detections(
        [logits, boxes], orig_hw=(100, 200), conf_threshold=0.5, config=PresenceConfig()
    )

    assert len(dets) == 1
    d = dets[0]
    assert d.class_id == 1
    assert d.confidence == pytest.approx(_sigmoid(5.0), abs=1e-3)
    # Scaled to width=200, height=100.
    assert d.box == pytest.approx((50.0, 25.0, 150.0, 75.0))


def test_decode_handles_swapped_output_order():
    # The output whose last dim is 4 is the boxes tensor, regardless of order.
    logits = np.array([[[-5.0, 5.0]]], dtype=np.float32)  # [1, 1, 2]
    boxes = np.array([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32)  # [1, 1, 4]

    dets = decode_detections(
        [boxes, logits], orig_hw=(100, 100), conf_threshold=0.5, config=PresenceConfig()
    )

    assert len(dets) == 1
    assert dets[0].class_id == 1


def _white_png(size=(10, 10)):
    import cv2
    img = np.full((size[0], size[1], 3), 255, dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_preprocess_normalizes_and_shapes_tensor():
    from inference.rfdetr import preprocess_image

    config = PresenceConfig(input_size=(20, 20))
    tensor, original = preprocess_image(_white_png((10, 10)), config)

    assert tensor.shape == (1, 3, 20, 20)
    assert original.shape == (10, 10, 3)
    # White (1.0 after /255) normalised per-channel: (1.0 - mean) / std.
    expected_r = (1.0 - config.mean[0]) / config.std[0]
    assert tensor[0, 0].mean() == pytest.approx(expected_r, abs=1e-3)


def test_preprocess_skips_normalization_when_disabled():
    from inference.rfdetr import preprocess_image

    config = PresenceConfig(input_size=(20, 20), normalize=False)
    tensor, _ = preprocess_image(_white_png((10, 10)), config)

    # Only scaled to [0,1]; white stays 1.0.
    assert tensor.max() == pytest.approx(1.0, abs=1e-6)
    assert tensor.min() == pytest.approx(1.0, abs=1e-6)


def test_draw_detections_returns_decodable_image():
    import cv2
    from inference.rfdetr import draw_detections

    original = np.zeros((50, 50, 3), dtype=np.uint8)
    dets = [Detection(1, 0.9, (5.0, 5.0, 20.0, 20.0))]
    out = draw_detections(original, dets, labels={1: "cap"})

    assert isinstance(out, bytes) and len(out) > 0
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (50, 50, 3)
