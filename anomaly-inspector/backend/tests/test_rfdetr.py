import numpy as np
import pytest

from inference.rfdetr import (
    Detection,
    PresenceConfig,
    decode_detections,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _det(class_id, confidence, box=(0.0, 0.0, 10.0, 10.0)):
    return Detection(class_id=class_id, confidence=confidence, box=box)


# Parity: these four cases are the pre-seam `evaluate_presence` contract,
# now asserted against the Expected Classes Decision Rule that replaced it.
# Same inputs, same Verdicts.


def _presence_verdict(dets, expected_classes, threshold, labels=None):
    from inference.decision import KIND_DETECTIONS, DecisionContext, DecodedOutput, get

    rule = get("expected_classes")
    ctx = DecisionContext(
        output=DecodedOutput(
            kinds=frozenset({KIND_DETECTIONS}), image_hw=(10, 10), detections=dets
        ),
        image_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        labels=labels or {1: "cap", 2: "logo"},
        params={"expected_classes": expected_classes},
        threshold=threshold,
    )
    return rule.evaluate(ctx).verdict


def test_presence_ok_when_expected_class_present():
    assert _presence_verdict([_det(1, 0.9)], [1], 0.5) == "ok"


def test_presence_nok_when_expected_class_missing():
    assert _presence_verdict([_det(2, 0.9)], [1], 0.5) == "not_ok"


def test_presence_nok_when_expected_below_threshold():
    assert _presence_verdict([_det(1, 0.3)], [1], 0.5) == "not_ok"


def test_presence_requires_all_expected_classes():
    assert _presence_verdict([_det(1, 0.9)], [1, 2], 0.5) == "not_ok"


def test_presence_resolves_expected_classes_by_name():
    # Config carries names, not ids, so a retrain that reorders classes survives.
    assert _presence_verdict([_det(1, 0.9)], ["cap"], 0.5) == "ok"
    assert _presence_verdict([_det(1, 0.9)], ["logo"], 0.5) == "not_ok"


def test_presence_unknown_class_name_raises_loudly():
    from inference.decision import ClassNotFound

    with pytest.raises(ClassNotFound, match="bottle_cap"):
        _presence_verdict([_det(1, 0.9)], ["bottle_cap"], 0.5)


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


def test_load_config_parses_sidecar(tmp_path):
    import json
    from inference.rfdetr import load_config

    sidecar = tmp_path / "rfdetr-nano.json"
    sidecar.write_text(json.dumps({
        "labels": {"0": "gasket", "1": "no-gasket", "2": "background"},
        "input_size": [384, 384],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "normalize": True,
    }))

    config = load_config(str(sidecar))

    # JSON object keys are strings; they must be coerced to int class ids.
    assert config.labels == {0: "gasket", 1: "no-gasket", 2: "background"}
    assert config.input_size == (384, 384)
    assert config.normalize is True


def test_load_config_falls_back_to_defaults_when_absent(tmp_path):
    from inference.rfdetr import load_config

    config = load_config(str(tmp_path / "missing.json"))

    # No sidecar -> usable defaults, empty catalog.
    assert config.labels == {}
    assert isinstance(config.input_size, tuple)


def test_load_config_partial_sidecar_keeps_defaults(tmp_path):
    import json
    from inference.rfdetr import load_config, PresenceConfig

    sidecar = tmp_path / "labels-only.json"
    sidecar.write_text(json.dumps({"labels": {"0": "gasket"}}))

    config = load_config(str(sidecar))

    assert config.labels == {0: "gasket"}
    # Unspecified fields fall back to RF-DETR defaults.
    assert config.mean == PresenceConfig().mean
    assert config.normalize == PresenceConfig().normalize


def test_load_config_reads_expected_classes(tmp_path):
    import json
    from inference.rfdetr import load_config

    sidecar = tmp_path / "with-expected.json"
    sidecar.write_text(json.dumps({
        "labels": {"0": "gasket", "1": "no-gasket", "2": "background"},
        "expected_classes": [0],
    }))

    config = load_config(str(sidecar))

    # The Expected Class policy default for this model: gasket must be present.
    assert config.expected_classes == [0]


def test_load_config_defaults_expected_classes_to_empty(tmp_path):
    import json
    from inference.rfdetr import load_config

    sidecar = tmp_path / "no-expected.json"
    sidecar.write_text(json.dumps({"labels": {"0": "gasket"}}))

    config = load_config(str(sidecar))

    # Absent -> empty; the pipeline treats "no expected classes configured"
    # as a distinct case rather than a trivially-OK verdict.
    assert config.expected_classes == []
