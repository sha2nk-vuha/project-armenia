"""Segmentation decode: output identification, mask handling, and the real
model's I/O contract.
"""
import numpy as np
import pytest

from inference.decision.base import Instance
from inference.model_config import ModelConfig
from inference.rfdetr_seg import decode_instances, draw_instances_rgb, split_outputs


def _outputs(num_queries=2, num_classes=3, mask_hw=(8, 8), favored=1, mask_box=None,
             logit=8.0):
    """dets/labels/masks in the real export's shapes, query 0 confident."""
    logits = np.full((1, num_queries, num_classes), -8.0, dtype=np.float32)
    logits[0, 0, favored] = logit
    boxes = np.tile(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (1, num_queries, 1))
    masks = np.full((1, num_queries, *mask_hw), -8.0, dtype=np.float32)
    y0, y1, x0, x1 = mask_box or (2, 6, 2, 6)
    masks[0, 0, y0:y1, x0:x1] = 8.0
    return [dets_labels_masks for dets_labels_masks in (boxes, logits, masks)]


def test_split_outputs_identifies_by_rank_not_position():
    boxes, logits, masks = _outputs()
    for ordering in ((logits, boxes, masks), (masks, logits, boxes), (boxes, masks, logits)):
        lg, bx, mk = split_outputs(list(ordering))
        assert mk.ndim == 4
        assert bx.shape[-1] == 4
        assert lg.shape[-1] == 3


def test_split_outputs_rejects_a_detection_only_export():
    # The detection decode's box-vs-logits heuristic would take `masks` as the
    # logits tensor here; routing the wrong export must fail loudly instead.
    boxes, logits, _ = _outputs()
    with pytest.raises(ValueError, match="detection-only"):
        split_outputs([logits, boxes])


def test_decode_produces_instances_at_original_resolution():
    insts = decode_instances(_outputs(), (40, 60), 0.5, ModelConfig())

    assert len(insts) == 1
    inst = insts[0]
    assert isinstance(inst, Instance)
    assert inst.class_id == 1
    assert inst.mask.shape == (40, 60)
    assert inst.mask.dtype == bool
    # Box decoded from normalised cxcywh into original pixels.
    assert inst.box == pytest.approx((15.0, 10.0, 45.0, 30.0))


def test_decode_drops_queries_below_confidence():
    # sigmoid(1.0) ~= 0.73, below the 0.9 confidence floor.
    outputs = _outputs(logit=1.0)
    assert decode_instances(outputs, (40, 60), 0.9, ModelConfig()) == []
    assert len(decode_instances(outputs, (40, 60), 0.7, ModelConfig())) == 1


def test_decode_drops_empty_masks():
    """A detection with no pixels cannot yield a centre; dropping it here means
    no geometry rule has to re-handle the same degenerate case."""
    boxes, logits, masks = _outputs()
    masks[:] = -8.0  # every mask binarises to empty
    assert decode_instances([boxes, logits, masks], (40, 60), 0.5, ModelConfig()) == []


def test_mask_threshold_comes_from_config():
    boxes, logits, masks = _outputs()
    masks[0, 0, 2:6, 2:6] = 0.5  # sigmoid(0.5) ~= 0.62

    lenient = decode_instances([boxes, logits, masks], (16, 16), 0.5, ModelConfig(mask_threshold=0.5))
    strict = decode_instances([boxes, logits, masks], (16, 16), 0.5, ModelConfig(mask_threshold=0.8))

    assert len(lenient) == 1
    assert strict == []


def test_draw_instances_tints_masked_pixels_only():
    base = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True
    inst = Instance(class_id=0, confidence=0.9, box=(4, 4, 8, 8), mask=mask)

    out = draw_instances_rgb(base, [inst], {0: "bottle_cap"})

    assert out.shape == base.shape
    assert (base == 0).all(), "caller's image must not be mutated"
    assert out[6, 6].any(), "masked pixel should be tinted"
