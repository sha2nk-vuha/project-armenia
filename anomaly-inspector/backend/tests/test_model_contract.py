"""Contract tests against the real bundled ONNX exports.

These assert the I/O signature each decode was written against, so swapping a
model file fails here with a clear message rather than silently mis-decoding
into confident nonsense. Skipped when a model file is absent (e.g. a lean
checkout), because a missing file is not a broken contract.
"""
import pytest

from config import FEATURES, PRESENCE_FEATURE, SEGMENTATION_FEATURE
from inference.model_config import load_config, sidecar_path

onnx = pytest.importorskip("onnx")


def _graph(feature):
    path = FEATURES[feature]["model_path"]
    if not path.exists():
        pytest.skip(f"model not present: {path}")
    return onnx.load(str(path), load_external_data=False).graph


def _shape(value_info):
    return [
        d.dim_value if d.HasField("dim_value") else None
        for d in value_info.type.tensor_type.shape.dim
    ]


def test_segmentation_export_signature():
    g = _graph(SEGMENTATION_FEATURE)
    outputs = {o.name: _shape(o) for o in g.output}
    inputs = {i.name: _shape(i) for i in g.input}

    (in_shape,) = inputs.values()
    assert in_shape[1] == 3, "expected a 3-channel image input"

    # Three outputs: boxes (rank 3, last dim 4), logits (rank 3), masks (rank 4).
    ranks = sorted(len(s) for s in outputs.values())
    assert ranks == [3, 3, 4], f"unexpected output ranks: {outputs}"
    assert any(len(s) == 3 and s[-1] == 4 for s in outputs.values()), "no cxcywh boxes"


def test_segmentation_masks_are_stride_four_full_frame():
    """The decode resizes masks straight to the original image with no
    box-relative paste. That is only correct if masks are full-frame; the
    stride-4 relationship to the input size is the structural signal."""
    g = _graph(SEGMENTATION_FEATURE)
    (in_shape,) = [_shape(i) for i in g.input]
    masks = next(_shape(o) for o in g.output if len(_shape(o)) == 4)

    input_hw, mask_hw = in_shape[2:4], masks[2:4]
    assert all(m * 4 == i for m, i in zip(mask_hw, input_hw)), (
        f"masks {mask_hw} are not stride-4 of input {input_hw}; the full-frame "
        "assumption in rfdetr_seg.decode_instances no longer holds"
    )


def test_segmentation_sidecar_labels_match_the_export():
    """The sidecar carries only the Class Catalog now (input size comes from the
    session, pre/post from the ONNX signature). A catalog that names classes the
    export does not have would mislabel detections silently."""
    g = _graph(SEGMENTATION_FEATURE)
    path = FEATURES[SEGMENTATION_FEATURE]["model_path"]
    config = load_config(sidecar_path(str(path)))

    num_classes = next(_shape(o)[-1] for o in g.output if len(_shape(o)) == 3 and _shape(o)[-1] != 4)
    assert len(config.labels) == num_classes
    # Verified empirically against the sample cap images.
    assert config.labels[0] == "bottle_cap"
    assert config.labels[1] == "logo"


def test_detection_export_signature():
    g = _graph(PRESENCE_FEATURE)
    ranks = sorted(len(_shape(o)) for o in g.output)
    # Detection-only: exactly two rank-3 outputs, no mask tensor.
    assert ranks == [3, 3], "detection export unexpectedly has a mask output"
