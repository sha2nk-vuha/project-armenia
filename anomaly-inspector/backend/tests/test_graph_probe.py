"""Probe an ONNX graph for baked-in preprocessing.

Each test builds a minimal graph with a known preamble and asserts what the
probe reads back, including how it resolves what it cannot see.
"""
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from inference.graph_probe import GraphPreprocessing, probe

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _conv(input_name, output_name="out"):
    """A weight-bearing op, which is what tells the probe the preamble ended."""
    return helper.make_node("Conv", [input_name, "w"], [output_name])


_CONV_WEIGHT = numpy_helper.from_array(np.zeros((3, 3, 1, 1), dtype=np.float32), "w")


def _build(nodes, initializers, output_name="out"):
    """Serialise a graph with one [1,3,8,8] float input and the given nodes."""
    graph = helper.make_graph(
        nodes,
        "probe-test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 3, 8, 8])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model.SerializeToString()


def _const(name, array):
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def test_detects_scaling_and_normalization():
    model = _build(
        [
            helper.make_node("Div", ["input", "c255"], ["scaled"]),
            helper.make_node("Sub", ["scaled", "mean"], ["centred"]),
            helper.make_node("Div", ["centred", "std"], ["normed"]),
            _conv("normed"),
        ],
        [_const("c255", 255.0), _const("mean", _MEAN), _const("std", _STD), _CONV_WEIGHT],
    )
    result = probe(model)
    assert result.scales is True
    assert result.normalizes is True
    assert result.needs_scaling is False
    assert result.needs_normalization is False


def test_detects_normalization_without_scaling():
    """The Anomalib shape: the graph normalises but expects [0,1] input."""
    model = _build(
        [
            helper.make_node("Sub", ["input", "mean"], ["centred"]),
            helper.make_node("Div", ["centred", "std"], ["normed"]),
            _conv("normed"),
        ],
        [_const("mean", _MEAN), _const("std", _STD), _CONV_WEIGHT],
    )
    result = probe(model)
    assert result.scales is False
    assert result.normalizes is True
    assert result.needs_scaling is True
    assert result.needs_normalization is False


def test_bare_graph_needs_both_steps():
    """A stock export with no preamble: the caller must do all of it."""
    model = _build([_conv("input")], [_CONV_WEIGHT])
    result = probe(model)
    assert result.scales is False
    assert result.normalizes is False
    assert result.needs_scaling is True
    assert result.needs_normalization is True


def test_recognises_reciprocal_scaling_form():
    model = _build(
        [
            helper.make_node("Mul", ["input", "inv255"], ["scaled"]),
            _conv("scaled"),
        ],
        [_const("inv255", 1.0 / 255.0), _CONV_WEIGHT],
    )
    assert probe(model).scales is True


def test_scalar_that_is_not_255_counts_as_normalization():
    """It isn't the rescale, but it is a value transform we must not repeat."""
    model = _build(
        [
            helper.make_node("Mul", ["input", "two"], ["doubled"]),
            _conv("doubled"),
        ],
        [_const("two", 2.0), _CONV_WEIGHT],
    )
    result = probe(model)
    assert result.scales is False
    assert result.normalizes is True


def test_reads_through_constant_nodes_and_passthrough_ops():
    """Constants can be nodes rather than initializers, behind a Cast."""
    mean_const = helper.make_node(
        "Constant", [], ["mean"], value=numpy_helper.from_array(_MEAN, "mean_v")
    )
    model = _build(
        [
            mean_const,
            helper.make_node("Cast", ["input"], ["cast"], to=TensorProto.FLOAT),
            helper.make_node("Sub", ["cast", "mean"], ["centred"]),
            _conv("centred"),
        ],
        [_CONV_WEIGHT],
    )
    assert probe(model).normalizes is True


def test_forked_input_is_unknown_not_guessed():
    """Two value consumers: we cannot say which path is preprocessing."""
    model = _build(
        [
            helper.make_node("Relu", ["input"], ["a"]),
            helper.make_node("Sigmoid", ["input"], ["b"]),
            helper.make_node("Add", ["a", "b"], ["out"]),
        ],
        [],
    )
    result = probe(model)
    assert result.scales is None
    assert result.normalizes is None
    assert "partial probe" in result.detail


def test_shape_consumers_do_not_count_as_a_fork():
    """A Shape node reads metadata, not values, so the chain is still linear."""
    model = _build(
        [
            helper.make_node("Shape", ["input"], ["shape"]),
            helper.make_node("Div", ["input", "c255"], ["scaled"]),
            _conv("scaled"),
        ],
        [_const("c255", 255.0), _CONV_WEIGHT],
    )
    assert probe(model).scales is True


def test_partial_probe_keeps_what_it_saw_and_unknowns_the_rest():
    """Scaling is confirmed; the fork after it leaves normalisation unknown."""
    model = _build(
        [
            helper.make_node("Div", ["input", "c255"], ["scaled"]),
            helper.make_node("Relu", ["scaled"], ["a"]),
            helper.make_node("Sigmoid", ["scaled"], ["b"]),
            helper.make_node("Add", ["a", "b"], ["out"]),
        ],
        [_const("c255", 255.0)],
    )
    result = probe(model)
    assert result.scales is True
    assert result.normalizes is None
    assert result.needs_normalization is False  # unknown → skip


def test_unreadable_bytes_are_unknown_not_an_error():
    result = probe(b"not an onnx model")
    assert result.scales is None
    assert result.normalizes is None
    assert result.needs_scaling is True       # unknown → scale, as before
    assert result.needs_normalization is False  # unknown → skip, as before


def test_default_is_unknown_and_matches_pre_probe_behaviour():
    default = GraphPreprocessing()
    assert default.needs_scaling is True
    assert default.needs_normalization is False


def test_reads_through_anomalibs_shape_plumbing_preamble():
    """Anomalib buries `Normalize` behind Reshape/Resize/Slice nodes.

    Regression guard: an early stop here reports "no normalisation" for a model
    that plainly normalises, and the caller then double-normalises it.
    """
    model = _build(
        [
            helper.make_node("Reshape", ["input", "shape"], ["reshaped"]),
            helper.make_node("Resize", ["reshaped", "", "", "size"], ["resized"]),
            helper.make_node("Slice", ["resized", "s0", "s1", "ax"], ["cropped"]),
            helper.make_node("Sub", ["cropped", "mean"], ["centred"]),
            helper.make_node("Div", ["centred", "std"], ["normed"]),
            _conv("normed"),
        ],
        [
            numpy_helper.from_array(np.array([1, 3, 8, 8], dtype=np.int64), "shape"),
            numpy_helper.from_array(np.array([1, 3, 8, 8], dtype=np.int64), "size"),
            numpy_helper.from_array(np.array([0], dtype=np.int64), "s0"),
            numpy_helper.from_array(np.array([8], dtype=np.int64), "s1"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "ax"),
            _const("mean", _MEAN),
            _const("std", _STD),
            _CONV_WEIGHT,
        ],
    )
    result = probe(model)
    assert result.scales is False
    assert result.normalizes is True


def test_unrecognised_op_is_unknown_rather_than_a_confident_no():
    """We stopped understanding the graph; that is not evidence of absence."""
    model = _build([helper.make_node("Softmax", ["input"], ["out"])], [])
    result = probe(model)
    assert result.normalizes is None
    assert "unrecognised op" in result.detail
