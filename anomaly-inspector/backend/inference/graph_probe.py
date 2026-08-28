"""Detect which preprocessing steps an ONNX export bakes into its own graph.

Exports differ in where preprocessing lives: Anomalib folds its `Normalize`
transform into the graph, while a stock RF-DETR export expects the caller to
normalise. Guessing wrong is silent — double-normalising saturates anomaly
scores, under-normalising degrades them — so instead of assuming, we read the
graph and let the caller apply only the steps the model does not already do.

The probe walks forward from the graph's data input through the leading chain of
shape-plumbing and constant-operand elementwise ops, classifying each constant:
a scalar 255 (or 1/255) is the [0,255] -> [0,1] scaling, any other constant
operand is a value transform we count as normalisation. The walk only concludes
"this graph does no preprocessing" once it reaches a weight-bearing op (Conv,
Gemm, MatMul, ...) — proof that the preamble is behind us. Stopping anywhere
else means we ran out of understanding, not that nothing was there, and is
reported as unknown.

That asymmetry is deliberate. Anomalib's export buries its `Normalize` behind a
dozen Reshape/Resize/Slice nodes, so a probe that treated "unrecognised op" as
"no normalisation" would confidently double-normalise the models most likely to
be loaded. Every caller resolves unknown to the pre-probe status quo: apply
scaling, skip normalisation.
"""
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Ops that can carry a preprocessing constant.
_ELEMENTWISE = {"Sub", "Add", "Mul", "Div"}
# Ops that move or reshape values without changing them, so preprocessing may
# still sit past them. Anomalib's export_transform is mostly made of these.
_PASSTHROUGH = {
    "Identity", "Cast", "Reshape", "Transpose", "Squeeze", "Unsqueeze",
    "Resize", "Slice", "Pad", "Flatten",
}
# Reaching one of these means the model proper has begun, so anything we did not
# see along the way genuinely is not in the graph. Only here can we conclude a
# definite "no".
_MODEL_START = {"Conv", "ConvTranspose", "Gemm", "MatMul", "Einsum", "LSTM", "GRU"}
# Consumers that read a tensor's metadata, not its values; they do not fork the
# value chain and are ignored when deciding whether a tensor has one consumer.
_METADATA_ONLY = {"Shape", "Size"}
# Generous enough for Anomalib's shape-plumbing preamble, bounded so a cyclic or
# pathological graph cannot spin.
_MAX_HOPS = 32


@dataclass(frozen=True)
class GraphPreprocessing:
    """What the model's own graph does to its input, as far as we can tell.

    `True` = the graph does this step, `False` = it provably does not, `None` =
    we could not determine it. Callers should use the `needs_*` properties
    rather than the raw fields, so that unknown resolves consistently.
    """

    scales: bool | None = None       # divides by 255
    normalizes: bool | None = None   # applies per-channel mean/std
    detail: str = "not probed"

    @property
    def needs_scaling(self) -> bool:
        """Should the caller divide by 255? Unknown resolves to yes.

        Feeding a [0,255] tensor to a model trained on [0,1] is a far worse
        failure than the reverse, and no export we have seen bakes the /255 in,
        so an unreadable graph keeps the long-standing behaviour of scaling.
        """
        return self.scales is not True

    @property
    def needs_normalization(self) -> bool:
        """Should the caller apply mean/std? Only when the graph provably won't.

        Anomalib and most detector exports embed `Normalize`, so unknown
        resolves to skipping — normalising a second time saturates the output.
        """
        return self.normalizes is False


def probe(model_bytes: bytes) -> GraphPreprocessing:
    """Inspect ONNX bytes for a baked-in preprocessing preamble.

    Never raises: any failure to parse or walk the graph yields an all-unknown
    result, so a model that loads for inference still loads here.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return GraphPreprocessing(detail="onnx package unavailable")

    try:
        graph = onnx.load_from_string(model_bytes).graph
    except Exception as e:  # malformed, truncated, or an external-data model
        return GraphPreprocessing(detail=f"graph unreadable: {e}")

    try:
        return _walk(graph, numpy_helper)
    except Exception as e:
        logger.warning("Preprocessing probe failed; assuming unknown: %s", e)
        return GraphPreprocessing(detail=f"probe error: {e}")


def _walk(graph, numpy_helper) -> GraphPreprocessing:
    initializers = {init.name for init in graph.initializer}
    data_inputs = [i.name for i in graph.input if i.name not in initializers]
    if not data_inputs:
        return GraphPreprocessing(detail="no data input")

    constants = _constants(graph, numpy_helper)
    consumers = _consumers(graph)

    current = data_inputs[0]
    scales = False
    normalizes = False
    truncated_reason = None

    for _ in range(_MAX_HOPS):
        nodes = [n for n in consumers.get(current, []) if n.op_type not in _METADATA_ONLY]
        if len(nodes) != 1:
            truncated_reason = f"{len(nodes)} value consumers of {current!r}"
            break
        node = nodes[0]

        if node.op_type in _PASSTHROUGH:
            current = node.output[0]
            continue
        if node.op_type in _MODEL_START:
            break  # weights: the preamble is behind us and fully seen
        if node.op_type not in _ELEMENTWISE:
            truncated_reason = f"unrecognised op {node.op_type!r}"
            break

        const = _constant_operand(node, current, constants)
        if const is None:
            truncated_reason = f"{node.op_type} has no constant operand"
            break

        if const.size == 1 and _is_scaling(node.op_type, float(const.reshape(-1)[0])):
            scales = True
        else:
            # Any other constant applied to the input is a value transform the
            # caller must not repeat, whether per-channel mean/std or a scalar
            # equivalent on a single-channel model.
            normalizes = True
        current = node.output[0]
    else:
        truncated_reason = f"chain longer than {_MAX_HOPS} ops"

    if truncated_reason is None:
        return GraphPreprocessing(
            scales=scales,
            normalizes=normalizes,
            detail=f"probed: scales={scales} normalizes={normalizes}",
        )
    # A truncated walk can confirm what it saw but cannot rule out what it did
    # not reach, so unseen steps stay unknown rather than becoming False.
    return GraphPreprocessing(
        scales=True if scales else None,
        normalizes=True if normalizes else None,
        detail=f"partial probe ({truncated_reason})",
    )


def _is_scaling(op_type: str, value: float) -> bool:
    """Is this op the [0,255] -> [0,1] rescale, in either of its two forms?"""
    if op_type == "Div":
        return abs(value - 255.0) < 1e-3
    if op_type == "Mul":
        return abs(value - 1.0 / 255.0) < 1e-6
    return False


def _constants(graph, numpy_helper) -> dict:
    """Every tensor whose value is known statically, by name."""
    consts = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    consts[node.output[0]] = numpy_helper.to_array(attr.t)
    return consts


def _consumers(graph) -> dict:
    """Map each tensor name to the nodes that take it as an input."""
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    return consumers


def _constant_operand(node, data_input: str, constants: dict):
    """The constant side of a binary op, or None if the other side isn't one."""
    for name in node.input:
        if name != data_input and name in constants:
            return constants[name]
    return None
