"""Contracts for the Decision Rule seam.

A Decision Rule is the plug-in post-processing step that turns a model's decoded
output into a Verdict. Every Feature's pipeline ends in one, so nothing in the
app special-cases "the Feature that has post-processing". See docs/adr/0005.

Rules are typed on *what the decode produced* (`DecodedOutput.kinds`), not on the
Feature name — so a rule written against `detections` works on any model whose
decode emits detections, including a segmentation model that emits both masks
and boxes.
"""
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

# ── Decoded model output ────────────────────────────────────────────────────

# The output kinds a decode can advertise. A rule declares which it consumes.
KIND_ANOMALY_MAP = "anomaly_map"
KIND_DETECTIONS = "detections"
KIND_MASKS = "masks"


@dataclass
class Instance:
    """One segmented object, in original-image pixel coordinates."""

    class_id: int
    confidence: float
    box: tuple[float, float, float, float]  # xyxy
    mask: np.ndarray  # bool [H, W] at original-image resolution


@dataclass
class DecodedOutput:
    """The Feature-agnostic intermediate a Decision Rule consumes.

    `kinds` is a set, not a single tag: a segmentation decode advertises both
    `masks` and `detections`, which is what lets a detections-only rule run
    unchanged on a segmentation model.
    """

    kinds: frozenset[str]
    image_hw: tuple[int, int]
    detections: list = field(default_factory=list)
    instances: list[Instance] = field(default_factory=list)
    anomaly_map: np.ndarray | None = None
    anomaly_score: float | None = None


# ── Annotation primitives ───────────────────────────────────────────────────
# Rules return declarative shapes, not rendered bytes, so they stay pure
# functions over numbers: tests assert on geometry, drawing style stays
# consistent across rules, and the same list can later ship to the frontend as
# an SVG overlay instead of being burned into a JPEG.

# RGB. Rules pick by meaning, not by literal colour.
COLOR_OK = (0, 200, 0)
COLOR_NOK = (220, 0, 0)
COLOR_REFERENCE = (0, 140, 255)
COLOR_TARGET = (255, 180, 0)
COLOR_NEUTRAL = (200, 200, 200)


@dataclass
class Point:
    """A filled or hollow dot at (x, y) in original-image pixels."""

    x: float
    y: float
    color: tuple[int, int, int] = COLOR_NEUTRAL
    radius: int = 4
    filled: bool = True


@dataclass
class Circle:
    """A circle of radius `r` centred at (cx, cy) in original-image pixels."""

    cx: float
    cy: float
    r: float
    color: tuple[int, int, int] = COLOR_NEUTRAL
    thickness: int = 2


@dataclass
class Line:
    """A straight segment between two points in original-image pixels."""

    x1: float
    y1: float
    x2: float
    y2: float
    color: tuple[int, int, int] = COLOR_NEUTRAL
    thickness: int = 2


@dataclass
class Polygon:
    """A polyline or closed polygon through `points`, in pixel coordinates."""

    points: list[tuple[float, float]]
    color: tuple[int, int, int] = COLOR_NEUTRAL
    thickness: int = 2
    closed: bool = True


@dataclass
class Text:
    """A text label drawn with its anchor at (x, y) in pixel coordinates."""

    x: float
    y: float
    text: str
    color: tuple[int, int, int] = COLOR_NEUTRAL
    scale: float = 0.5


Annotation = Point | Circle | Line | Polygon | Text


# ── Parameter schema ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParamSpec:
    """One user-tunable parameter, rendered generically by the GUI.

    `type` drives the control: "number" -> slider, "enum" -> select,
    "class" -> select populated from the Class Catalog, "class_list" ->
    multi-select, "bool" -> checkbox. No rule ever needs frontend code.
    """

    name: str
    label: str
    type: str
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None
    help: str | None = None

    def as_dict(self) -> dict:
        """The JSON shape the GUI renders a control from.

        Returns:
            Dict of every ParamSpec field, ready to ship as rule metadata.
        """
        return {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "options": self.options,
            "help": self.help,
        }


# ── Rule input / output ─────────────────────────────────────────────────────


@dataclass
class DecisionContext:
    """Everything a rule may look at.

    `image_rgb` is carried even though geometry rules ignore it, so future rules
    (colour checks inside a mask, OCR on a region) need no contract change.
    `threshold` is the model-level confidence/score floor from the main slider.
    """

    output: DecodedOutput
    image_rgb: np.ndarray
    labels: dict[int, str]
    params: dict[str, Any]
    threshold: float


@dataclass
class DecisionResult:
    """What one rule decided: the Verdict plus everything that explains it.

    `metrics` and `reason` are persisted for report traceability;
    `annotations` are declarative shapes rendered by inference.visualizer.
    """

    verdict: str  # Verdict.OK | Verdict.NOT_OK (inference.verdict)
    reason: str
    score: float | None = None
    score_label: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    annotations: list = field(default_factory=list)


@dataclass(frozen=True)
class Calibration:
    """How a rule learns a per-SKU baseline from known-good samples.

    Some artwork is not symmetric about its own centre, so it measures non-zero
    even when correctly placed, by a margin that differs per SKU. Rather than
    special-casing that in the API, a rule declares which of its params holds
    that baseline and which measured metric feeds it; the calibrate endpoint
    then works for any rule without knowing what the rule does.
    """

    param: str  # the param the taught value is written to
    metric: str  # the DecisionResult.metrics key it is computed from
    label: str = "Nominal"


@runtime_checkable
class DecisionRule(Protocol):
    """The contract every pluggable post-processing step satisfies.

    Rules are plain stateful-small objects registered at import; `consumes`
    types them on decoded-output kinds so only compatible rules are offered.
    """

    name: str
    label: str
    consumes: frozenset[str]
    params: list[ParamSpec]
    # Rules without a per-SKU baseline leave this None and are not calibratable.
    calibration: Calibration | None

    def evaluate(self, ctx: DecisionContext) -> DecisionResult:
        """Judge one inspection's decoded output under its parameters.

        Args:
            ctx: Decision context (decoded output, Threshold, params, catalog).

        Returns:
            DecisionResult with the Verdict, metrics, and overlay geometry.
        """
        ...


# ── Class-reference resolution ──────────────────────────────────────────────


class ClassNotFound(ValueError):
    """A rule param names a class the loaded model's Class Catalog lacks."""


def resolve_class(ref: Any, labels: dict[int, str]) -> int:
    """Resolve a class reference (name or id) to a class id.

    Config and GUI carry class *names*, so a retrain that reorders classes keeps
    working. A name the catalog lacks raises rather than silently resolving to
    the wrong object — a wrong class reference inverts a verdict invisibly.

    Args:
        ref: Class reference — an id, a numeric string, or a catalog name.
        labels: The loaded model's Class Catalog (id -> name).

    Returns:
        The resolved class id.

    Raises:
        ClassNotFound: If the reference is malformed or names a class the
            catalog does not contain.
    """
    if isinstance(ref, bool):
        raise ClassNotFound(f"Invalid class reference: {ref!r}")
    if isinstance(ref, int):
        return ref
    if isinstance(ref, str):
        if ref.lstrip("-").isdigit():
            return int(ref)
        for cid, name in labels.items():
            if name == ref:
                return cid
        known = ", ".join(sorted(labels.values())) or "<empty catalog>"
        raise ClassNotFound(f"Class {ref!r} is not in the model's Class Catalog ({known}).")
    raise ClassNotFound(f"Invalid class reference: {ref!r}")


def class_name(class_id: int, labels: dict[int, str]) -> str:
    """Human label for a class id.

    Args:
        class_id: Numeric class id from a decode.
        labels: The loaded model's Class Catalog (id -> name).

    Returns:
        The catalog label, or the id as a string when unlabelled.
    """
    return labels.get(class_id, str(class_id))
