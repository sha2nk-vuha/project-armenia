"""Concentric-placement rule: is the target centred on the reference?

Class-agnostic by construction. `reference_class` and `target_class` are params,
so the same module answers "is the logo centred on the bottle cap", "is the seal
centred on the can", or "is the label centred on the jar" with no new code.

The offset is reported as a **ratio of the reference's radius**, not in pixels.
A pixel tolerance tuned at one working distance or resolution silently changes
meaning at another; a dimensionless eccentricity survives both, and maps onto the
0-1 control the GUI already has. Raw pixels are still reported in `metrics` for
operator intuition.
"""
import math

import numpy as np

from inference.decision.base import (
    COLOR_NEUTRAL,
    COLOR_NOK,
    COLOR_OK,
    COLOR_REFERENCE,
    COLOR_TARGET,
    KIND_MASKS,
    Circle,
    DecisionContext,
    DecisionResult,
    Line,
    ParamSpec,
    Point,
    Text,
    class_name,
    resolve_class,
)
from inference.decision.registry import register

CENTER_METHODS = [
    "min_enclosing_circle",
    "convex_hull_centroid",
    "mask_centroid",
    "bbox_center",
]


def _mask_points(mask: np.ndarray) -> np.ndarray:
    """Non-zero mask pixels as an OpenCV-style [N,1,2] int32 array of (x, y)."""
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.int32)


def compute_center(mask: np.ndarray, method: str) -> tuple[float, float]:
    """The centre of a boolean mask under the chosen method.

    The methods differ in what they are robust to, which is why this is a
    parameter rather than a hardcoded choice:
    - `min_enclosing_circle`: the centre of the smallest circle containing the
      mask. Best for round parts; insensitive to lopsided mask density but
      sensitive to a single stray pixel.
    - `convex_hull_centroid`: centroid of the mask's convex hull. Ignores
      concavities (a logo's cut-outs) that would drag a centre of mass.
    - `mask_centroid`: plain centre of mass. Pulled by asymmetric detail.
    - `bbox_center`: centre of the axis-aligned bounding box. Cheapest, and the
      most affected by rotation.
    """
    import cv2

    if not mask.any():
        raise ValueError("Cannot compute the centre of an empty mask.")

    if method == "mask_centroid":
        ys, xs = np.nonzero(mask)
        return float(xs.mean()), float(ys.mean())

    if method == "bbox_center":
        ys, xs = np.nonzero(mask)
        return float((xs.min() + xs.max()) / 2.0), float((ys.min() + ys.max()) / 2.0)

    points = _mask_points(mask)

    if method == "min_enclosing_circle":
        (cx, cy), _ = cv2.minEnclosingCircle(points)
        return float(cx), float(cy)

    if method == "convex_hull_centroid":
        hull = cv2.convexHull(points)
        m = cv2.moments(hull)
        if m["m00"] == 0:  # degenerate hull (collinear pixels)
            return float(hull[:, 0, 0].mean()), float(hull[:, 0, 1].mean())
        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])

    raise ValueError(f"Unknown centre method: {method!r} (known: {', '.join(CENTER_METHODS)})")


def enclosing_radius(mask: np.ndarray) -> float:
    """Radius of the mask's minimum enclosing circle.

    Always used as the normalising scale, whichever centre method is selected,
    so changing how the centre is found never silently rescales the tolerance.
    """
    import cv2

    _, radius = cv2.minEnclosingCircle(_mask_points(mask))
    return float(radius)


class ConcentricityRule:
    name = "concentricity"
    label = "Concentric Placement"
    consumes = frozenset({KIND_MASKS})
    params = [
        ParamSpec(
            name="reference_class",
            label="Reference Class",
            type="class",
            default="bottle_cap",
            help="The part the target must be centred on.",
        ),
        ParamSpec(
            name="target_class",
            label="Target Class",
            type="class",
            default="logo",
            help="The part whose placement is being judged.",
        ),
        ParamSpec(
            name="max_offset_ratio",
            label="Max Offset",
            type="number",
            default=0.10,
            min=0.0,
            max=1.0,
            step=0.01,
            help="Centre offset as a fraction of the reference radius. Above this, NOK.",
        ),
        ParamSpec(
            name="reference_center_method",
            label="Reference Centre",
            type="enum",
            default="min_enclosing_circle",
            options=CENTER_METHODS,
        ),
        ParamSpec(
            name="target_center_method",
            label="Target Centre",
            type="enum",
            default="convex_hull_centroid",
            options=CENTER_METHODS,
        ),
    ]

    def evaluate(self, ctx: DecisionContext) -> DecisionResult:
        labels = ctx.labels
        ref_id = resolve_class(ctx.params["reference_class"], labels)
        tgt_id = resolve_class(ctx.params["target_class"], labels)
        tolerance = float(ctx.params["max_offset_ratio"])

        ref_all = [i for i in ctx.output.instances if i.class_id == ref_id]
        tgt_all = [i for i in ctx.output.instances if i.class_id == tgt_id]
        counts = {
            "reference_instances": len(ref_all),
            "target_instances": len(tgt_all),
            "tolerance": tolerance,
        }

        # Fail safe: a missing part is NOK, and the reason says which kind of
        # failure it was rather than inventing a third Verdict state.
        missing = [
            class_name(cid, labels)
            for cid, found in ((ref_id, ref_all), (tgt_id, tgt_all))
            if not found
        ]
        if missing:
            return DecisionResult(
                verdict="not_ok",
                reason=f"not detected above {ctx.threshold:.2f} confidence: {', '.join(missing)}",
                score=None,
                score_label="Offset Ratio",
                metrics=counts,
            )

        # More than one of a class: take the most confident and say so.
        reference = max(ref_all, key=lambda i: i.confidence)
        target = max(tgt_all, key=lambda i: i.confidence)

        ref_radius = enclosing_radius(reference.mask)
        if ref_radius <= 0:
            return DecisionResult(
                verdict="not_ok",
                reason=f"{class_name(ref_id, labels)} mask has no measurable size",
                score=None,
                score_label="Offset Ratio",
                metrics=counts,
            )

        ref_cx, ref_cy = compute_center(reference.mask, ctx.params["reference_center_method"])
        tgt_cx, tgt_cy = compute_center(target.mask, ctx.params["target_center_method"])
        tgt_radius = enclosing_radius(target.mask)

        offset_px = math.hypot(tgt_cx - ref_cx, tgt_cy - ref_cy)
        offset_ratio = offset_px / ref_radius
        ok = offset_ratio <= tolerance

        metrics = {
            **counts,
            "offset_ratio": round(offset_ratio, 4),
            "offset_px": round(offset_px, 2),
            "reference_radius_px": round(ref_radius, 2),
            "target_radius_px": round(tgt_radius, 2),
            "reference_center": [round(ref_cx, 1), round(ref_cy, 1)],
            "target_center": [round(tgt_cx, 1), round(tgt_cy, 1)],
        }

        reason = (
            f"offset {offset_ratio:.3f} {'<=' if ok else '>'} tolerance {tolerance:.3f} "
            f"({offset_px:.1f}px of {ref_radius:.1f}px radius)"
        )

        # Guard against a mislabelled Class Catalog: a target larger than the
        # part it sits on is physically backwards, and would otherwise produce a
        # plausible-looking ratio from a swapped reference/target pair.
        if tgt_radius > ref_radius:
            metrics["warning"] = (
                f"{class_name(tgt_id, labels)} is larger than "
                f"{class_name(ref_id, labels)} — reference and target may be swapped"
            )
            reason = f"{reason}; {metrics['warning']}"

        verdict_color = COLOR_OK if ok else COLOR_NOK
        annotations = [
            Circle(ref_cx, ref_cy, ref_radius, COLOR_REFERENCE, thickness=2),
            # The tolerance the operator set, drawn where they can see it.
            Circle(ref_cx, ref_cy, tolerance * ref_radius, COLOR_NEUTRAL, thickness=1),
            Point(ref_cx, ref_cy, COLOR_REFERENCE, radius=5),
            Point(tgt_cx, tgt_cy, COLOR_TARGET, radius=5),
            Line(ref_cx, ref_cy, tgt_cx, tgt_cy, verdict_color, thickness=2),
            Text(
                ref_cx + 8,
                ref_cy - 8,
                f"{offset_ratio:.3f}",
                verdict_color,
                scale=0.6,
            ),
        ]

        return DecisionResult(
            verdict="ok" if ok else "not_ok",
            reason=reason,
            score=offset_ratio,
            score_label="Offset Ratio",
            metrics=metrics,
            annotations=annotations,
        )


register(ConcentricityRule())
