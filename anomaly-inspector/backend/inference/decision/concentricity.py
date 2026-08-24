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
    Calibration,
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
    "outer_circle_fit",
]

# `outer_circle_fit` tuning. These are deliberately not exposed as rule params:
# they describe how the estimator works, not a process tolerance, and giving an
# operator two extra knobs that interact with the offset threshold invites
# mistuning. Measured stable across 0.45-0.65 on the sample caps.
_ENVELOPE_BINS = 180
_ENVELOPE_KEEP = 0.55
_FIT_REFINE_ITERS = 5


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
    - `outer_circle_fit`: fits a circle to the mask's outer edge. The only
      method that survives a mask covering just part of a ring-shaped print,
      where every centroid-style estimator is dragged off by the missing side.
      Meaningless on free-form artwork with no circular outer edge.
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

    if method == "outer_circle_fit":
        cx, cy, _ = fit_outer_circle(mask)
        return cx, cy

    if method == "convex_hull_centroid":
        hull = cv2.convexHull(points)
        m = cv2.moments(hull)
        if m["m00"] == 0:  # degenerate hull (collinear pixels)
            return float(hull[:, 0, 0].mean()), float(hull[:, 0, 1].mean())
        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])

    raise ValueError(f"Unknown centre method: {method!r} (known: {', '.join(CENTER_METHODS)})")


def _fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle through `points` [N,2]."""
    x, y = points[:, 0], points[:, 1]
    design = np.stack([x, y, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(design, x**2 + y**2, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    return cx, cy, float(np.sqrt(max(sol[2] + cx**2 + cy**2, 0.0)))


def _outer_envelope(mask: np.ndarray) -> np.ndarray:
    """The outermost mask pixel per angular bin, keeping only the far ones.

    Seeded from the mask's own minimum-enclosing-circle centre, so the estimate
    never depends on the reference part -- a centre estimator that peeked at the
    cap centre could not be trusted to measure its distance from it.

    The quantile filter is what makes this work on a partial ring: where the
    print exists these points lie on its outer edge, while bins that only reach
    an interior boundary (the flat side of a crescent) fall short and are cut.
    """
    seed_x, seed_y = compute_center(mask, "min_enclosing_circle")
    ys, xs = np.nonzero(mask)
    dx, dy = xs - seed_x, ys - seed_y
    radius = np.hypot(dx, dy)
    bins = (
        (np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * _ENVELOPE_BINS
    ).astype(int) % _ENVELOPE_BINS

    # Farthest pixel per bin: sort by (bin, -radius) and take each bin's first.
    order = np.lexsort((-radius, bins))
    sorted_bins = bins[order]
    is_first = np.ones(len(sorted_bins), dtype=bool)
    is_first[1:] = sorted_bins[1:] != sorted_bins[:-1]
    idx = order[is_first]

    far = radius[idx]
    kept = idx[far >= np.quantile(far, _ENVELOPE_KEEP)]
    return np.stack([xs[kept], ys[kept]], axis=1).astype(float)


def fit_outer_circle(mask: np.ndarray) -> tuple[float, float, float]:
    """Fit a circle to the outer edge of a mask; returns (cx, cy, r).

    For a print whose artwork sits on a circle -- a badge, or text arced around
    a rim -- this recovers the design centre even when the mask covers only part
    of that circle. That is the case a centroid cannot handle: the centroid of a
    crescent is nowhere near the ring it came from.

    It is *not* suitable for free-form artwork with no circular outer edge; on
    such a mask the fitted circle is arbitrary. Hence a selectable method rather
    than the default.
    """
    points = _outer_envelope(mask)
    if len(points) < 3:
        raise ValueError("Too few boundary points to fit a circle.")
    for _ in range(_FIT_REFINE_ITERS):
        cx, cy, r = _fit_circle(points)
        residual = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
        keep = residual <= max(float(np.median(residual)) * 2.0, 1.0)
        if keep.sum() < 12:
            break
        points = points[keep]
    return _fit_circle(points)


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
            help="Tolerance on the offset, as a fraction of the reference radius.",
        ),
        ParamSpec(
            name="nominal_offset",
            label="Nominal Offset",
            type="number",
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.005,
            help=(
                "This SKU's offset when correctly placed; the tolerance applies "
                "to the deviation from it. Leave at 0 unless calibrated."
            ),
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

    # Teaching `nominal_offset` from the median measured offset of known-good
    # caps is what lets one tolerance serve SKUs whose artwork is not symmetric.
    calibration = Calibration(
        param="nominal_offset", metric="offset_ratio", label="Nominal Offset"
    )

    def evaluate(self, ctx: DecisionContext) -> DecisionResult:
        labels = ctx.labels
        ref_id = resolve_class(ctx.params["reference_class"], labels)
        tgt_id = resolve_class(ctx.params["target_class"], labels)
        tolerance = float(ctx.params["max_offset_ratio"])
        # Artwork that is not symmetric about its own centre has a non-zero
        # offset even when perfectly placed, and that bias differs per SKU. The
        # nominal absorbs it so one tolerance serves every SKU; it stays 0 for
        # artwork the estimator already centres correctly.
        nominal = float(ctx.params.get("nominal_offset") or 0.0)

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
        deviation = abs(offset_ratio - nominal)
        ok = deviation <= tolerance

        metrics = {
            **counts,
            "offset_ratio": round(offset_ratio, 4),
            "nominal_offset": round(nominal, 4),
            "deviation": round(deviation, 4),
            "offset_px": round(offset_px, 2),
            "reference_radius_px": round(ref_radius, 2),
            "target_radius_px": round(tgt_radius, 2),
            "reference_center": [round(ref_cx, 1), round(ref_cy, 1)],
            "target_center": [round(tgt_cx, 1), round(tgt_cy, 1)],
        }

        if nominal:
            reason = (
                f"offset {offset_ratio:.3f} deviates {deviation:.3f} from nominal "
                f"{nominal:.3f}, {'within' if ok else 'over'} tolerance {tolerance:.3f}"
            )
        else:
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
            # Tolerance band. With a nominal it is an annulus, since any offset
            # within `tolerance` of the nominal passes.
            Circle(ref_cx, ref_cy, (nominal + tolerance) * ref_radius, COLOR_NEUTRAL, thickness=1),
            *(
                [Circle(ref_cx, ref_cy, max(nominal - tolerance, 0.0) * ref_radius,
                        COLOR_NEUTRAL, thickness=1)]
                if nominal
                else []
            ),
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
            score=deviation,
            score_label="Offset Deviation" if nominal else "Offset Ratio",
            metrics=metrics,
            annotations=annotations,
        )


register(ConcentricityRule())
