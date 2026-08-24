"""Concentric-placement rule.

Geometry over masks is pure arithmetic, so the expected answers here are
analytic: a disc of known radius offset by a known number of pixels has exactly
one correct ratio. No model is involved.
"""
import numpy as np
import pytest

from inference.decision import (
    KIND_DETECTIONS,
    KIND_MASKS,
    Circle,
    ClassNotFound,
    DecisionContext,
    DecodedOutput,
    Instance,
    Line,
    get,
)
from inference.decision.concentricity import CENTER_METHODS, compute_center, enclosing_radius

CANVAS = 400
LABELS = {0: "bottle_cap", 1: "logo"}
RULE = get("concentricity")

DEFAULTS = {
    "reference_class": "bottle_cap",
    "target_class": "logo",
    "max_offset_ratio": 0.10,
    "reference_center_method": "min_enclosing_circle",
    "target_center_method": "convex_hull_centroid",
}


def _disc(cx, cy, r, size=CANVAS):
    ys, xs = np.ogrid[:size, :size]
    return ((xs - cx) ** 2 + (ys - cy) ** 2) <= r * r


def _inst(class_id, mask, confidence=0.9):
    ys, xs = np.nonzero(mask)
    return Instance(
        class_id=class_id,
        confidence=confidence,
        box=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
        mask=mask,
    )


def _evaluate(instances, **param_overrides):
    params = {**DEFAULTS, **param_overrides}
    ctx = DecisionContext(
        output=DecodedOutput(
            kinds=frozenset({KIND_DETECTIONS, KIND_MASKS}),
            image_hw=(CANVAS, CANVAS),
            instances=instances,
        ),
        image_rgb=np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8),
        labels=LABELS,
        params=params,
        threshold=0.5,
    )
    return RULE.evaluate(ctx)


# ── Centre methods, analytically ────────────────────────────────────────────


@pytest.mark.parametrize("method", CENTER_METHODS)
def test_every_centre_method_finds_the_centre_of_a_disc(method):
    # A disc is symmetric, so all four methods must agree on its centre.
    cx, cy = compute_center(_disc(200, 150, 60), method)
    assert cx == pytest.approx(200, abs=1.5)
    assert cy == pytest.approx(150, abs=1.5)


def test_enclosing_radius_matches_the_drawn_radius():
    assert enclosing_radius(_disc(200, 200, 80)) == pytest.approx(80, abs=1.5)


def test_centre_methods_disagree_on_an_asymmetric_shape():
    """The reason this is a parameter and not a hardcoded choice: on a lopsided
    mask the methods genuinely differ, so the operator can pick what suits."""
    mask = _disc(200, 200, 60) | _disc(260, 250, 25)  # a lobe hanging off one side

    com = compute_center(mask, "mask_centroid")
    circle = compute_center(mask, "min_enclosing_circle")

    assert abs(com[0] - circle[0]) > 3 or abs(com[1] - circle[1]) > 3


def test_empty_mask_centre_raises():
    with pytest.raises(ValueError, match="empty mask"):
        compute_center(np.zeros((10, 10), dtype=bool), "mask_centroid")


def test_unknown_centre_method_lists_the_known_ones():
    with pytest.raises(ValueError, match="min_enclosing_circle"):
        compute_center(_disc(50, 50, 10, size=100), "centre_of_vibes")


# ── The verdict, analytically ───────────────────────────────────────────────


def test_perfectly_concentric_is_ok_with_zero_offset():
    res = _evaluate([_inst(0, _disc(200, 200, 100)), _inst(1, _disc(200, 200, 40))])

    assert res.verdict == "ok"
    assert res.score == pytest.approx(0.0, abs=0.02)
    assert res.score_label == "Offset Ratio"


def test_offset_is_reported_as_a_fraction_of_reference_radius():
    # Logo centre displaced 12px inside a 100px-radius cap -> ratio 0.12.
    res = _evaluate([_inst(0, _disc(200, 200, 100)), _inst(1, _disc(212, 200, 40))])

    assert res.score == pytest.approx(0.12, abs=0.02)
    assert res.metrics["offset_px"] == pytest.approx(12, abs=1.5)
    assert res.metrics["reference_radius_px"] == pytest.approx(100, abs=1.5)


def test_verdict_flips_at_the_tolerance():
    instances = [_inst(0, _disc(200, 200, 100)), _inst(1, _disc(212, 200, 40))]

    assert _evaluate(instances, max_offset_ratio=0.15).verdict == "ok"
    assert _evaluate(instances, max_offset_ratio=0.10).verdict == "not_ok"


def test_ratio_is_scale_invariant():
    """The whole point of normalising: the same part imaged twice as large must
    produce the same ratio, so a tuned tolerance survives a resolution change."""
    small = _evaluate([_inst(0, _disc(200, 200, 50)), _inst(1, _disc(206, 200, 20))])
    large = _evaluate([_inst(0, _disc(200, 200, 100)), _inst(1, _disc(212, 200, 40))])

    assert small.score == pytest.approx(large.score, abs=0.02)
    assert small.metrics["offset_px"] != large.metrics["offset_px"]


def test_diagonal_offset_uses_euclidean_distance():
    # 30px right and 40px down inside a 100px radius -> hypotenuse 50 -> 0.50.
    res = _evaluate([_inst(0, _disc(200, 200, 100)), _inst(1, _disc(230, 240, 20))])
    assert res.metrics["offset_px"] == pytest.approx(50, abs=2)
    assert res.score == pytest.approx(0.50, abs=0.03)


# ── Degenerate scenes fail safe ─────────────────────────────────────────────


def test_missing_target_is_nok_and_says_so():
    res = _evaluate([_inst(0, _disc(200, 200, 100))])

    assert res.verdict == "not_ok"
    assert "logo" in res.reason
    assert res.score is None


def test_missing_reference_is_nok_and_says_so():
    res = _evaluate([_inst(1, _disc(200, 200, 40))])

    assert res.verdict == "not_ok"
    assert "bottle_cap" in res.reason


def test_multiple_instances_use_the_most_confident_and_report_the_count():
    res = _evaluate([
        _inst(0, _disc(200, 200, 100), confidence=0.99),
        _inst(1, _disc(200, 200, 40), confidence=0.95),   # concentric, most confident
        _inst(1, _disc(320, 200, 20), confidence=0.60),   # a far-off distractor
    ])

    assert res.verdict == "ok"
    assert res.metrics["target_instances"] == 2


def test_swapped_reference_and_target_is_flagged():
    """Guards a mislabelled Class Catalog: a target bigger than the part it sits
    on is physically backwards, and would otherwise yield a plausible ratio."""
    res = _evaluate([_inst(0, _disc(200, 200, 40)), _inst(1, _disc(200, 200, 100))])

    assert "swapped" in res.metrics["warning"]
    assert "swapped" in res.reason


def test_unknown_class_name_raises_rather_than_measuring_the_wrong_object():
    with pytest.raises(ClassNotFound, match="seal"):
        _evaluate([_inst(0, _disc(200, 200, 100))], reference_class="seal")


# ── Class-agnostic by construction ──────────────────────────────────────────


def test_same_rule_works_for_any_class_pair():
    # No code change to inspect a different product: only params differ.
    labels = {3: "can", 7: "seal"}
    ctx = DecisionContext(
        output=DecodedOutput(
            kinds=frozenset({KIND_MASKS}),
            image_hw=(CANVAS, CANVAS),
            instances=[_inst(3, _disc(200, 200, 100)), _inst(7, _disc(260, 200, 30))],
        ),
        image_rgb=np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8),
        labels=labels,
        params={**DEFAULTS, "reference_class": "can", "target_class": "seal"},
        threshold=0.5,
    )
    res = RULE.evaluate(ctx)

    assert res.verdict == "not_ok"
    assert res.score == pytest.approx(0.60, abs=0.03)


# ── Annotations ─────────────────────────────────────────────────────────────


def test_annotations_show_the_measurement_and_the_tolerance():
    res = _evaluate([_inst(0, _disc(200, 200, 100)), _inst(1, _disc(212, 200, 40))],
                    max_offset_ratio=0.10)

    circles = [a for a in res.annotations if isinstance(a, Circle)]
    lines = [a for a in res.annotations if isinstance(a, Line)]

    # The reference outline and the tolerance circle the operator configured.
    radii = sorted(c.r for c in circles)
    assert radii[0] == pytest.approx(10, abs=1.5)   # 0.10 * 100px
    assert radii[-1] == pytest.approx(100, abs=1.5)
    # The measured offset, drawn between the two centres.
    assert len(lines) == 1
    assert lines[0].x2 - lines[0].x1 == pytest.approx(12, abs=1.5)


def test_no_annotations_when_the_measurement_could_not_be_made():
    res = _evaluate([_inst(0, _disc(200, 200, 100))])
    assert res.annotations == []


# ── outer_circle_fit: the partial-ring case ─────────────────────────────────


def _ring(cx, cy, r_outer, r_inner, size=CANVAS):
    ys, xs = np.ogrid[:size, :size]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return (d2 <= r_outer**2) & (d2 >= r_inner**2)


def _wedge(cx, cy, r_outer, r_inner, a0, a1, size=CANVAS):
    """A partial ring spanning [a0, a1] radians — the shape a segmentation model
    produces when it captures only part of an arced print."""
    ys, xs = np.ogrid[:size, :size]
    ring = _ring(cx, cy, r_outer, r_inner, size)
    ang = np.arctan2(ys - cy, xs - cx)
    return ring & (ang >= a0) & (ang <= a1)


def test_outer_circle_fit_recovers_the_centre_of_a_full_ring():
    cx, cy = compute_center(_ring(200, 200, 90, 60), "outer_circle_fit")
    assert cx == pytest.approx(200, abs=2)
    assert cy == pytest.approx(200, abs=2)


def test_outer_circle_fit_recovers_the_centre_from_only_part_of_a_ring():
    """The failure this method exists for: a mask covering a third of an arced
    print. Its centroid lands out on the arc, nowhere near the ring's centre —
    which is exactly how a correctly-placed asymmetric logo reads as off-centre."""
    partial = _wedge(200, 200, 90, 60, -0.6, 1.5)

    fitted_x, fitted_y = compute_center(partial, "outer_circle_fit")
    centroid_x, centroid_y = compute_center(partial, "convex_hull_centroid")

    fit_err = np.hypot(fitted_x - 200, fitted_y - 200)
    centroid_err = np.hypot(centroid_x - 200, centroid_y - 200)

    assert fit_err < 5, f"circle fit should recover the centre, off by {fit_err:.1f}px"
    assert centroid_err > 40, "centroid should be badly biased (that is the bug)"


def test_outer_circle_fit_tracks_a_genuinely_offset_partial_ring():
    # Same partial shape, ring centre moved 30px: the fit must follow it, so the
    # method measures real displacement rather than merely ignoring shape.
    moved = _wedge(230, 200, 90, 60, -0.6, 1.5)
    cx, cy = compute_center(moved, "outer_circle_fit")
    assert cx == pytest.approx(230, abs=5)
    assert cy == pytest.approx(200, abs=5)


def test_outer_circle_fit_is_seeded_from_the_mask_not_the_reference():
    """The estimator must not consult the reference part: a centre estimator
    that peeked at the cap centre could not be trusted to measure distance
    from it. Same mask in two scenes must give the same answer."""
    mask = _wedge(230, 200, 90, 60, -0.6, 1.5)
    first = compute_center(mask, "outer_circle_fit")
    second = compute_center(mask, "outer_circle_fit")
    assert first == second
    # And it is reachable through the rule with any reference position.
    res = _evaluate(
        [_inst(0, _disc(200, 200, 150)), _inst(1, mask)],
        target_center_method="outer_circle_fit",
    )
    assert res.metrics["offset_px"] == pytest.approx(30, abs=6)


def test_outer_circle_fit_is_offered_as_a_centre_method():
    spec = next(p for p in RULE.params if p.name == "target_center_method")
    assert "outer_circle_fit" in spec.options


# ── Per-SKU nominal ─────────────────────────────────────────────────────────


def test_nominal_defaults_to_zero_so_behaviour_is_unchanged():
    plain = _evaluate([_inst(0, _disc(200, 200, 100)), _inst(1, _disc(212, 200, 40))])
    explicit = _evaluate(
        [_inst(0, _disc(200, 200, 100)), _inst(1, _disc(212, 200, 40))],
        nominal_offset=0.0,
    )
    assert plain.verdict == explicit.verdict == "not_ok"
    assert plain.score_label == "Offset Ratio"


def test_nominal_absorbs_a_per_sku_bias():
    """Artwork that sits 0.12 off-centre when correct passes once calibrated,
    while the tolerance still applies to the deviation."""
    instances = [_inst(0, _disc(200, 200, 100)), _inst(1, _disc(212, 200, 40))]

    uncalibrated = _evaluate(instances, max_offset_ratio=0.05)
    calibrated = _evaluate(instances, max_offset_ratio=0.05, nominal_offset=0.12)

    assert uncalibrated.verdict == "not_ok"
    assert calibrated.verdict == "ok"
    assert calibrated.score == pytest.approx(0.0, abs=0.02)
    assert calibrated.score_label == "Offset Deviation"


def test_nominal_still_fails_a_part_that_drifts_from_it():
    # Calibrated at 0.12; this part sits at ~0.30, deviation ~0.18 > 0.05.
    res = _evaluate(
        [_inst(0, _disc(200, 200, 100)), _inst(1, _disc(230, 200, 40))],
        max_offset_ratio=0.05,
        nominal_offset=0.12,
    )
    assert res.verdict == "not_ok"
    assert res.metrics["nominal_offset"] == 0.12
    assert res.metrics["deviation"] == pytest.approx(0.18, abs=0.02)


def test_nominal_catches_a_part_that_is_too_centred():
    """Deviation is two-sided on purpose: artwork that should sit 0.30 off and
    arrives centred is just as wrong as one that drifted outward."""
    res = _evaluate(
        [_inst(0, _disc(200, 200, 100)), _inst(1, _disc(200, 200, 40))],
        max_offset_ratio=0.05,
        nominal_offset=0.30,
    )
    assert res.verdict == "not_ok"


def test_score_is_always_the_quantity_thresholded():
    # The badge must never show a number that disagrees with the Verdict.
    res = _evaluate(
        [_inst(0, _disc(200, 200, 100)), _inst(1, _disc(212, 200, 40))],
        max_offset_ratio=0.05,
        nominal_offset=0.12,
    )
    assert (res.score <= 0.05) == (res.verdict == "ok")
