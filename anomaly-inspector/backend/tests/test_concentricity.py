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
