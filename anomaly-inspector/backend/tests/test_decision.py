"""The Decision Rule seam: registry, compatibility, param layering, rendering."""
import numpy as np
import pytest

from inference import decision
from inference.decision import (
    KIND_ANOMALY_MAP,
    KIND_DETECTIONS,
    KIND_MASKS,
    Circle,
    ClassNotFound,
    DecisionContext,
    DecodedOutput,
    ParamSpec,
)
from inference.rfdetr import Detection


def _ctx(output, params=None, labels=None, threshold=0.5):
    return DecisionContext(
        output=output,
        image_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        labels=labels or {},
        params=params or {},
        threshold=threshold,
    )


# ── Registry and compatibility ──────────────────────────────────────────────


def test_builtin_rules_are_registered():
    names = {r.name for r in decision.all_rules()}
    assert {"anomaly_threshold", "expected_classes"} <= names


def test_unknown_rule_raises_with_known_names_listed():
    with pytest.raises(ValueError, match="expected_classes"):
        decision.get("no_such_rule")


def test_compatibility_filters_by_output_kind():
    anomaly_rules = {r.name for r in decision.compatible_with(frozenset({KIND_ANOMALY_MAP}))}
    detector_rules = {r.name for r in decision.compatible_with(frozenset({KIND_DETECTIONS}))}

    assert anomaly_rules == {"anomaly_threshold"}
    assert "expected_classes" in detector_rules
    assert "anomaly_threshold" not in detector_rules


def test_segmentation_kinds_inherit_detection_rules():
    """The payoff of typing on output kind: a segmentation model advertises both
    kinds, so a detections-only rule runs on it with no new code."""
    seg_kinds = frozenset({KIND_DETECTIONS, KIND_MASKS})
    names = {r.name for r in decision.compatible_with(seg_kinds)}
    assert "expected_classes" in names


def test_describe_exposes_param_schema_for_the_gui():
    described = decision.describe(decision.get("expected_classes"))
    assert described["name"] == "expected_classes"
    assert described["consumes"] == ["detections"]
    assert described["params"][0]["type"] == "class_list"


# ── Param layering ──────────────────────────────────────────────────────────


class _FakeRule:
    name = "_fake"
    label = "Fake"
    consumes = frozenset({KIND_DETECTIONS})
    params = [
        ParamSpec("a", "A", "number", default=1),
        ParamSpec("b", "B", "number", default=2),
    ]

    def evaluate(self, ctx):  # pragma: no cover - not exercised
        raise NotImplementedError


def test_param_layers_apply_lowest_first_without_reseeding():
    # Regression: resolving each layer separately re-seeded unspecified params
    # from their defaults and silently discarded the layer beneath.
    resolved = decision.resolve_params(_FakeRule(), {"a": 10, "b": 20}, {"b": 30})
    assert resolved == {"a": 10, "b": 30}


def test_param_layers_ignore_unknown_keys():
    # A stale param from a previously-selected rule must not leak into this one.
    resolved = decision.resolve_params(_FakeRule(), {"a": 10, "leftover": 99})
    assert resolved == {"a": 10, "b": 2}


def test_none_layers_are_skipped():
    assert decision.resolve_params(_FakeRule(), None, None) == {"a": 1, "b": 2}


# ── Class reference resolution ──────────────────────────────────────────────


def test_resolve_class_by_name_and_id():
    labels = {0: "background", 1: "bottle_cap", 2: "logo"}
    assert decision.resolve_class("logo", labels) == 2
    assert decision.resolve_class(1, labels) == 1
    assert decision.resolve_class("1", labels) == 1


def test_resolve_class_unknown_name_lists_the_catalog():
    with pytest.raises(ClassNotFound, match="bottle_cap"):
        decision.resolve_class("bottle_cap", {0: "gasket"})


# ── Built-in rule behaviour ─────────────────────────────────────────────────


def test_anomaly_rule_ok_below_threshold_and_reports_score():
    out = DecodedOutput(
        kinds=frozenset({KIND_ANOMALY_MAP}), image_hw=(10, 10), anomaly_score=0.2
    )
    res = decision.get("anomaly_threshold").evaluate(_ctx(out, threshold=0.5))

    assert res.verdict == "ok"
    assert res.score == pytest.approx(0.2)
    assert res.score_label == "Anomaly Score"
    assert "0.2000" in res.reason


def test_anomaly_rule_nok_at_or_above_threshold():
    out = DecodedOutput(
        kinds=frozenset({KIND_ANOMALY_MAP}), image_hw=(10, 10), anomaly_score=0.5
    )
    res = decision.get("anomaly_threshold").evaluate(_ctx(out, threshold=0.5))
    assert res.verdict == "not_ok"


def test_expected_classes_rule_names_what_is_missing():
    out = DecodedOutput(
        kinds=frozenset({KIND_DETECTIONS}),
        image_hw=(10, 10),
        detections=[Detection(1, 0.9, (0, 0, 5, 5))],
    )
    res = decision.get("expected_classes").evaluate(
        _ctx(out, params={"expected_classes": ["cap", "logo"]},
             labels={1: "cap", 2: "logo"})
    )

    assert res.verdict == "not_ok"
    assert res.metrics["missing"] == ["logo"]
    assert "logo" in res.reason


def test_expected_classes_rule_with_no_policy_passes_everything():
    out = DecodedOutput(kinds=frozenset({KIND_DETECTIONS}), image_hw=(10, 10))
    res = decision.get("expected_classes").evaluate(_ctx(out, params={"expected_classes": []}))
    assert res.verdict == "ok"


# ── Annotation rendering ────────────────────────────────────────────────────


def test_render_annotations_draws_and_leaves_original_untouched():
    from inference.visualizer import render_annotations

    base = np.zeros((40, 40, 3), dtype=np.uint8)
    out = render_annotations(base, [Circle(20, 20, 10, (255, 0, 0))])

    assert out.shape == base.shape
    assert (base == 0).all(), "the caller's image must not be mutated"
    # Rules pick colours by meaning; the renderer owns the RGB/BGR conversion.
    assert out[:, :, 0].max() == 255 and out[:, :, 1].max() == 0


def test_render_annotations_is_a_noop_without_annotations():
    from inference.visualizer import render_annotations

    base = np.zeros((8, 8, 3), dtype=np.uint8)
    assert render_annotations(base, []) is base
