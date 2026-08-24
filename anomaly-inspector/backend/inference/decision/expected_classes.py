"""Presence/Absence's verdict rule, as a Decision Rule.

Behaviour is identical to the pre-seam `evaluate_presence`: OK only when every
Expected Class has at least one detection at or above the confidence Threshold.

Because it consumes `detections`, it runs unchanged on any model whose decode
emits detections — including a segmentation model, which advertises both
`detections` and `masks`.
"""
from inference.decision.base import (
    KIND_DETECTIONS,
    DecisionContext,
    DecisionResult,
    ParamSpec,
    class_name,
    resolve_class,
)
from inference.decision.registry import register


class ExpectedClassesRule:
    name = "expected_classes"
    label = "Expected Classes Present"
    consumes = frozenset({KIND_DETECTIONS})
    params = [
        ParamSpec(
            name="expected_classes",
            label="Expected Classes",
            type="class_list",
            default=[],
            help="Every class listed must be detected for an OK Verdict.",
        )
    ]

    def evaluate(self, ctx: DecisionContext) -> DecisionResult:
        expected = [
            resolve_class(ref, ctx.labels)
            for ref in (ctx.params.get("expected_classes") or [])
        ]
        present = {
            d.class_id for d in ctx.output.detections if d.confidence >= ctx.threshold
        }
        missing = [c for c in expected if c not in present]
        missing_names = [class_name(c, ctx.labels) for c in missing]

        if not expected:
            reason = "No Expected Classes configured; every image passes."
        elif missing:
            reason = f"missing: {', '.join(missing_names)}"
        else:
            reason = f"all {len(expected)} expected class(es) present"

        return DecisionResult(
            verdict="not_ok" if missing else "ok",
            reason=reason,
            score=None,  # no single scalar for a presence check
            score_label="",
            metrics={
                "expected": [class_name(c, ctx.labels) for c in expected],
                "missing": missing_names,
                "detections": len(ctx.output.detections),
            },
        )


register(ExpectedClassesRule())
