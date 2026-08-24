"""Anomaly Detection's verdict rule, as a Decision Rule.

Behaviour is identical to the pre-seam inline comparison in `AnomalyPipeline`:
OK when the image score is below the Threshold. It takes no params of its own —
the main Threshold slider *is* its tuning surface.
"""
from inference.decision.base import (
    KIND_ANOMALY_MAP,
    DecisionContext,
    DecisionResult,
)
from inference.decision.registry import register


class AnomalyThresholdRule:
    name = "anomaly_threshold"
    label = "Anomaly Score Threshold"
    consumes = frozenset({KIND_ANOMALY_MAP})
    params: list = []
    calibration = None

    def evaluate(self, ctx: DecisionContext) -> DecisionResult:
        score = ctx.output.anomaly_score
        if score is None:
            return DecisionResult(
                verdict="not_ok",
                reason="No anomaly score produced by the model.",
                score_label="Anomaly Score",
            )
        ok = score < ctx.threshold
        return DecisionResult(
            verdict="ok" if ok else "not_ok",
            reason=(
                f"anomaly score {score:.4f} "
                f"{'<' if ok else '>='} threshold {ctx.threshold:.4f}"
            ),
            score=score,
            score_label="Anomaly Score",
            metrics={"anomaly_score": round(score, 4), "threshold": ctx.threshold},
        )


register(AnomalyThresholdRule())
