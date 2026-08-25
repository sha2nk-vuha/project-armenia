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
from inference.verdict import NOT_OK, OK


class AnomalyThresholdRule:
    """OK while the image's anomaly score stays below the Threshold."""

    name = "anomaly_threshold"
    label = "Anomaly Score Threshold"
    consumes = frozenset({KIND_ANOMALY_MAP})
    params: list = []
    calibration = None

    def evaluate(self, ctx: DecisionContext) -> DecisionResult:
        """Compare the decoded anomaly score against the Threshold.

        Args:
            ctx: Decision context carrying the decoded output and Threshold.

        Returns:
            OK while score < threshold; NOK with a "no score" reason when the
            decode produced no scalar.
        """
        score = ctx.output.anomaly_score
        if score is None:
            return DecisionResult(
                verdict=NOT_OK,
                reason="No anomaly score produced by the model.",
                score_label="Anomaly Score",
            )
        ok = score < ctx.threshold
        return DecisionResult(
            verdict=OK if ok else NOT_OK,
            reason=(
                f"anomaly score {score:.4f} "
                f"{'<' if ok else '>='} threshold {ctx.threshold:.4f}"
            ),
            score=score,
            score_label="Anomaly Score",
            metrics={"anomaly_score": round(score, 4), "threshold": ctx.threshold},
        )


register(AnomalyThresholdRule())
