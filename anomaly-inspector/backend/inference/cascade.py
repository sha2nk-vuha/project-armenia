"""Cascade Inspection: an ordered ensemble of stages combined into one Verdict.

Each stage runs its own single-Feature pipeline on the *original* image and
yields its own Verdict; a Combinator reduces those to the cascade Verdict. Stages
are independent -- no stage consumes another's output -- so a cascade is just a
list of the existing pipelines plus a reduce step. See docs/adr/0006.
"""
from dataclasses import dataclass

from inference.combine.base import Combinator
from inference.pipeline import (
    InferenceResult,
    StageResult,
    _PipelineBase,
    labelled_images,
)


@dataclass
class CascadeStage:
    """One configured stage: a built sub-pipeline plus how to run it."""

    feature: str
    pipeline: _PipelineBase
    rule: str
    threshold: float
    params: dict


class CascadePipeline:
    """Runs staged sub-pipelines and reduces their Verdicts with a Combinator.

    Short-circuit: stages run in order and, when `short_circuit` is on, the run
    stops as soon as a stage's Verdict is *decisive* for the Combinator (a NOK
    under AND, an OK under OR). Later stages are recorded as skipped rather than
    run, which is the whole point of letting the operator order them.
    """

    feature = "cascade"

    def __init__(
        self,
        stages: list[CascadeStage],
        combinator: Combinator,
        short_circuit: bool = True,
    ):
        if not stages:
            raise ValueError("A cascade needs at least one stage.")
        self.stages = stages
        self.combinator = combinator
        self.short_circuit = short_circuit

    @property
    def model_version(self) -> str:
        """A composite identifier naming the ordered member models."""
        return "cascade[" + ", ".join(
            f"{s.feature}:{s.pipeline.model_version}" for s in self.stages
        ) + "]"

    def infer(
        self,
        image_bytes: bytes,
        threshold: float,  # ignored: each stage carries its own threshold
        rule_name: str | None = None,
        rule_params: dict | None = None,
    ) -> InferenceResult:
        stage_results: list[StageResult] = []
        evaluated_verdicts: list[str] = []
        decided = False

        for stage in self.stages:
            if decided:
                # Short-circuited past: recorded, not run, and never counted as OK.
                stage_results.append(
                    StageResult(
                        feature=stage.feature,
                        decision_rule=stage.rule,
                        verdict="skipped",
                        evaluated=False,
                        reason="not evaluated (cascade already decided)",
                    )
                )
                continue

            result = stage.pipeline.infer(
                image_bytes, stage.threshold, stage.rule, stage.params
            )
            evaluated_verdicts.append(result.verdict)
            stage_results.append(
                StageResult(
                    feature=stage.feature,
                    decision_rule=result.decision_rule or stage.rule,
                    verdict=result.verdict,
                    evaluated=True,
                    score=result.score,
                    score_label=result.score_label,
                    reason=result.reason,
                    images=labelled_images(result.images),
                    detections=result.detections,
                )
            )
            if self.short_circuit and self.combinator.decisive(result.verdict):
                decided = True

        verdict = self.combinator.combine(evaluated_verdicts)
        decisive_stage = _decisive_stage(stage_results, verdict)
        reason = _cascade_reason(self.combinator, verdict, stage_results, decisive_stage)

        return InferenceResult(
            verdict=verdict,
            score=None,  # a cascade has no single scalar
            images={},
            decision_rule=self.combinator.name,
            score_label="",
            reason=reason,
            metrics={
                "combinator": self.combinator.name,
                "short_circuit": self.short_circuit,
                "decisive_stage": decisive_stage,
                "stages": [
                    {
                        "feature": s.feature,
                        "rule": s.decision_rule,
                        "verdict": s.verdict,
                        "evaluated": s.evaluated,
                        "score": s.score,
                        "reason": s.reason,
                        # Per-stage threshold + model version so a customer-facing
                        # report can explain, per stage, under what tolerance and
                        # with which model the verdict was reached.
                        "threshold": stage.threshold,
                        "model_version": stage.pipeline.model_version,
                    }
                    for s, stage in zip(stage_results, self.stages)
                ],
            },
            stages=stage_results,
        )


def _decisive_stage(stage_results: list[StageResult], verdict: str) -> int | None:
    """Index of the stage that settled the cascade, for the report.

    On a NOK it is the first failing stage; on an OK under OR it is the first
    passing stage; otherwise there is no single decisive stage.
    """
    for i, s in enumerate(stage_results):
        if s.evaluated and s.verdict == verdict:
            return i
    return None


def _cascade_reason(
    combinator: Combinator,
    verdict: str,
    stage_results: list[StageResult],
    decisive: int | None,
) -> str:
    evaluated = [s for s in stage_results if s.evaluated]
    if decisive is not None and verdict == "not_ok":
        s = stage_results[decisive]
        return f"stage {decisive + 1} ({s.feature}/{s.decision_rule}) NOK: {s.reason}"
    if verdict == "ok":
        return f"all {len(evaluated)} evaluated stage(s) passed ({combinator.name.upper()})"
    return f"no stage passed ({combinator.name.upper()})"
