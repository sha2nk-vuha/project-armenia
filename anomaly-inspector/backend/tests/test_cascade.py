"""CascadePipeline: verdict combination, short-circuit, and per-stage results."""
import pytest

from inference import combine
from inference.cascade import CascadePipeline, CascadeStage
from inference.pipeline import InferenceResult


class _StubPipeline:
    """A sub-pipeline that returns a fixed verdict and records if it ran."""

    def __init__(self, feature, verdict, model_version="v", score=None):
        self.feature = feature
        self._verdict = verdict
        self.model_version = model_version
        self._score = score
        self.calls = 0

    def infer(self, image_bytes, threshold, rule_name=None, rule_params=None):
        self.calls += 1
        return InferenceResult(
            verdict=self._verdict,
            score=self._score,
            images={"annotated": b"img-" + self.feature.encode()},
            decision_rule=rule_name or "rule",
            score_label="S",
            reason=f"{self.feature} says {self._verdict}",
            detections=[],
        )


def _stage(feature, verdict, rule="rule", score=None):
    return CascadeStage(
        feature=feature,
        pipeline=_StubPipeline(feature, verdict, score=score),
        rule=rule,
        threshold=0.5,
        params={},
    )


def _cascade(stages, combinator="and", short_circuit=True):
    return CascadePipeline(stages, combine.get(combinator), short_circuit)


def test_and_all_pass_is_ok():
    pipeline = _cascade([_stage("a", "ok"), _stage("b", "ok")])
    assert pipeline.feature == "cascade"
    result = pipeline.infer(b"x", 0.5)
    assert result.verdict == "ok"
    assert result.score is None
    assert len(result.stages) == 2
    assert all(s.evaluated for s in result.stages)


def test_and_one_fails_is_nok_and_names_the_stage():
    stages = [_stage("anomaly", "ok"), _stage("seg", "not_ok")]
    result = _cascade(stages).infer(b"x", 0.5)
    assert result.verdict == "not_ok"
    assert result.metrics["decisive_stage"] == 1
    assert "seg" in result.reason


def test_short_circuit_skips_later_stages():
    first_nok = _stage("a", "not_ok")
    later = _stage("b", "ok")
    result = _cascade([first_nok, later], short_circuit=True).infer(b"x", 0.5)

    assert result.verdict == "not_ok"
    assert later.pipeline.calls == 0, "later stage must not run after a decisive NOK"
    assert result.stages[1].verdict == "skipped"
    assert result.stages[1].evaluated is False


def test_no_short_circuit_runs_every_stage():
    first_nok = _stage("a", "not_ok")
    later = _stage("b", "ok")
    result = _cascade([first_nok, later], short_circuit=False).infer(b"x", 0.5)

    assert result.verdict == "not_ok"
    assert later.pipeline.calls == 1, "full-evidence mode runs all stages"
    assert all(s.evaluated for s in result.stages)


def test_or_short_circuits_on_first_pass():
    first_ok = _stage("a", "ok")
    later = _stage("b", "not_ok")
    result = _cascade([first_ok, later], combinator="or").infer(b"x", 0.5)

    assert result.verdict == "ok"
    assert later.pipeline.calls == 0
    assert result.stages[1].verdict == "skipped"


def test_stage_results_carry_labelled_images_and_reason():
    result = _cascade([_stage("a", "ok"), _stage("b", "not_ok")]).infer(b"x", 0.5)
    s0 = result.stages[0]
    assert s0.feature == "a"
    # The stub emits an "annotated" image -> labelled "Detections".
    assert s0.images == [("Detections", b"img-a")]
    assert "says ok" in s0.reason


def test_metrics_capture_every_stage_for_the_report():
    result = _cascade(
        [_stage("a", "ok"), _stage("b", "not_ok"), _stage("c", "ok")]
    ).infer(b"x", 0.5)
    stages_meta = result.metrics["stages"]
    assert [m["verdict"] for m in stages_meta] == ["ok", "not_ok", "skipped"]
    assert result.metrics["combinator"] == "and"


def test_empty_cascade_is_rejected():
    with pytest.raises(ValueError, match="at least one stage"):
        _cascade([])


def test_model_version_names_the_members():
    result_pipeline = _cascade([_stage("a", "ok"), _stage("b", "ok")])
    assert "a:" in result_pipeline.model_version
    assert "b:" in result_pipeline.model_version
