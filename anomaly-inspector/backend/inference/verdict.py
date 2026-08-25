"""One home for the Verdict vocabulary and the Cascade stage-metrics shape.

A Verdict is the pass/fail outcome of one inspection: OK or NOK (see
CONTEXT.md). The database stores these exact strings, so the constants here
are the single spelling authority -- no module should write 'ok'/'not_ok'
as a bare literal. A Cascade stage that was short-circuited past records
SKIPPED, which is deliberately *not* a Verdict: it never counts as OK and
must never be pooled into pass-rate statistics.

The stage-metrics shape written by CascadePipeline and read back by the
report layer is owned here too: `stage_metrics_row` builds one row,
`cascade_nok_reason` reads a parsed metrics dict back. Producers and
consumers depend on this module instead of on each other's private keys.
Deliberately dependency-free (stdlib only) so reports/pdf_generator can
import it without pulling numpy or cv2.

See docs/adr/0004 (records scoped by Feature) and docs/adr/0006 (Cascade).
"""
from typing import Any

OK = "ok"
NOT_OK = "not_ok"
# Recorded for a Cascade stage the Combinator short-circuited past: not run,
# so no Verdict of its own (see CascadeStage docstring in inference/cascade.py).
SKIPPED = "skipped"


def stage_metrics_row(
    feature: str,
    decision_rule: str,
    verdict: str,
    evaluated: bool,
    score: float | None,
    reason: str,
    threshold: float,
    model_version: str,
) -> dict[str, Any]:
    """Build one Cascade stage's row as stored in InferenceResult.metrics.

    Built here rather than inline in cascade.py so the report layer's keys
    can never drift from what the Cascade writes.

    Args:
        feature: The stage's Feature name.
        decision_rule: The Decision Rule the stage ran.
        verdict: The stage's Verdict string ("ok" | "not_ok" | "skipped").
        evaluated: False when the Combinator short-circuited past this stage.
        score: The rule's primary scalar, or None.
        reason: Why the stage decided as it did.
        threshold: The stage's Threshold at run time.
        model_version: Version of the model the stage ran.

    Returns:
        Dict in the exact shape persisted to InferenceResult.metrics["stages"].
    """
    return {
        "feature": feature,
        "rule": decision_rule,
        "verdict": verdict,
        "evaluated": evaluated,
        "score": score,
        "reason": reason,
        # Per-stage threshold + model version so a customer-facing report can
        # explain, per stage, under what tolerance and with which model the
        # verdict was reached.
        "threshold": threshold,
        "model_version": model_version,
    }


def cascade_nok_reason(metrics: Any) -> str:
    """The decisive stage's reason from a parsed cascade metrics dict.

    Best-effort by design: returns '' for anything malformed, missing, or
    legacy rather than raising -- a truncated historical row must not break
    customer report generation.

    Args:
        metrics: Parsed "metrics" payload from an inspection record (a dict
            with "stages" and "decisive_stage", or anything else).

    Returns:
        The decisive stage's reason string; '' when it cannot be recovered.
    """
    if not isinstance(metrics, dict):
        return ""
    stages = metrics.get("stages") or []
    decisive = metrics.get("decisive_stage")
    if isinstance(decisive, int) and 0 <= decisive < len(stages):
        return str(stages[decisive].get("reason") or "")
    return ""
