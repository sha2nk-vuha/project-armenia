# Design: Cascade Inspection

Status: ready-for-agent
Settled via a `/grill-me` session (12 questions, all resolved).

## Problem

Combine several models' verdicts into one OK/NOK. First case: anomaly AND
segmentation. Modular: the operator picks which Features, in which order, and how
their verdicts combine.

## Model (settled)

A **Cascade** is an ordered **ensemble** of **stages**. Each stage runs its own
full pipeline (model -> decode -> Decision Rule) on the *original* image and
yields its own OK/NOK. A **Combinator** reduces the per-stage verdicts to one.

- **Not** a data pipeline: stage N does not consume stage N-1's output. The ask
  was "AND of the two individual outputs" -- independent checks, combined. (Q1)
- **Stage** = (feature, decision rule, threshold, rule params). The same model
  may appear in two stages with different rules, because residency is keyed by
  model, not feature. (Q6)
- **Combinator** is a pluggable seam mirroring Decision Rules: ship `and`
  (OK iff all OK) and `or` (OK iff any OK). Each declares a decisive verdict for
  short-circuit -- AND stops at the first NOK, OR at the first OK. (Q4)
- **Order** = execution order + short-circuit: run cheapest/most-discriminating
  first, skip the rest once the result is decided. (Q3)

## Model residency -- revises ADR 0001

ADR 0001 kept exactly one model resident to save memory. A cascade needs several
at once, so it is held resident together (Q2). Pipelines already take a
`ModelSession` in their constructor, so a cascade is just sub-pipelines built
over different sessions -- no pipeline rewrite.

- `engine` gains a session store keyed by model identity; building a session no
  longer forces it to be *the* global one.
- On each cascade run, reconcile: ensure a session exists for every model the
  spec names, evict resident sessions no active spec references. Memory stays
  bounded to the current cascade; leaving cascade mode frees the rest. (Q11)
- A new ADR amends 0001: single-resident remains the rule for single-Feature
  mode; a cascade holds its member models together, bounded by reconciliation.

## Orchestration (Q5)

`feature = "cascade"`, a Feature-like mode. A `CascadePipeline` holds the ordered
sub-pipelines and returns the standard `InferenceResult` envelope, so
`/api/infer`, the `feature` column, stats, and reports work unchanged. Selecting
Cascade in the UI shows a stage builder instead of one rule control.

The cascade has no single default model, so activating it loads nothing; models
load on the first run via reconciliation.

## Transport & persistence (Q7)

The cascade spec travels per request on `/api/infer`, exactly like `rule_params`
today; the backend stays stateless. Persisted named cascades are a clean
follow-up, not built now.

Spec shape (JSON):

    {
      "combinator": "and",
      "short_circuit": true,
      "stages": [
        {"feature": "anomaly_detection", "rule": "anomaly_threshold",
         "threshold": 0.5, "params": {}},
        {"feature": "segmentation", "rule": "concentricity",
         "threshold": 0.5, "params": {"target_center_method": "outer_circle_fit",
                                      "max_offset_ratio": 0.06}}
      ]
    }

## Record (Q8)

One inspection row: `feature='cascade'`, `score=null`,
`decision_rule=<combinator>`, `params=<the spec>`, and `metrics` carries each
stage's feature/rule/verdict/score plus which stage was decisive and which were
skipped. Reuses the columns already added; no migration. Stats scope by feature.

## Per-stage config & thresholds (Q10)

Each stage owns its threshold (an anomaly cutoff and a detection-confidence floor
are different quantities). The global threshold slider is hidden in cascade mode;
`/api/infer` still takes its `threshold` field for single-Feature mode, and the
cascade ignores it in favour of per-stage values.

Per-stage calibration reuse: a stage's rule still reads its SKU calibration
(the `nominal_offset` teach) through the same server-side layering.

## Stage failure (Q12)

- Runtime/degenerate (part not detected, unexpected model output) -> that stage
  is NOK with a reason. Fail-safe, consistent with the concentricity rule.
- Configuration error (unknown rule, class not in the model's catalog, missing
  model file) -> 4xx, so the operator fixes it rather than shipping a silent NOK.

## Result & UI (Q9)

`InferenceResult` gains an ordered `stages` list: each stage's feature, rule,
verdict, score, reason, and its own annotated/heatmap image. `/api/infer`
base64s the per-stage images. The Results panel shows one combined badge, then
each stage in order with its verdict/score/reason/image; short-circuited stages
render greyed as "not evaluated".

## Building the stage editor without loading models

Populating the builder (rules per candidate feature, class catalogs) must not
load every ONNX. A new `/api/cascade/options` returns, per cascadable feature:
its output kinds, its compatible rules (with param schemas + sidecar-merged
defaults), and its Class Catalog read from the sidecar JSON only. Rules are typed
on output kinds, which are static per feature, so this needs no ONNX load.

## Delivery

1. Combinator seam + engine multi-session + `CascadePipeline` + features wiring.
   Pure backend, unit-tested (synthetic sessions).
2. API: `/api/infer` cascade path, `/api/cascade/options`, the cascade record.
   Contract-tested end to end against the real models.
3. Frontend: stage builder, per-stage result breakdown, client. Browser-verified
   on anomaly AND segmentation over the sample caps.

## Tests

- Combinator truth tables (AND/OR) and short-circuit decisiveness.
- Residency reconcile: sessions loaded for spec models, unreferenced evicted,
  a model shared by two stages loaded once.
- CascadePipeline over synthetic sessions: AND/OR verdicts, short-circuit skips
  later stages, a degenerate stage -> NOK, a config error -> raised.
- End to end: anomaly(OK) AND segmentation(concentricity) over the sample caps
  reproduces the segmentation verdicts, and an anomaly NOK forces cascade NOK.
