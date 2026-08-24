# Cascade Inspection and multi-model residency

A Cascade is an ordered ensemble of stages, each a full single-Feature pipeline
(model -> decode -> Decision Rule) run on the *original* image. A Combinator
reduces the per-stage Verdicts to one cascade Verdict.

## Why an ensemble, not a data pipeline

The requirement was "an AND of the two individual outputs" -- independent checks
(anomaly score, logo placement) on the same image, combined. No stage consumes
another's output, so a cascade is literally a list of the existing pipelines plus
a reduce step. Pipelines already take a `ModelSession` in their constructor
(from the Decision Rule work), so this needed no pipeline rewrite: a cascade is
sub-pipelines built over different sessions.

## Revising ADR 0001

ADR 0001 kept exactly one model resident to bound memory. A cascade needs its
members together, so `engine` gains a session builder that does not clobber the
global (`build_session`), and `features` holds a store keyed by model path.
Single-Feature mode is unchanged -- one resident model via the global session.
Cascade mode reconciles its store to the models the active spec names: missing
ones load, unreferenced ones evict, and a model shared by two stages loads once.
Memory is therefore bounded to the current cascade, not unbounded -- the concern
ADR 0001 raised is preserved, its literal one-resident rule is relaxed only for
the cascade.

## Combinator seam

Combinators mirror Decision Rules: a registry of small operators, shipping `and`
(OK iff all OK) and `or` (OK iff any OK). Each declares which single Verdict is
*decisive* -- one that settles the outcome alone -- so the cascade can
short-circuit: run stages in order and stop once the result is fixed (a NOK under
AND, an OK under OR). Order is thus execution order plus early exit; skipped
stages are recorded as such, never counted as OK.

## Orchestration and transport

`feature = "cascade"`, a Feature-like mode with no model of its own, so
activating it loads nothing. A `CascadePipeline` returns the standard
`InferenceResult` envelope (with an added per-stage `stages` list), so
`/api/infer`, the `feature` column, stats, and reports work unchanged. The spec
travels per request like `rule_params`, keeping the backend stateless; named
persisted cascades are a deliberate follow-up. Each stage still reads its SKU
calibration through the same server-side layering.

## Failure handling

A degenerate scene makes a stage NOK with a reason (fail-safe, as the rules
already do); a configuration error -- unknown feature/rule, a rule incompatible
with a stage's Feature, a missing model file -- surfaces as 4xx rather than a
silent NOK, so the operator fixes it.
