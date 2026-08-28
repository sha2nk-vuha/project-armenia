# Pluggable Decision Rules

The step that turns a model's decoded output into a Verdict is a swappable
module — a **Decision Rule** — rather than logic baked into each pipeline. Every
Feature's pipeline ends by delegating to one:

```
preprocess -> model -> decode -> DecodedOutput -> DecisionRule -> Verdict
```

Rules are typed on **what the decode produced** (`DecodedOutput.kinds`), not on
the Feature name. A rule declares `consumes = {"masks"}` or `{"detections"}`;
the GUI offers only rules the active model can feed.

We chose a universal seam over a segmentation-only one because a seam that
covers a single Feature guarantees branching everywhere else: the API would
conditionally accept rule params, the UI would conditionally render the rule
selector, and reporting would conditionally scope. Retrofitting the two existing
rules cost ~40 lines — the anomaly rule is one comparison, and `evaluate_presence`
already had the right shape. `evaluate_presence` was **removed** rather than kept
alongside its replacement, so there is exactly one implementation of that verdict.

We made `kinds` a set rather than a single tag so a segmentation decode can
advertise both `masks` and `detections`. The existing Expected Classes rule then
runs on a segmentation model with no new code. This is what makes the plug-in
claim structural rather than nominal, and it is the reason Features are named for
the model family rather than for an inspection intent (see CONTEXT.md): a Feature
named "Placement Inspection" would offer rules that contradict its own name.

Rules return **declarative annotation primitives** (`Circle`, `Line`, `Point`,
`Polygon`, `Text`), not rendered image bytes. `visualizer.render_annotations`
owns drawing and encoding. Rules therefore stay pure functions over numbers:
their tests assert geometry rather than diffing JPEGs, drawing style stays
consistent across rules, and the same annotation list can later be shipped to the
frontend as a zoomable SVG overlay instead of being burned into an image. The
cost is a small primitive vocabulary to extend when a rule needs something exotic.

Rule selection and params travel **per request**, exactly as `threshold` already
does; the model sidecar supplies the defaults the GUI seeds from. This keeps the
backend stateless — no new table, no session state, no migration for tuning
values. Params are layered lowest-first (rule defaults < sidecar < request) in a
single pass; resolving each layer independently would re-seed unspecified params
from their defaults and silently discard the layer beneath.

Class references in params are **names, not ids** (`"bottle_cap"`, not `1`),
resolved against the Class Catalog at inference. A retrain that reorders classes
keeps working, and a rename fails loudly instead of measuring the wrong object —
a wrong class id inverts a Verdict with no visible symptom.

Records store `decision_rule`, `params`, and `metrics`, and `anomaly_score` was
renamed to `score`. This app emits customer-facing PDF reports, so a record has
to state which rule ran, under what tolerances, and what it measured. A column
named `anomaly_score` holding a logo-offset ratio would be a lie in a repo that
maintains a deliberate ubiquitous-language doc. Stats and reports scope by
(Feature, Decision Rule): ADR 0004 established that pooling incomparable OKs
misleads, and two rules on the *same* Feature are exactly that case.
