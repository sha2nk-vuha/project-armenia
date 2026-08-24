# Design: Decision Rules + Segmentation Feature

Status: ready-for-agent
Settled via a `/grill-me` session. 18 questions, all resolved. Two inputs still
needed from the user (see "Open inputs").

## Problem

Add a third inspection Feature backed by an RF-DETR **segmentation** model, whose
OK/NOK verdict comes from a **post-processing step** that is plug-and-play: the
model's output feeds the module, the module's result feeds the verdict. The
first module answers "is the logo concentric with the bottle cap?". Other
use-cases must be able to supply other modules, selectable from the GUI.

## Core decision: a universal Decision Rule seam

Every pipeline becomes:

```
preprocess -> model -> decode -> DecodedOutput -> DecisionRule -> verdict
```

The existing verdict logic in `AnomalyPipeline` (score vs threshold) and
`evaluate_presence` are **retrofitted as built-in Decision Rules**, so there is
no special-casing anywhere for "the Feature that has post-processing".

Rules are typed on **what the decode produced**, not on the Feature name. A
segmentation model advertises `kinds = {"detections", "masks"}`, so a
detections-only rule (e.g. Expected Classes Present) is automatically available
on it with no new code. This is what makes the plug-and-play claim real.

## Contracts

```python
@dataclass
class Instance:                      # one segmented object
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]     # xyxy, original-image px
    mask: np.ndarray                            # bool [H, W], original-image res

@dataclass
class DecodedOutput:
    kinds: frozenset[str]            # {"anomaly_map"} | {"detections"} | {"detections","masks"}
    image_hw: tuple[int, int]
    detections: list[Detection] | None = None
    instances: list[Instance] | None = None
    anomaly_map: np.ndarray | None = None

@dataclass
class DecisionResult:
    verdict: str                     # "ok" | "not_ok"
    score: float | None              # primary scalar -> VerdictBadge + DB `score`
    score_label: str                 # e.g. "Offset Ratio"
    metrics: dict[str, float | str]  # named measurements shown in Results
    reason: str                      # "logo offset 0.21 > 0.10 tolerance"
    annotations: list[Annotation]    # declarative primitives, rendered by visualizer

class DecisionRule(Protocol):
    name: str                        # "concentricity"
    label: str                       # "Logo Concentricity"
    consumes: frozenset[str]         # {"masks"} -- GUI filters on this
    params: list[ParamSpec]          # drives generic GUI controls
    def evaluate(self, ctx: DecisionContext) -> DecisionResult: ...
```

`DecisionContext` carries the `DecodedOutput`, the original RGB pixels (unused by
geometry rules, needed by future colour/OCR rules), the resolved Class Catalog,
and the resolved params.

**Annotations are declarative primitives** (`Circle`, `Line`, `Point`, `Polygon`,
`Text`), not finished JPEGs. `visualizer.py` owns rendering and encoding. Rules
stay pure functions over numbers, so tests assert `Circle(cx=51.2, ...)` rather
than diffing image bytes, drawing style stays consistent across modules, and the
same list can later ship to the frontend as a zoomable SVG overlay.

## The Concentricity rule

Class-agnostic by construction -- `reference_class` and `target_class` are
params, so the identical module answers "is the seal centred on the can" with no
new code.

| Param | Type | Default | Notes |
|---|---|---|---|
| `reference_class` | class name | `bottle_cap` | resolved against Class Catalog |
| `target_class` | class name | `logo` | |
| `max_offset_ratio` | number 0-1 | 0.10 | **the NOK threshold** |
| `reference_center_method` | enum | `min_enclosing_circle` | |
| `target_center_method` | enum | `convex_hull_centroid` | |

Centre methods: `min_enclosing_circle`, `convex_hull_centroid`, `mask_centroid`,
`bbox_center`.

**Units: normalised ratio**, `offset / reference_radius`. Raw pixels break the
moment camera distance, zoom, or resolution changes -- a 12px tolerance tuned on
a 1024px image is twice as strict on a 2048px one. The ratio is dimensionless
and drops straight into the existing 0-1 `ThresholdControl`. Raw pixel offset is
still reported in `metrics` for operator intuition.

**"Aligned" means centred.** Rotation is out of scope: a bottle cap is
rotationally symmetric, so there is no cap feature to measure the logo's angle
against -- it could only be judged versus an operator-set reference angle, which
is a separate measurement with its own tolerance. Revisit only if real NOK
images demand it.

**Degenerate scenes fail safe.** Missing reference or target -> NOK with an
explicit `reason`. Multiple instances -> highest-confidence per class, noted in
`metrics`. No third verdict state: the DB, stats, and PDF report all stay binary.

## Segmentation Feature

Verified against `model/three_cee_caps_rfdetr-seg-nano_v0.0.1.onnx`:

| Tensor | Shape | Reading |
|---|---|---|
| `input` | `[1,3,312,312]` | **not 384** -- differs from rfdetr-nano |
| `dets` | `[1,100,4]` | 100 queries, cxcywh normalised |
| `labels` | `[1,100,3]` | 3 classes |
| `masks` | `[1,100,78,78]` | per-query logits, stride-4 full-frame (312/4 = 78) |

Two consequences:

1. **The existing decode would silently mis-parse this.**
   `_split_logits_and_boxes` picks boxes by `last dim == 4`, then takes the
   *first other array* as logits -- with three outputs that grabs `masks`
   instead of `labels`. A segmentation-specific decode is mandatory.
2. **78 = 312/4 implies full-frame masks at quarter resolution**, not
   ROI-cropped mask heads. Decode is `sigmoid -> threshold -> resize to original
   HxW`, no box-relative paste. This is inference from arithmetic and **must be
   verified against a real image** before geometry is built on it.

**Mask binarisation cutoff** (default 0.5) is a **sidecar constant, not a GUI
control**. It is a property of the export's calibration, not a process
tolerance; on the operator panel it would silently shift every centre
measurement and invalidate every tuned offset threshold.

The existing `threshold` slider keeps meaning **detection confidence**, same as
Presence/Absence. The offset tolerance is a rule param, not a second meaning for
the same field.

## Sidecar: PresenceConfig -> ModelConfig

`expected_classes` is a **rule param**, not model config -- it belongs to the
Expected Classes Present rule. Sidecars supply *defaults* the GUI seeds from.

```json
{ "labels": {"0": "bottle_cap", "1": "logo", "2": "background"},
  "input_size": [312, 312],
  "mask_threshold": 0.5,
  "default_rule": "concentricity",
  "rule_params": { "concentricity": { "reference_class": "bottle_cap",
                                      "target_class": "logo",
                                      "max_offset_ratio": 0.10 } } }
```

Backward compatible: a top-level `expected_classes` in an old sidecar is read
and mapped into the presence rule's params.

**Classes are referenced by name in config, resolved to ids at runtime.** A
retrain that reorders classes keeps working; a rename fails loudly with
"class bottle_cap not in catalog" instead of silently measuring the wrong object.

Default rule resolution: sidecar `default_rule` -> Feature's declared default ->
first compatible rule.

## Params transport

The GUI sends `decision_rule` + `rule_params` JSON on each `/api/infer`, exactly
as `threshold` is sent today. Backend stays stateless -- no new table, no
migration, no session state. Sidecars supply the initial values the GUI seeds
from. Per-SKU persistence stays a separate, later concern.

## Persistence and reporting

`db.py` already has an idempotent additive migration path (`_add_missing_columns`).

- Rename `anomaly_score` -> `score` (generic primary scalar) via
  `ALTER TABLE RENAME COLUMN`, guarded so it runs once.
- Add nullable `decision_rule` (String), `params` (JSON text), `metrics` (JSON text).

The audit trail matters because this app emits **customer-facing PDF reports**: a
record must say which rule ran, with what tolerances, and what it measured.

**Stats and reports scope by (Feature, Decision Rule).** ADR 0004 established
that pooling an anomaly OK with a presence OK yields a misleading pass rate.
Under this seam, "logo is concentric" and "logo is present" are two rules on the
*same* Feature producing incomparable OKs -- scoping by `feature` alone
reintroduces exactly the bug ADR 0004 exists to prevent.

## Domain language

Choosing the universal seam moves the *capability* into the rule, so CONTEXT.md's
"Feature = a selectable inspection capability" no longer holds. Resolution:

- **Feature** is redefined as *model family + decode* (what the model can see).
- **Decision Rule** is added as the term for the capability layer (what turns
  decoded output into a Verdict).
- New Feature is user-facing **"Segmentation"**.
- The two existing Feature names are kept as-is (no DB value migration); the
  residual inconsistency -- Presence/Absence is really a rule name -- is
  documented as accepted debt in the ADR.

This trade is deliberate: Features named by *intent* would make the rule dropdown
offer rules that contradict the Feature's own name, which forfeits the
kinds-based compatibility that makes new modules plug in without touching the
Feature layer.

## GUI

Rule dropdown sits directly under Feature in the Setup panel. Params render
beneath it **generically from the ParamSpec schema** -- number -> slider,
enum -> select, class -> select populated from the Class Catalog. No per-module
frontend code, ever. `VerdictBadge` already accepts a `scoreLabel` prop; App.tsx
stops hardcoding "Anomaly Score" and uses the rule's `score_label`.

## Delivery: three commits

1. Decision Rule seam + retrofit anomaly/presence + DB migration. Behaviour
   identical, proven by existing tests.
2. Segmentation Feature + decode, verified against the real ONNX and sample
   images.
3. Concentricity rule + schema-driven GUI + report scoping.

## Tests

- **Synthetic masks, analytic answers**: a circle offset by exactly 12px inside a
  100px-radius cap -> ratio 0.12, asserted across all four centre methods. No
  model needed.
- **Parity**: retrofitted anomaly/presence verdicts unchanged.
- **Real-model contract**: assert the segmentation ONNX's I/O signature so a
  model swap fails loudly rather than silently mis-decoding.

## Open inputs

1. **Class map for the 3 classes.** `labels` has width 3; the two named classes
   are `bottle_cap` and `logo`. The detection sidecar's third slot was
   `background`, so the same is likely -- but guessing class ids is precisely how
   a NOK rule silently inverts. Needed before the sidecar is written.
2. **Sample cap images** under `data/three_cee_caps/test/`. Any directory
   containing a `test/` subfolder becomes a browsable SKU automatically. Needed
   to verify the stride-4 full-frame mask assumption and to confirm the class map
   empirically.

Neither blocks commit 1.
