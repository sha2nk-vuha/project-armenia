# Design: Decision Rules + Segmentation Feature

Status: implemented
Settled via a `/grill-me` session (18 questions), then delivered in three
commits. Both open inputs are resolved -- see "Resolved inputs".

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

## Resolved inputs

1. **Class map — corrected from the working assumption.** Running the model over
   the sample images shows **class 0 = bottle_cap, class 1 = logo**; class 2
   never fires. The assumed `0=background, 1=bottle_cap, 2=logo` would have
   pointed the rule at class 2 for the logo, so every part would have failed with
   "logo not detected". Confirmed both numerically (class 0 is the larger filled
   blob, never touching the image border; class 1 is contained within it) and
   visually against all three samples.

2. **Mask geometry — verified, not assumed.** Each query's binarised mask bbox
   agrees with its own predicted box to within ~0.005 normalised across every
   query and image. ROI-cropped masks would instead give a mask bbox of
   ~(0,0,1,1) every time. Stride-4 full-frame confirmed; a contract test now
   asserts the 4x relationship so a model swap fails loudly.

## Outcome on the sample images

With the default 0.10 tolerance, the rule separates the samples cleanly:

| Image | Offset ratio | Verdict |
|---|---|---|
| `1a791f7d-radico_purple_anomaly_119.png` | 0.300 | NOK |
| `3a08ab93-radico_blue_image_116.png` | 0.054 | OK |
| `3adf5c74-image_095.png` | 0.027 | OK |

The one file named "anomaly" is the one that fails, and the margin between the
NOK (0.300) and the worst OK (0.054) is wide enough that the tolerance is not
balanced on a knife edge.

## Known follow-ups

- **`DATA_ROOT` points at `data/MVTecAD`,** so `data/three_cee_caps` is not
  browsable in the UI without `DATA_ROOT=<repo>/data`. That setting in turn hides
  the MVTec SKUs, because `MVTecAD` has no `test/` directory of its own. One root
  cannot serve both dataset layouts; either move the cap folder under
  `data/MVTecAD/`, or teach SKU discovery about nested collections.
- **`/model/` is gitignored,** so neither sidecar is version-controlled. A fresh
  clone gets no sidecar, and the segmentation model would then preprocess at the
  384 default instead of its actual 312 — silently wrong rather than broken.

---

# Follow-up: the multi-SKU threshold problem

## Symptom

With 3 cap designs and one tolerance, no cut existed. Passing the purple OK
needed `> 0.2412`; failing the blue NOK needed `< 0.0544`.

## Cause

Mask shape, not the metric. The purple print is a ring -- text arced around the
rim plus an emblem -- but the model segments only a crescent of it. A crescent's
centroid sits far from the ring it came from, so a correctly placed purple cap
measured 0.24 off-centre. Every centroid-style estimator inherits that bias, and
because it depends on the artwork it differs per SKU. That per-SKU offset is
precisely what a single threshold cannot absorb.

## Fix: `outer_circle_fit`

Fit a circle to the mask's outer edge instead of averaging its area. Where the
artwork sits on a circle, that circle is concentric with the design centre
whether or not the mask covers the whole ring.

Seeded from the mask's own minimum enclosing circle, never the reference part --
an estimator that consulted the cap centre could not be trusted to measure
distance from it. Self-seeding also proved far more stable: the cap-seeded
variant swung the purple OK between 0.02 and 0.21 across quantile settings,
while self-seeded stays within 0.017-0.027 over 0.35-0.75.

| method | max OK | min NOK | |
|---|---|---|---|
| min_enclosing_circle | 0.1499 | 0.0641 | overlaps |
| convex_hull_centroid | 0.2412 | 0.0544 | overlaps |
| mask_centroid | 0.2640 | 0.0541 | overlaps |
| bbox_center | 0.2240 | 0.0545 | overlaps |
| **outer_circle_fit** | **0.0531** | **0.0634** | **separates** |

Sidecar adopts it at tolerance 0.06: **6/6 correct end to end**.

It is a selectable method, not the default: it is meaningless on free-form
artwork with no circular outer edge, where the fitted circle is arbitrary.

## Fallback: per-SKU nominal

`nominal_offset` holds a SKU's offset when correctly placed; the tolerance
applies to the deviation from it. Two-sided on purpose -- artwork that should sit
off-centre and arrives centred is equally wrong. Defaults to 0, so behaviour is
unchanged until a SKU is taught.

Taught via `POST /api/calibrate` from images known to be good. The median is used
rather than the mean so one mislabelled sample cannot drag the baseline, and the
spread is recorded because a tolerance below it would be measuring sample noise.

Generic at the seam: a rule declares `Calibration(param=..., metric=...)` and the
endpoint works without knowing what the rule does. Rules with no baseline
(anomaly threshold, expected classes) declare `None` and are rejected.

## Open issue: SKU granularity

**Calibration is per-SKU, but all three cap designs currently live inside one SKU
folder (`data/three_cee_caps`).** Teaching across them pools three different
artworks into one baseline, which is meaningless -- and actively harmful:
calibrating on the pooled `ok_case` gave nominal 0.0201, which moved the
tolerance window enough to pass the blue NOK. **5/6, worse than not calibrating.**

So with the data as it stands, uncalibrated `outer_circle_fit` at 0.06 is the
correct configuration. To use calibration, each design needs its own SKU
directory (`data/<design>/test/{ok_case,nok_case}`).

## Evidence limits

One OK and one NOK per design. The separating window is 0.0531-0.0634, about
0.010 wide -- roughly 16%. That is not validated; it is the best configuration
consistent with six images. `test_sample_caps.py` asserts the window still
exists, so it fails while the margin is closing rather than after a verdict
flips. Setting a defensible tolerance needs ~20-30 known-good caps per design to
measure natural spread, and the spread measured on the three OK caps here
(0.0394) is already two-thirds of the 0.06 tolerance.
