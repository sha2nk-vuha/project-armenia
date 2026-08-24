# Anomaly Inspector

The domain language for the inspection app: a tool that runs a machine-vision
model over product images and returns a pass/fail verdict, with supporting
visualisations and reports.

## Language

**Feature**:
A selectable model family and its decode — what the loaded model can *see*.
Exactly one Feature is active at a time. Anomaly Detection, Presence/Absence,
and Segmentation.
_Avoid_: mode, task, model type (the Feature is the capability, not the file).
(Note: the Feature no longer carries the pass/fail *policy* — that moved to the
Decision Rule, see docs/adr/0005. "Presence/Absence" is therefore a legacy name
describing a rule rather than a model family; it is kept because it is a stored
value in the `feature` column, and renaming it would migrate data for no
operator-visible gain.)

**Decision Rule**:
The pluggable post-processing step that turns a Feature's decoded output into a
Verdict, selectable from the GUI. Each rule declares which output kinds it
consumes (`anomaly_map`, `detections`, `masks`), so only rules the active model
can feed are offered. Adding an inspection policy means adding a rule, not
touching a pipeline.
_Avoid_: post-processor (acceptable synonym in code comments), verdict logic,
check.

**Anomaly Detection**:
The Feature that scores how unusual an image is (Dinomaly model). Produces a
pixel-level anomaly map and a scalar score; the Verdict is the score compared to
a Threshold.
_Avoid_: defect detection, Dinomaly (that's the model, not the Feature).

**Presence/Absence**:
The Feature that checks whether an Expected Class is present in the image
(RF-DETR detector). Produces object detections; the Verdict is OK when every
Expected Class is present, NOK when any is missing.
_Avoid_: object detection (that's the technique), presence check.

**Verdict**:
The pass/fail outcome of one inspection: OK or NOK. Every Feature/Decision Rule
pairing produces a Verdict so the user sees a uniform result regardless of what
is active. An OK from one rule is not comparable to an OK from another, so
statistics and reports are scoped by (Feature, Decision Rule).
_Avoid_: result, prediction, label. (Note: the current code stores the strings
`ok` / `not_ok` — "OK/NOK" is the user-facing spelling.)

**Expected Class**:
A detector class that must be present for the Expected Classes Decision Rule to
return OK. Its absence yields NOK. It is a *rule parameter*, referenced by class
name so a retrain that reorders classes cannot silently invert the check.
_Avoid_: target, required object.

**Class Catalog**:
The index-to-name map of the classes a detector model can emit, shipped as a
sidecar beside the model. Feeds the per-SKU editor where Expected Classes are
chosen.
_Avoid_: label map (acceptable synonym), classes list.

**SKU**:
A product/category under inspection. Existing term: dataset subdirectories
containing a `test/` folder. For Presence/Absence, a SKU also carries its set of
Expected Classes.
_Avoid_: product, part, category.

**Threshold**:
The single model-level cut-off from the main slider. For Anomaly Detection it
bounds the anomaly score; for detector-backed Features it is the minimum
detection confidence. Tolerances belonging to a specific policy (an offset
limit, a size band) are **Decision Rule params**, not the Threshold.
_Avoid_: cutoff, sensitivity.
