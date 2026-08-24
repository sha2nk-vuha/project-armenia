# Anomaly Inspector

The domain language for the inspection app: a tool that runs a machine-vision
model over product images and returns a pass/fail verdict, with supporting
visualisations and reports.

## Language

**Feature**:
A selectable inspection capability, backed by one model architecture and one
inference pipeline. Exactly one Feature is active at a time. Two exist: Anomaly
Detection and Presence/Absence.
_Avoid_: mode, task, model type (the Feature is the capability, not the file).

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
The pass/fail outcome of one inspection: OK or NOK. Both Features produce a
Verdict so the user sees a uniform result regardless of which is active.
_Avoid_: result, prediction, label. (Note: the current code stores the strings
`ok` / `not_ok` — "OK/NOK" is the user-facing spelling.)

**Expected Class**:
A detector class that must be present for a Presence/Absence Verdict to be OK.
Configured per SKU. Its absence yields NOK.
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
The cut-off a Feature applies to turn a model output into a Verdict. For Anomaly
Detection it bounds the anomaly score; for Presence/Absence it is the minimum
detection confidence for a class to count as present.
_Avoid_: cutoff, sensitivity.
