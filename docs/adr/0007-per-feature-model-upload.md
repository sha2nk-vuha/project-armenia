# Per-Feature model upload; no default models

Models are never auto-loaded. On startup the app selects a default *Feature* (so
the UI has a starting selection) but loads no model; the operator uploads one
model per Feature. A single per-Feature store is the source of truth for both
single-Feature mode and Cascade mode.

## Why

A cascade runs several models, and the previous single "Model" upload gave no way
to say *which* model it was for. Bundled defaults also masked misconfiguration --
an inspection ran against whatever happened to be loaded. Requiring an explicit
per-Feature upload makes the loaded model unambiguous and visible, and lets each
cascade stage get its own model. A Feature used by two stages (two rules over one
segmentation model) shares that one uploaded model.

## Store

`features._models: {feature: (ModelSession, ModelConfig)}`. `upload_model`
replaces a Feature's entry; `current_pipeline` builds over the active Feature's
entry; `build_cascade_pipeline` reads each stage's Feature entry and raises a
configuration error (surfaced as 4xx) when a stage's Feature has no model. This
retires the separate cascade session store from ADR 0006 -- there is now one
store, keyed by Feature, and the single global `engine._current_session` is no
longer the model of record.

## Config for an uploaded model

An uploaded `.onnx` carries no sidecar, so the Class Catalog, rule defaults, and
preprocessing come from: an optional sidecar JSON uploaded alongside the model;
else the bundled sidecar as a template. Either way the preprocessing input size
is read from the uploaded model itself, so a model of a different size than the
template still preprocesses correctly. A sidecar is configuration, not model
weights, so reading the bundled one as a template does not violate "no defaults";
it only seeds class names so the builder is usable before an upload, and an
uploaded model's own sidecar overrides it.

## Consequence

Inference before an upload returns a clear "no model loaded" error rather than
running against a default. This is a deliberate behaviour change for
single-Feature mode too, chosen for one consistent rule across both modes.
