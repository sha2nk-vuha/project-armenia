# Per-Feature inference pipeline abstraction

Each Feature is implemented as a self-contained inference pipeline
(`AnomalyPipeline`, `PresenceAbsencePipeline`) that owns its own preprocessing,
model-output interpretation, verdict rule, and visualization. The active session
holds one pipeline; `/api/infer` delegates to it. Adding a Feature means adding a
pipeline class.

We chose this over inline `if feature == ...` branching because the two Features
differ at *every* step — input normalization, output structure, verdict logic,
and visualization — so branching would scatter each Feature's logic across the
codebase. A pipeline seam keeps each Feature's knowledge in one module, makes each
independently testable, and localizes the blast radius of a third Feature.
