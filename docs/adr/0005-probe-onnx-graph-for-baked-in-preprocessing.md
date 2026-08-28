# Probe the ONNX graph for baked-in preprocessing

Preprocessing runs only where the model does not already do it. At load time
`inference.graph_probe.probe` reads the .onnx bytes and reports whether the
graph itself performs the [0,255]→[0,1] rescale and per-channel mean/std
normalization. `AnomalyPipeline` and `PresenceAbsencePipeline` then apply only
the missing steps.

We chose this over the previous per-Feature constants — `preprocessor.preprocess`
hardcoded "Anomalib always bakes normalization in", RF-DETR's sidecar declared it
by hand — because both were assertions no one verified. A mismatch in either
direction fails silently: double-normalizing saturates every anomaly score to
1.0, under-normalizing quietly degrades detections. Neither raises, so the first
symptom is bad inspection results in the field.

The probe is deliberately asymmetric about what it will conclude. It reports "the
graph does not do this" only after reaching a weight-bearing op (Conv, Gemm,
MatMul, …), which proves the preamble is behind it; stopping anywhere else — an
unrecognized op, a forked tensor, a non-constant operand — is reported as unknown
rather than as absence. This matters concretely: Anomalib's `export_transform`
buries its `Normalize` behind Reshape/Resize/Slice shape-plumbing, and a probe
that treated "op I don't know" as "no normalization" double-normalizes exactly
the models we ship.

Unknown resolves to the pre-probe status quo, per step: apply the rescale (a
[0,255] tensor into a model trained on [0,1] is the worse failure), skip
normalization (the exports we load embed it). So an unreadable or unparseable
model behaves exactly as it did before the probe existed.

The `PresenceConfig` sidecar keys `scale` and `normalize` became tri-state.
Absent means "defer to the probe"; an explicit true/false overrides it, leaving
an escape hatch for an export the probe reads wrongly without requiring a code
change.

Probing the .onnx bytes rather than the loaded session means the answer does not
depend on which runtime (CUDA / OpenVINO / CPU) `_select_runtime` picked, and
needs no OpenVINO graph introspection. It costs one `onnx` dependency and one
protobuf parse per model load.
