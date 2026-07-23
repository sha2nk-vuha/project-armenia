# Explicit Feature type over ONNX output auto-detection

The inference pipeline is selected from the user's explicit Feature choice, not
inferred from the model's ONNX output signature. The Feature type is an input to
model loading and activation.

The existing anomaly path auto-classifies outputs (`_classify_outputs`) by name
and shape, and that heuristic is already fragile across Anomalib export versions.
Adding a second, architecturally different model would make silent misdetection
more likely and harder to debug. An explicit Feature type is self-documenting,
keeps each pipeline isolated, and gives the user an unambiguous mental model. The
within-anomaly output classification stays; it is not extended to discriminate
between Features.
