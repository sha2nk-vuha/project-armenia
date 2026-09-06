"""Decision Rules: the pluggable post-processing step that produces a Verdict.

Importing this package registers the built-in rules. New rules are added by
dropping a module here and calling `register()` — no changes to pipelines, the
API, or the frontend. See docs/adr/0005.
"""
from inference.decision.base import (  # noqa: F401
    KIND_ANOMALY_MAP,
    KIND_DETECTIONS,
    KIND_MASKS,
    Annotation,
    Circle,
    ClassNotFound,
    DecisionContext,
    DecisionResult,
    DecisionRule,
    DecodedOutput,
    Instance,
    Line,
    ParamSpec,
    Point,
    Polygon,
    Text,
    class_name,
    resolve_class,
)
from inference.decision.registry import (  # noqa: F401
    all_rules,
    compatible_with,
    describe,
    get,
    register,
    resolve_params,
)

# Built-in rules — imported for their registration side effect.
from inference.decision import (  # noqa: F401,E402
    anomaly_threshold,
    concentricity,
    expected_and_forbidden,
    expected_classes,
    forbidden_classes,
)
