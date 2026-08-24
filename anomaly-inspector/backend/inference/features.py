"""Active-Feature registry: which inspection Feature is live, and its pipeline.

A thin coordinator over `engine` (which owns the single loaded model session).
It records the active Feature, loads the RF-DETR sidecar when Presence/Absence
is active, and builds the matching pipeline on demand. `/api/infer` asks for
`current_pipeline()` and stays agnostic to which Feature is running.

Exactly one Feature is active at a time; switching reloads that Feature's
default model (see docs/adr/0001).
"""
import logging
from pathlib import Path

from config import (
    ANOMALY_FEATURE as ANOMALY,
    CASCADE_FEATURE,
    FEATURES,
    PRESENCE_FEATURE,
    SEGMENTATION_FEATURE,
)
from inference import combine, decision, engine
from inference.engine import ModelSession
from inference.cascade import CascadePipeline, CascadeStage
from inference.pipeline import (
    AnomalyPipeline,
    PresenceAbsencePipeline,
    SegmentationPipeline,
)
from inference.model_config import ModelConfig, load_config, sidecar_path

logger = logging.getLogger(__name__)

# Features whose model carries a sidecar (Class Catalog, preprocessing, rule
# param defaults). Anomaly Detection needs none.
_SIDECAR_FEATURES = (PRESENCE_FEATURE, SEGMENTATION_FEATURE)

_active_feature: str | None = None
_model_config: ModelConfig | None = None


def get_active_feature() -> str | None:
    return _active_feature


def get_model_config() -> ModelConfig | None:
    return _model_config


def activate(feature: str) -> ModelSession:
    """Make `feature` active by loading its default model (swaps the session).

    For sidecar-backed Features, also loads the model's sidecar so the Class
    Catalog, preprocessing, and rule param defaults are ready. Raises ValueError for an unknown
    Feature and FileNotFoundError when the default model file is missing.
    """
    global _active_feature, _model_config
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature: {feature!r}")

    if feature == CASCADE_FEATURE:
        # A cascade has no single default model; members load per request.
        _active_feature = feature
        _model_config = None
        return engine.get_session()

    spec = FEATURES[feature]
    model_path = Path(spec["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Default model for {feature!r} not found at {model_path}."
        )

    # Load the sidecar first so a malformed one fails before we swap the session
    # or flip active state (no half-applied activation).
    model_config = (
        load_config(sidecar_path(str(model_path)))
        if feature in _SIDECAR_FEATURES
        else None
    )
    engine.load_model_from_path(str(model_path), spec["model_version"])
    _active_feature = feature
    _model_config = model_config
    logger.info("Feature activated: %s (model=%s)", feature, model_path.name)
    return engine.get_session()


def register_upload(feature: str) -> None:
    """Record that the currently-loaded (uploaded) session belongs to `feature`.

    Used by /api/load-model after `engine` has loaded the uploaded bytes. Unlike
    `activate`, it does not reload a default model. An uploaded model carries no
    sidecar, so the existing Presence config (if any) is kept, else defaults.
    """
    global _active_feature, _model_config
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature: {feature!r}")
    _active_feature = feature
    if feature in _SIDECAR_FEATURES and _model_config is None:
        _model_config = ModelConfig()


def _build_pipeline_for(feature: str, session: ModelSession, config: ModelConfig):
    """Construct the right single-Feature pipeline over a session + config."""
    if feature == PRESENCE_FEATURE:
        return PresenceAbsencePipeline(session, config, config.expected_classes)
    if feature == SEGMENTATION_FEATURE:
        return SegmentationPipeline(session, config)
    return AnomalyPipeline(session)


def current_pipeline():
    """Build the pipeline for the active Feature over the loaded session.

    Returns None when no Feature is active or no model is loaded. The Cascade
    Feature is not built here: it has no single model and is assembled per
    request from its spec by `build_cascade_pipeline`.
    """
    sess = engine.get_session()
    if sess is None or _active_feature is None or _active_feature == CASCADE_FEATURE:
        return None
    return _build_pipeline_for(_active_feature, sess, _model_config or ModelConfig())


# ── Cascade: multi-model residency + assembly ───────────────────────────────
# ADR 0001 keeps one model resident for single-Feature mode; a Cascade needs its
# members together. They live in a store keyed by model path, reconciled to the
# active spec so memory stays bounded to the current cascade (see docs/adr/0006).
_cascade_sessions: dict[str, ModelSession] = {}
_cascade_configs: dict[str, ModelConfig] = {}


def _model_path_for(feature: str) -> str:
    spec = FEATURES[feature]
    path = Path(spec["model_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Model for stage feature {feature!r} not found at {path}."
        )
    return str(path)


def _ensure_cascade_member(feature: str) -> tuple[ModelSession, ModelConfig]:
    """Return a resident (session, config) for a stage's model, loading if new."""
    path = _model_path_for(feature)
    if path not in _cascade_sessions:
        version = FEATURES[feature]["model_version"]
        _cascade_sessions[path] = engine.build_session_from_path(path, version)
        _cascade_configs[path] = (
            load_config(sidecar_path(path))
            if feature in _SIDECAR_FEATURES
            else ModelConfig()
        )
        logger.info("Cascade member loaded: %s (%s)", feature, path)
    return _cascade_sessions[path], _cascade_configs[path]


def _retain_cascade_members(paths: set[str]) -> None:
    """Evict resident member sessions no active spec references."""
    for path in list(_cascade_sessions):
        if path not in paths:
            _cascade_sessions.pop(path, None)
            _cascade_configs.pop(path, None)
            logger.info("Cascade member evicted: %s", path)


def build_cascade_pipeline(spec: dict) -> CascadePipeline:
    """Assemble a CascadePipeline from a per-request spec.

    Reconciles residency to the models the spec names. Raises ValueError for a
    malformed spec, an unknown feature/rule, or a rule incompatible with a
    stage's Feature; FileNotFoundError when a stage's model file is missing.
    These are configuration errors the API surfaces rather than silent NOKs.
    """
    combinator = combine.get(spec.get("combinator", "and"))
    short_circuit = bool(spec.get("short_circuit", True))
    raw_stages = spec.get("stages") or []
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("A cascade needs a non-empty 'stages' list.")

    referenced: set[str] = set()
    stages: list[CascadeStage] = []
    for i, raw in enumerate(raw_stages):
        feature = raw.get("feature")
        if feature not in FEATURES or feature == CASCADE_FEATURE:
            raise ValueError(f"Stage {i + 1}: invalid feature {feature!r}.")
        session, config = _ensure_cascade_member(feature)
        referenced.add(_model_path_for(feature))
        pipeline = _build_pipeline_for(feature, session, config)

        rule_name = raw.get("rule") or pipeline.default_rule
        # Fail fast on an incompatible rule rather than at run time.
        pipeline._resolve_rule(rule_name)
        # Sidecar defaults under the stage's explicit params.
        params = decision.resolve_params(
            decision.get(rule_name),
            pipeline.default_rule_params(rule_name),
            raw.get("params") or {},
        )
        stages.append(
            CascadeStage(
                feature=feature,
                pipeline=pipeline,
                rule=rule_name,
                threshold=float(raw.get("threshold", 0.5)),
                params=params,
            )
        )

    _retain_cascade_members(referenced)
    return CascadePipeline(stages, combinator, short_circuit)


def cascade_options() -> dict:
    """Everything the stage builder needs, without loading any ONNX model.

    Rules are typed on output kinds, which are static per Feature, and Class
    Catalogs come from the sidecar JSON -- so the builder is populated without
    paying a model load for every candidate stage.
    """
    features_out = []
    for name in FEATURES:
        if name == CASCADE_FEATURE:
            continue
        kinds = _FEATURE_KINDS.get(name, frozenset())
        cfg = (
            load_config(sidecar_path(str(FEATURES[name]["model_path"])))
            if name in _SIDECAR_FEATURES
            else ModelConfig()
        )
        rules = []
        for rule in decision.compatible_with(kinds):
            described = decision.describe(rule)
            described["defaults"] = decision.resolve_params(
                rule, _sidecar_rule_defaults(name, rule.name, cfg)
            )
            rules.append(described)
        features_out.append(
            {
                "name": name,
                "label": FEATURES[name]["label"],
                "threshold_label": FEATURES[name]["threshold_label"],
                "labels": {str(k): v for k, v in cfg.labels.items()},
                "default_rule": cfg.default_rule,
                "rules": rules,
            }
        )
    return {
        "features": features_out,
        "combinators": [combine.describe(c) for c in combine.all_combinators()],
    }


def _sidecar_rule_defaults(feature: str, rule_name: str, cfg: ModelConfig) -> dict:
    """The sidecar's default params for a rule on a feature (empty if none)."""
    if feature == PRESENCE_FEATURE and rule_name == "expected_classes":
        return {"expected_classes": cfg.expected_classes}
    return cfg.rule_params.get(rule_name, {})


# Output kinds each Feature's decode advertises -- static, so the builder needs
# no model load to know which rules a Feature can run.
_FEATURE_KINDS = {
    ANOMALY: frozenset({decision.KIND_ANOMALY_MAP}),
    PRESENCE_FEATURE: frozenset({decision.KIND_DETECTIONS}),
    SEGMENTATION_FEATURE: frozenset({decision.KIND_DETECTIONS, decision.KIND_MASKS}),
}


def current_decision_rules() -> list[dict]:
    """UI metadata for the Decision Rules the active pipeline can run.

    Filtered by the output kinds the active Feature's decode advertises, so the
    GUI never offers a rule that cannot consume this model's output. Empty when
    no Feature is active.
    """
    pipeline = current_pipeline()
    if pipeline is None:
        return []
    rules = []
    for rule in pipeline.compatible_rules():
        described = decision.describe(rule)
        # The *effective* defaults, i.e. the rule's schema defaults with this
        # model's sidecar layered on. The GUI seeds from these and echoes them
        # back on every request, so seeding from the bare schema instead would
        # silently override whatever the sidecar configured.
        described["defaults"] = decision.resolve_params(
            rule, pipeline.default_rule_params(rule.name)
        )
        rules.append(described)
    return rules


def reset() -> None:
    """Clear active-Feature state (test hook; does not unload the engine session)."""
    _cascade_sessions.clear()
    _cascade_configs.clear()
    global _active_feature, _model_config
    _active_feature = None
    _model_config = None
