"""Active-Feature registry and the per-Feature model store.

Models are never auto-loaded: the operator uploads one model per Feature, and it
is held here as a live session keyed by Feature (see docs/adr/0007). Both
single-Feature mode and Cascade mode read from this one store, so a model
uploaded for Segmentation serves a single-Feature Segmentation inspection and a
Segmentation stage inside a cascade alike. A Feature used by two cascade stages
(e.g. two rules over one segmentation model) shares that one model.

`/api/infer` asks for `current_pipeline()` (single) or `build_cascade_pipeline`
(cascade) and stays agnostic to which Feature is running.
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
from inference.cascade import CascadePipeline, CascadeStage
from inference.engine import ModelSession
from inference.model_config import (
    ModelConfig,
    config_from_dict,
    load_config,
    sidecar_path,
)
from inference.pipeline import (
    AnomalyPipeline,
    PresenceAbsencePipeline,
    SegmentationPipeline,
)

logger = logging.getLogger(__name__)

# Features whose model carries a sidecar (Class Catalog, preprocessing, rule
# param defaults). Anomaly Detection needs none.
_SIDECAR_FEATURES = (PRESENCE_FEATURE, SEGMENTATION_FEATURE)

# The single source of truth: one uploaded (session, config) per Feature. No
# default models are ever placed here.
_models: dict[str, tuple[ModelSession, ModelConfig]] = {}
_active_feature: str | None = None


def get_active_feature() -> str | None:
    return _active_feature


def get_model_config() -> ModelConfig | None:
    """Config of the active single-Feature model, or None if none is loaded."""
    if _active_feature and _active_feature != CASCADE_FEATURE:
        entry = _models.get(_active_feature)
        if entry:
            return entry[1]
    return None


def activate(feature: str) -> None:
    """Make `feature` active. Does not load a model -- the operator uploads one.

    Raises ValueError for an unknown Feature. The Cascade Feature is valid; its
    stage models load per request from the same store.
    """
    global _active_feature
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature: {feature!r}")
    _active_feature = feature


def _bundled_template(feature: str) -> ModelConfig:
    """The bundled sidecar as a config template (Class Catalog, rule defaults).

    A sidecar is configuration, not a model, so reading it does not load default
    weights. It seeds class names and rule defaults so the builder is usable
    before a model is uploaded; an uploaded model's own sidecar overrides it.
    """
    path = FEATURES[feature].get("model_path")
    if path and Path(sidecar_path(str(path))).is_file():
        return load_config(sidecar_path(str(path)))
    return ModelConfig()


def _config_for_upload(
    feature: str, session: ModelSession, sidecar: dict | None
) -> ModelConfig:
    """Resolve an uploaded model's config: its own sidecar, else the template.

    The preprocessing size always comes from the uploaded model itself, so a
    model of a different input size than the template still preprocesses right.
    """
    if feature not in _SIDECAR_FEATURES:
        return ModelConfig()
    config = config_from_dict(sidecar) if sidecar is not None else _bundled_template(feature)
    config.input_size = tuple(session.input_shape)
    return config


def upload_model(
    feature: str,
    model_bytes: bytes,
    filename: str,
    model_version: str,
    sidecar: dict | None = None,
) -> ModelSession:
    """Load an uploaded model for `feature`, replacing any previous one.

    Raises ValueError for an unknown or non-uploadable Feature (the Cascade
    Feature has no model of its own), or when the bytes are not a valid .onnx.
    """
    if feature not in FEATURES or feature == CASCADE_FEATURE:
        raise ValueError(f"Cannot upload a model for feature {feature!r}.")
    session = engine.build_session(model_bytes, filename, model_version)
    _models[feature] = (session, _config_for_upload(feature, session, sidecar))
    logger.info("Model uploaded for %s: version=%s", feature, model_version)
    return session


def get_model(feature: str) -> tuple[ModelSession, ModelConfig] | None:
    return _models.get(feature)


def active_model() -> ModelSession | None:
    """The active single-Feature model session, for status and reporting."""
    if _active_feature and _active_feature != CASCADE_FEATURE:
        entry = _models.get(_active_feature)
        if entry:
            return entry[0]
    return None


def loaded_models() -> dict[str, dict]:
    """Which Features have a model loaded, for the per-stage/per-Feature indicator."""
    return {
        feature: {
            "model_version": session.model_version,
            "runtime": session.runtime,
            "input_shape": list(session.input_shape),
        }
        for feature, (session, _) in _models.items()
    }


def clear_model(feature: str) -> bool:
    """Unload a Feature's model. Returns True if one was present."""
    return _models.pop(feature, None) is not None


def _build_pipeline_for(feature: str, session: ModelSession, config: ModelConfig):
    """Construct the right single-Feature pipeline over a session + config."""
    if feature == PRESENCE_FEATURE:
        return PresenceAbsencePipeline(session, config, config.expected_classes)
    if feature == SEGMENTATION_FEATURE:
        return SegmentationPipeline(session, config)
    return AnomalyPipeline(session)


def current_pipeline():
    """Build the pipeline for the active Feature over its uploaded model.

    Returns None when no Feature is active, the active Feature has no model
    uploaded, or the Cascade Feature is active (it is assembled per request from
    its spec by `build_cascade_pipeline`).
    """
    if _active_feature is None or _active_feature == CASCADE_FEATURE:
        return None
    entry = _models.get(_active_feature)
    if entry is None:
        return None
    return _build_pipeline_for(_active_feature, entry[0], entry[1])


# ── Cascade assembly ────────────────────────────────────────────────────────
# A cascade reads the same per-Feature store: each stage's model is the model
# uploaded for that stage's Feature. Stages sharing a Feature share its model.


def build_cascade_pipeline(spec: dict) -> CascadePipeline:
    """Assemble a CascadePipeline from a per-request spec.

    Raises ValueError for a malformed spec, an unknown feature/rule, a rule
    incompatible with a stage's Feature, or a stage whose Feature has no model
    uploaded -- configuration errors the API surfaces rather than silent NOKs.
    """
    combinator = combine.get(spec.get("combinator", "and"))
    short_circuit = bool(spec.get("short_circuit", True))
    raw_stages = spec.get("stages") or []
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("A cascade needs a non-empty 'stages' list.")

    stages: list[CascadeStage] = []
    for i, raw in enumerate(raw_stages):
        feature = raw.get("feature")
        if feature not in FEATURES or feature == CASCADE_FEATURE:
            raise ValueError(f"Stage {i + 1}: invalid feature {feature!r}.")
        entry = _models.get(feature)
        if entry is None:
            raise ValueError(
                f"Stage {i + 1}: no model loaded for {feature!r}; upload one first."
            )
        session, config = entry
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

    return CascadePipeline(stages, combinator, short_circuit)


# Output kinds each Feature's decode advertises -- static, so the builder needs
# no model load to know which rules a Feature can run.
_FEATURE_KINDS = {
    ANOMALY: frozenset({decision.KIND_ANOMALY_MAP}),
    PRESENCE_FEATURE: frozenset({decision.KIND_DETECTIONS}),
    SEGMENTATION_FEATURE: frozenset({decision.KIND_DETECTIONS, decision.KIND_MASKS}),
}


def cascade_options() -> dict:
    """Everything the stage builder needs, plus which Features have a model.

    Rules are typed on output kinds (static per Feature). Class Catalogs and
    rule defaults come from the uploaded model's config when one is loaded, else
    the bundled sidecar template -- so the builder is usable before upload and
    reflects the real model afterwards.
    """
    features_out = []
    for name in FEATURES:
        if name == CASCADE_FEATURE:
            continue
        entry = _models.get(name)
        cfg = entry[1] if entry else _bundled_template(name)
        rules = []
        for rule in decision.compatible_with(_FEATURE_KINDS.get(name, frozenset())):
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
                "model": loaded_models().get(name),  # None until a model is uploaded
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


def current_decision_rules() -> list[dict]:
    """UI metadata for the Decision Rules the active pipeline can run.

    Filtered by the output kinds the active Feature's decode advertises, so the
    GUI never offers a rule that cannot consume this model's output. Empty when
    no Feature is active or no model is loaded for it.
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
    """Clear all loaded models and active-Feature state (test hook)."""
    global _active_feature
    _models.clear()
    _active_feature = None
