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
    PipelineBase,
    PresenceAbsencePipeline,
    SegmentationPipeline,
)

logger = logging.getLogger(__name__)

# Features whose model carries a sidecar (Class Catalog, preprocessing, rule
# param defaults). Anomaly Detection needs none.
_SIDECAR_FEATURES = (PRESENCE_FEATURE, SEGMENTATION_FEATURE)

# The single source of truth: one uploaded (session, config) per Feature. No
# default models are ever placed here.
# Encapsulated for future dependency injection (design-no-global-singleton) —
# a FeatureStore instance will be provided via FastAPI Depends in production;
# module-level globals remain as the composition-root default for this demo app
# and for tests that patch _models directly.
from dataclasses import dataclass, field as _field


@dataclass
class FeatureStore:
    """Mutable Feature state. In production this is provided via Depends."""

    models: dict[str, tuple[ModelSession, ModelConfig]] = _field(default_factory=dict)
    active_feature: str | None = None


_STORE = FeatureStore()
# Back-compat aliases — tests patch these directly (e.g. _features._models[...])
_models = _STORE.models
_active_feature = _STORE.active_feature  # kept in sync via property below


def _sync_store_aliases() -> None:
    global _models, _active_feature
    _models = _STORE.models
    _active_feature = _STORE.active_feature


def get_feature_store() -> FeatureStore:
    """Composition-root accessor for DI (future FastAPI Depends)."""
    return _STORE


def get_active_feature() -> str | None:
    """The Feature selected in the GUI, or None before any selection.

    Returns:
        A Feature name from config.FEATURES, or None.
    """
    return _active_feature


def get_model_config() -> ModelConfig | None:
    """Config of the active single-Feature model.

    Returns:
        The uploaded model's ModelConfig, or None when no Feature is active,
        the Cascade Feature is active (stages carry their own), or nothing
        has been uploaded yet.
    """
    if _active_feature and _active_feature != CASCADE_FEATURE:
        entry = _models.get(_active_feature)
        if entry:
            return entry[1]
    return None


def activate(feature: str) -> None:
    """Make `feature` active. Does not load a model -- the operator uploads one.

    Raises ValueError for an unknown Feature. The Cascade Feature is valid; its
    stage models load per request from the same store.

    Args:
        feature: Feature name to make active.

    Raises:
        ValueError: If `feature` is not in FEATURES.
    """
    global _active_feature
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature: {feature!r}")
    _active_feature = feature
    _STORE.active_feature = feature


def _bundled_template(feature: str) -> ModelConfig:
    """The bundled sidecar as a config template (Class Catalog, rule defaults).

    A sidecar is configuration, not a model, so reading it does not load default
    weights. It seeds class names and rule defaults so the builder is usable
    before a model is uploaded; an uploaded model's own sidecar overrides it.

    Args:
        feature: Feature whose bundled sidecar to read.

    Returns:
        The sidecar's ModelConfig, or a default ModelConfig when the Feature
        has no bundled sidecar.
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

    Args:
        feature: Feature the model was uploaded for.
        session: Session built from the uploaded bytes.
        sidecar: Parsed sidecar JSON sent alongside the upload, or None.

    Returns:
        The ModelConfig to pair with this session in the store.
    """
    if feature not in _SIDECAR_FEATURES:
        return ModelConfig()
    config = config_from_dict(sidecar) if sidecar is not None else _bundled_template(feature)
    config.input_size = tuple(session.input_shape)
    # Resolve auto (None) pre/post-processing from the ONNX signature: a
    # channels-last raw-image input bakes preprocessing; decoded (`xyxy`)
    # outputs bake postprocessing. An explicit sidecar True/False wins.
    if config.preprocess is None:
        config.preprocess = not session.input_channels_last
    if config.postprocess is None:
        config.postprocess = not session.outputs_decoded
    return config


def upload_model(
    feature: str,
    model_bytes: bytes,
    filename: str,
    model_version: str,
    sidecar: dict | None = None,
) -> ModelSession:
    """Load an uploaded model for `feature`, replacing any previous one.

    Replaces any previous entry for the Feature in the store.

    Args:
        feature: Feature to load the model for.
        model_bytes: Raw .onnx contents.
        filename: Original filename; must end in ".onnx".
        model_version: Version string recorded on the session.
        sidecar: Optional parsed sidecar JSON overriding the bundled template.

    Returns:
        The ModelSession now stored for this Feature.

    Raises:
        ValueError: If `feature` is unknown or non-uploadable (the Cascade
            Feature has no model of its own), or the bytes are not a valid
            .onnx payload.
    """
    if feature not in FEATURES or feature == CASCADE_FEATURE:
        raise ValueError(f"Cannot upload a model for feature {feature!r}.")
    session = engine.build_session(model_bytes, filename, model_version)
    _models[feature] = (session, _config_for_upload(feature, session, sidecar))
    logger.info("Model uploaded for %s: version=%s", feature, model_version)
    return session


def get_model(feature: str) -> tuple[ModelSession, ModelConfig] | None:
    """The (session, config) uploaded for `feature`.

    Args:
        feature: Feature name to look up.

    Returns:
        Tuple of (ModelSession, ModelConfig), or None when nothing is stored.
    """
    return _models.get(feature)


def active_model() -> ModelSession | None:
    """The active single-Feature model session, for status and reporting.

    Returns:
        The active Feature's ModelSession, or None when no Feature is active,
        the Cascade Feature is active, or it has no uploaded model.
    """
    if _active_feature and _active_feature != CASCADE_FEATURE:
        entry = _models.get(_active_feature)
        if entry:
            return entry[0]
    return None


def loaded_models() -> dict[str, dict]:
    """Which Features have a model loaded.

    Returns:
        Map of Feature name -> {model_version, runtime, input_shape} for each
        uploaded model, driving the per-stage/per-Feature indicators.
    """
    return {
        feature: {
            "model_version": session.model_version,
            "runtime": session.runtime,
            "input_shape": list(session.input_shape),
        }
        for feature, (session, _) in _models.items()
    }


_PIPELINE_FACTORIES: dict[str, object] = {
    ANOMALY: lambda s, c: AnomalyPipeline(s),
    PRESENCE_FEATURE: lambda s, c: PresenceAbsencePipeline(s, c),
    SEGMENTATION_FEATURE: lambda s, c: SegmentationPipeline(s, c),  # type: ignore[arg-type]
}


def _build_pipeline_for(
    feature: str, session: ModelSession, config: ModelConfig
) -> PipelineBase:
    """Construct the right single-Feature pipeline over a session + config.

    Closed for modification — adding a Feature registers a factory rather than
    editing a branching chain (Open/Closed Principle).

    Args:
        feature: Feature name (Anomaly Detection needs no sidecar config).
        session: The Feature's loaded model session.
        config: The Feature's ModelConfig (ignored for Anomaly Detection).

    Returns:
        The pipeline instance implementing the Feature.

    Raises:
        ValueError: If `feature` is unknown.
    """
    try:
        factory = _PIPELINE_FACTORIES[feature]  # type: ignore[assignment]
    except KeyError:
        raise ValueError(f"Unknown feature: {feature!r}") from None
    return factory(session, config)  # type: ignore[operator]


def current_pipeline() -> PipelineBase | None:
    """Build the pipeline for the active Feature over its uploaded model.

    Returns None when no Feature is active, the active Feature has no model
    uploaded, or the Cascade Feature is active (it is assembled per request from
    its spec by `build_cascade_pipeline`).

    Returns:
        The active Feature's pipeline, or None as described above.
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

    Each stage's model comes from the same per-Feature store the operator
    uploads into; stages sharing a Feature share its model.

    Args:
        spec: Parsed cascade spec with "combinator", optional "short_circuit",
            and a non-empty "stages" list of {feature, rule?, threshold?,
            params?} dicts.

    Returns:
        A CascadePipeline ready to run on one image.

    Raises:
        ValueError: For a malformed spec, an unknown Feature or Decision Rule,
            a rule incompatible with a stage's Feature, or a stage whose
            Feature has no model uploaded — configuration errors surfaced by
            the API rather than silent NOKs.
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
        pipeline.resolve_rule(rule_name)
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

    Returns:
        {"features": [...], "combinators": [...]}; each feature entry carries
        name/label/threshold_label, its Class Catalog as label strings, and
        its compatible rules with effective default params.
    """
    features_out = []
    for name in FEATURES:
        if name == CASCADE_FEATURE:
            continue
        entry = _models.get(name)
        cfg = entry[1] if entry else _bundled_template(name)
        kinds = _FEATURE_KINDS.get(name, frozenset())
        rules = []
        for rule in decision.compatible_with(kinds):
            # Describe against the feature's output kinds so a rule can tailor
            # its controls (e.g. only box-compatible centre methods without
            # masks). Defaults come from those adapted params — the sidecar
            # carries only the Class Catalog now.
            described = decision.describe(rule, kinds)
            _apply_catalog_class_defaults(described, cfg.labels)
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


def current_decision_rules() -> list[dict]:
    """UI metadata for the Decision Rules the active pipeline can run.

    Filtered by the output kinds the active Feature's decode advertises, so the
    GUI never offers a rule that cannot consume this model's output.

    Returns:
        UI metadata dicts with schema + effective defaults; empty when no
        Feature is active or no model is loaded for it.
    """
    pipeline = current_pipeline()
    if pipeline is None:
        return []
    rules = []
    for rule in pipeline.compatible_rules():
        # Describe against this model's output kinds so a rule can tailor its
        # controls (e.g. only box-compatible centre methods without masks). The
        # GUI seeds from these defaults and echoes them back on every request,
        # so they must match the offered options.
        described = decision.describe(rule, pipeline.kinds)
        _apply_catalog_class_defaults(described, pipeline.labels)
        rules.append(described)
    return rules


def _apply_catalog_class_defaults(described: dict, labels: dict[int, str]) -> None:
    """Point `class` param defaults at real catalog classes, then set defaults.

    A rule's schema default for a `class` param names a class from whatever model
    it was first written against (concentricity's `bottle_cap`/`logo`), which is
    not in another model's catalog — the GUI would seed an invalid class and
    inference would reject it. Any `class` default absent from this model's
    catalog is reassigned to a catalog class, distinct per class-param position
    when the catalog is large enough. Defaults valid for this model (and
    `class_list` defaults) are left untouched.

    Args:
        described: A `describe(...)` dict; mutated in place, and its `defaults`
            recomputed from the (possibly rewritten) params.
        labels: The loaded model's Class Catalog (id -> name).
    """
    names = list(labels.values())
    class_ordinal = 0
    for p in described["params"]:
        if p["type"] == "class":
            if names and p["default"] not in names:
                p["default"] = names[min(class_ordinal, len(names) - 1)]
            class_ordinal += 1
    described["defaults"] = {p["name"]: p["default"] for p in described["params"]}


def reset() -> None:
    """Clear all loaded models and active-Feature state (test hook).

    Production never resets mid-process; this exists so tests start from a
    clean Feature store.
    """
    global _active_feature
    _models.clear()
    _STORE.models.clear()
    _active_feature = None
    _STORE.active_feature = None
