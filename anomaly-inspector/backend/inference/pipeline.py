"""Per-Feature inference pipelines.

Each Feature owns its preprocessing, model-output decode, and visualization. The
Verdict itself is *not* the pipeline's job: every pipeline ends by delegating to
a Decision Rule, the pluggable post-processing step. `/api/infer` holds one
active pipeline and delegates to it, so it stays agnostic to which Feature is
running and which rule is selected. See docs/adr/0002 and docs/adr/0005.
"""
from dataclasses import dataclass, field

from inference import decision, engine
from inference.decision.base import (
    KIND_ANOMALY_MAP,
    KIND_DETECTIONS,
    DecisionContext,
    DecodedOutput,
)
from inference.engine import ModelSession
from inference.preprocessor import preprocess
from inference.rfdetr import (
    PresenceConfig,
    decode_detections,
    draw_detections_rgb,
    preprocess_image,
)
from inference.visualizer import (
    encode_jpeg,
    generate_heatmap,
    generate_segmentation,
    render_annotations,
)


@dataclass
class InferenceResult:
    """The Feature-agnostic envelope every pipeline returns.

    - `verdict`: "ok" | "not_ok" (existing DB spelling for OK/NOK).
    - `score`: the Decision Rule's primary scalar, or None for rules without
      one. `score_label` names it for display.
    - `images`: Feature-specific visualizations keyed by name, e.g.
      {"heatmap":.., "segmentation":..} or {"annotated":..}.
    - `detections`: optional structured detections.
    - `decision_rule` / `metrics` / `reason`: which rule ran, what it measured,
      and why it decided as it did. Persisted for report traceability.
    """

    verdict: str
    score: float | None
    images: dict[str, bytes]
    detections: list[dict] | None = None
    decision_rule: str = ""
    score_label: str = ""
    metrics: dict = field(default_factory=dict)
    reason: str = ""


class _PipelineBase:
    """Shared Decision Rule plumbing: rule selection, invocation, envelope."""

    feature: str = ""
    kinds: frozenset[str] = frozenset()
    default_rule: str = ""

    model: ModelSession

    @property
    def model_version(self) -> str:
        return self.model.model_version

    @property
    def labels(self) -> dict[int, str]:
        return {}

    def default_rule_params(self, rule_name: str) -> dict:
        """Sidecar-sourced defaults for a rule, merged under caller overrides."""
        return {}

    def compatible_rules(self) -> list:
        return decision.compatible_with(self.kinds)

    def _resolve_rule(self, rule_name: str | None):
        rule = decision.get(rule_name or self.default_rule)
        if not rule.consumes <= self.kinds:
            raise ValueError(
                f"Decision Rule {rule.name!r} needs {sorted(rule.consumes)}, "
                f"but the {self.feature!r} Feature produces {sorted(self.kinds)}."
            )
        return rule

    def _decide(
        self,
        rule_name: str | None,
        rule_params: dict | None,
        output: DecodedOutput,
        image_rgb,
        threshold: float,
    ):
        rule = self._resolve_rule(rule_name)
        # Layered lowest-first: rule defaults < sidecar defaults < request params.
        params = decision.resolve_params(
            rule, self.default_rule_params(rule.name), rule_params
        )
        ctx = DecisionContext(
            output=output,
            image_rgb=image_rgb,
            labels=self.labels,
            params=params,
            threshold=threshold,
        )
        return rule, rule.evaluate(ctx)


class AnomalyPipeline(_PipelineBase):
    """Anomaly Detection: pixel anomaly map + scalar score, judged by a rule."""

    feature = "anomaly_detection"
    kinds = frozenset({KIND_ANOMALY_MAP})
    default_rule = "anomaly_threshold"

    def __init__(self, model: ModelSession):
        self.model = model

    def infer(
        self,
        image_bytes: bytes,
        threshold: float,
        rule_name: str | None = None,
        rule_params: dict | None = None,
    ) -> InferenceResult:
        tensor, original_rgb = preprocess(image_bytes, self.model.input_shape)
        anomaly_map, pred_score = engine.run_inference_on(self.model, tensor)
        output = DecodedOutput(
            kinds=self.kinds,
            image_hw=original_rgb.shape[:2],
            anomaly_map=anomaly_map,
            anomaly_score=pred_score,
        )
        rule, res = self._decide(
            rule_name, rule_params, output, original_rgb, threshold
        )

        images = {
            "heatmap": generate_heatmap(anomaly_map, original_rgb),
            "segmentation": generate_segmentation(anomaly_map, original_rgb, threshold),
        }
        if res.annotations:
            images["overlay"] = encode_jpeg(
                render_annotations(original_rgb, res.annotations)
            )

        return InferenceResult(
            verdict=res.verdict,
            score=res.score,
            images=images,
            decision_rule=rule.name,
            score_label=res.score_label,
            metrics=res.metrics,
            reason=res.reason,
        )


class PresenceAbsencePipeline(_PipelineBase):
    """Presence/Absence: RF-DETR detections, judged by a Decision Rule.

    Holds the model plus its model-scoped `PresenceConfig` (Class Catalog +
    preprocessing) and the SKU's `expected_classes`, which seeds the default
    params of the Expected Classes rule. The Threshold passed to `infer` is the
    detection-confidence floor.
    """

    feature = "presence_absence"
    kinds = frozenset({KIND_DETECTIONS})
    default_rule = "expected_classes"

    def __init__(
        self,
        model: ModelSession,
        config: PresenceConfig,
        expected_classes: list[int],
    ):
        self.model = model
        self.config = config
        self.expected_classes = expected_classes

    @property
    def labels(self) -> dict[int, str]:
        return self.config.labels

    def default_rule_params(self, rule_name: str) -> dict:
        if rule_name == "expected_classes":
            return {"expected_classes": self.expected_classes}
        return {}

    def infer(
        self,
        image_bytes: bytes,
        threshold: float,
        rule_name: str | None = None,
        rule_params: dict | None = None,
    ) -> InferenceResult:
        tensor, original_rgb = preprocess_image(image_bytes, self.config)
        outputs = engine.run_raw(self.model, tensor)
        orig_h, orig_w = original_rgb.shape[:2]
        detections = decode_detections(outputs, (orig_h, orig_w), threshold, self.config)
        output = DecodedOutput(
            kinds=self.kinds,
            image_hw=(orig_h, orig_w),
            detections=detections,
        )
        rule, res = self._decide(
            rule_name, rule_params, output, original_rgb, threshold
        )

        # Boxes first, then the rule's own annotations on top, then one encode.
        canvas = draw_detections_rgb(original_rgb, detections, self.config.labels)
        annotated = encode_jpeg(render_annotations(canvas, res.annotations))

        return InferenceResult(
            verdict=res.verdict,
            score=res.score,
            images={"annotated": annotated},
            detections=[
                {
                    "class_id": d.class_id,
                    "label": self.config.labels.get(d.class_id, str(d.class_id)),
                    "confidence": round(d.confidence, 4),
                    "box": [round(v, 1) for v in d.box],
                }
                for d in detections
            ],
            decision_rule=rule.name,
            score_label=res.score_label,
            metrics=res.metrics,
            reason=res.reason,
        )
