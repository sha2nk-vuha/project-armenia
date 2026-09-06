"""Per-Feature inference pipelines.

Each Feature owns its preprocessing, model-output decode, and visualization. The
Verdict itself is *not* the pipeline's job: every pipeline ends by delegating to
a Decision Rule, the pluggable post-processing step. `/api/infer` holds one
active pipeline and delegates to it, so it stays agnostic to which Feature is
running and which rule is selected. See docs/adr/0002 and docs/adr/0005.
"""
from dataclasses import dataclass, field
import logging

import numpy as np

from inference import decision, engine
from inference.decision.base import (
    KIND_ANOMALY_MAP,
    KIND_DETECTIONS,
    KIND_MASKS,
    DecisionContext,
    DecisionRule,
    DecodedOutput,
    Instance,
)
from inference.engine import ModelSession
from inference.preprocessor import preprocess
from inference.rfdetr import (
    Detection,
    ModelConfig,
    decode_detections,
    draw_detections_rgb,
    preprocess_image,
)
from inference.rfdetr_seg import decode_instances, draw_instances_rgb
from inference.visualizer import (
    encode_jpeg,
    generate_heatmap,
    generate_segmentation,
    render_annotations,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """The Feature-agnostic envelope every pipeline returns.

    - `verdict`: the Verdict string (see inference.verdict for the spelling).
    - `score`: the Decision Rule's primary scalar, or None for rules without
      one. `score_label` names it for display.
    - `images`: Feature-specific visualizations keyed by name, e.g.
      {"heatmap":.., "segmentation":..} or {"annotated":..}.
    - `detections`: optional structured detections.
    - `decision_rule` / `metrics` / `reason`: which rule ran, what it measured,
      and why it decided as it did. Persisted for report traceability.
    - `stages`: for a Cascade, the ordered per-stage results (each carrying its
      own image bytes); empty for a single Feature.
    """

    verdict: str
    score: float | None
    images: dict[str, bytes]
    detections: list[dict] | None = None
    decision_rule: str = ""
    score_label: str = ""
    metrics: dict = field(default_factory=dict)
    reason: str = ""
    stages: list["StageResult"] = field(default_factory=list)


# Human labels for the image keys pipelines emit. One source of truth so a
# cascade can present a stage's images generically; a new image kind is labelled
# by adding one entry here, with no frontend change (see docs/adr/0002, 0006).
IMAGE_LABELS = {
    "heatmap": "Heatmap",
    "segmentation": "Segmentation",
    "annotated": "Detections",
    "overlay": "Overlay",
}


def labelled_images(images: dict[str, bytes]) -> list[tuple[str, bytes]]:
    """Order-preserving (label, bytes) pairs for a pipeline's image dict.

    Args:
        images: Image kind key -> JPEG bytes, as a pipeline emits them.

    Returns:
        (human label, bytes) pairs; unknown keys get title-cased.
    """
    return [(IMAGE_LABELS.get(key, key.replace("_", " ").title()), img)
            for key, img in images.items()]


@dataclass
class StageResult:
    """One stage's outcome within a Cascade.

    `evaluated` is False for a stage the cascade short-circuited past: it did not
    run, so it has no Verdict of its own and must not be read as OK. `images` is
    that stage's full set of labelled visualizations (a cascade shows them all,
    not just one).
    """

    feature: str
    decision_rule: str
    # A Verdict ("ok" | "not_ok"); SKIPPED (see inference.verdict) when the
    # cascade short-circuited past this stage.
    verdict: str
    evaluated: bool
    score: float | None = None
    score_label: str = ""
    reason: str = ""
    images: list[tuple[str, bytes]] = field(default_factory=list)
    detections: list[dict] | None = None


class PipelineBase:
    """Shared Decision Rule plumbing: rule selection, invocation, envelope."""

    feature: str = ""
    kinds: frozenset[str] = frozenset()
    default_rule: str = ""

    model: ModelSession

    @property
    def model_version(self) -> str:
        """Identifier of the loaded model this pipeline runs.

        Returns:
            The ModelSession's version string.
        """
        return self.model.model_version

    @property
    def labels(self) -> dict[int, str]:
        """The Class Catalog of the loaded model.

        Returns:
            Map of class id -> label; empty for Features without a catalog.
        """
        return {}

    def default_rule_params(self, rule_name: str) -> dict:
        """Sidecar-sourced defaults for a rule, merged under caller overrides.

        Args:
            rule_name: Decision Rule name the params are for.

        Returns:
            Default params dict; empty when this Feature seeds nothing.
        """
        return {}

    def compatible_rules(self) -> list:
        """Decision Rules whose consumed output kinds this Feature feeds.

        Returns:
            Registered rules compatible with this pipeline's decode.
        """
        return decision.compatible_with(self.kinds)

    def resolve_rule(self, rule_name: str | None) -> "DecisionRule":
        """The Decision Rule this pipeline would run for `rule_name`.

        Public interface (the API resolves rules to validate requests and to
        look up SKU calibration).

        Args:
            rule_name: Requested Decision Rule name; None selects the
                pipeline's default.

        Returns:
            The resolved Decision Rule instance.

        Raises:
            ValueError: If the name is unknown, or the rule consumes output
                kinds this Feature's decode cannot feed.
        """
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
        rule = self.resolve_rule(rule_name)
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

    # ── Shared detector flow ────────────────────────────────────────────────
    # Every RF-DETR-family Feature runs the same skeleton: preprocess -> raw
    # model run -> per-Feature decode -> DecodedOutput -> Decision Rule ->
    # rule annotations over a drawn canvas -> one JPEG encode -> envelope.
    # Only the decode step (and what it feeds the canvas) varies per Feature;
    # subclasses supply it and call `annotated_result`.

    def _preprocess_and_run(self, image_bytes: bytes):
        """Preprocess an image for this Feature and run its model.

        Args:
            image_bytes: Encoded image (any PIL-decodable format).

        Returns:
            Tuple of (original_rgb [H,W,3] uint8, raw output tensors in port
            order).
        """
        tensor, original_rgb = preprocess_image(image_bytes, self.config)
        return original_rgb, engine.run_raw(self.model, tensor)

    def annotated_result(
        self,
        original_rgb: np.ndarray,
        output: DecodedOutput,
        canvas: np.ndarray,
        detection_rows: list[dict],
        threshold: float,
        rule_name: str | None,
        rule_params: dict | None,
    ) -> InferenceResult:
        """Decide on `output`, annotate `canvas`, and assemble the envelope.

        The Decision Rule's declarative annotations are rendered over whatever
        the Feature drew (boxes or masks), then encoded exactly once.

        Args:
            original_rgb: The preprocessed-original RGB frame annotations are
                positioned against.
            output: The decoded model output the rule will judge.
            canvas: Feature-specific drawing (boxes or masks) to annotate over.
            detection_rows: API-shaped dicts describing each decoded item.
            threshold: Model-level Threshold passed through to the rule.
            rule_name: Requested Decision Rule; None uses the default.
            rule_params: Caller overrides layered over sidecar defaults.

        Returns:
            The InferenceResult envelope with verdict, annotated JPEG, and
            decision traceability fields.
        """
        rule, res = self._decide(rule_name, rule_params, output, original_rgb, threshold)
        annotated = encode_jpeg(render_annotations(canvas, res.annotations))
        return InferenceResult(
            verdict=res.verdict,
            score=res.score,
            images={"annotated": annotated},
            detections=detection_rows,
            decision_rule=rule.name,
            score_label=res.score_label,
            metrics=res.metrics,
            reason=res.reason,
        )


class AnomalyPipeline(PipelineBase):
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
        *,
        letterbox: bool = False,
    ) -> InferenceResult:
        """Run one Anomaly Detection inspection and return its envelope.

        Args:
            image_bytes: Encoded image to inspect.
            threshold: Threshold bounding the anomaly score (main slider).
            rule_name: Decision Rule to judge with; defaults to
                anomaly_threshold.
            rule_params: Overrides layered over sidecar defaults.
            letterbox: When True, scale uniformly and pad to model size before
                inference so overlays align on non-square images.

        Returns:
            InferenceResult with heatmap/segmentation images and the Verdict.
        """
        tensor, original_rgb, transform = preprocess(
            image_bytes, self.model.input_shape, letterbox=letterbox
        )
        anomaly_map, pred_score = engine.run_inference_on(self.model, tensor)
        map_h, map_w = anomaly_map.squeeze().shape
        orig_h, orig_w = original_rgb.shape[:2]
        model_h, model_w = self.model.input_shape
        logger.info(
            "Anomaly overlay shapes: original=%dx%d model=%dx%d map=%dx%d "
            "letterbox=%s pad=(%d,%d)",
            orig_h,
            orig_w,
            model_h,
            model_w,
            map_h,
            map_w,
            transform.letterboxed,
            transform.pad_top,
            transform.pad_left,
        )
        if (map_h, map_w) != (model_h, model_w):
            logger.warning(
                "Anomaly map resolution %dx%d differs from model input %dx%d; "
                "overlay uses a two-step resize through model space",
                map_h,
                map_w,
                model_h,
                model_w,
            )
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
            "heatmap": generate_heatmap(
                anomaly_map, original_rgb, transform=transform
            ),
            "segmentation": generate_segmentation(
                anomaly_map, original_rgb, threshold, transform=transform
            ),
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


class PresenceAbsencePipeline(PipelineBase):
    """Presence/Absence: RF-DETR detections, judged by a Decision Rule.

    Holds the model plus its model-scoped `ModelConfig` (only the Class Catalog
    is model-specific now). Decision Rule params come from each rule's own schema
    defaults, layered under the operator's GUI values — the sidecar no longer
    seeds them. The Threshold passed to `infer` is the detection-confidence floor.
    """

    feature = "presence_absence"
    kinds = frozenset({KIND_DETECTIONS})
    default_rule = "expected_and_forbidden"

    def __init__(self, model: ModelSession, config: ModelConfig):
        self.model = model
        self.config = config

    @property
    def labels(self) -> dict[int, str]:
        """The Class Catalog from the model's sidecar config.

        Returns:
            Map of class id -> label.
        """
        return self.config.labels

    def infer(
        self,
        image_bytes: bytes,
        threshold: float,
        rule_name: str | None = None,
        rule_params: dict | None = None,
        *,
        letterbox: bool = False,
    ) -> InferenceResult:
        """Run one Presence/Absence inspection and return its envelope.

        Args:
            image_bytes: Encoded image to inspect.
            threshold: Minimum detection confidence (the main slider).
            rule_name: Decision Rule to judge with; defaults to
                expected_and_forbidden.
            rule_params: Overrides layered over sidecar + SKU defaults.

        Returns:
            InferenceResult with the annotated detections JPEG and Verdict.
        """
        original_rgb, outputs = self._preprocess_and_run(image_bytes)
        orig_h, orig_w = original_rgb.shape[:2]
        detections = decode_detections(
            outputs, (orig_h, orig_w), threshold, self.config, self.model.output_names
        )
        output = DecodedOutput(
            kinds=self.kinds,
            image_hw=(orig_h, orig_w),
            detections=detections,
        )

        # Boxes first, then the rule's own annotations on top, then one encode.
        canvas = draw_detections_rgb(original_rgb, detections, self.labels)
        return self.annotated_result(
            original_rgb,
            output,
            canvas,
            [self._detection_row(d) for d in detections],
            threshold,
            rule_name,
            rule_params,
        )

    def _detection_row(self, d: Detection) -> dict:
        """One Detection shaped for the API response.

        Args:
            d: Decoded detection in original-image pixels.

        Returns:
            Dict with class_id, label, confidence, and rounded box.
        """
        return {
            "class_id": d.class_id,
            "label": self.config.labels.get(d.class_id, str(d.class_id)),
            "confidence": round(d.confidence, 4),
            "box": [round(v, 1) for v in d.box],
        }


class SegmentationPipeline(PipelineBase):
    """Segmentation: RF-DETR per-instance masks, judged by a Decision Rule.

    Advertises both `masks` and `detections`, so geometry rules and the existing
    Expected Classes rule are equally selectable against it. Rule params come
    from each rule's schema defaults, layered under the operator's GUI values.
    The Threshold passed to `infer` is the detection-confidence floor; the mask
    binarisation cutoff comes from the model sidecar (see docs/adr/0005).
    """

    feature = "segmentation"
    kinds = frozenset({KIND_DETECTIONS, KIND_MASKS})
    default_rule = "expected_classes"

    def __init__(self, model: ModelSession, config: ModelConfig):
        self.model = model
        self.config = config

    @property
    def labels(self) -> dict[int, str]:
        """The Class Catalog from the model's sidecar config.

        Returns:
            Map of class id -> label.
        """
        return self.config.labels

    def resolve_rule(self, rule_name: str | None) -> "DecisionRule":
        """Sidecar-aware rule resolution.

        An unset name falls back to the model sidecar's default_rule before
        the Feature-level default.

        Args:
            rule_name: Requested Decision Rule name; None to use defaults.

        Returns:
            The resolved Decision Rule instance.

        Raises:
            ValueError: If the name is unknown or the rule is incompatible.
        """
        return super().resolve_rule(rule_name or self.config.default_rule)

    def infer(
        self,
        image_bytes: bytes,
        threshold: float,
        rule_name: str | None = None,
        rule_params: dict | None = None,
        *,
        letterbox: bool = False,
    ) -> InferenceResult:
        """Run one Segmentation inspection and return its envelope.

        Args:
            image_bytes: Encoded image to inspect.
            threshold: Minimum detection confidence; mask binarisation comes
                from the model sidecar instead.
            rule_name: Decision Rule to judge with; falls back to the sidecar
                default_rule before the Feature-level default.
            rule_params: Overrides layered over sidecar + SKU defaults.

        Returns:
            InferenceResult with the annotated instances JPEG and Verdict.
        """
        original_rgb, outputs = self._preprocess_and_run(image_bytes)
        orig_h, orig_w = original_rgb.shape[:2]
        instances = decode_instances(outputs, (orig_h, orig_w), threshold, self.config)
        # Masks carry boxes too, so detection-only rules run unchanged.
        output = DecodedOutput(
            kinds=self.kinds,
            image_hw=(orig_h, orig_w),
            instances=instances,
            detections=[
                Detection(class_id=i.class_id, confidence=i.confidence, box=i.box)
                for i in instances
            ],
        )

        # Masks first, then the rule's own geometry on top, then one encode.
        canvas = draw_instances_rgb(original_rgb, instances, self.labels)
        return self.annotated_result(
            original_rgb,
            output,
            canvas,
            [self._instance_row(i) for i in instances],
            threshold,
            rule_name,
            rule_params,
        )

    def _instance_row(self, i: Instance) -> dict:
        """One Instance shaped for the API response.

        Args:
            i: Decoded instance with pixel-space box and boolean mask.

        Returns:
            Dict with class_id, label, confidence, rounded box, and
            mask_area_px.
        """
        return {
            "class_id": i.class_id,
            "label": self.config.labels.get(i.class_id, str(i.class_id)),
            "confidence": round(i.confidence, 4),
            "box": [round(v, 1) for v in i.box],
            "mask_area_px": int(i.mask.sum()),
        }
