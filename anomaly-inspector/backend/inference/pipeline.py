"""Per-Feature inference pipelines.

Each Feature (Anomaly Detection, Presence/Absence) is a self-contained pipeline
that owns its preprocessing, model-output interpretation, verdict rule, and
visualization. `/api/infer` holds one active pipeline and delegates to it, so it
stays agnostic to which Feature is running. See docs/adr/0002.
"""
from dataclasses import dataclass

from inference import engine
from inference.engine import ModelSession
from inference.preprocessor import preprocess
from inference.rfdetr import (
    PresenceConfig,
    decode_detections,
    draw_detections,
    evaluate_presence,
    preprocess_image,
)
from inference.visualizer import generate_heatmap, generate_segmentation


@dataclass
class InferenceResult:
    """The Feature-agnostic envelope every pipeline returns.

    - `verdict`: "ok" | "not_ok" (existing DB spelling for OK/NOK).
    - `score`: a scalar summarising the inspection — the anomaly score for
      Anomaly Detection, or None for Features without a single score.
    - `images`: Feature-specific visualizations keyed by name, e.g.
      {"heatmap":.., "segmentation":..} or {"annotated":..}.
    - `detections`: optional structured detections (Presence/Absence).
    """

    verdict: str
    score: float | None
    images: dict[str, bytes]
    detections: list[dict] | None = None


class AnomalyPipeline:
    """Anomaly Detection: pixel anomaly map + scalar score vs Threshold."""

    feature = "anomaly_detection"

    def __init__(self, model: ModelSession):
        self.model = model

    @property
    def model_version(self) -> str:
        return self.model.model_version

    def infer(self, image_bytes: bytes, threshold: float) -> InferenceResult:
        tensor, original_rgb = preprocess(
            image_bytes, self.model.input_shape, self.model.graph_preprocessing
        )
        anomaly_map, pred_score = engine.run_inference_on(self.model, tensor)
        verdict = "ok" if pred_score < threshold else "not_ok"
        heatmap = generate_heatmap(anomaly_map, original_rgb)
        segmentation = generate_segmentation(anomaly_map, original_rgb, threshold)
        return InferenceResult(
            verdict=verdict,
            score=pred_score,
            images={"heatmap": heatmap, "segmentation": segmentation},
        )


class PresenceAbsencePipeline:
    """Presence/Absence: RF-DETR detections + expected-object verdict rule.

    Holds the model plus its per-run policy: the model-scoped `PresenceConfig`
    (Class Catalog + preprocessing) and the SKU's `expected_classes`. The
    Threshold passed to `infer` is the detection-confidence floor.
    """

    feature = "presence_absence"

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
    def model_version(self) -> str:
        return self.model.model_version

    def infer(self, image_bytes: bytes, threshold: float) -> InferenceResult:
        tensor, original_rgb = preprocess_image(
            image_bytes, self.config, self.model.graph_preprocessing
        )
        outputs = engine.run_raw(self.model, tensor)
        orig_h, orig_w = original_rgb.shape[:2]
        detections = decode_detections(outputs, (orig_h, orig_w), threshold, self.config)
        verdict = evaluate_presence(detections, self.expected_classes, threshold)
        annotated = draw_detections(original_rgb, detections, self.config.labels)
        return InferenceResult(
            verdict=verdict,
            score=None,  # Presence/Absence has no single anomaly score
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
        )
