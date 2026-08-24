"""Model-scoped configuration, sourced from a model's JSON sidecar.

Shared by every RF-DETR-family Feature (detection and segmentation). Carries the
Class Catalog, the preprocessing parameters, the mask binarisation cutoff, and
the *default* Decision Rule params the GUI seeds from.

Rule params live here as defaults only — the live values travel per-request, so
the backend stays stateless (see docs/adr/0005).
"""
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Standard ImageNet normalisation, the RF-DETR default. Overridable via sidecar.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DEFAULT_INPUT_SIZE = (384, 384)


@dataclass
class ModelConfig:
    """Everything about a model that is not in the .onnx itself.

    Fields left unset fall back to RF-DETR defaults so the common case is
    zero-config — but note `input_size` genuinely differs between exports
    (rfdetr-nano is 384, the segmentation export is 312), so a sidecar that
    omits it will preprocess at the wrong resolution.
    """

    labels: dict[int, str] = field(default_factory=dict)
    input_size: tuple[int, int] = _DEFAULT_INPUT_SIZE  # (height, width)
    mean: tuple[float, float, float] = _IMAGENET_MEAN
    std: tuple[float, float, float] = _IMAGENET_STD
    normalize: bool = True  # False when normalisation is baked into the graph
    # Sigmoid cutoff turning per-query mask logits into a boolean mask. A
    # property of the export's calibration, not a process tolerance, so it is
    # configured here rather than exposed as an operator control.
    mask_threshold: float = 0.5
    # Which Decision Rule to preselect, and the param defaults the GUI seeds
    # from, keyed by rule name.
    default_rule: str | None = None
    rule_params: dict[str, dict] = field(default_factory=dict)

    @property
    def expected_classes(self) -> list:
        """Back-compat view of the Expected Classes rule's default param."""
        return self.rule_params.get("expected_classes", {}).get("expected_classes", [])


def sidecar_path(model_path: str) -> str:
    """The sidecar JSON path for a model: same directory and stem, `.json`.

    e.g. `model/rfdetr-nano.onnx` -> `model/rfdetr-nano.json`.
    """
    return str(Path(model_path).with_suffix(".json"))


def load_config(path: str) -> ModelConfig:
    """Load a `ModelConfig` from a sidecar JSON, defaulting when absent.

    Missing files or fields fall back to RF-DETR defaults so a Feature stays
    usable (with raw class indices) even without a sidecar. JSON object keys are
    strings, so `labels` keys are coerced to int.

    Back-compat: a top-level `expected_classes` from the pre-Decision-Rule
    sidecar format is mapped into the Expected Classes rule's params.
    """
    p = Path(path)
    if not p.is_file():
        logger.info("No model sidecar at %s; using defaults (raw class indices).", path)
        return ModelConfig()

    data = json.loads(p.read_text())
    defaults = ModelConfig()
    labels = {int(k): str(v) for k, v in data.get("labels", {}).items()}
    input_size = data.get("input_size")
    mean = data.get("mean")
    std = data.get("std")

    rule_params = {k: dict(v) for k, v in (data.get("rule_params") or {}).items()}
    legacy_expected = data.get("expected_classes")
    if legacy_expected and "expected_classes" not in rule_params:
        rule_params["expected_classes"] = {
            "expected_classes": [int(c) for c in legacy_expected]
        }

    return ModelConfig(
        labels=labels,
        input_size=tuple(input_size) if input_size else defaults.input_size,
        mean=tuple(mean) if mean else defaults.mean,
        std=tuple(std) if std else defaults.std,
        normalize=data.get("normalize", defaults.normalize),
        mask_threshold=float(data.get("mask_threshold", defaults.mask_threshold)),
        default_rule=data.get("default_rule"),
        rule_params=rule_params,
    )
