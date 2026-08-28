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
from typing import Any

from pydantic import BaseModel, Field, ValidationError

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
    # Full preprocessing/postprocessing bypass, for exports that bake both into
    # the ONNX graph (e.g. an RF-DETR export whose input is the raw [1,H,W,3]
    # image and whose outputs are already-decoded xyxy boxes + sigmoid scores).
    # When `preprocess` is False the raw image is fed as-is (no resize/scale/
    # normalise, and `normalize`/`mean`/`std`/`input_size` are moot); when
    # `postprocess` is False the decode reads the model's boxes/scores directly
    # instead of applying sigmoid and un-normalising cxcywh boxes.
    #
    # None means "auto": the value is resolved from the ONNX signature when the
    # config is paired with its session (see features._config_for_upload) — a
    # channels-last input means baked preprocessing, `xyxy`-named outputs mean
    # baked postprocessing. The decode/preprocess helpers treat only an explicit
    # False as baked, so an unresolved None behaves as the normal (RF-DETR) path.
    preprocess: bool | None = None
    postprocess: bool | None = None
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
        """Back-compat view of the Expected Classes rule's default param.

    Returns:
        List of Expected Class ids from the rule_params defaults; empty when
        none configured.
    """
        return self.rule_params.get("expected_classes", {}).get("expected_classes", [])


def sidecar_path(model_path: str) -> str:
    """The sidecar JSON path for a model: same directory and stem, `.json`.

    e.g. `model/rfdetr-nano.onnx` -> `model/rfdetr-nano.json`.

    Args:
        model_path: Filesystem path of the .onnx model.

    Returns:
        Path of the sidecar JSON beside it.
    """
    return str(Path(model_path).with_suffix(".json"))


def load_config(path: str) -> ModelConfig:
    """Load a `ModelConfig` from a sidecar JSON, defaulting when absent.

    Missing files or fields fall back to RF-DETR defaults so a Feature stays
    usable (with raw class indices) even without a sidecar. JSON object keys are
    strings, so `labels` keys are coerced to int.

    Back-compat: a top-level `expected_classes` from the pre-Decision-Rule
    sidecar format is mapped into the Expected Classes rule's params.

    Args:
        path: Sidecar JSON path; may not exist.

    Returns:
        Parsed ModelConfig, or bare defaults when the file is absent.
    """
    p = Path(path)
    if not p.is_file():
        logger.info("No model sidecar at %s; using defaults (raw class indices).", path)
        return ModelConfig()

    return config_from_dict(json.loads(p.read_text()))


class _SidecarSchema(BaseModel):
    """Validated shape of a sidecar JSON (trust boundary: uploaded file)."""

    labels: dict[str, str] = Field(default_factory=dict)
    input_size: tuple[int, int] | None = None
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    normalize: bool = True
    preprocess: bool | None = None
    postprocess: bool | None = None
    mask_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    default_rule: str | None = None
    rule_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    expected_classes: list[int] | None = None  # legacy top-level

    model_config = {"extra": "ignore"}


def config_from_dict(data: dict) -> ModelConfig:
    """Build a ModelConfig from an already-parsed sidecar dict.

    Shared by `load_config` (bundled sidecar files) and model upload (a sidecar
    sent alongside the .onnx), so both honour the same defaults and back-compat.
    The sidecar JSON is a trust boundary (uploaded file) and is validated via
    Pydantic; invalid fields are logged and fall back to safe defaults rather
    than failing the upload, with strict mask_threshold bounds enforced.

    Args:
        data: Parsed sidecar JSON (possibly partial).

    Returns:
        ModelConfig with unset fields falling back to RF-DETR defaults.
    """
    defaults = ModelConfig()
    try:
        parsed = _SidecarSchema.model_validate(data)
    except ValidationError as exc:
        logger.warning("Sidecar validation failed (%s); using defaults for invalid fields: %s", exc.errors()[0]["msg"] if exc.errors() else exc, data)
        # Fall back to lenient partial parse for back-compat: coerce what we can
        parsed = _SidecarSchema.model_validate({})
        # Preserve raw rule_params/labels that are at least dict-typed
        if isinstance(data.get("rule_params"), dict):
            parsed = parsed.model_copy(update={"rule_params": {k: dict(v) for k, v in data["rule_params"].items() if isinstance(v, dict)}})
        if isinstance(data.get("labels"), dict):
            parsed = parsed.model_copy(update={"labels": {str(k): str(v) for k, v in data["labels"].items()}})

    labels = {int(k): str(v) for k, v in parsed.labels.items()}

    rule_params = {k: dict(v) for k, v in parsed.rule_params.items()}
    legacy_expected = parsed.expected_classes if parsed.expected_classes is not None else data.get("expected_classes")
    if legacy_expected and "expected_classes" not in rule_params:
        try:
            rule_params["expected_classes"] = {
                "expected_classes": [int(c) for c in legacy_expected]  # type: ignore[arg-type]
            }
        except (TypeError, ValueError):
            logger.warning("Legacy expected_classes could not be coerced to ints: %r", legacy_expected)

    return ModelConfig(
        labels=labels,
        input_size=parsed.input_size if parsed.input_size else defaults.input_size,
        mean=parsed.mean if parsed.mean else defaults.mean,
        std=parsed.std if parsed.std else defaults.std,
        normalize=parsed.normalize,
        preprocess=parsed.preprocess,
        postprocess=parsed.postprocess,
        mask_threshold=float(parsed.mask_threshold),
        default_rule=parsed.default_rule,
        rule_params=rule_params,
    )
