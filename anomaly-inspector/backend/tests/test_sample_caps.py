"""Ground-truth regression over the labelled sample caps.

This is the test that guards the thing the Feature exists to do: separate good
placement from bad across every SKU with one threshold. It loads the real ONNX
and runs the real pipeline, so it is slow (~30s, dominated by model load) and is
opt-in:

    INSPECT_SAMPLES=1 pytest tests/test_sample_caps.py

Ground truth lives in the directory names under data/three_cee_caps/test/
(`ok_case` / `nok_case`), so adding samples needs no change here. As more caps
arrive this is where a tuned threshold gets validated -- or falsified.
"""
import glob
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("INSPECT_SAMPLES") != "1",
    reason="slow: set INSPECT_SAMPLES=1 to run against the real model",
)

_SAMPLES = Path(__file__).resolve().parents[3] / "data" / "three_cee_caps" / "test"


def _upload_bundled(feature):
    """Upload a Feature's bundled model + sidecar into the per-Feature store.

    Nothing auto-loads (docs/adr/0007), so these real-model tests upload the
    bundled files the same way the GUI uploads an operator's model.
    """
    import json as _json

    from config import FEATURES
    from inference import features
    from inference.model_config import sidecar_path

    path = FEATURES[feature]["model_path"]
    sidecar_path_str = sidecar_path(str(path))
    sidecar = None
    if Path(sidecar_path_str).is_file():
        sidecar = _json.loads(Path(sidecar_path_str).read_text())
    features.upload_model(
        feature, path.read_bytes(), path.name, FEATURES[feature]["model_version"], sidecar
    )


def _labelled():
    for folder, expected in (("ok_case", "ok"), ("nok_case", "not_ok")):
        for path in sorted(glob.glob(str(_SAMPLES / folder / "*.png"))):
            yield path, expected


@pytest.fixture(scope="module")
def pipeline():
    from config import FEATURES, SEGMENTATION_FEATURE
    from inference import features

    if not FEATURES[SEGMENTATION_FEATURE]["model_path"].exists():
        pytest.skip("segmentation model not present")
    if not _SAMPLES.is_dir():
        pytest.skip("sample caps not present")
    features.reset()
    _upload_bundled(SEGMENTATION_FEATURE)
    features.activate(SEGMENTATION_FEATURE)
    return features.current_pipeline()


def test_every_labelled_cap_gets_the_right_verdict(pipeline):
    """One rule, one threshold, all SKUs -- using the sidecar's own defaults, so
    this fails if the shipped configuration drifts away from the ground truth."""
    wrong = []
    for path, expected in _labelled():
        result = pipeline.infer(
            Path(path).read_bytes(), threshold=0.5, rule_name="concentricity"
        )
        if result.verdict != expected:
            wrong.append(
                f"{os.path.basename(path)}: expected {expected}, got "
                f"{result.verdict} (score {result.score:.4f}) -- {result.reason}"
            )
    assert not wrong, "misclassified:\n  " + "\n  ".join(wrong)


def test_a_separating_threshold_still_exists(pipeline):
    """Guards the margin, not just the verdicts: reports the window a universal
    threshold has to sit in. A shrinking window is the early warning that the
    estimator is losing its grip before any verdict actually flips."""
    scores = {"ok": [], "not_ok": []}
    for path, expected in _labelled():
        result = pipeline.infer(
            Path(path).read_bytes(), threshold=0.5, rule_name="concentricity"
        )
        scores[expected].append(result.metrics["offset_ratio"])

    worst_ok, best_nok = max(scores["ok"]), min(scores["not_ok"])
    assert worst_ok < best_nok, (
        f"no universal threshold separates these samples: worst OK {worst_ok:.4f} "
        f">= best NOK {best_nok:.4f}. Per-SKU calibration (nominal_offset) is "
        "then required."
    )


def test_cascade_anomaly_and_segmentation_over_the_samples():
    """The headline case: anomaly AND segmentation as a cascade, over the real
    models. Segmentation's concentricity verdict must drive the cascade when
    anomaly passes, and any anomaly NOK must force the cascade NOK."""
    import json as _json

    from config import ANOMALY_FEATURE, FEATURES, SEGMENTATION_FEATURE
    from inference import features

    for feat in (ANOMALY_FEATURE, SEGMENTATION_FEATURE):
        if not FEATURES[feat]["model_path"].exists():
            pytest.skip(f"{feat} model not present")
    if not _SAMPLES.is_dir():
        pytest.skip("sample caps not present")

    features.reset()
    _upload_bundled(ANOMALY_FEATURE)
    _upload_bundled(SEGMENTATION_FEATURE)
    spec = {
        "combinator": "and",
        "short_circuit": False,  # evaluate both stages so we can inspect each
        "stages": [
            {"feature": ANOMALY_FEATURE, "rule": "anomaly_threshold", "threshold": 0.95},
            {"feature": SEGMENTATION_FEATURE, "rule": "concentricity", "threshold": 0.5,
             "params": {"target_center_method": "outer_circle_fit", "max_offset_ratio": 0.06}},
        ],
    }
    pipeline = features.build_cascade_pipeline(_json.loads(_json.dumps(spec)))

    from pathlib import Path
    for path, seg_expected in _labelled():
        result = pipeline.infer(Path(path).read_bytes(), 0.5)
        assert len(result.stages) == 2
        anomaly_stage, seg_stage = result.stages
        assert anomaly_stage.feature == ANOMALY_FEATURE
        assert seg_stage.feature == SEGMENTATION_FEATURE
        # Cascade verdict is the AND of the two stages.
        combined = "ok" if anomaly_stage.verdict == "ok" and seg_stage.verdict == "ok" else "not_ok"
        assert result.verdict == combined
        # When anomaly passes, segmentation's concentricity drives the result.
        if anomaly_stage.verdict == "ok":
            assert result.verdict == seg_stage.verdict == seg_expected

    features.reset()
