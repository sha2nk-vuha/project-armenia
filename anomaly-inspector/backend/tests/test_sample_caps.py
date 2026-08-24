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
