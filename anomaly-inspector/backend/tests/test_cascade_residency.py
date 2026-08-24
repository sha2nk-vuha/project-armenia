"""Cascade model residency: reconcile to the spec, share models, evict the rest.

ADR 0006 revises ADR 0001's single-resident rule for cascades. These tests use a
stub session builder so no ONNX is loaded.
"""
from unittest.mock import MagicMock, patch

import pytest

import inference.features as features
from config import ANOMALY_FEATURE, PRESENCE_FEATURE, SEGMENTATION_FEATURE
from inference.engine import ModelSession
from inference.model_config import ModelConfig


@pytest.fixture(autouse=True)
def _clean():
    features.reset()
    yield
    features.reset()


def _fake_session(version):
    return ModelSession(
        session=MagicMock(), runtime="cpu", input_name="input",
        input_shape=(32, 32), model_version=version,
    )


def _spec(*features_in, combinator="and"):
    return {
        "combinator": combinator,
        "stages": [{"feature": f, "rule": None, "threshold": 0.5} for f in features_in],
    }


def _build(spec):
    # Stub session building and sidecar loading; every model path "exists".
    with patch("inference.features.engine.build_session_from_path",
               side_effect=lambda path, ver: _fake_session(ver)) as build, \
         patch("inference.features._model_path_for", side_effect=lambda f: f"/models/{f}.onnx"), \
         patch("inference.features.load_config", return_value=ModelConfig(labels={0: "bottle_cap", 1: "logo"})):
        pipeline = features.build_cascade_pipeline(spec)
    return pipeline, build


def test_builds_a_session_per_distinct_model():
    pipeline, build = _build(_spec(ANOMALY_FEATURE, SEGMENTATION_FEATURE))
    assert len(pipeline.stages) == 2
    assert build.call_count == 2


def test_a_model_shared_by_two_stages_loads_once():
    # Segmentation twice with different rules -> one session, two stages.
    spec = {
        "combinator": "and",
        "stages": [
            {"feature": SEGMENTATION_FEATURE, "rule": "concentricity", "threshold": 0.5},
            {"feature": SEGMENTATION_FEATURE, "rule": "expected_classes", "threshold": 0.5},
        ],
    }
    pipeline, build = _build(spec)
    assert len(pipeline.stages) == 2
    assert build.call_count == 1, "the shared model must load only once"
    # Both stages share the same live session.
    assert pipeline.stages[0].pipeline.model is pipeline.stages[1].pipeline.model


def test_unreferenced_members_are_evicted_on_reconcile():
    _build(_spec(ANOMALY_FEATURE, SEGMENTATION_FEATURE))
    assert len(features._cascade_sessions) == 2

    # A new spec referencing only segmentation must evict the anomaly session.
    _build(_spec(SEGMENTATION_FEATURE))
    assert set(features._cascade_sessions) == {"/models/segmentation.onnx"}


def test_rebuilding_the_same_spec_reuses_sessions():
    _build(_spec(ANOMALY_FEATURE, SEGMENTATION_FEATURE))
    _, build = _build(_spec(ANOMALY_FEATURE, SEGMENTATION_FEATURE))
    # Second build loads nothing new.
    assert build.call_count == 0


def test_unknown_feature_in_spec_is_a_config_error():
    with pytest.raises(ValueError, match="invalid feature"):
        _build(_spec("not_a_feature"))


def test_incompatible_rule_for_stage_is_a_config_error():
    # anomaly_threshold consumes an anomaly map; segmentation cannot feed it.
    spec = {
        "combinator": "and",
        "stages": [{"feature": SEGMENTATION_FEATURE, "rule": "anomaly_threshold", "threshold": 0.5}],
    }
    with pytest.raises(ValueError):
        _build(spec)


def test_empty_stages_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        _build({"combinator": "and", "stages": []})


def test_leaving_cascade_mode_frees_members():
    _build(_spec(ANOMALY_FEATURE, SEGMENTATION_FEATURE))
    assert features._cascade_sessions
    features.reset()
    assert not features._cascade_sessions
