"""Cascade model sourcing: stages read the per-Feature store; a missing model is
a configuration error; a Feature shared by two stages shares its one model.
"""
from unittest.mock import MagicMock

import pytest

import inference.features as features
from config import ANOMALY_FEATURE, SEGMENTATION_FEATURE
from inference.engine import ModelSession
from inference.model_config import ModelConfig


@pytest.fixture(autouse=True)
def _clean():
    features.reset()
    yield
    features.reset()


def _session(version="v"):
    return ModelSession(
        session=MagicMock(), runtime="cpu", input_name="input",
        input_shape=(32, 32), model_version=version,
    )


def _install(feature, labels=None):
    features._models[feature] = (
        _session(feature),
        ModelConfig(labels=labels or {0: "bottle_cap", 1: "logo"}),
    )


def _spec(*features_in, combinator="and"):
    return {
        "combinator": combinator,
        "stages": [{"feature": f, "rule": None, "threshold": 0.5} for f in features_in],
    }


def test_stages_use_the_uploaded_model_for_their_feature():
    _install(ANOMALY_FEATURE)
    _install(SEGMENTATION_FEATURE)
    pipeline = features.build_cascade_pipeline(_spec(ANOMALY_FEATURE, SEGMENTATION_FEATURE))
    assert len(pipeline.stages) == 2
    assert pipeline.stages[0].pipeline.model.model_version == ANOMALY_FEATURE


def test_a_feature_shared_by_two_stages_shares_its_one_model():
    _install(SEGMENTATION_FEATURE)
    spec = {
        "combinator": "and",
        "stages": [
            {"feature": SEGMENTATION_FEATURE, "rule": "concentricity", "threshold": 0.5},
            {"feature": SEGMENTATION_FEATURE, "rule": "expected_classes", "threshold": 0.5},
        ],
    }
    pipeline = features.build_cascade_pipeline(spec)
    assert len(pipeline.stages) == 2
    # One uploaded model, two rules -> the same live session in both stages.
    assert pipeline.stages[0].pipeline.model is pipeline.stages[1].pipeline.model


def test_a_stage_without_an_uploaded_model_is_a_config_error():
    _install(ANOMALY_FEATURE)  # segmentation deliberately not uploaded
    with pytest.raises(ValueError, match="no model loaded for 'segmentation'"):
        features.build_cascade_pipeline(_spec(ANOMALY_FEATURE, SEGMENTATION_FEATURE))


def test_unknown_feature_in_spec_is_a_config_error():
    with pytest.raises(ValueError, match="invalid feature"):
        features.build_cascade_pipeline(_spec("not_a_feature"))


def test_incompatible_rule_for_stage_is_a_config_error():
    _install(SEGMENTATION_FEATURE)
    spec = {
        "combinator": "and",
        "stages": [{"feature": SEGMENTATION_FEATURE, "rule": "anomaly_threshold", "threshold": 0.5}],
    }
    with pytest.raises(ValueError):
        features.build_cascade_pipeline(spec)


def test_empty_stages_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        features.build_cascade_pipeline({"combinator": "and", "stages": []})


def test_reset_frees_all_models():
    _install(ANOMALY_FEATURE)
    _install(SEGMENTATION_FEATURE)
    assert features.loaded_models()
    features.reset()
    assert features.loaded_models() == {}
