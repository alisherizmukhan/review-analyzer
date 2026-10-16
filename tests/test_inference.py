"""Inference service: output schema, validation, explicit backends, missing artifacts, caching."""

from __future__ import annotations

import pytest

from review_intel.inference.service import ServiceConfig, SentimentService
from review_intel.models.base import SCORE_TYPES, ModelNotAvailableError


@pytest.fixture
def trained_service(trained):
    models_dir, _, _ = trained
    return SentimentService.from_config(ServiceConfig(model_dir=models_dir / "tfidf_logreg", max_text_chars=200))


def test_prediction_output_schema(trained_service):
    result = trained_service.analyze("Fast performance, apps open instantly")
    assert set(result) == {"text", "sentiment", "scores", "score_type", "score_description", "model",
                           "processing", "warning", "details"}
    assert result["sentiment"] in ("positive", "negative", "neutral")
    assert result["score_type"] == "probability"
    assert result["score_description"] == SCORE_TYPES["probability"]
    assert result["model"]["is_trained_model"] is True and result["model"]["name"] == "tfidf_logreg"
    assert result["warning"] is None
    assert result["processing"]["characters"] == len(result["text"])


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_input_rejected(trained_service, text):
    with pytest.raises(ValueError, match="empty"):
        trained_service.analyze(text)


def test_too_long_input_rejected(trained_service):
    with pytest.raises(ValueError, match="maximum is 200"):
        trained_service.analyze("good " * 100)


def test_keyword_backend_must_be_explicit_and_is_labelled():
    service = SentimentService.from_config(ServiceConfig(backend="keyword"))
    result = service.analyze("great and fast")
    assert result["model"]["is_trained_model"] is False
    assert "not a trained NLP model" in result["warning"]
    assert result["score_type"] == "keyword_count"


def test_trained_backend_without_artifact_does_not_fall_back(tmp_path):
    config = ServiceConfig(backend="trained", selected_pointer=tmp_path / "selected.json")
    with pytest.raises(ModelNotAvailableError, match="REVIEW_ANALYZER_SENTIMENT_BACKEND=keyword"):
        SentimentService.from_config(config)


def test_config_from_env():
    config = ServiceConfig.from_env({"REVIEW_ANALYZER_SENTIMENT_BACKEND": "Keyword",
                                     "REVIEW_ANALYZER_MAX_TEXT_CHARS": "100"})
    assert config.backend == "keyword" and config.max_text_chars == 100
    with pytest.raises(ValueError, match="must be one of"):
        ServiceConfig.from_env({"REVIEW_ANALYZER_SENTIMENT_BACKEND": "gpt"})
    with pytest.raises(ValueError, match="positive"):
        ServiceConfig.from_env({"REVIEW_ANALYZER_MAX_TEXT_CHARS": "0"})


def test_batch_predictions_are_memoized(trained_service):
    calls = []
    original = trained_service.model.predict
    trained_service.model.predict = lambda texts: calls.append(list(texts)) or original(texts)
    trained_service.predict_texts(["a good one", "a bad one", "a good one"])
    trained_service.predict_texts(["a good one"])
    assert calls == [["a good one", "a bad one"]]
