"""Keyword-counting sentiment baseline (the original v1 demo logic, unchanged).

This is NOT machine learning. It is kept as (1) a transparent reference
point and (2) an explicitly configured demo fallback. Known limitations:
negation ("not good" counts as positive), sarcasm and context are ignored;
the vocabulary is tiny, English-only, and every keyword has equal weight.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal, TypedDict

from review_intel.data.labels import LABELS
from review_intel.models.base import Prediction

Sentiment = Literal["positive", "negative", "neutral"]

POSITIVE_WORDS = frozenset(
    {
        "amazing", "awesome", "best", "excellent", "fantastic", "fast", "good", "great",
        "happy", "love", "loved", "perfect", "quick", "recommend", "reliable", "satisfied",
    }
)
NEGATIVE_WORDS = frozenset(
    {
        "awful", "bad", "blurry", "broke", "broken", "disappointed", "disappointing", "drains",
        "hate", "late", "poor", "slow", "terrible", "useless", "worse", "worst",
    }
)

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

BASELINE_WARNING = (
    "Demonstration baseline: keyword counting, not a trained NLP model. "
    "It ignores negation, sarcasm and context, so predictions can be wrong."
)


class SentimentPrediction(TypedDict):
    sentiment: Sentiment
    positive_matches: list[str]
    negative_matches: list[str]


class KeywordSentimentClassifier:
    """More positive hits -> positive, more negative hits -> negative, tie (incl. zero) -> neutral."""

    name = "keyword_baseline_v1"

    def predict(self, text: str) -> SentimentPrediction:
        words = _WORD_RE.findall(text.lower())
        positive = [w for w in words if w in POSITIVE_WORDS]
        negative = [w for w in words if w in NEGATIVE_WORDS]
        if len(positive) > len(negative):
            sentiment: Sentiment = "positive"
        elif len(negative) > len(positive):
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return {"sentiment": sentiment, "positive_matches": positive, "negative_matches": negative}


class KeywordSentimentModel:
    """Adapter exposing the keyword baseline through the common ``SentimentModel`` interface."""

    name = KeywordSentimentClassifier.name
    version = "1.0"
    model_type = "keyword_baseline"
    labels = LABELS
    score_type = "keyword_count"
    is_trained_model = False

    def __init__(self) -> None:
        self._classifier = KeywordSentimentClassifier()

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        out = []
        for text in texts:
            p = self._classifier.predict(text)
            out.append(Prediction(
                label=p["sentiment"],
                scores={"negative": float(len(p["negative_matches"])), "neutral": 0.0,
                        "positive": float(len(p["positive_matches"]))},
                details={"positive_matches": p["positive_matches"], "negative_matches": p["negative_matches"]},
            ))
        return out

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name, "version": self.version, "model_type": self.model_type,
            "is_trained_model": False, "labels": list(self.labels), "score_type": self.score_type,
            "warning": BASELINE_WARNING,
        }


class KeywordSentimentResult(TypedDict):
    text: str
    sentiment: Sentiment
    method: str
    positive_matches: list[str]
    negative_matches: list[str]
    warning: str


def analyze_text_sentiment(text: str) -> KeywordSentimentResult:
    """v1 API: keyword-baseline prediction for one text. Raises ``ValueError`` for empty input."""
    if not text or not text.strip():
        raise ValueError("Text must not be empty or whitespace only.")
    prediction = KeywordSentimentClassifier().predict(text)
    return {
        "text": text,
        "sentiment": prediction["sentiment"],
        "method": KeywordSentimentClassifier.name,
        "positive_matches": prediction["positive_matches"],
        "negative_matches": prediction["negative_matches"],
        "warning": BASELINE_WARNING,
    }
