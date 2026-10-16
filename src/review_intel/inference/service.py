"""Sentiment inference service: one stable interface over any ``SentimentModel``.

Independent of MCP. The model is loaded once when the service is created;
predictions on repeated texts (common in analytics) are memoized.

Configuration (environment variables, all optional):
    REVIEW_ANALYZER_SENTIMENT_BACKEND   "trained" (default) or "keyword" (explicit demo fallback)
    REVIEW_ANALYZER_MODEL_DIR           artifact dir; default: the model named in artifacts/models/selected.json
    REVIEW_ANALYZER_MAX_TEXT_CHARS      max accepted input length (default 5000)
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from review_intel.models.artifacts import SELECTED_POINTER, load_model, resolve_selected_model_dir
from review_intel.models.base import SCORE_TYPES, Prediction, SentimentModel
from review_intel.models.keyword import BASELINE_WARNING, KeywordSentimentModel
from review_intel.utils import PROJECT_ROOT, relative_to_root

logger = logging.getLogger(__name__)

DEFAULT_MAX_TEXT_CHARS = 5000
BACKENDS = ("trained", "keyword")


@dataclass
class ServiceConfig:
    backend: str = "trained"
    model_dir: Path | None = None
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS
    selected_pointer: Path = SELECTED_POINTER

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> ServiceConfig:
        env = os.environ if environ is None else environ
        backend = env.get("REVIEW_ANALYZER_SENTIMENT_BACKEND", "trained").strip().lower()
        if backend not in BACKENDS:
            raise ValueError(f"REVIEW_ANALYZER_SENTIMENT_BACKEND must be one of {BACKENDS}, got {backend!r}")
        model_dir = env.get("REVIEW_ANALYZER_MODEL_DIR", "").strip()
        path = Path(model_dir) if model_dir else None
        if path is not None and not path.is_absolute():
            path = PROJECT_ROOT / path
        max_chars = int(env.get("REVIEW_ANALYZER_MAX_TEXT_CHARS", DEFAULT_MAX_TEXT_CHARS))
        if max_chars < 1:
            raise ValueError("REVIEW_ANALYZER_MAX_TEXT_CHARS must be positive")
        return cls(backend=backend, model_dir=path, max_text_chars=max_chars)


class SentimentService:
    def __init__(self, model: SentimentModel, max_text_chars: int = DEFAULT_MAX_TEXT_CHARS, cache_size: int = 100_000):
        self.model = model
        self.max_text_chars = max_text_chars
        self._cache: OrderedDict[str, Prediction] = OrderedDict()
        self._cache_size = cache_size

    @classmethod
    def from_config(cls, config: ServiceConfig) -> SentimentService:
        """Build the service. Raises ``ModelNotAvailableError`` with instructions if no trained model exists."""
        if config.backend == "keyword":
            logger.warning("Sentiment backend explicitly set to the keyword DEMO baseline (not a trained model)")
            return cls(KeywordSentimentModel(), config.max_text_chars)
        model_dir = config.model_dir or resolve_selected_model_dir(config.selected_pointer)
        started = time.perf_counter()
        model = load_model(model_dir)
        model.predict(["warm-up"])  # one-off kernel/JIT warm-up so the first real call is not slow
        logger.info("Loaded model %s from %s in %.2fs", model.version, relative_to_root(model_dir),
                    time.perf_counter() - started)
        return cls(model, config.max_text_chars)

    # -- validation ---------------------------------------------------------

    def validate_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must not be empty or whitespace only.")
        if len(text) > self.max_text_chars:
            raise ValueError(f"Text is {len(text)} characters long; the maximum is {self.max_text_chars}. "
                             "Shorten it or analyze it in parts.")
        return text

    # -- prediction ---------------------------------------------------------

    def predict_texts(self, texts: Sequence[str]) -> list[Prediction]:
        """Batch prediction for internal use (analytics). Memoized; no length limit."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for text, prediction in zip(missing, self.model.predict(missing)):
                self._cache[text] = prediction
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return [self._cache[t] for t in texts]

    def analyze(self, text: str) -> dict[str, Any]:
        """Validated single-text prediction with scores, model identity and processing metadata."""
        self.validate_text(text)
        started = time.perf_counter()
        prediction = self.model.predict([text])[0]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "text": text,
            "sentiment": prediction.label,
            "scores": prediction.scores,
            "score_type": self.model.score_type,
            "score_description": SCORE_TYPES[self.model.score_type],
            "model": self.model_identity(),
            "processing": {"latency_ms": round(elapsed_ms, 3), "characters": len(text)},
            "warning": None if self.model.is_trained_model else BASELINE_WARNING,
            "details": prediction.details,
        }

    def model_identity(self) -> dict[str, Any]:
        return {"name": self.model.name, "version": self.model.version, "model_type": self.model.model_type,
                "is_trained_model": self.model.is_trained_model}

    def info(self) -> dict[str, Any]:
        return {**self.model.info(), "max_text_chars": self.max_text_chars}
