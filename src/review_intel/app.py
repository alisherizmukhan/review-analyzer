"""Application facade wiring the services together (used by the MCP server, usable anywhere).

Holds long-lived objects so they are built once per process:
the sentiment service (model loaded once), the aspect extractor and one
similarity index per corpus file. Configuration comes from environment
variables (see ``ServiceConfig`` and ``AppConfig``).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from review_intel.analytics.aspects import DEFAULT_LEXICON_PATH, AspectAnalyzer, LexiconAspectExtractor
from review_intel.analytics.corpus import corpus_fingerprint, load_reviews
from review_intel.analytics.stats import DEFAULT_MIN_PRODUCT_REVIEWS
from review_intel.inference.service import SentimentService, ServiceConfig
from review_intel.retrieval.semantic import SimilaritySearch, make_retriever
from review_intel.utils import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    service: ServiceConfig = field(default_factory=ServiceConfig)
    lexicon_path: Path = DEFAULT_LEXICON_PATH
    retriever: str = "lsa"
    embedding_model: str | None = None
    min_product_reviews: int = DEFAULT_MIN_PRODUCT_REVIEWS

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> AppConfig:
        env = os.environ if environ is None else environ
        lexicon = env.get("REVIEW_ANALYZER_ASPECTS", "").strip()
        lexicon_path = Path(lexicon) if lexicon else DEFAULT_LEXICON_PATH
        if not lexicon_path.is_absolute():
            lexicon_path = PROJECT_ROOT / lexicon_path
        min_reviews = int(env.get("REVIEW_ANALYZER_MIN_PRODUCT_REVIEWS", DEFAULT_MIN_PRODUCT_REVIEWS))
        if min_reviews < 1:
            raise ValueError("REVIEW_ANALYZER_MIN_PRODUCT_REVIEWS must be positive")
        return cls(service=ServiceConfig.from_env(env), lexicon_path=lexicon_path,
                   retriever=env.get("REVIEW_ANALYZER_RETRIEVER", "lsa"),
                   embedding_model=env.get("REVIEW_ANALYZER_EMBEDDING_MODEL") or None,
                   min_product_reviews=min_reviews)


class ReviewIntelligence:
    def __init__(self, config: AppConfig):
        self.config = config
        self._service: SentimentService | None = None
        self._extractor = LexiconAspectExtractor.from_file(config.lexicon_path)
        self._search_cache: dict[tuple, SimilaritySearch] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> ReviewIntelligence:
        return cls(AppConfig.from_env())

    @property
    def service(self) -> SentimentService:
        """The sentiment service, created on first use and then reused.

        A failed load (e.g. no trained model yet) is not cached, so training a
        model and calling again works without restarting the process.
        """
        with self._lock:
            if self._service is None:
                self._service = SentimentService.from_config(self.config.service)
            return self._service

    def analyzer(self) -> AspectAnalyzer:
        service = self.service
        return AspectAnalyzer(self._extractor, service.predict_texts, service.model_identity())

    def similarity_search(self, corpus_path: Path) -> SimilaritySearch:
        key = (*corpus_fingerprint(corpus_path), self.config.retriever, self.config.embedding_model)
        with self._lock:
            if key not in self._search_cache:
                reviews = load_reviews(corpus_path)
                logger.info("Building %s index over %d reviews", self.config.retriever, len(reviews))
                self._search_cache = {key: SimilaritySearch(
                    reviews, make_retriever(self.config.retriever, self.config.embedding_model))}
            return self._search_cache[key]

    @property
    def extractor(self) -> LexiconAspectExtractor:
        return self._extractor
