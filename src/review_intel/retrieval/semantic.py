"""Similarity search over the review corpus.

Two interchangeable retrievers, both ranking by cosine similarity:

* ``LsaRetriever`` (default, offline): TF-IDF + truncated SVD (latent semantic
  analysis). Captures co-occurrence-based similarity ("drains" ~ "battery
  life") without any pretrained model. Not true semantic understanding.
* ``SentenceTransformerRetriever`` (optional): dense sentence embeddings,
  e.g. ``sentence-transformers/all-MiniLM-L6-v2``, computed locally. Needs
  ``pip install -e .[semantic]``; the model is downloaded once on first use.
  An ``encoder`` callable can be injected (used by tests: no downloads).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any, Protocol

import numpy as np

from review_intel.models.base import Prediction

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# BLAS threads used for the truncated SVD. With all 16 OpenBLAS threads on the development machine (hybrid
# Intel CPU) the randomized SVD over the 9,659-review corpus took ~94 s; with 4 threads 12.9 s and the same
# vectors (max abs difference 7e-13), so the cap changes speed, not results.
LSA_BLAS_THREADS = 4


class Retriever(Protocol):
    method: str

    def fit(self, texts: Sequence[str]) -> None: ...

    def similarities(self, query: str) -> np.ndarray:
        """Cosine similarity of the query to every fitted text."""


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class LsaRetriever:
    def __init__(self, n_components: int = 200, seed: int = 42):
        self.n_components = n_components
        self.seed = seed
        self.method = "lsa"
        self._vectorizer = None
        self._svd = None
        self._doc_vectors: np.ndarray | None = None

    def fit(self, texts: Sequence[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from threadpoolctl import threadpool_limits  # installed with scikit-learn

        self._vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english", min_df=1)
        tfidf = self._vectorizer.fit_transform(texts)
        k = min(self.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        if k >= 2:
            self._svd = TruncatedSVD(n_components=k, random_state=self.seed)
            with threadpool_limits(limits=LSA_BLAS_THREADS, user_api="blas"):
                vectors = self._svd.fit_transform(tfidf)
            self.method = f"lsa (tf-idf + truncated SVD, {k} dims, cosine)"
        else:  # tiny corpus: SVD impossible, fall back to plain TF-IDF cosine
            self._svd = None
            vectors = tfidf.toarray()
            self.method = "tfidf_cosine (corpus too small for SVD)"
        self._doc_vectors = _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def similarities(self, query: str) -> np.ndarray:
        if self._doc_vectors is None:
            raise RuntimeError("Retriever is not fitted.")
        q = self._vectorizer.transform([query])
        q = self._svd.transform(q) if self._svd is not None else q.toarray()
        return self._doc_vectors @ _l2_normalize(np.asarray(q, dtype=np.float32))[0]


class SentenceTransformerRetriever:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL,
                 encoder: Callable[[Sequence[str]], np.ndarray] | None = None, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.method = f"dense embeddings ({model_name}, cosine)"
        self._encoder = encoder
        self._doc_vectors: np.ndarray | None = None

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is not installed; pip install -e .[semantic] "
                                   "or use REVIEW_ANALYZER_RETRIEVER=lsa") from exc
            model = SentenceTransformer(self.model_name)
            self._encoder = lambda batch: model.encode(list(batch), batch_size=self.batch_size,
                                                       normalize_embeddings=True, show_progress_bar=False)
        return _l2_normalize(np.asarray(self._encoder(texts), dtype=np.float32))

    def fit(self, texts: Sequence[str]) -> None:
        self._doc_vectors = self._encode(texts)

    def similarities(self, query: str) -> np.ndarray:
        if self._doc_vectors is None:
            raise RuntimeError("Retriever is not fitted.")
        return self._doc_vectors @ self._encode([query])[0]


def make_retriever(kind: str, model_name: str | None = None) -> Retriever:
    kind = kind.strip().lower()
    if kind == "lsa":
        return LsaRetriever()
    if kind in ("sentence-transformers", "sentence_transformers", "dense"):
        return SentenceTransformerRetriever(model_name or DEFAULT_EMBEDDING_MODEL)
    raise ValueError(f"Unknown retriever {kind!r}; use 'lsa' or 'sentence-transformers'.")


class SimilaritySearch:
    """A retriever fitted on a list of corpus reviews, with product filtering."""

    def __init__(self, reviews: Sequence[dict[str, Any]], retriever: Retriever):
        self.reviews = list(reviews)
        self.retriever = retriever
        if self.reviews:
            retriever.fit([r["text"] for r in self.reviews])

    def ranked(self, query: str,
               product_filter: Callable[[dict[str, Any]], bool] | None = None) -> Iterator[dict[str, Any]]:
        """Every (product-filtered) review with its similarity, most similar first (ties: corpus order)."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty or whitespace only.")
        if not self.reviews:
            return
        scores = self.retriever.similarities(query)
        for i in np.argsort(-scores, kind="stable"):
            review = self.reviews[int(i)]
            if product_filter is None or product_filter(review):
                yield {**review, "similarity": round(float(scores[i]), 4)}

    def search(self, query: str, product_filter: Callable[[dict[str, Any]], bool] | None = None,
               limit: int = 5) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query must not be empty or whitespace only.")
        return list(islice(self.ranked(query, product_filter), limit))


# ---------------------------------------------------------------------------
# Optional sentiment filter (two-stage: similarity ranking, then a sentiment filter)
# ---------------------------------------------------------------------------

SENTIMENT_FILTERS = ("any", "negative", "neutral", "positive")
FILTER_SOURCES = ("star_rating_label", "model_prediction")
DEFAULT_CANDIDATE_POOL = 200


@dataclass
class FilteredSearch:
    results: list[dict[str, Any]]
    sentiment: str
    source: str | None
    candidate_pool: int | None  # None: the whole ranking was scanned
    candidates_scanned: int
    candidates_matching: int


def search_with_sentiment(search: SimilaritySearch, query: str, *, sentiment: str = "any", source: str | None = None,
                          predict: Callable[[Sequence[str]], list[Prediction]] | None = None,
                          product_filter: Callable[[dict[str, Any]], bool] | None = None, limit: int = 5,
                          candidate_pool: int = DEFAULT_CANDIDATE_POOL) -> FilteredSearch:
    """Similarity search that keeps only reviews with the requested sentiment.

    The ranking is unchanged (the retriever's similarity); the filter only removes reviews. ``sentiment="any"``
    is exactly ``SimilaritySearch.search``. The two filter sources are never mixed:

    * ``star_rating_label``: the review's stored label (from its star rating). Every review is a candidate,
      because labels are already known, so the result is the true top-``limit`` with that label;
    * ``model_prediction``: the sentiment model's prediction for the review text. Predicting the whole corpus is
      too slow per query, so only the ``candidate_pool`` most similar reviews are classified; fewer than
      ``limit`` results means the pool held fewer matches, not that the corpus has none.
    """
    if sentiment not in SENTIMENT_FILTERS:
        raise ValueError(f"Unsupported sentiment_filter {sentiment!r}; use one of: {', '.join(SENTIMENT_FILTERS)}.")
    if sentiment == "any":
        results = search.search(query, product_filter, limit)
        return FilteredSearch(results, "any", None, None, len(results), len(results))
    if source not in FILTER_SOURCES:
        raise ValueError(f"sentiment_filter={sentiment!r} needs sentiment_source set to 'star_rating_label' "
                         "(the review's star rating, not its text) or 'model_prediction' (the sentiment model's "
                         "prediction for the review text).")
    if limit < 1 or candidate_pool < 1:
        raise ValueError("limit and candidate_pool must be >= 1")

    if source == "star_rating_label":
        results, scanned, matching = [], 0, 0
        for review in search.ranked(query, product_filter):
            scanned += 1
            if review["sentiment"] == sentiment:
                matching += 1
                results.append(review)
                if len(results) == limit:
                    break
        return FilteredSearch(results, sentiment, source, None, scanned, matching)

    if predict is None:
        raise ValueError("model_prediction filtering needs a sentiment model (predict function).")
    pool = list(islice(search.ranked(query, product_filter), candidate_pool))
    predictions = predict([r["text"] for r in pool]) if pool else []
    matching = [{**r, "predicted_sentiment": p.label, "predicted_score": round(float(p.scores.get(p.label, 0.0)), 6)}
                for r, p in zip(pool, predictions) if p.label == sentiment]
    return FilteredSearch(matching[:limit], sentiment, source, candidate_pool, len(pool), len(matching))
