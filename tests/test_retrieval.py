"""Similarity search: offline LSA retriever and the dense retriever with an injected encoder (no download)."""

from __future__ import annotations

import numpy as np
import pytest

from review_intel.retrieval.semantic import LsaRetriever, SentenceTransformerRetriever, SimilaritySearch, make_retriever

CORPUS = [
    {"id": 1, "product": "A", "text": "The battery drains quickly and charging takes hours", "sentiment": "negative"},
    {"id": 2, "product": "A", "text": "Gorgeous bright screen with vivid colors", "sentiment": "positive"},
    {"id": 3, "product": "B", "text": "Battery life is short, it needs charging twice a day", "sentiment": "negative"},
    {"id": 4, "product": "B", "text": "Arrived late and the box was crushed", "sentiment": "negative"},
    {"id": 5, "product": "B", "text": "Colors on the screen look washed out", "sentiment": "negative"},
]


def test_lsa_ranks_relevant_reviews_first():
    search = SimilaritySearch(CORPUS, LsaRetriever(n_components=3))
    results = search.search("battery charging problems", limit=2)
    assert {r["id"] for r in results} == {1, 3}
    assert results[0]["similarity"] >= results[1]["similarity"]


def test_product_filter_and_limit():
    search = SimilaritySearch(CORPUS, LsaRetriever(n_components=3))
    results = search.search("screen colors", product_filter=lambda r: r["product"] == "B", limit=5)
    assert results and all(r["product"] == "B" for r in results)
    assert results[0]["id"] == 5


def test_tiny_corpus_falls_back_to_tfidf():
    retriever = LsaRetriever()
    SimilaritySearch(CORPUS[:2], retriever)
    assert retriever.method.startswith("tfidf_cosine")


def test_empty_query_rejected():
    with pytest.raises(ValueError, match="empty"):
        SimilaritySearch(CORPUS, LsaRetriever(n_components=3)).search("  ")


def test_dense_retriever_with_injected_encoder():
    vocab = ["battery", "screen", "box"]

    def encoder(texts):  # bag-of-keywords "embedding": deterministic, no model download
        return np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=float)

    search = SimilaritySearch(CORPUS, SentenceTransformerRetriever("fake", encoder=encoder))
    assert search.search("battery", limit=1)[0]["id"] in (1, 3)
    assert "fake" in search.retriever.method


def test_make_retriever():
    assert isinstance(make_retriever("lsa"), LsaRetriever)
    assert isinstance(make_retriever("sentence-transformers"), SentenceTransformerRetriever)
    with pytest.raises(ValueError):
        make_retriever("bm42")


# --- optional sentiment filter (two-stage) ---------------------------------------

from review_intel.models.base import Prediction  # noqa: E402
from review_intel.retrieval.semantic import search_with_sentiment  # noqa: E402

POLARITY_CORPUS = [
    {"id": 1, "product": "A", "text": "The battery drains quickly and dies by noon", "sentiment": "negative", "rating": 1.0},
    {"id": 2, "product": "A", "text": "Battery lasts forever, great battery life", "sentiment": "positive", "rating": 5.0},
    {"id": 3, "product": "B", "text": "Battery life is okay, average battery", "sentiment": "neutral", "rating": 3.0},
    # Star label and text disagree: rated 5 stars, but the text is a complaint.
    {"id": 4, "product": "B", "text": "The battery drains quickly, it stopped charging", "sentiment": "positive", "rating": 5.0},
    {"id": 5, "product": "B", "text": "Gorgeous bright screen with vivid colors", "sentiment": "positive", "rating": 5.0},
]


def fake_predict(texts):
    """Deterministic stand-in model: complaint words -> negative, praise words -> positive, else neutral."""
    out = []
    for text in texts:
        low = text.lower()
        label = ("negative" if any(w in low for w in ("drains", "dies", "stopped")) else
                 "positive" if any(w in low for w in ("great", "forever", "gorgeous")) else "neutral")
        out.append(Prediction(label, {label: 0.9, **{k: 0.05 for k in ("negative", "neutral", "positive") if k != label}}))
    return out


@pytest.fixture
def polarity_search():
    return SimilaritySearch(POLARITY_CORPUS, LsaRetriever(n_components=3))


def test_filter_any_is_exactly_the_old_search(polarity_search):
    found = search_with_sentiment(polarity_search, "battery drains", limit=3)
    assert found.results == polarity_search.search("battery drains", limit=3)
    assert found.source is None and found.candidate_pool is None
    assert all("predicted_sentiment" not in r for r in found.results)


def test_star_label_filter_excludes_other_labels(polarity_search):
    negative = search_with_sentiment(polarity_search, "battery", sentiment="negative", source="star_rating_label",
                                     limit=5).results
    assert [r["id"] for r in negative] == [1]
    positive = search_with_sentiment(polarity_search, "battery", sentiment="positive", source="star_rating_label",
                                     limit=5).results
    assert positive and all(r["sentiment"] == "positive" for r in positive)
    assert 1 not in {r["id"] for r in positive}


def test_model_filter_uses_text_predictions_not_star_labels(polarity_search):
    found = search_with_sentiment(polarity_search, "battery drains quickly", sentiment="negative",
                                  source="model_prediction", predict=fake_predict, limit=5)
    ids = {r["id"] for r in found.results}
    assert ids == {1, 4}  # review 4 is rated 5 stars but its text is a complaint
    assert all(r["predicted_sentiment"] == "negative" and r["predicted_score"] == 0.9 for r in found.results)
    star = search_with_sentiment(polarity_search, "battery drains quickly", sentiment="negative",
                                 source="star_rating_label", limit=5)
    assert {r["id"] for r in star.results} == {1}  # the two sources give different answers, kept apart
    positive = search_with_sentiment(polarity_search, "battery", sentiment="positive", source="model_prediction",
                                     predict=fake_predict, limit=5).results
    assert positive and all(r["predicted_sentiment"] == "positive" for r in positive)
    assert not {1, 4} & {r["id"] for r in positive}


def test_ranking_order_is_preserved_after_filtering(polarity_search):
    ranked = [r["id"] for r in polarity_search.search("battery drains quickly", limit=5)]
    found = search_with_sentiment(polarity_search, "battery drains quickly", sentiment="negative",
                                  source="model_prediction", predict=fake_predict, limit=5)
    assert [r["id"] for r in found.results] == [i for i in ranked if i in {1, 4}]


def test_candidate_pool_bounds_model_filtering(polarity_search):
    calls = []

    def counting_predict(texts):
        calls.append(len(texts))
        return fake_predict(texts)

    found = search_with_sentiment(polarity_search, "screen colors", sentiment="negative", source="model_prediction",
                                  predict=counting_predict, limit=5, candidate_pool=2)
    assert calls == [2] and found.candidates_scanned == 2 and found.candidate_pool == 2


def test_empty_results_are_reported(polarity_search):
    found = search_with_sentiment(polarity_search, "battery", sentiment="neutral", source="star_rating_label",
                                  product_filter=lambda r: r["product"] == "A", limit=5)
    assert found.results == [] and found.candidates_matching == 0 and found.candidates_scanned == 2
    empty = SimilaritySearch([], LsaRetriever())
    assert search_with_sentiment(empty, "battery", sentiment="negative", source="model_prediction",
                                 predict=fake_predict).results == []


@pytest.mark.parametrize("kwargs, message", [
    ({"sentiment": "angry", "source": "model_prediction"}, "Unsupported sentiment_filter"),
    ({"sentiment": "negative"}, "needs sentiment_source"),
    ({"sentiment": "negative", "source": "stars"}, "needs sentiment_source"),
    ({"sentiment": "negative", "source": "model_prediction"}, "needs a sentiment model"),
])
def test_invalid_filter_arguments_are_rejected(polarity_search, kwargs, message):
    with pytest.raises(ValueError, match=message):
        search_with_sentiment(polarity_search, "battery", **kwargs)
