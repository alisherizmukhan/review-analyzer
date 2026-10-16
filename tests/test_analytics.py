"""Aspect extraction, aspect aggregation, corpus loading, product summaries and comparison."""

from __future__ import annotations

import pytest

from review_intel.analytics.aspects import (
    METHOD_LIMITATIONS,
    AspectAnalyzer,
    LexiconAspectExtractor,
    extract_mentions,
    top_aspects,
)
from review_intel.analytics.corpus import ReviewDataError, load_reviews
from review_intel.analytics.stats import (
    UnknownProductError,
    compare_products,
    compute_statistics,
    filter_by_product,
    label_source,
    list_products,
    negative_aspects,
    prediction_source,
    product_overview,
    product_summary,
    sample_assessment,
    wilson_interval,
)
from review_intel.models.base import Prediction
from review_intel.models.keyword import KeywordSentimentModel


@pytest.fixture
def extractor():
    return LexiconAspectExtractor({"battery": ["battery", "battery life", "charging"], "screen": ["screen", "display"],
                                   "price": ["price", "value"]}, version="test")


def test_extraction_uses_word_boundaries_and_multiword_terms(extractor):
    assert extractor.find("The battery life is great") == [("battery", "battery life")]
    assert extractor.find("Took a SCREENSHOT of the display") == [("screen", "display")]
    assert extractor.find("Screenshots everywhere") == []
    assert extractor.find("Charging is slow and the price is high") == [("battery", "charging"), ("price", "price")]


def test_default_lexicon_loads():
    ex = LexiconAspectExtractor.from_file()
    assert {"battery", "camera", "screen", "performance", "price", "design", "delivery"} <= set(ex.aspects)


def test_invalid_lexicon_rejected():
    with pytest.raises(ValueError):
        LexiconAspectExtractor({})
    with pytest.raises(ValueError):
        LexiconAspectExtractor({"battery": []})


REVIEWS = [
    {"id": 1, "product": "A", "text": "Great screen. The battery is terrible.", "sentiment": "neutral", "rating": 3.0},
    {"id": 2, "product": "A", "text": "Battery is terrible and charging is slow!", "sentiment": "negative", "rating": 1.0},
    {"id": 3, "product": "A", "text": "Good value. Great display.", "sentiment": "positive", "rating": 5.0},
    {"id": 4, "product": "B", "text": "The screen is bad. Price is good.", "sentiment": "neutral", "rating": 3.0,
     "product_name": "Bravo Tablet"},
]


def fake_predict(texts):
    """Deterministic stand-in model: 'terrible'/'bad'/'slow' -> negative, 'great'/'good' -> positive."""
    out = []
    for t in texts:
        low = t.lower()
        if any(w in low for w in ("terrible", "bad", "slow")):
            out.append(Prediction("negative", {"negative": 0.9, "neutral": 0.05, "positive": 0.05}))
        elif any(w in low for w in ("great", "good")):
            out.append(Prediction("positive", {"negative": 0.05, "neutral": 0.05, "positive": 0.9}))
        else:
            out.append(Prediction("neutral", {"negative": 0.1, "neutral": 0.8, "positive": 0.1}))
    return out


@pytest.fixture
def analyzer(extractor):
    return AspectAnalyzer(extractor, fake_predict, {"name": "fake", "version": "0", "is_trained_model": True})


def test_mentions_are_sentence_level(extractor):
    mentions = extract_mentions(REVIEWS[:1], extractor)
    assert [(m.aspect, m.sentence) for m in mentions] == [("screen", "Great screen."),
                                                           ("battery", "The battery is terrible.")]


def test_aspect_aggregation_counts_and_evidence(analyzer):
    result = analyzer.analyze(REVIEWS)
    battery = result["aspects"]["battery"]
    assert battery["reviews_mentioning"] == 2 and battery["mentions"] == 2
    assert battery["sentiment_counts"] == {"positive": 0, "neutral": 0, "negative": 2}
    assert battery["negative_share"] == 1.0
    assert battery["examples"]["negative"][0]["review_id"] in (1, 2)
    screen = result["aspects"]["screen"]
    assert screen["sentiment_counts"] == {"positive": 2, "neutral": 0, "negative": 1}
    assert result["method"]["limitations"] == METHOD_LIMITATIONS
    assert [a["aspect"] for a in top_aspects(result, 2, by="negative")] == ["battery", "screen"]


def test_product_summary_uses_real_counts(analyzer):
    summary = product_summary(REVIEWS, "a", analyzer)
    assert summary["total_reviews"] == 3
    assert summary["average_rating"] == 3.0
    assert summary["rating_distribution"] == {"1": 1, "2": 0, "3": 1, "4": 0, "5": 1}
    assert summary["sentiment_distribution"]["negative"] == {"count": 1, "percent": 33.33}
    assert summary["most_negative_aspects"][0]["aspect"] == "battery"
    assert summary["representative_reviews"]["positive"][0]["review_id"] == 3


def test_product_matching_by_name(analyzer):
    assert product_summary(REVIEWS, "bravo tablet", analyzer)["product"] == "B"


def test_unknown_product_raises(analyzer):
    with pytest.raises(UnknownProductError, match="list_products"):
        product_summary(REVIEWS, "Z", analyzer)


def test_negative_aspects_whole_corpus(analyzer):
    result = negative_aspects(REVIEWS, "", analyzer, limit=1)
    assert result["reviews_analyzed"] == 4
    assert [a["aspect"] for a in result["negative_aspects"]] == ["battery"]
    assert result["negative_aspects"][0]["negative_examples"]


def test_compare_products(analyzer):
    result = compare_products(REVIEWS, "A", "B", analyzer)
    assert result["differences"]["average_rating_a_minus_b"] == 0.0
    shared = {s["aspect"]: s for s in result["shared_aspects"]}
    assert set(shared) == {"screen", "price"}
    assert shared["screen"]["b"]["negative_share"] == 1.0
    with pytest.raises(ValueError, match="different"):
        compare_products(REVIEWS, "A", "a", analyzer)


def test_list_products_and_statistics_with_ratings():
    rows = list_products(REVIEWS)
    assert rows[0] == {"product": "A", "product_name": "", "review_count": 3, "average_rating": 3.0}
    stats = compute_statistics(REVIEWS, "A")
    assert stats["average_rating"] == 3.0 and stats["rated_reviews"] == 3


def test_analyzer_with_keyword_model_runs(extractor):
    model = KeywordSentimentModel()
    result = AspectAnalyzer(extractor, model.predict, {"name": model.name}).analyze(REVIEWS)
    assert result["aspects"]["battery"]["sentiment_counts"]["negative"] == 2


# --- corpus loading ---------------------------------------------------------

def test_processed_csv_corpus_loads(fixture_corpus):
    reviews = load_reviews(fixture_corpus)
    assert len(reviews) == 64
    assert {"id", "product", "text", "sentiment", "rating"} <= set(reviews[0])
    assert load_reviews(fixture_corpus) is reviews  # cached until the file changes


def test_csv_corpus_validation(tmp_path):
    bad = tmp_path / "c.csv"
    bad.write_text("review_id,product_id,text\n1,A,hi\n", encoding="utf-8")
    with pytest.raises(ReviewDataError, match="missing column"):
        load_reviews(bad)
    bad.write_text("review_id,product_id,text,label\nx,A,hi,positive\n", encoding="utf-8")
    with pytest.raises(ReviewDataError, match="integer"):
        load_reviews(bad)


def test_json_rating_validation(tmp_path):
    path = tmp_path / "r.json"
    path.write_text('[{"id": 1, "product": "P", "text": "t", "sentiment": "neutral", "rating": 9}]', encoding="utf-8")
    with pytest.raises(ReviewDataError, match="'rating'"):
        load_reviews(path)


# --- sentiment sources, sample sizes, product matching ------------------------

def test_label_source_is_checked_against_ratings():
    assert label_source(REVIEWS)["source"] == "star_rating_label"
    relabelled = [*REVIEWS[:3], {**REVIEWS[3], "sentiment": "positive"}]  # 3 stars but labelled positive
    assert label_source(relabelled)["source"] == "dataset_label"
    unrated = [{k: v for k, v in r.items() if k != "rating"} for r in REVIEWS]
    assert label_source(unrated)["source"] == "dataset_label"


def test_prediction_source_distinguishes_trained_model_and_keyword_baseline():
    trained = prediction_source({"version": "m@1", "is_trained_model": True})
    assert trained["source"] == "model_prediction" and trained["model_version"] == "m@1"
    assert prediction_source({"version": "kw", "is_trained_model": False})["source"] == "keyword_baseline_prediction"


def test_wilson_interval_reference_values():
    assert wilson_interval(0, 10) == [0.0, 27.75]
    assert wilson_interval(5, 10) == [23.66, 76.34]
    assert wilson_interval(0, 0) is None


def test_sample_assessment_threshold():
    assert sample_assessment(30, 30)["sufficient"] is True and sample_assessment(30, 30)["warning"] is None
    single = sample_assessment(1, 30)
    assert single["sufficient"] is False and "single review" in single["warning"]


def test_summary_flags_small_sample_and_counts_are_ints(analyzer):
    summary = product_summary(REVIEWS, "B", analyzer, min_reviews=2)
    assert summary["sample"]["sufficient"] is False
    assert all(type(v["count"]) is int for v in summary["sentiment_distribution"].values())
    assert summary["sentiment_sources"]["most_negative_aspects"]["source"] == "model_prediction"
    assert product_summary(REVIEWS, "A", analyzer, min_reviews=2)["sample"]["sufficient"] is True


def test_statistics_counts_are_ints():
    stats = compute_statistics(REVIEWS)
    assert all(type(v) is int for v in stats["counts"].values())


def test_shared_product_name_is_ambiguous_and_id_wins(analyzer):
    reviews = [*REVIEWS, {"id": 5, "product": "C", "text": "Good screen.", "sentiment": "positive", "rating": 5.0,
                          "product_name": "Bravo Tablet"}]
    with pytest.raises(ValueError, match="matches 2 different product ids"):
        product_summary(reviews, "Bravo Tablet", analyzer)
    named_like_id = [*REVIEWS, {"id": 6, "product": "D", "text": "Fine.", "sentiment": "neutral", "rating": 3.0,
                                "product_name": "a"}]
    assert {r["product"] for r in filter_by_product(named_like_id, "a")} == {"A"}


def test_compare_rejects_same_product_by_id_and_name(analyzer):
    with pytest.raises(ValueError, match="both refer to product 'B'"):
        compare_products(REVIEWS, "B", "Bravo Tablet", analyzer)


def test_product_overview_counts():
    overview = product_overview(REVIEWS, min_reviews=2)
    assert (overview["products_in_corpus"], overview["products_with_one_review"], overview["products_with_name"],
            overview["products_meeting_min_reviews"], overview["max_reviews_per_product"]) == (2, 1, 1, 1, 3)
