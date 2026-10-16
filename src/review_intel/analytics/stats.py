"""Review statistics, keyword search, product summaries and product comparison.

Every analytical output carries its evidence (counts, review ids, excerpts)
and says where the numbers come from. Nothing here generates explanations of
*why* customers feel a certain way.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal, TypedDict

from review_intel.analytics.aspects import AspectAnalyzer, top_aspects
from review_intel.analytics.corpus import SENTIMENTS, CorpusReview
from review_intel.data.labels import rating_to_label
from review_intel.models.base import Prediction

LABEL_SOURCE = ("Stored dataset labels. For Amazon data these come from star ratings "
                "(1-2 negative, 3 neutral, 4-5 positive); they are not model predictions.")

# Where a sentiment value in a tool response comes from (see ``label_source`` / ``prediction_source``).
SentimentSourceName = Literal["star_rating_label", "dataset_label", "model_prediction", "keyword_baseline_prediction"]

# Reviews a product needs before its summary is not flagged as a small sample. A heuristic for
# warnings (n >= 30 is the usual rule of thumb for normal approximations), not a statistical guarantee.
DEFAULT_MIN_PRODUCT_REVIEWS = 30


class SentimentSource(TypedDict):
    source: SentimentSourceName
    description: str
    model_version: str | None


def label_source(reviews: Sequence[CorpusReview]) -> SentimentSource:
    """Describe the corpus's stored labels, checked against the data rather than assumed.

    ``star_rating_label`` only if every review has a whole-star rating and its label equals the documented
    rating rule; otherwise the labels are just whatever the corpus file contains (``dataset_label``).
    """
    if reviews and all(r.get("rating") is not None and rating_to_label(r["rating"]) == r["sentiment"]
                       for r in reviews):
        return {"source": "star_rating_label", "model_version": None,
                "description": "Stored labels derived from each review's star rating (1-2 negative, 3 neutral, "
                               "4-5 positive), consistent for every review in this corpus. Not model predictions."}
    return {"source": "dataset_label", "model_version": None,
            "description": "Labels stored in the corpus file (not derivable from star ratings for every review). "
                           "Not model predictions."}


def prediction_source(model_identity: dict[str, Any]) -> SentimentSource:
    """Describe predictions made by the configured sentiment model (or the explicit keyword demo baseline)."""
    if model_identity.get("is_trained_model"):
        return {"source": "model_prediction", "model_version": model_identity["version"],
                "description": f"Predicted by the trained sentiment model {model_identity['version']} "
                               "(see get_model_info for its measured test metrics). Not the star rating."}
    return {"source": "keyword_baseline_prediction", "model_version": model_identity.get("version"),
            "description": "Predicted by the keyword-counting DEMO baseline (explicitly configured); "
                           "not a trained model and not the star rating."}


class SentimentCounts(TypedDict):
    positive: int
    negative: int
    neutral: int


class UnknownProductError(ValueError):
    """No review in the corpus matches the requested product."""


def product_matches(review: CorpusReview, wanted_casefolded: str) -> bool:
    return (review["product"].strip().casefold() == wanted_casefolded
            or review.get("product_name", "").strip().casefold() == wanted_casefolded)


def filter_by_product(reviews: Sequence[CorpusReview], product: str) -> list[CorpusReview]:
    """Reviews whose product id or product name equals ``product`` (case-insensitive). Empty = all.

    An exact product-id match wins over a name match, and a name shared by several product ids is rejected
    (``ValueError``) instead of silently pooling different products.
    """
    wanted = product.strip().casefold()
    if not wanted:
        return list(reviews)
    by_id = [r for r in reviews if r["product"].strip().casefold() == wanted]
    if by_id:
        return by_id
    by_name = [r for r in reviews if r.get("product_name", "").strip().casefold() == wanted]
    ids = sorted({r["product"] for r in by_name})
    if len(ids) > 1:
        raise ValueError(f"Product name {product.strip()!r} matches {len(ids)} different product ids "
                         f"({', '.join(ids[:10])}); pass one product id instead.")
    return by_name


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a proportion, in percent (None when n = 0)."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(100 * max(0.0, centre - half), 2), round(100 * min(1.0, centre + half), 2)]


def sample_assessment(n: int, min_reviews: int) -> dict[str, Any]:
    """Whether ``n`` reviews are enough for a product-level summary to be more than anecdotal."""
    warning = None
    if n < min_reviews:
        what = "a single review, not the product" if n == 1 else f"only {n} reviews"
        warning = (f"Small sample: this summary describes {what} (threshold {min_reviews}). Percentages, "
                   "averages and aspect counts are not reliable; do not draw product-level conclusions from them.")
    return {"reviews": n, "min_reviews": min_reviews, "sufficient": n >= min_reviews, "warning": warning}


def require_product(reviews: Sequence[CorpusReview], product: str) -> list[CorpusReview]:
    if not product.strip():
        raise ValueError("Product must not be empty.")
    matching = filter_by_product(reviews, product)
    if not matching:
        raise UnknownProductError(f"No reviews found for product {product.strip()!r}. "
                                  "Use list_products to see the available product ids and names.")
    return matching


def _rating_stats(reviews: Sequence[CorpusReview]) -> dict[str, Any]:
    ratings = [r["rating"] for r in reviews if r.get("rating") is not None]
    distribution = Counter(int(x) for x in ratings if float(x).is_integer())
    return {
        "average_rating": round(sum(ratings) / len(ratings), 3) if ratings else None,
        "rated_reviews": len(ratings),
        "rating_distribution": {str(star): distribution.get(star, 0) for star in range(1, 6)},
    }


# ---------------------------------------------------------------------------
# v1 tools (behaviour preserved)
# ---------------------------------------------------------------------------


class SentimentBreakdown(TypedDict):
    positive: float
    negative: float
    neutral: float


class StatisticsResult(TypedDict):
    product: str
    total_reviews: int
    counts: SentimentCounts
    percentages: SentimentBreakdown
    average_rating: float | None
    rated_reviews: int
    sentiment_sources: dict[str, SentimentSource]


def compute_statistics(reviews: Sequence[CorpusReview], product: str = "") -> StatisticsResult:
    """Count reviews per labelled sentiment, optionally for a single product.

    Percentages are rounded to 2 decimals. With no matching reviews every
    count and percentage is 0 (no division by zero).
    """
    matching = filter_by_product(reviews, product)
    total = len(matching)
    counts = {s: sum(1 for r in matching if r["sentiment"] == s) for s in SENTIMENTS}
    percentages = {s: round(counts[s] / total * 100, 2) if total else 0.0 for s in SENTIMENTS}
    ratings = _rating_stats(matching)
    return {
        "product": product.strip() or "all",
        "total_reviews": total,
        "counts": counts,  # type: ignore[typeddict-item]
        "percentages": percentages,  # type: ignore[typeddict-item]
        "average_rating": ratings["average_rating"],
        "rated_reviews": ratings["rated_reviews"],
        "sentiment_sources": {"counts": label_source(reviews), "percentages": label_source(reviews)},
    }


class SearchResult(TypedDict):
    keyword: str
    product: str
    match_count: int
    reviews: list[CorpusReview]
    sentiment_sources: dict[str, SentimentSource]


def find_reviews(reviews: Sequence[CorpusReview], keyword: str, product: str = "",
                 limit: int | None = None) -> SearchResult:
    """Return reviews whose text contains ``keyword`` (case-insensitive substring).

    Raises ``ValueError`` if ``keyword`` is empty. No matches -> empty list.
    ``match_count`` is the total number of matches even when ``limit`` truncates the list.
    """
    needle = keyword.strip().casefold()
    if not needle:
        raise ValueError("Keyword must not be empty or whitespace only.")
    matches = [r for r in filter_by_product(reviews, product) if needle in r["text"].casefold()]
    return {
        "keyword": keyword.strip(),
        "product": product.strip() or "all",
        "match_count": len(matches),
        "reviews": matches[:limit] if limit else matches,
        "sentiment_sources": {"reviews.sentiment": label_source(reviews)},
    }


# ---------------------------------------------------------------------------
# v2 analytics
# ---------------------------------------------------------------------------


def list_products(reviews: Sequence[CorpusReview], limit: int = 20) -> list[dict[str, Any]]:
    """Products ordered by review count (ties: product id)."""
    groups: dict[str, list[CorpusReview]] = {}
    for review in reviews:
        groups.setdefault(review["product"], []).append(review)
    rows = []
    for product, items in groups.items():
        ratings = _rating_stats(items)
        rows.append({"product": product, "product_name": items[0].get("product_name", ""),
                     "review_count": len(items), "average_rating": ratings["average_rating"]})
    rows.sort(key=lambda r: (-r["review_count"], r["product"]))
    return rows[:limit]


def product_overview(reviews: Sequence[CorpusReview], min_reviews: int = DEFAULT_MIN_PRODUCT_REVIEWS) -> dict[str, Any]:
    """How much per-product evidence the corpus holds (counts only; nothing is inferred)."""
    counts = Counter(r["product"] for r in reviews)
    named = {r["product"] for r in reviews if r.get("product_name", "").strip()}
    meeting = sum(1 for n in counts.values() if n >= min_reviews)
    notes = []
    if meeting < len(counts):
        notes.append(f"{len(counts) - meeting} of {len(counts)} products have fewer than {min_reviews} reviews; "
                     "their summaries and comparisons are flagged as small samples.")
    if len(named) < len(counts):
        notes.append(f"{len(counts) - len(named)} products have no product_name in this corpus (no title "
                     "available); refer to them by product id. Names are never guessed.")
    return {"products_in_corpus": len(counts),
            "products_with_one_review": sum(1 for n in counts.values() if n == 1),
            "products_with_name": len(named),
            "min_reviews_for_summary": min_reviews,
            "products_meeting_min_reviews": meeting,
            "max_reviews_per_product": max(counts.values(), default=0),
            "notes": notes}


def _representative_reviews(reviews: Sequence[CorpusReview], predictions: list[Prediction], label: str,
                            limit: int) -> list[dict[str, Any]]:
    """Reviews whose stored label is ``label``, ranked by the model's score for that label.

    Prefers readable lengths (40-400 chars); ties broken by review id for determinism.
    """
    pool = [(r, p) for r, p in zip(reviews, predictions) if r["sentiment"] == label]
    pool.sort(key=lambda rp: (not 40 <= len(rp[0]["text"]) <= 400, -rp[1].scores.get(label, 0.0), rp[0]["id"]))
    return [{"review_id": r["id"], "rating": r.get("rating"), "text": r["text"][:400]} for r, _ in pool[:limit]]


def product_summary(reviews: Sequence[CorpusReview], product: str, analyzer: AspectAnalyzer,
                    top_n_aspects: int = 5, examples: int = 2,
                    min_reviews: int = DEFAULT_MIN_PRODUCT_REVIEWS) -> dict[str, Any]:
    matching = require_product(reviews, product)
    total = len(matching)
    label_counts = Counter(r["sentiment"] for r in matching)
    aspect_analysis = analyzer.analyze(matching)
    predictions = analyzer.predict([r["text"] for r in matching])
    sample = sample_assessment(total, min_reviews)
    labels = label_source(reviews)
    predicted = prediction_source(analyzer.model_identity)
    product_name = matching[0].get("product_name", "").strip()
    notes = [sample["warning"]] if sample["warning"] else []
    if not product_name:
        notes.append("No product name is available for this product in the corpus; it is identified by id only.")
    return {
        "product": matching[0]["product"],
        "product_name": product_name,
        "total_reviews": total,
        **_rating_stats(matching),
        "sentiment_distribution": {
            s: {"count": label_counts.get(s, 0), "percent": round(100 * label_counts.get(s, 0) / total, 2)}
            for s in SENTIMENTS
        },
        "sentiment_ci95": {s: wilson_interval(label_counts.get(s, 0), total) for s in SENTIMENTS},
        "sample": sample,
        "notes": notes,
        "sentiment_label_source": LABEL_SOURCE,
        "sentiment_sources": {
            "sentiment_distribution": labels,
            "representative_reviews": {**labels, "description": labels["description"] + " Within each label, "
                                       "reviews are ordered by the sentiment model's score for that label."},
            "most_mentioned_aspects": predicted,
            "most_negative_aspects": predicted,
        },
        "most_mentioned_aspects": [_aspect_brief(a) for a in top_aspects(aspect_analysis, top_n_aspects)],
        "most_negative_aspects": [_aspect_brief(a) for a in top_aspects(aspect_analysis, top_n_aspects, by="negative")],
        "representative_reviews": {
            "positive": _representative_reviews(matching, predictions, "positive", examples),
            "negative": _representative_reviews(matching, predictions, "negative", examples),
        },
        "method": {
            "statistics": "Counts and averages over all matching reviews in the corpus; sentiment_ci95 is the "
                          "95% Wilson score interval of each percentage.",
            "representative_reviews": "Reviews with that stored label, ranked by the sentiment model's score for "
                                      "the label, preferring 40-400 characters.",
            "aspects": aspect_analysis["method"],
        },
    }


def _aspect_brief(aspect: dict[str, Any]) -> dict[str, Any]:
    return {k: aspect[k] for k in ("aspect", "reviews_mentioning", "mentions", "sentiment_counts", "negative_share")}


def negative_aspects(reviews: Sequence[CorpusReview], product: str, analyzer: AspectAnalyzer,
                     limit: int = 5, examples: int = 2,
                     min_reviews: int = DEFAULT_MIN_PRODUCT_REVIEWS) -> dict[str, Any]:
    matching = require_product(reviews, product) if product.strip() else list(reviews)
    analysis = analyzer.analyze(matching, examples_per_sentiment=examples)
    ranked = top_aspects(analysis, limit, by="negative")
    return {
        "product": product.strip() or "all",
        "reviews_analyzed": len(matching),
        "sample": sample_assessment(len(matching), min_reviews),
        "negative_aspects": [{**_aspect_brief(a), "negative_examples": a["examples"]["negative"]} for a in ranked],
        "sentiment_sources": {"negative_aspects": prediction_source(analyzer.model_identity)},
        "method": analysis["method"],
    }


def compare_products(reviews: Sequence[CorpusReview], product_a: str, product_b: str,
                     analyzer: AspectAnalyzer, min_reviews: int = DEFAULT_MIN_PRODUCT_REVIEWS) -> dict[str, Any]:
    if product_a.strip().casefold() == product_b.strip().casefold():
        raise ValueError("product_a and product_b must be different products.")
    a = product_summary(reviews, product_a, analyzer, top_n_aspects=50, examples=1, min_reviews=min_reviews)
    b = product_summary(reviews, product_b, analyzer, top_n_aspects=50, examples=1, min_reviews=min_reviews)
    if a["product"] == b["product"]:  # e.g. the id and the name of the same product
        raise ValueError(f"product_a and product_b both refer to product {a['product']!r}; "
                         "choose two different products.")
    aspects_a = {x["aspect"]: x for x in a["most_mentioned_aspects"]}
    aspects_b = {x["aspect"]: x for x in b["most_mentioned_aspects"]}
    shared = sorted(set(aspects_a) & set(aspects_b),
                    key=lambda k: -(aspects_a[k]["mentions"] + aspects_b[k]["mentions"]))

    def neg_pct(summary: dict[str, Any]) -> float:
        return summary["sentiment_distribution"]["negative"]["percent"]

    def brief(s: dict[str, Any]) -> dict[str, Any]:
        return {k: s[k] for k in ("product", "product_name", "total_reviews", "average_rating", "rated_reviews",
                                  "sentiment_distribution", "sentiment_ci95", "sample")}

    def overlap(x: list[float] | None, y: list[float] | None) -> bool:
        return x is not None and y is not None and x[0] <= y[1] and y[0] <= x[1]

    small = [s for s in (a, b) if not s["sample"]["sufficient"]]
    negative_overlap = overlap(a["sentiment_ci95"]["negative"], b["sentiment_ci95"]["negative"])

    return {
        "product_a": brief(a),
        "product_b": brief(b),
        "differences": {
            "average_rating_a_minus_b": (round(a["average_rating"] - b["average_rating"], 3)
                                         if a["average_rating"] is not None and b["average_rating"] is not None else None),
            "negative_percent_a_minus_b": round(neg_pct(a) - neg_pct(b), 2),
            "positive_percent_a_minus_b": round(a["sentiment_distribution"]["positive"]["percent"]
                                                - b["sentiment_distribution"]["positive"]["percent"], 2),
        },
        "shared_aspects": [
            {"aspect": k,
             "a": {"mentions": aspects_a[k]["mentions"], "negative_share": aspects_a[k]["negative_share"]},
             "b": {"mentions": aspects_b[k]["mentions"], "negative_share": aspects_b[k]["negative_share"]},
             "negative_share_a_minus_b": round(aspects_a[k]["negative_share"] - aspects_b[k]["negative_share"], 4)}
            for k in shared
        ],
        "comparison_reliable": not small,
        "negative_percent_ci95_overlap": negative_overlap,
        "notes": [
            *(f"{s['product']} has only {s['total_reviews']} review(s) (threshold {min_reviews}); the comparison "
              "is not meaningful for it." for s in small),
            *(["The 95% intervals of the negative percentages overlap, so this corpus does not show a clear "
               "difference in negative share."] if negative_overlap else []),
            "Differences are descriptive statistics over this corpus, not significance-tested.",
            "Small review counts make percentages unstable; check total_reviews and mentions before drawing conclusions.",
            LABEL_SOURCE,
        ],
        "sentiment_sources": {"sentiment_distribution": a["sentiment_sources"]["sentiment_distribution"],
                              "shared_aspects": a["sentiment_sources"]["most_mentioned_aspects"]},
        "method": a["method"]["aspects"],
    }
