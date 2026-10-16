"""Review Analyzer MCP server (v2).

A thin MCP integration layer over the ``review_intel`` package at the
project root, which holds every piece of NLP/ML and analytics logic. This
file only declares tools, validates arguments, and turns expected failures
into MCP tool errors.

Tools
-----
v1 (behaviour preserved): analyze_sentiment, get_review_statistics, search_reviews
v2: list_products, get_product_summary, get_negative_aspects, compare_products,
    search_similar_reviews, get_model_info

Configuration (environment variables; relative paths resolve from the project root)
-----------------------------------------------------------------------------------
REVIEW_ANALYZER_DATA               review corpus (.json v1 format or processed .csv);
                                   default: data/processed/amazon_electronics/analytics_corpus.csv;
                                   "demo" = the 5-review demo file reviews.json next to this server
REVIEW_ANALYZER_SENTIMENT_BACKEND  "trained" (default) | "keyword" (explicit demo baseline)
REVIEW_ANALYZER_MODEL_DIR          model artifact dir; default: artifacts/models/selected.json
REVIEW_ANALYZER_MAX_TEXT_CHARS     max analyze_sentiment input length (default 5000)
REVIEW_ANALYZER_RETRIEVER          "lsa" (default, offline) | "sentence-transformers"
REVIEW_ANALYZER_EMBEDDING_MODEL    embedding model for the dense retriever
REVIEW_ANALYZER_MIN_PRODUCT_REVIEWS  reviews a product needs before its summary is not flagged as a small
                                   sample (default 30; a heuristic, not a statistical guarantee)

IMPORTANT for stdio transport: stdout carries JSON-RPC messages. Never
``print()`` here; diagnostics go to stderr through ``logging``.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

try:
    from review_intel.analytics import stats
    from review_intel.analytics.corpus import CorpusReview, ReviewDataError, load_reviews as _load_corpus
    from review_intel.analytics.corpus import validate_reviews  # noqa: F401  (v1 re-export)
    from review_intel.app import ReviewIntelligence
    from review_intel.analytics.stats import (
        SearchResult,
        SentimentSource,
        StatisticsResult,
        compute_statistics,
        find_reviews,
    )
    from review_intel.data.pipeline import DatasetConfigError
    from review_intel.models.base import ModelNotAvailableError
    from review_intel.models.keyword import (  # noqa: F401  (v1 re-exports)
        BASELINE_WARNING,
        KeywordSentimentClassifier,
        analyze_text_sentiment,
    )
    from review_intel.retrieval.semantic import search_with_sentiment
    from review_intel.utils import PROJECT_ROOT, REPORTS_DIR, read_json, relative_to_root
except ImportError as exc:  # pragma: no cover - environment problem, reported on stderr
    sys.stderr.write(
        f"review_intel is not importable ({exc}). Run this server with the project's virtual environment, "
        "after `pip install -e .[mcp]` in the project root.\n"
    )
    raise

logger = logging.getLogger("review_analyzer")

Sentiment = Literal["positive", "negative", "neutral"]


DEFAULT_DATA = "data/processed/amazon_electronics/analytics_corpus.csv"
DEMO_DATA_FILE = Path(__file__).resolve().parent / "reviews.json"


def resolve_data_file(environ: dict[str, str] | None = None) -> Path:
    """The corpus named by REVIEW_ANALYZER_DATA, resolved from the project root (never the working directory).

    Unset means the Amazon analytics corpus; the 5-review demo file is used only when explicitly asked for
    with REVIEW_ANALYZER_DATA=demo (or its path). A missing file is reported by the tools, never replaced.
    """
    env = os.environ if environ is None else environ
    configured = env.get("REVIEW_ANALYZER_DATA", "").strip()
    if configured.casefold() == "demo":
        return DEMO_DATA_FILE
    path = Path(configured or DEFAULT_DATA)
    return path if path.is_absolute() else PROJECT_ROOT / path


DATA_FILE = resolve_data_file()
app = ReviewIntelligence.from_env()


def missing_data_message(path: Path) -> str:
    return (f"Review dataset not found: {relative_to_root(path)}. Set REVIEW_ANALYZER_DATA to an existing "
            f".csv/.json corpus (relative paths resolve from the project root). The default corpus "
            f"{DEFAULT_DATA} is built by scripts/build_corpus.py (see README); REVIEW_ANALYZER_DATA=demo selects "
            "the 5-review demo file.")


def load_reviews(path: Path | None = None) -> list[CorpusReview]:
    """Load and validate the review corpus (v1 API; defaults to DATA_FILE)."""
    path = DATA_FILE if path is None else path
    if not path.exists():
        raise ReviewDataError(missing_data_message(path))
    return _load_corpus(path)


@contextmanager
def tool_errors() -> Iterator[None]:
    """Turn anticipated failures into ToolError so the model sees a readable message (isError=true).

    Anything else propagates; the SDK then logs the traceback to stderr and
    returns only a generic error to the client (no internals leak).
    """
    try:
        yield
    except (ValueError, ReviewDataError, ModelNotAvailableError, DatasetConfigError) as exc:
        raise ToolError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Output schemas (TypedDicts become the tools' outputSchema)
# ---------------------------------------------------------------------------


class SentimentResult(TypedDict):
    text: str
    sentiment: Sentiment
    method: str
    positive_matches: list[str]
    negative_matches: list[str]
    warning: str
    scores: dict[str, float]
    score_type: str
    score_description: str
    model_version: str
    is_trained_model: bool
    sentiment_sources: dict[str, SentimentSource]


class ProductRow(TypedDict):
    product: str
    product_name: str
    review_count: int
    average_rating: float | None


class ProductList(TypedDict):
    corpus: str
    total_reviews: int
    products_in_corpus: int
    products_with_one_review: int
    products_with_name: int
    min_reviews_for_summary: int
    products_meeting_min_reviews: int
    max_reviews_per_product: int
    notes: list[str]
    products: list[ProductRow]


class SentimentShare(TypedDict):
    count: int
    percent: float


class SampleInfo(TypedDict):
    reviews: int
    min_reviews: int
    sufficient: bool
    warning: str | None


class ProductSummary(TypedDict):
    product: str
    product_name: str
    total_reviews: int
    average_rating: float | None
    rated_reviews: int
    rating_distribution: dict[str, int]
    sentiment_distribution: dict[str, SentimentShare]
    sentiment_ci95: dict[str, list[float] | None]
    sample: SampleInfo
    notes: list[str]
    sentiment_label_source: str
    sentiment_sources: dict[str, SentimentSource]
    most_mentioned_aspects: list[dict[str, Any]]
    most_negative_aspects: list[dict[str, Any]]
    representative_reviews: dict[str, list[dict[str, Any]]]
    method: dict[str, Any]


class NegativeAspects(TypedDict):
    product: str
    reviews_analyzed: int
    sample: SampleInfo
    negative_aspects: list[dict[str, Any]]
    sentiment_sources: dict[str, SentimentSource]
    method: dict[str, Any]


class ProductComparison(TypedDict):
    product_a: dict[str, Any]
    product_b: dict[str, Any]
    differences: dict[str, float | None]
    shared_aspects: list[dict[str, Any]]
    comparison_reliable: bool
    negative_percent_ci95_overlap: bool
    notes: list[str]
    sentiment_sources: dict[str, SentimentSource]
    method: dict[str, Any]


class SimilarReview(TypedDict):
    id: int
    product: str
    product_name: str
    text: str
    sentiment: Sentiment
    rating: float | None
    similarity: float
    predicted_sentiment: Sentiment | None
    predicted_score: float | None


class SentimentFilterInfo(TypedDict):
    sentiment: Literal["any", "negative", "neutral", "positive"]
    source: Literal["star_rating_label", "model_prediction"] | None
    model_version: str | None
    description: str
    candidate_pool: int | None
    candidates_scanned: int
    candidates_matching: int
    note: str | None


class SimilarReviews(TypedDict):
    query: str
    product: str
    method: str
    sentiment_filter: SentimentFilterInfo
    results: list[SimilarReview]
    sentiment_sources: dict[str, SentimentSource]


class ModelInfo(TypedDict):
    status: Literal["ready", "unavailable"]
    model: dict[str, Any]
    test_metrics: dict[str, Any] | None
    corpus: dict[str, Any]
    retrieval: str
    aspect_lexicon: str
    message: str


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

mcp = MCPServer(
    "Review Analyzer MCP",
    instructions=(
        "Tools for analyzing product reviews. Sentiment comes from the trained model reported by get_model_info "
        "(or an explicitly configured keyword demo baseline). Statistics come from stored dataset labels. "
        "Every response's `sentiment_sources` says which values are star-rating labels and which are model "
        "predictions; do not present one as the other. Product summaries with `sample.sufficient=false` describe "
        "too few reviews for product-level conclusions; say so. search_similar_reviews ranks by topic, not "
        "polarity: to find complaints or praise, pass sentiment_filter with sentiment_source. "
        "Call list_products to find valid product ids/names. Report numbers with their evidence; do not invent "
        "reasons that the returned reviews do not state."
    ),
)

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)

ProductArg = Annotated[str, Field(max_length=300, description="Product id or exact product name (case-insensitive).")]
OptionalProductArg = Annotated[str, Field(max_length=300, description="Optional product id or name; empty = all.")]


@mcp.tool(annotations=READ_ONLY)
def analyze_sentiment(text: str) -> SentimentResult:
    """Predict whether a piece of review text is positive, negative or neutral.

    Uses the trained sentiment model selected on validation data (see get_model_info), or the keyword demo
    baseline if the server was explicitly configured that way (then is_trained_model=false and warning says so).
    `scores` are explained by `score_type`/`score_description`: SVM decision scores are not probabilities.

    Args:
        text: The review text to classify. Must not be empty; maximum length is configurable (default 5000 chars).
    """
    with tool_errors():
        result = app.service.analyze(text)
    details = result["details"]
    model = result["model"]
    return {
        "text": result["text"],
        "sentiment": result["sentiment"],
        "method": model["name"],
        "positive_matches": details.get("positive_matches", []),
        "negative_matches": details.get("negative_matches", []),
        "warning": result["warning"] or (
            f"Prediction from trained model {model['version']}. Trained on star-rating-labelled reviews; accuracy "
            "outside that domain is unknown. See get_model_info for measured metrics."),
        "scores": result["scores"],
        "score_type": result["score_type"],
        "score_description": result["score_description"],
        "model_version": model["version"],
        "is_trained_model": model["is_trained_model"],
        "sentiment_sources": {"sentiment": stats.prediction_source(model)},
    }


@mcp.tool(annotations=READ_ONLY)
def get_review_statistics(product: OptionalProductArg = "") -> StatisticsResult:
    """Get sentiment statistics (counts and percentages) and average star rating for the stored reviews.

    Uses the dataset's stored sentiment labels (for Amazon data: derived from star ratings), not predictions.

    Args:
        product: Optional product id or name to filter by (case-insensitive exact match).
            Leave empty to get statistics for all reviews.
    """
    with tool_errors():
        return compute_statistics(load_reviews(DATA_FILE), product)


@mcp.tool(annotations=READ_ONLY)
def search_reviews(
    keyword: Annotated[str, Field(max_length=200)],
    product: OptionalProductArg = "",
    limit: Annotated[int, Field(ge=1, le=100, description="Maximum reviews returned.")] = 20,
) -> SearchResult:
    """Find stored reviews whose text contains a keyword (case-insensitive substring match).

    Returns matching reviews with their id, product, text and sentiment label. match_count is the total
    number of matches even if `limit` truncates the list. An empty list means nothing matched.

    Args:
        keyword: Word or phrase to look for in the review text. Must not be empty.
        product: Optional product id or name to filter by (case-insensitive exact match).
            Leave empty to search all products.
        limit: Maximum number of reviews to return (1-100, default 20).
    """
    with tool_errors():
        return find_reviews(load_reviews(DATA_FILE), keyword, product, limit)


@mcp.tool(annotations=READ_ONLY)
def list_products(limit: Annotated[int, Field(ge=1, le=200)] = 20) -> ProductList:
    """List products in the review corpus, most-reviewed first, with ids, names, review counts and average rating.

    Use the returned `product` id or `product_name` as the product argument of other tools.

    Args:
        limit: Maximum number of products to return (1-200, default 20).
    """
    with tool_errors():
        reviews = load_reviews(DATA_FILE)
        return {"corpus": DATA_FILE.name, "total_reviews": len(reviews),
                **stats.product_overview(reviews, app.config.min_product_reviews),
                "products": stats.list_products(reviews, limit)}  # type: ignore[typeddict-item]


@mcp.tool(annotations=READ_ONLY)
def get_product_summary(product: ProductArg) -> ProductSummary:
    """Summarize one product from its reviews: counts, average rating, sentiment distribution, most-mentioned
    and most-negative aspects, and representative positive/negative reviews as evidence.

    All numbers are computed from the stored reviews; aspect sentiment comes from the sentiment model applied to
    sentences that mention each aspect (see `method.limitations`).

    Args:
        product: Product id or exact product name (case-insensitive). See list_products.
    """
    with tool_errors():
        return stats.product_summary(load_reviews(DATA_FILE), product, app.analyzer(),  # type: ignore[return-value]
                                     min_reviews=app.config.min_product_reviews)


@mcp.tool(annotations=READ_ONLY)
def get_negative_aspects(
    product: OptionalProductArg = "",
    limit: Annotated[int, Field(ge=1, le=20, description="Maximum aspects returned.")] = 5,
) -> NegativeAspects:
    """Find the product aspects (battery, screen, price, ...) most often mentioned in negatively classified
    sentences, with counts, negative share and example excerpts as evidence.

    Args:
        product: Optional product id or name; empty analyzes the whole corpus.
        limit: Maximum number of aspects to return (1-20, default 5).
    """
    with tool_errors():
        return stats.negative_aspects(load_reviews(DATA_FILE), product, app.analyzer(), limit,  # type: ignore[return-value]
                                      min_reviews=app.config.min_product_reviews)


@mcp.tool(annotations=READ_ONLY)
def compare_products(product_a: ProductArg, product_b: ProductArg) -> ProductComparison:
    """Compare two products: review counts, average ratings, sentiment distributions and the negative share of
    aspects both products have in common. Descriptive statistics only; no significance testing.

    Args:
        product_a: First product id or name.
        product_b: Second product id or name (must differ from product_a).
    """
    with tool_errors():
        return stats.compare_products(load_reviews(DATA_FILE), product_a, product_b,  # type: ignore[return-value]
                                      app.analyzer(), min_reviews=app.config.min_product_reviews)


@mcp.tool(annotations=READ_ONLY)
def search_similar_reviews(
    query: Annotated[str, Field(max_length=1000, description="Free-text description of what to look for.")],
    product: OptionalProductArg = "",
    limit: Annotated[int, Field(ge=1, le=50)] = 5,
    sentiment_filter: Annotated[Literal["any", "negative", "neutral", "positive"], Field(
        description="Keep only reviews with this sentiment; 'any' (default) = no filter.")] = "any",
    sentiment_source: Annotated[Literal["", "star_rating_label", "model_prediction"], Field(
        description="Required when sentiment_filter is not 'any': 'star_rating_label' (the review's star "
                    "rating) or 'model_prediction' (the sentiment model's prediction for the review text).")] = "",
    candidate_pool: Annotated[int, Field(ge=1, le=1000, description=(
        "model_prediction only: how many of the most similar reviews are classified before filtering."))] = 200,
) -> SimilarReviews:
    """Find reviews similar to a free-text query (ranked by cosine similarity of LSA vectors), e.g.
    "remote stopped working after a month". Unlike search_reviews, words do not need to match exactly.

    Similarity is lexical-topical: it does NOT understand negation or polarity, so a complaint like "battery
    drains fast" can match "battery lasts forever". To find complaints or praise, set sentiment_filter together
    with sentiment_source. The ranking itself is unchanged; the filter only removes reviews:
    - star_rating_label: filters by the review's stored star-rating label (what the customer rated, not
      necessarily what the text says); every review is a candidate;
    - model_prediction: classifies the candidate_pool most similar reviews with the trained sentiment model
      (see get_model_info) and keeps the matching ones; fewer results than `limit` means the pool held fewer
      matches, not that none exist.

    Args:
        query: What to look for. Must not be empty.
        product: Optional product id or name to restrict the search.
        limit: Maximum number of reviews to return (1-50, default 5).
        sentiment_filter: any | negative | neutral | positive (default any).
        sentiment_source: star_rating_label | model_prediction; required unless sentiment_filter is 'any'.
        candidate_pool: Reviews classified for model_prediction filtering (1-1000, default 200).
    """
    with tool_errors():
        reviews = load_reviews(DATA_FILE)  # actionable error if the corpus is missing
        labels = stats.label_source(reviews)
        if sentiment_filter == "any" and sentiment_source:
            raise ValueError("sentiment_source only applies when sentiment_filter is negative, neutral or positive.")
        if sentiment_source == "star_rating_label" and labels["source"] != "star_rating_label":
            raise ValueError("This corpus's labels are not derived from star ratings for every review "
                             f"({labels['source']}), so a star_rating_label filter would be mislabelled. "
                             "Use sentiment_source='model_prediction'.")
        search = app.similarity_search(DATA_FILE)
        product_filter = None
        if product.strip():
            stats.require_product(search.reviews, product)
            wanted = product.strip().casefold()
            product_filter = lambda r: stats.product_matches(r, wanted)  # noqa: E731
        predict = app.service.predict_texts if sentiment_source == "model_prediction" else None
        found = search_with_sentiment(search, query, sentiment=sentiment_filter, source=sentiment_source or None,
                                      predict=predict, product_filter=product_filter, limit=limit,
                                      candidate_pool=candidate_pool)

    sources: dict[str, Any] = {"results.sentiment": labels}
    model_version = None
    if found.source == "model_prediction":
        prediction = stats.prediction_source(app.service.model_identity())
        sources["results.predicted_sentiment"] = prediction
        model_version = prediction["model_version"]
    description = {
        None: "No sentiment filter: results are ranked by similarity only and may have any sentiment.",
        "star_rating_label": f"Kept only reviews whose stored star-rating label is {sentiment_filter!r}. This "
                             "reflects what the customer rated, not necessarily what the text says.",
        "model_prediction": f"Kept only reviews the sentiment model {model_version} predicts as "
                            f"{sentiment_filter!r} from their text, among the {found.candidate_pool} most similar.",
    }[found.source]
    note = None
    if found.source == "model_prediction" and len(found.results) < limit:
        note = (f"Only {found.candidates_matching} of the {found.candidates_scanned} most similar reviews are predicted "
                f"{sentiment_filter!r}; raise candidate_pool to look further down the ranking.")
    elif found.source == "star_rating_label" and len(found.results) < limit:
        note = f"Only {found.candidates_matching} reviews in the searched set have this label."
    return {
        "query": query,
        "product": product.strip() or "all",
        "method": search.retriever.method,
        "sentiment_filter": {"sentiment": sentiment_filter, "source": found.source, "model_version": model_version,
                             "description": description, "candidate_pool": found.candidate_pool,
                             "candidates_scanned": found.candidates_scanned,
                             "candidates_matching": found.candidates_matching, "note": note},
        "results": [{"id": r["id"], "product": r["product"], "product_name": r.get("product_name", ""),
                     "text": r["text"], "sentiment": r["sentiment"], "rating": r.get("rating"),
                     "similarity": r["similarity"], "predicted_sentiment": r.get("predicted_sentiment"),
                     "predicted_score": r.get("predicted_score")} for r in found.results],
        "sentiment_sources": sources,
    }


def _test_metrics_for(info: dict[str, Any]) -> dict[str, Any] | None:
    """Held-out test metrics from reports/, only if they belong to exactly this model version."""
    dataset = (info.get("dataset") or {}).get("name")
    path = REPORTS_DIR / dataset / "test_results.json" if dataset else None
    if not path or not path.exists():
        return None
    for family, result in read_json(path).get("results", {}).items():
        if result.get("model_version") == info.get("version"):
            test = result["test"]
            return {"split": "test", "n": test["n"], "accuracy": test["accuracy"], "macro_f1": test["macro_f1"],
                    "per_class_f1": {k: v["f1"] for k, v in test["per_class"].items()}, "role": result["role"]}
    return None


@mcp.tool(annotations=READ_ONLY)
def get_model_info() -> ModelInfo:
    """Describe the active sentiment model (name, version, training data, validation and test metrics, score
    type), the review corpus, the retrieval method and the aspect lexicon. Works even if no model is loaded."""
    corpus: dict[str, Any] = {"file": relative_to_root(DATA_FILE), "is_demo": DATA_FILE == DEMO_DATA_FILE}
    try:
        reviews = load_reviews(DATA_FILE)
        corpus.update(reviews=len(reviews), products=len({r["product"] for r in reviews}))
        sidecar = DATA_FILE.with_suffix(".json")  # written by scripts/build_corpus.py
        manifest = DATA_FILE.parent / "manifest.json"  # written by scripts/prepare_data.py
        if DATA_FILE.suffix == ".csv" and sidecar.exists():
            built = read_json(sidecar)
            corpus.update(source_split=built.get("source_split"), note=built.get("note"))
        elif DATA_FILE.suffix == ".csv" and manifest.exists():
            built = read_json(manifest)
            corpus.update(dataset=built.get("name"), note=built.get("source", {}).get("origin"))
    except ReviewDataError as exc:
        corpus["error"] = str(exc)
    common = {"corpus": corpus, "retrieval": app.config.retriever, "aspect_lexicon": app.extractor.version}
    try:
        info = app.service.info()
    except ModelNotAvailableError as exc:
        return {"status": "unavailable", "model": {}, "test_metrics": None, "message": str(exc), **common}
    message = ("Trained model loaded." if info.get("is_trained_model")
               else "Keyword DEMO baseline explicitly configured; this is not a trained model.")
    return {"status": "ready", "model": info, "test_metrics": _test_metrics_for(info), "message": message, **common}


def main() -> None:
    # Logs go to stderr; stdout is reserved for MCP protocol messages.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    source = "REVIEW_ANALYZER_DATA" if os.environ.get("REVIEW_ANALYZER_DATA", "").strip() else "default"
    logger.info("Starting Review Analyzer MCP (stdio), corpus (%s): %s", source, relative_to_root(DATA_FILE))
    if DATA_FILE == DEMO_DATA_FILE:
        logger.warning("Using the 5-review DEMO corpus (REVIEW_ANALYZER_DATA=demo); for demonstration/testing only.")
    elif not DATA_FILE.exists():
        logger.error("%s Corpus-based tools will return this error until it is fixed.", missing_data_message(DATA_FILE))
    try:  # load the model once at startup
        logger.info("Sentiment model: %s", app.service.model.version)
    except ModelNotAvailableError as exc:
        logger.warning("Sentiment model unavailable; sentiment/aspect tools will return an error until fixed: %s", exc)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
