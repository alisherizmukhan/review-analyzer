"""MCP tests for the data-source configuration, output types, sentiment-source labels and small-sample handling.

Uses the fixture-trained LogReg model (offline) and small hand-written corpora; nothing depends on local
Amazon runs, the network or a GPU.
"""

from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

import server
from review_intel.app import AppConfig, ReviewIntelligence
from review_intel.inference.service import ServiceConfig
from review_intel.testing import prepare_fixture, train_fixture_models
from review_intel.utils import PROJECT_ROOT

PROJECT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def fixture_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("mcp_fixes")
    train_fixture_models(prepare_fixture(root / "processed"), root / "models", root / "reports")
    return {"model_dir": root / "models" / "tfidf_logreg"}


def call(name: str, arguments: dict):
    async def go():
        async with Client(server.mcp) as client:
            return await client.call_tool(name, arguments)
    return asyncio.run(go())


def list_tools():
    async def go():
        async with Client(server.mcp) as client:
            return {t.name: t for t in (await client.list_tools()).tools}
    return asyncio.run(go())

# Small corpus in the processed-CSV format: one product with a single unnamed review, two named products.
SMALL_ROWS = [
    (1, "SOLO1", "", "The battery died after a week.", 1, "negative"),
    (2, "P2", "Alpha Speaker", "Great sound and the price is fair.", 5, "positive"),
    (3, "P2", "Alpha Speaker", "The battery is weak.", 2, "negative"),
    (4, "P2", "Alpha Speaker", "Good speaker, solid battery.", 4, "positive"),
    (5, "P3", "Beta Speaker", "Terrible battery and bad sound.", 1, "negative"),
    (6, "P3", "Beta Speaker", "It is okay.", 3, "neutral"),
    (7, "P3", "Beta Speaker", "Great value for the price.", 5, "positive"),
]


@pytest.fixture
def small_corpus(tmp_path):
    path = tmp_path / "small.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["review_id", "product_id", "product_name", "text", "rating", "label", "source_row"])
        for i, row in enumerate(SMALL_ROWS):
            writer.writerow([*row, i])
    return path


def make_server(monkeypatch, fixture_env, data_file, min_reviews=30):
    config = AppConfig(service=ServiceConfig(model_dir=fixture_env["model_dir"]), min_product_reviews=min_reviews)
    monkeypatch.setattr(server, "app", ReviewIntelligence(config))
    monkeypatch.setattr(server, "DATA_FILE", data_file)
    return server


@pytest.fixture
def small_server(monkeypatch, fixture_env, small_corpus):
    return make_server(monkeypatch, fixture_env, small_corpus)


# --- Task 1: data-source configuration -------------------------------------------

def test_default_data_is_amazon_corpus_not_demo():
    path = server.resolve_data_file({})
    assert path == PROJECT_ROOT / "data/processed/amazon_electronics/analytics_corpus.csv"
    assert path != server.DEMO_DATA_FILE


def test_demo_mode_must_be_explicit():
    assert server.resolve_data_file({"REVIEW_ANALYZER_DATA": "demo"}) == server.DEMO_DATA_FILE
    assert server.resolve_data_file({"REVIEW_ANALYZER_DATA": " DEMO "}) == server.DEMO_DATA_FILE
    assert server.DATA_FILE == server.DEMO_DATA_FILE  # the test session sets REVIEW_ANALYZER_DATA=demo


def test_relative_path_resolves_from_project_root_not_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = server.resolve_data_file({"REVIEW_ANALYZER_DATA": "review-analyzer-mcp/reviews.json"})
    assert path == PROJECT_ROOT / "review-analyzer-mcp" / "reviews.json" and path.exists()
    absolute = tmp_path / "x.csv"
    assert server.resolve_data_file({"REVIEW_ANALYZER_DATA": str(absolute)}) == absolute


@pytest.mark.parametrize("tool,args", [
    ("get_review_statistics", {}),
    ("search_reviews", {"keyword": "battery"}),
    ("list_products", {}),
    ("get_product_summary", {"product": "P2"}),
    ("get_negative_aspects", {}),
    ("compare_products", {"product_a": "P2", "product_b": "P3"}),
    ("search_similar_reviews", {"query": "battery"}),
])
def test_missing_data_file_gives_actionable_error(monkeypatch, fixture_env, tmp_path, tool, args):
    make_server(monkeypatch, fixture_env, tmp_path / "missing.csv")
    result = call(tool, args)
    assert result.is_error
    message = result.content[0].text
    assert "Review dataset not found" in message and "REVIEW_ANALYZER_DATA" in message
    assert "build_corpus.py" in message and "demo" in message
    assert "Traceback" not in message


def test_model_info_reports_missing_corpus(monkeypatch, fixture_env, tmp_path):
    make_server(monkeypatch, fixture_env, tmp_path / "missing.csv")
    out = call("get_model_info", {}).structured_content
    assert out["status"] == "ready" and "not found" in out["corpus"]["error"]


def test_model_info_describes_corpus(small_server):
    corpus = call("get_model_info", {}).structured_content["corpus"]
    assert corpus["reviews"] == 7 and corpus["products"] == 3 and corpus["is_demo"] is False


def test_model_info_reads_provenance_from_pipeline_manifest(small_server, small_corpus):
    (small_corpus.parent / "manifest.json").write_text(
        '{"name": "products_x", "source": {"origin": "collected per product"}}', encoding="utf-8")
    corpus = call("get_model_info", {}).structured_content["corpus"]
    assert corpus["dataset"] == "products_x" and corpus["note"] == "collected per product"


def test_stdio_missing_configured_file_is_an_error_not_demo_fallback(tmp_path):
    params = StdioServerParameters(
        command=sys.executable, args=[str(PROJECT_DIR / "server.py")], cwd=str(tmp_path),
        env={"REVIEW_ANALYZER_DATA": "data/does-not-exist.csv", "REVIEW_ANALYZER_SENTIMENT_BACKEND": "keyword"})

    async def session():
        async with Client(params) as client:
            return await client.call_tool("get_review_statistics", {})

    result = asyncio.run(session())
    assert result.is_error
    assert "data/does-not-exist.csv" in result.content[0].text


# --- Task 2: integer counts --------------------------------------------------------

def test_statistics_counts_are_integers(small_server):
    tools = list_tools()
    schema = tools["get_review_statistics"].output_schema
    counts_schema = schema["$defs"][schema["properties"]["counts"]["$ref"].split("/")[-1]]
    assert {p["type"] for p in counts_schema["properties"].values()} == {"integer"}

    out = call("get_review_statistics", {}).structured_content
    assert out["counts"] == {"positive": 3, "negative": 3, "neutral": 1}
    assert all(type(v) is int for v in out["counts"].values())
    assert out["percentages"] == {"positive": 42.86, "negative": 42.86, "neutral": 14.29}
    assert all(isinstance(v, float) for v in out["percentages"].values())
    assert type(out["total_reviews"]) is int and type(out["rated_reviews"]) is int


def test_other_tools_return_integer_counts(small_server):
    summary = call("get_product_summary", {"product": "P2"}).structured_content
    assert summary["sentiment_distribution"]["positive"] == {"count": 2, "percent": 66.67}
    assert all(type(v["count"]) is int for v in summary["sentiment_distribution"].values())
    assert all(type(v) is int for v in summary["rating_distribution"].values())
    products = call("list_products", {}).structured_content
    assert all(type(p["review_count"]) is int for p in products["products"])
    comparison = call("compare_products", {"product_a": "P2", "product_b": "P3"}).structured_content
    assert all(type(v["count"]) is int for v in comparison["product_a"]["sentiment_distribution"].values())


# --- Task 3: sentiment sources -------------------------------------------------------

def test_analyze_sentiment_is_marked_as_model_prediction(small_server):
    out = call("analyze_sentiment", {"text": "The battery died fast."}).structured_content
    source = out["sentiment_sources"]["sentiment"]
    assert source["source"] == "model_prediction" and source["model_version"] == out["model_version"]


def test_keyword_baseline_is_not_marked_as_model_prediction():
    # The test session configures the keyword demo baseline explicitly (see conftest.py).
    out = call("analyze_sentiment", {"text": "Great battery"}).structured_content
    assert out["is_trained_model"] is False
    assert out["sentiment_sources"]["sentiment"]["source"] == "keyword_baseline_prediction"


def test_statistics_and_search_are_marked_as_star_rating_labels(small_server):
    stats = call("get_review_statistics", {}).structured_content
    assert stats["sentiment_sources"]["counts"]["source"] == "star_rating_label"
    search = call("search_reviews", {"keyword": "battery"}).structured_content
    assert search["sentiment_sources"]["reviews.sentiment"]["source"] == "star_rating_label"
    similar = call("search_similar_reviews", {"query": "battery"}).structured_content
    assert similar["sentiment_sources"]["results.sentiment"]["source"] == "star_rating_label"


def test_demo_corpus_without_ratings_is_marked_as_dataset_label():
    out = call("get_review_statistics", {}).structured_content  # demo corpus: labels, no star ratings
    assert out["sentiment_sources"]["counts"]["source"] == "dataset_label"


def test_mixed_tools_separate_label_and_prediction_sources(small_server):
    summary = call("get_product_summary", {"product": "P2"}).structured_content
    sources = summary["sentiment_sources"]
    assert sources["sentiment_distribution"]["source"] == "star_rating_label"
    assert sources["representative_reviews"]["source"] == "star_rating_label"
    assert sources["most_negative_aspects"]["source"] == "model_prediction"
    assert sources["most_negative_aspects"]["model_version"].startswith("tfidf_logreg@")
    aspects = call("get_negative_aspects", {}).structured_content
    assert aspects["sentiment_sources"]["negative_aspects"]["source"] == "model_prediction"
    comparison = call("compare_products", {"product_a": "P2", "product_b": "P3"}).structured_content
    assert comparison["sentiment_sources"]["sentiment_distribution"]["source"] == "star_rating_label"
    assert comparison["sentiment_sources"]["shared_aspects"]["source"] == "model_prediction"


# --- Task 5: small samples and missing names ------------------------------------------

def test_list_products_reports_sample_coverage(small_server):
    out = call("list_products", {}).structured_content
    assert (out["products_in_corpus"], out["products_with_one_review"], out["products_with_name"]) == (3, 1, 2)
    assert out["min_reviews_for_summary"] == 30 and out["products_meeting_min_reviews"] == 0
    assert out["max_reviews_per_product"] == 3
    assert any("no product_name" in note for note in out["notes"])
    solo = next(p for p in out["products"] if p["product"] == "SOLO1")
    assert solo["product_name"] == ""


def test_one_review_unnamed_product_is_flagged(small_server):
    out = call("get_product_summary", {"product": "SOLO1"}).structured_content
    assert out["product_name"] == "" and out["total_reviews"] == 1
    assert out["sample"] == {"reviews": 1, "min_reviews": 30, "sufficient": False, "warning": out["sample"]["warning"]}
    assert "single review" in out["sample"]["warning"]
    assert out["sentiment_ci95"]["negative"][0] < 100 and out["sentiment_ci95"]["negative"][1] == 100.0
    assert any("No product name" in note for note in out["notes"])


def test_comparison_with_insufficient_data_is_not_reliable(small_server):
    out = call("compare_products", {"product_a": "SOLO1", "product_b": "P2"}).structured_content
    assert out["comparison_reliable"] is False
    assert out["product_a"]["sample"]["sufficient"] is False
    assert any(note.startswith("SOLO1 has only 1 review") for note in out["notes"])


def test_comparison_with_enough_data_is_reliable(monkeypatch, fixture_env, small_corpus):
    make_server(monkeypatch, fixture_env, small_corpus, min_reviews=3)
    out = call("compare_products", {"product_a": "alpha speaker", "product_b": "P3"}).structured_content
    assert out["comparison_reliable"] is True
    assert out["product_a"]["product"] == "P2" and out["product_b"]["product"] == "P3"
    assert out["negative_percent_ci95_overlap"] is True  # 1/3 vs 1/3: no evidence of a difference
    assert not any("has only" in note for note in out["notes"])


def test_same_product_by_id_and_name_is_rejected(small_server):
    result = call("compare_products", {"product_a": "P2", "product_b": "Alpha Speaker"})
    assert result.is_error and "both refer to product 'P2'" in result.content[0].text


def test_min_product_reviews_is_configurable():
    assert AppConfig.from_env({"REVIEW_ANALYZER_MIN_PRODUCT_REVIEWS": "5"}).min_product_reviews == 5
    assert AppConfig.from_env({}).min_product_reviews == 30
    with pytest.raises(ValueError, match="REVIEW_ANALYZER_MIN_PRODUCT_REVIEWS"):
        AppConfig.from_env({"REVIEW_ANALYZER_MIN_PRODUCT_REVIEWS": "0"})


# --- search_similar_reviews: optional sentiment filter ------------------------------------

def test_similar_default_is_unfiltered_and_backward_compatible(small_server):
    default = call("search_similar_reviews", {"query": "battery", "limit": 7}).structured_content
    explicit = call("search_similar_reviews", {"query": "battery", "limit": 7, "sentiment_filter": "any"}).structured_content
    assert [r["id"] for r in default["results"]] == [r["id"] for r in explicit["results"]]
    assert len(default["results"]) == 7  # nothing filtered out
    assert default["sentiment_filter"]["sentiment"] == "any" and default["sentiment_filter"]["source"] is None
    assert all(r["predicted_sentiment"] is None and r["predicted_score"] is None for r in default["results"])
    assert set(default["sentiment_sources"]) == {"results.sentiment"}
    for key in ("id", "product", "product_name", "text", "sentiment", "rating", "similarity"):  # v2 fields kept
        assert key in default["results"][0]


def test_similar_star_label_filters(small_server):
    negative = call("search_similar_reviews", {"query": "battery", "limit": 7, "sentiment_filter": "negative",
                                               "sentiment_source": "star_rating_label"}).structured_content
    assert {r["id"] for r in negative["results"]} == {1, 3, 5}  # the three 1-2 star reviews
    assert negative["sentiment_filter"]["source"] == "star_rating_label"
    assert "star-rating label" in negative["sentiment_filter"]["description"]
    assert negative["sentiment_filter"]["model_version"] is None
    assert all(r["predicted_sentiment"] is None for r in negative["results"])
    positive = call("search_similar_reviews", {"query": "battery", "limit": 7, "sentiment_filter": "positive",
                                               "sentiment_source": "star_rating_label"}).structured_content
    assert positive["results"] and all(r["sentiment"] == "positive" for r in positive["results"])


def test_similar_model_filter_is_labelled_with_model_version(small_server):
    out = call("search_similar_reviews", {"query": "battery", "limit": 7, "sentiment_filter": "negative",
                                          "sentiment_source": "model_prediction"}).structured_content
    info = out["sentiment_filter"]
    assert info["source"] == "model_prediction" and info["model_version"].startswith("tfidf_logreg@")
    assert info["candidate_pool"] == 200 and info["candidates_scanned"] == 7
    assert all(r["predicted_sentiment"] == "negative" for r in out["results"])
    assert out["sentiment_sources"]["results.predicted_sentiment"]["source"] == "model_prediction"
    assert out["sentiment_sources"]["results.sentiment"]["source"] == "star_rating_label"  # kept separate
    positive = call("search_similar_reviews", {"query": "battery", "limit": 7, "sentiment_filter": "positive",
                                               "sentiment_source": "model_prediction"}).structured_content
    assert all(r["predicted_sentiment"] == "positive" for r in positive["results"])
    assert not {r["id"] for r in positive["results"]} & {r["id"] for r in out["results"]}


def test_similar_empty_result_is_reported(small_server):
    out = call("search_similar_reviews", {"query": "battery", "product": "SOLO1", "sentiment_filter": "positive",
                                          "sentiment_source": "star_rating_label"})
    assert not out.is_error
    data = out.structured_content
    assert data["results"] == [] and data["sentiment_filter"]["candidates_matching"] == 0
    assert data["sentiment_filter"]["note"].startswith("Only 0 reviews")


@pytest.mark.parametrize("args, message", [
    ({"sentiment_filter": "angry"}, "sentiment_filter"),
    ({"sentiment_filter": "negative", "sentiment_source": "stars"}, "sentiment_source"),
    ({"sentiment_filter": "negative"}, "needs sentiment_source"),
    ({"sentiment_source": "model_prediction"}, "only applies when sentiment_filter"),
    ({"sentiment_filter": "negative", "sentiment_source": "model_prediction", "candidate_pool": 0}, "candidate_pool"),
])
def test_similar_invalid_filter_arguments(small_server, args, message):
    result = call("search_similar_reviews", {"query": "battery", **args})
    assert result.is_error and message in result.content[0].text


def test_star_label_filter_refused_when_labels_are_not_star_ratings():
    # The session's demo corpus has hand-written labels and no ratings.
    result = call("search_similar_reviews", {"query": "battery", "sentiment_filter": "negative",
                                             "sentiment_source": "star_rating_label"})
    assert result.is_error and "not derived from star ratings" in result.content[0].text


def test_all_tools_still_discoverable_with_filter_parameters():
    tools = list_tools()
    assert len(tools) == 9
    props = tools["search_similar_reviews"].input_schema["properties"]
    assert props["sentiment_filter"]["enum"] == ["any", "negative", "neutral", "positive"]
    assert props["sentiment_filter"]["default"] == "any"
    assert tools["search_similar_reviews"].input_schema["required"] == ["query"]
