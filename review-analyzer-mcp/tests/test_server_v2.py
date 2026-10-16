"""MCP tests for the v2 tools: discovery, schemas, valid and invalid calls, errors, and stdio.

Uses a model trained on the SYNTHETIC fixture (offline, a fraction of a
second) and the fixture as corpus, so nothing depends on local Amazon runs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

import server
from review_intel.app import AppConfig, ReviewIntelligence
from review_intel.inference.service import ServiceConfig
from review_intel.testing import prepare_fixture, train_fixture_models

PROJECT_DIR = Path(__file__).resolve().parent.parent
V2_TOOLS = {"list_products", "get_product_summary", "get_negative_aspects", "compare_products",
            "search_similar_reviews", "get_model_info"}


@pytest.fixture(scope="module")
def fixture_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("mcp_v2")
    data = prepare_fixture(root / "processed")
    corpus = prepare_fixture(root / "corpus", split_strategy="none") / "corpus.csv"
    train_fixture_models(data, root / "models", root / "reports")
    return {"corpus": corpus, "model_dir": root / "models" / "tfidf_logreg"}


@pytest.fixture
def trained_server(fixture_env, monkeypatch):
    app = ReviewIntelligence(AppConfig(service=ServiceConfig(model_dir=fixture_env["model_dir"], max_text_chars=300)))
    monkeypatch.setattr(server, "app", app)
    monkeypatch.setattr(server, "DATA_FILE", fixture_env["corpus"])
    return server


@pytest.fixture
def server_without_model(fixture_env, monkeypatch, tmp_path):
    config = AppConfig(service=ServiceConfig(backend="trained", selected_pointer=tmp_path / "selected.json"))
    monkeypatch.setattr(server, "app", ReviewIntelligence(config))
    monkeypatch.setattr(server, "DATA_FILE", fixture_env["corpus"])
    return server


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


# --- discovery & schemas ----------------------------------------------------

def test_all_tools_discoverable_with_schemas():
    tools = list_tools()
    assert V2_TOOLS | {"analyze_sentiment", "get_review_statistics", "search_reviews"} == set(tools)
    for tool in tools.values():
        assert tool.description and tool.output_schema and tool.annotations.read_only_hint
    assert tools["get_product_summary"].input_schema["required"] == ["product"]
    assert tools["compare_products"].input_schema["required"] == ["product_a", "product_b"]
    limit = tools["get_negative_aspects"].input_schema["properties"]["limit"]
    assert (limit["minimum"], limit["maximum"], limit["default"]) == (1, 20, 5)
    assert tools["search_reviews"].input_schema["properties"]["limit"]["maximum"] == 100
    assert "score_type" in tools["analyze_sentiment"].output_schema["properties"]


# --- valid calls ------------------------------------------------------------

def test_analyze_sentiment_with_trained_model(trained_server):
    result = call("analyze_sentiment", {"text": "The battery drains way too fast, very disappointing."})
    out = result.structured_content
    assert not result.is_error
    assert out["is_trained_model"] is True and out["method"] == "tfidf_logreg"
    assert out["score_type"] == "probability"
    assert sum(out["scores"].values()) == pytest.approx(1.0, abs=1e-3)
    assert out["model_version"].startswith("tfidf_logreg@synthetic_fixture")
    assert "trained model" in out["warning"]


def test_list_products(trained_server):
    out = call("list_products", {"limit": 2}).structured_content
    assert out["total_reviews"] == 64 and len(out["products"]) == 2
    assert out["products"][0]["review_count"] >= out["products"][1]["review_count"]


def test_get_product_summary(trained_server):
    out = call("get_product_summary", {"product": "nova x1 phone"}).structured_content
    assert out["product"] == "Nova X1 Phone"
    assert out["total_reviews"] == sum(v["count"] for v in out["sentiment_distribution"].values())
    assert out["average_rating"] is not None
    assert out["most_mentioned_aspects"] and "limitations" in out["method"]["aspects"]


def test_get_negative_aspects(trained_server):
    out = call("get_negative_aspects", {"limit": 3}).structured_content
    assert out["product"] == "all" and out["reviews_analyzed"] == 64
    assert len(out["negative_aspects"]) <= 3
    for aspect in out["negative_aspects"]:
        assert aspect["sentiment_counts"]["negative"] > 0 and aspect["negative_examples"]


def test_compare_products(trained_server):
    out = call("compare_products", {"product_a": "Nova X1 Phone", "product_b": "Pulse Earbuds"}).structured_content
    assert out["product_a"]["product"] == "Nova X1 Phone" and out["product_b"]["product"] == "Pulse Earbuds"
    assert "negative_percent_a_minus_b" in out["differences"]
    assert out["shared_aspects"]


def test_search_similar_reviews(trained_server):
    out = call("search_similar_reviews", {"query": "battery dies quickly", "product": "Orbit Laptop", "limit": 3})
    data = out.structured_content
    assert data["method"].startswith("lsa") and len(data["results"]) == 3
    assert all(r["product"] == "Orbit Laptop" for r in data["results"])
    # LSA is lexical-latent: rare shared words ("quickly") can outrank the topic word, so we only require
    # a battery review among the top 3 (a documented limitation of the offline retriever).
    assert any("battery" in r["text"].lower() for r in data["results"])


def test_search_reviews_limit_keeps_total_count(trained_server):
    out = call("search_reviews", {"keyword": "the", "limit": 2}).structured_content
    assert len(out["reviews"]) == 2 and out["match_count"] > 2


def test_get_model_info_ready(trained_server):
    out = call("get_model_info", {}).structured_content
    assert out["status"] == "ready"
    assert out["model"]["name"] == "tfidf_logreg" and out["model"]["is_trained_model"] is True
    assert out["model"]["dataset"]["is_synthetic"] is True
    assert out["corpus"]["reviews"] == 64 and out["aspect_lexicon"] == "electronics-v1"


# --- invalid calls / errors -------------------------------------------------

@pytest.mark.parametrize(("tool", "args", "message"), [
    ("get_product_summary", {"product": "No Such Thing"}, "list_products"),
    ("get_product_summary", {"product": " "}, "must not be empty"),
    ("compare_products", {"product_a": "Pulse Earbuds", "product_b": "pulse earbuds"}, "different"),
    ("search_similar_reviews", {"query": "   "}, "must not be empty"),
    ("search_similar_reviews", {"query": "battery", "product": "Nope"}, "list_products"),
    ("analyze_sentiment", {"text": "x" * 301}, "maximum is 300"),
])
def test_expected_errors_are_tool_errors(trained_server, tool, args, message):
    result = call(tool, args)
    assert result.is_error
    assert message in result.content[0].text


@pytest.mark.parametrize(("tool", "args"), [
    ("get_negative_aspects", {"limit": 0}),
    ("get_negative_aspects", {"limit": 21}),
    ("list_products", {"limit": "many"}),
    ("compare_products", {"product_a": "A"}),
])
def test_schema_validation_rejects_bad_arguments(trained_server, tool, args):
    assert call(tool, args).is_error


def test_unexpected_exception_does_not_leak_internals(trained_server, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal path C:/very/private")
    monkeypatch.setattr(server.stats, "product_summary", boom)
    result = call("get_product_summary", {"product": "Pulse Earbuds"})
    assert result.is_error
    assert "secret" not in result.content[0].text


def test_missing_model_is_actionable_and_not_a_silent_fallback(server_without_model):
    result = call("analyze_sentiment", {"text": "great"})
    assert result.is_error
    assert "train_baselines.py" in result.content[0].text
    info = call("get_model_info", {}).structured_content
    assert info["status"] == "unavailable" and "REVIEW_ANALYZER_SENTIMENT_BACKEND=keyword" in info["message"]
    # Tools that need no model keep working.
    assert not call("get_review_statistics", {}).is_error


# --- real stdio subprocess --------------------------------------------------

def test_stdio_v2_with_trained_model(fixture_env, tmp_path):
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_DIR / "server.py")],
        cwd=str(tmp_path),
        env={"REVIEW_ANALYZER_DATA": str(fixture_env["corpus"]),
             "REVIEW_ANALYZER_MODEL_DIR": str(fixture_env["model_dir"]),
             "REVIEW_ANALYZER_SENTIMENT_BACKEND": "trained"},
    )

    async def session():
        async with Client(params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            sentiment = await client.call_tool("analyze_sentiment", {"text": "Love the sleek design"})
            summary = await client.call_tool("get_product_summary", {"product": "Pulse Earbuds"})
            bad = await client.call_tool("get_product_summary", {"product": "nope"})
            return names, sentiment, summary, bad

    names, sentiment, summary, bad = asyncio.run(session())
    assert V2_TOOLS <= names
    assert sentiment.structured_content["is_trained_model"] is True
    assert summary.structured_content["total_reviews"] > 0
    assert bad.is_error
