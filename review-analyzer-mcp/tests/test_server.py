"""Tests for the Review Analyzer MCP server.

Part 1 tests the plain Python functions directly (no MCP involved).
Part 2 tests the MCP layer: tool discovery, argument validation, results and
errors, both in-process and through a real stdio subprocess.
No network access or LLM is needed.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

import server
from server import (
    ReviewDataError,
    analyze_text_sentiment,
    compute_statistics,
    find_reviews,
    load_reviews,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def reviews():
    return load_reviews(server.DATA_FILE)


def write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ===========================================================================
# Part 1: pure Python business logic
# ===========================================================================


class TestSentiment:
    def test_positive(self):
        result = analyze_text_sentiment("Great phone, I love the excellent screen!")
        assert result["sentiment"] == "positive"
        assert result["method"] == "keyword_baseline_v1"
        assert "not a trained NLP model" in result["warning"]

    def test_negative(self):
        assert analyze_text_sentiment("Terrible camera and the battery drains fast.")["sentiment"] == "negative"

    def test_neutral_no_keywords(self):
        assert analyze_text_sentiment("The phone comes in a blue box.")["sentiment"] == "neutral"

    def test_neutral_on_tie(self):
        assert analyze_text_sentiment("Good screen, bad speaker.")["sentiment"] == "neutral"

    def test_returns_input_text(self):
        assert analyze_text_sentiment("I love it")["text"] == "I love it"

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_empty_input_rejected(self, text):
        with pytest.raises(ValueError, match="empty"):
            analyze_text_sentiment(text)

    def test_baseline_agrees_with_sample_labels(self, reviews):
        for review in reviews:
            assert analyze_text_sentiment(review["text"])["sentiment"] == review["sentiment"], review["id"]


class TestStatistics:
    def test_all_reviews(self, reviews):
        stats = compute_statistics(reviews)
        assert stats["product"] == "all"
        assert stats["total_reviews"] == 5
        assert stats["counts"] == {"positive": 2, "negative": 2, "neutral": 1}

    def test_specific_product_case_insensitive(self, reviews):
        stats = compute_statistics(reviews, "nova x1")
        assert stats["total_reviews"] == 3
        assert stats["counts"] == {"positive": 1, "negative": 2, "neutral": 0}

    def test_nonexistent_product(self, reviews):
        stats = compute_statistics(reviews, "No Such Product")
        assert stats["total_reviews"] == 0
        assert stats["counts"] == {"positive": 0, "negative": 0, "neutral": 0}
        assert stats["percentages"] == {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    def test_percentages(self, reviews):
        assert compute_statistics(reviews)["percentages"] == {"positive": 40.0, "negative": 40.0, "neutral": 20.0}
        # 1/3 and 2/3 are rounded to 2 decimals.
        assert compute_statistics(reviews, "Nova X1")["percentages"] == {
            "positive": 33.33,
            "negative": 66.67,
            "neutral": 0.0,
        }


class TestSearch:
    def test_keyword(self, reviews):
        result = find_reviews(reviews, "BATTERY")
        assert result["match_count"] == 2
        assert [r["id"] for r in result["reviews"]] == [1, 4]

    def test_keyword_with_product_filter(self, reviews):
        result = find_reviews(reviews, "the", product="pulse earbuds")
        assert {r["product"] for r in result["reviews"]} == {"Pulse Earbuds"}
        assert [r["id"] for r in result["reviews"]] == [3, 5]

    def test_no_results(self, reviews):
        result = find_reviews(reviews, "refund")
        assert result["match_count"] == 0
        assert result["reviews"] == []

    @pytest.mark.parametrize("keyword", ["", "   "])
    def test_empty_keyword_rejected(self, reviews, keyword):
        with pytest.raises(ValueError, match="empty"):
            find_reviews(reviews, keyword)


class TestDataLoading:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ReviewDataError, match="not found"):
            load_reviews(tmp_path / "missing.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("[{ not valid json", encoding="utf-8")
        with pytest.raises(ReviewDataError, match="not valid JSON"):
            load_reviews(path)

    def test_top_level_not_a_list(self, tmp_path):
        with pytest.raises(ReviewDataError, match="JSON list"):
            load_reviews(write_json(tmp_path / "r.json", {"id": 1}))

    @pytest.mark.parametrize(
        ("record", "message"),
        [
            ({"id": 1, "product": "P", "text": "t"}, "missing required field"),
            ({"id": "1", "product": "P", "text": "t", "sentiment": "neutral"}, "'id' must be an integer"),
            ({"id": 1, "product": " ", "text": "t", "sentiment": "neutral"}, "'product' must be"),
            ({"id": 1, "product": "P", "text": 42, "sentiment": "neutral"}, "'text' must be"),
            ({"id": 1, "product": "P", "text": "t", "sentiment": "angry"}, "'sentiment' must be one of"),
        ],
    )
    def test_malformed_record(self, tmp_path, record, message):
        path = write_json(tmp_path / "r.json", [record])
        with pytest.raises(ReviewDataError, match=message):
            load_reviews(path)


# ===========================================================================
# Part 2: MCP protocol layer
# ===========================================================================


def run(coro):
    return asyncio.run(coro)


async def call(name: str, arguments: dict):
    """Call a tool through an in-process MCP client (real MCP messages, no subprocess)."""
    async with Client(server.mcp) as client:
        return await client.call_tool(name, arguments)


class TestMcpTools:
    def test_tools_are_discoverable(self):
        async def list_tools():
            async with Client(server.mcp) as client:
                return (await client.list_tools()).tools

        tools = {t.name: t for t in run(list_tools())}
        # v2 added tools; the three v1 tools must still be present.
        assert {"analyze_sentiment", "get_review_statistics", "search_reviews"} <= set(tools)
        # Type hints become the JSON Schema the client uses to validate arguments.
        assert tools["analyze_sentiment"].input_schema["required"] == ["text"]
        assert tools["search_reviews"].input_schema["required"] == ["keyword"]
        assert "product" in tools["get_review_statistics"].input_schema["properties"]
        assert all(t.description and t.annotations.read_only_hint for t in tools.values())

    def test_analyze_sentiment_tool(self):
        result = run(call("analyze_sentiment", {"text": "Excellent and fast!"}))
        assert not result.is_error
        assert result.structured_content["sentiment"] == "positive"

    def test_statistics_tool(self):
        result = run(call("get_review_statistics", {"product": "Nova X1"}))
        assert result.structured_content["total_reviews"] == 3

    def test_search_tool(self):
        result = run(call("search_reviews", {"keyword": "camera"}))
        assert [r["id"] for r in result.structured_content["reviews"]] == [2]

    def test_empty_text_returns_tool_error(self):
        result = run(call("analyze_sentiment", {"text": "  "}))
        assert result.is_error
        assert "must not be empty" in result.content[0].text

    def test_missing_argument_rejected_by_schema(self):
        result = run(call("search_reviews", {}))
        assert result.is_error

    def test_missing_dataset_returns_tool_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(server, "DATA_FILE", tmp_path / "missing.json")
        result = run(call("get_review_statistics", {}))
        assert result.is_error
        assert "not found" in result.content[0].text

    def test_malformed_dataset_returns_tool_error(self, monkeypatch, tmp_path):
        bad = tmp_path / "reviews.json"
        bad.write_text("not json", encoding="utf-8")
        monkeypatch.setattr(server, "DATA_FILE", bad)
        result = run(call("search_reviews", {"keyword": "battery"}))
        assert result.is_error
        assert "not valid JSON" in result.content[0].text


def test_stdio_end_to_end(tmp_path):
    """Launch server.py as a subprocess over stdio, the way a real MCP host does.

    Runs from an unrelated working directory to prove the dataset path does not
    depend on the current directory, and fails if anything pollutes stdout.
    """
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_DIR / "server.py")],
        cwd=str(tmp_path),
        env={"REVIEW_ANALYZER_DATA": "demo"},  # the demo corpus must be requested explicitly
    )

    async def session():
        async with Client(params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            stats = await client.call_tool("get_review_statistics", {})
            return names, stats

    names, stats = run(session())
    assert {"analyze_sentiment", "get_review_statistics", "search_reviews"} <= names
    assert stats.structured_content["total_reviews"] == 5
