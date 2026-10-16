"""Per-product collection: bottom-k sampling, de-duplication, exclusions, chunking, resume, failures, MCP use.

Everything runs on local files and a local HTTP server; no network access.
"""

from __future__ import annotations

import http.server
import json
import threading
from collections import Counter
from pathlib import Path

import pytest

from review_intel.analytics.corpus import load_reviews
from review_intel.analytics.stats import product_overview, product_summary
from review_intel.data import amazon
from review_intel.data import product_collection as pc
from review_intel.data.pipeline import DatasetConfig, prepare_dataset

PRODUCTS = ["B000000001", "B000000002", "B000000003"]


def make_record(i: int, product: str, **overrides) -> dict:
    record = {"rating": float(1 + i % 5), "title": f"title {i}", "text": f"review number {i} about the battery",
              "images": [], "asin": product, "parent_asin": product, "user_id": f"U{i}", "timestamp": 1_600_000_000 + i,
              "verified_purchase": True, "helpful_vote": 0}
    record.update(overrides)
    return record


def write_source(path: Path, records: list[dict], blank_lines: bool = False) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
            if blank_lines:
                handle.write("\n")
    return path


@pytest.fixture
def source(tmp_path):
    """300 reviews: product 1 has 150, product 2 has 100, product 3 has 50; plus 200 reviews of other products."""
    records = []
    for i in range(500):
        product = (PRODUCTS[0] if i % 10 < 3 else PRODUCTS[1] if i % 10 < 5 else PRODUCTS[2] if i % 10 < 6
                   else f"B9{i:08d}")
        records.append(make_record(i, product))
    return write_source(tmp_path / "reviews.jsonl", records), records


def collect(src: Path, state_dir: Path, **kwargs):
    defaults = {"cap": 40, "seed": 7, "chunk_bytes": 4096, "workers": 3}
    defaults.update(kwargs)
    cap, seed = defaults.pop("cap"), defaults.pop("seed")
    return pc.collect_products(str(src), PRODUCTS, cap, seed, state_dir, **defaults)


# --- selection and validation ----------------------------------------------------------

def test_invalid_product_ids_rejected():
    with pytest.raises(ValueError, match="Invalid product id"):
        pc.validate_product_ids(["B075X8471B", "not-an-asin"])
    with pytest.raises(ValueError, match="No product ids"):
        pc.validate_product_ids([])
    assert pc.validate_product_ids(["B075X8471B", " B075X8471B ", "0123456789"]) == ["B075X8471B", "0123456789"]


def test_rank_products_in_sample(tmp_path):
    sample = write_source(tmp_path / "s.jsonl", [make_record(i, PRODUCTS[i % 3 != 0]) for i in range(9)])
    ranked = pc.rank_products_in_sample(sample, 5)
    assert ranked == [{"rank": 1, "parent_asin": PRODUCTS[1], "reviews_in_research_sample": 6},
                      {"rank": 2, "parent_asin": PRODUCTS[0], "reviews_in_research_sample": 3}]


# --- chunking ------------------------------------------------------------------------------

def test_chunks_cover_every_line_exactly_once(source):
    src, records = source
    size = src.stat().st_size
    for chunk_bytes in (97, 1000, 4096, size, size * 2):
        offsets = []
        for chunk in pc.plan_chunks(size, chunk_bytes):
            with pc.default_opener(str(src), max(chunk.start - 1, 0)) as stream:
                offsets += [o for o, _ in pc.iter_chunk_lines(stream, chunk.start, chunk.end)]
        assert len(offsets) == len(records) and offsets == sorted(set(offsets)), chunk_bytes


# --- sampling, caps, duplicates, exclusions ------------------------------------------------

def test_cap_counts_and_product_association(source, tmp_path):
    src, records = source
    result = collect(src, tmp_path / "state")
    rows = result.collector.rows()
    counts = Counter(r["parent_asin"] for r in rows)
    assert counts == {PRODUCTS[0]: 40, PRODUCTS[1]: 40, PRODUCTS[2]: 40}
    assert result.total_lines == 500
    assert result.collector.reviews_in_source == {PRODUCTS[0]: 150, PRODUCTS[1]: 100, PRODUCTS[2]: 50}
    by_offset = {}
    offset = 0
    for record in records:
        by_offset[offset] = record
        offset += len(json.dumps(record)) + 1
    for row in rows:  # every kept review is the exact source line at its recorded offset
        original = by_offset[row["source_offset"]]
        assert row["parent_asin"] == original["parent_asin"] and row["text"] == original["text"]
        assert "images" not in row and row["review_key"] == pc.review_key(original)


def test_products_below_cap_keep_all_reviews(source, tmp_path):
    src, _ = source
    rows = collect(src, tmp_path / "state", cap=1000).collector.rows()
    assert Counter(r["parent_asin"] for r in rows) == {PRODUCTS[0]: 150, PRODUCTS[1]: 100, PRODUCTS[2]: 50}


def test_sample_is_independent_of_chunking_and_workers(source, tmp_path):
    src, _ = source
    a = collect(src, tmp_path / "a", chunk_bytes=500, workers=1).collector.rows()
    b = collect(src, tmp_path / "b", chunk_bytes=50_000, workers=4).collector.rows()
    assert [r["review_key"] for r in a] == [r["review_key"] for r in b]
    c = collect(src, tmp_path / "c", seed=8).collector.rows()
    assert {r["review_key"] for r in a} != {r["review_key"] for r in c}  # the seed matters


def test_duplicate_reviews_removed(tmp_path):
    records = [make_record(i, PRODUCTS[0]) for i in range(20)]
    records += [dict(records[3]), dict(records[3]), dict(records[7])]  # exact duplicates (same review key)
    src = write_source(tmp_path / "dups.jsonl", records)
    result = collect(src, tmp_path / "state", cap=100, chunk_bytes=300)
    rows = result.collector.rows()
    assert len(rows) == 20 and len({r["review_key"] for r in rows}) == 20
    assert result.collector.duplicates[PRODUCTS[0]] == 3
    assert result.collector.reviews_in_source[PRODUCTS[0]] == 23


def test_research_sample_reviews_are_excluded(source, tmp_path):
    src, records = source
    sample = write_source(tmp_path / "sample.jsonl", [records[0], records[1], records[4]])  # products 1, 1, 2
    exclude = pc.load_exclusion_keys(sample)
    result = collect(src, tmp_path / "state", cap=1000, exclude_keys=exclude)
    keys = {r["review_key"] for r in result.collector.rows()}
    assert not keys & exclude
    assert result.collector.excluded == {PRODUCTS[0]: 2, PRODUCTS[1]: 1}


def test_missing_fields_and_bad_lines_are_handled(tmp_path):
    records = [make_record(0, PRODUCTS[0], rating=None), make_record(1, PRODUCTS[0], text=None),
               make_record(2, PRODUCTS[0])]
    records[2].pop("title")
    src = write_source(tmp_path / "missing.jsonl", records, blank_lines=True)
    with src.open("a", encoding="utf-8") as handle:
        handle.write('{"parent_asin": "B000000001", "broken json\n')
    result = collect(src, tmp_path / "state", cap=10)
    rows = result.collector.rows()
    assert len(rows) == 3 and result.total_lines == 4  # blank lines are not counted
    assert result.collector.stats["unparseable_lines"] == 1
    assert sorted(r["rating"] for r in rows if r["rating"] is not None) == [2.0, 3.0]
    assert sum(r["rating"] is None for r in rows) == 1 and sum(r["text"] is None for r in rows) == 1
    assert all(r["title"] is None for r in rows if r["user_id"] == "U2")  # missing -> None, never invented


def test_unknown_products_are_reported(source, tmp_path):
    src, _ = source
    result = pc.collect_products(str(src), [PRODUCTS[0], "B0NOTHERE1"], 5, 1, tmp_path / "state", chunk_bytes=4096)
    report = pc.collection_report(result.collector, [{"parent_asin": PRODUCTS[0]}, {"parent_asin": "B0NOTHERE1"}])
    assert report["products_not_found"] == ["B0NOTHERE1"]
    assert report["products"][1]["collected"] == 0 and report["rows"] == 5


# --- resume and failures ---------------------------------------------------------------------

class FlakyOpener:
    """Fails every chunk listed in ``fail`` (each time it is opened), counts which chunks were opened."""

    def __init__(self, fail_starts: set[int]):
        self.fail_starts = fail_starts
        self.opened: list[int] = []

    def __call__(self, source: str, start: int):
        self.opened.append(start)
        if start in self.fail_starts:
            raise ConnectionResetError("simulated network failure")
        return pc.default_opener(source, start)


def test_interrupted_collection_resumes_without_redoing_chunks(source, tmp_path):
    src, _ = source
    size = src.stat().st_size
    chunks = pc.plan_chunks(size, 4096)
    failing = {max(chunks[2].start - 1, 0)}
    state = tmp_path / "state"
    flaky = FlakyOpener(failing)
    with pytest.raises(pc.CollectionError, match="Run the same command again to resume"):
        collect(src, state, opener=flaky, chunk_retries=1)
    saved = sorted(state.glob("chunk_*.json"))
    assert len(saved) == len(chunks) - 1  # everything except the failed chunk was kept

    healthy = FlakyOpener(set())
    resumed = collect(src, state, opener=healthy)
    assert healthy.opened == [max(chunks[2].start - 1, 0)]  # only the missing chunk was read again
    assert resumed.chunks_resumed == len(chunks) - 1
    fresh = collect(src, tmp_path / "fresh")
    assert [r["review_key"] for r in resumed.collector.rows()] == [r["review_key"] for r in fresh.collector.rows()]


def test_resume_with_different_parameters_is_refused(source, tmp_path):
    src, _ = source
    collect(src, tmp_path / "state")
    with pytest.raises(pc.CollectionError, match="different parameters.*cap"):
        collect(src, tmp_path / "state", cap=41)


def _serve(handler_cls):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/reviews.jsonl"


def test_http_range_collection_matches_local(source, tmp_path):
    src, _ = source
    payload = src.read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            start = int(self.headers["Range"].split("=")[1].rstrip("-")) if self.headers.get("Range") else 0
            self.send_response(206 if start else 200)
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
            self.send_header("Content-Length", str(len(payload) - start))
            self.end_headers()
            self.wfile.write(payload[start:])

        def log_message(self, *args):
            pass

    server, url = _serve(Handler)
    try:
        assert pc.source_size(url)[0] == len(payload)
        remote = pc.collect_products(url, PRODUCTS, 40, 7, tmp_path / "remote", chunk_bytes=4096, workers=3)
    finally:
        server.shutdown()
    local = collect(src, tmp_path / "local")
    assert [r["review_key"] for r in remote.collector.rows()] == [r["review_key"] for r in local.collector.rows()]


def test_http_404_fails_fast_without_retries():
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            calls.append(1)
            self.send_error(404, "Not Found")

        def log_message(self, *args):
            pass

    server, url = _serve(Handler)
    try:
        with pytest.raises(amazon.SourceAccessError, match="HTTP 404"):
            pc.source_size(url)
    finally:
        server.shutdown()
    assert len(calls) == 1


def test_http_429_is_retried_with_retry_after(monkeypatch):
    calls = []
    sleeps = []
    monkeypatch.setattr(amazon.time, "sleep", sleeps.append)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            calls.append(1)
            if len(calls) == 1:
                self.send_response(429)
                self.send_header("Retry-After", "7")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", "bytes 1-9/10")
            self.send_header("Content-Length", "9")
            self.end_headers()
            self.wfile.write(b"x" * 9)

        def log_message(self, *args):
            pass

    server, url = _serve(Handler)
    try:
        assert pc.source_size(url)[0] == 10
    finally:
        server.shutdown()
    assert len(calls) == 2 and sleeps == [7.0]


# --- from collection to the MCP analytics layer ------------------------------------------------

def test_collected_reviews_feed_the_analytics_corpus(source, tmp_path):
    src, _ = source
    rows = collect(src, tmp_path / "state", cap=35).collector.rows()
    raw = write_source(tmp_path / "product_reviews.jsonl", rows)
    names = tmp_path / "names.json"
    names.write_text(json.dumps({PRODUCTS[0]: {"title": "Alpha"}}), encoding="utf-8")
    config = DatasetConfig.from_dict({
        "name": "products_test", "source_path": str(raw), "source_format": "jsonl",
        "columns": {"text": ["title", "text"], "product_id": "parent_asin", "rating": "rating",
                    "review_id": "source_offset"},
        "label_source": "rating", "product_names_path": str(names), "split": {"strategy": "none"}})
    prepared = prepare_dataset(config, tmp_path / "processed")
    corpus = load_reviews(tmp_path / "processed" / "corpus.csv")
    assert prepared.manifest["counts"]["rows_after_cleaning"] == len(corpus) == 105
    assert {r["id"] for r in corpus} == {r["source_offset"] for r in rows}  # stable ids survive
    overview = product_overview(corpus, min_reviews=30)
    assert overview["products_meeting_min_reviews"] == 3 and overview["products_with_name"] == 1


def test_enriched_corpus_works_with_product_summary(source, tmp_path):
    from review_intel.analytics.aspects import AspectAnalyzer, LexiconAspectExtractor
    from review_intel.models.keyword import KeywordSentimentModel

    src, _ = source
    rows = collect(src, tmp_path / "state", cap=35).collector.rows()
    corpus = [{"id": r["source_offset"], "product": r["parent_asin"], "text": r["text"],
               "sentiment": "positive" if r["rating"] >= 4 else "negative" if r["rating"] <= 2 else "neutral",
               "rating": r["rating"]} for r in rows]
    model = KeywordSentimentModel()
    analyzer = AspectAnalyzer(LexiconAspectExtractor.from_file(), model.predict,
                              {"name": model.name, "version": "kw", "is_trained_model": False})
    summary = product_summary(corpus, PRODUCTS[0], analyzer, min_reviews=30)
    assert summary["total_reviews"] == 35 and summary["sample"]["sufficient"] is True
