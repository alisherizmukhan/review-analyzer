"""Dataset pipeline: label mapping, cleaning, deduplication, splitting, manifest, Amazon sampling."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter

import pytest

from review_intel.data import amazon
from review_intel.data.labels import normalize_label, parse_rating, rating_to_label
from review_intel.data.pipeline import (
    DatasetConfig,
    DatasetConfigError,
    clean_records,
    deduplicate,
    prepare_dataset,
    read_processed_csv,
    split_rows,
    SplitConfig,
)
from review_intel.preprocessing.text import clean_text, combine_title_and_body, dedup_key, is_star_title, split_sentences


# --- labels -----------------------------------------------------------------

@pytest.mark.parametrize(("rating", "label"), [(1, "negative"), (2.0, "negative"), (3, "neutral"),
                                               (4, "positive"), (5.0, "positive")])
def test_rating_to_label_documented_rule(rating, label):
    assert rating_to_label(rating) == label


@pytest.mark.parametrize("rating", [None, 0, 6, 3.5, -1])
def test_invalid_ratings_rejected(rating):
    assert rating_to_label(rating) is None


@pytest.mark.parametrize(("raw", "parsed"), [("4", 4.0), (" 5.0 ", 5.0), ("", None), ("abc", None), (None, None),
                                             (True, None), ("nan", None)])
def test_parse_rating(raw, parsed):
    assert parse_rating(raw) == parsed


def test_normalize_label():
    assert normalize_label(" Positive ") == "positive"
    assert normalize_label("great") is None
    assert normalize_label(5) is None


# --- preprocessing ----------------------------------------------------------

def test_clean_text_strips_html_urls_and_whitespace():
    assert clean_text("Good&amp;cheap<br />see  https://x.com/a   now") == "Good&cheap see URL now"


@pytest.mark.parametrize("title", ["Five Stars", "one star", "3 Stars.", " FOUR STARS "])
def test_star_titles_detected(title):
    assert is_star_title(title)


def test_star_title_is_dropped_but_real_title_kept():
    assert combine_title_and_body("Five Stars", "Love it") == "Love it"
    assert combine_title_and_body("Meh!", "It works") == "Meh. It works"
    assert combine_title_and_body("", "") == ""


def test_dedup_key_ignores_case_and_whitespace():
    assert dedup_key("It is  a PHONE.") == dedup_key("it is a phone.")


def test_split_sentences():
    assert split_sentences("Great screen! Battery is poor.<br />Delivery fine") == [
        "Great screen!", "Battery is poor.", "Delivery fine"]


# --- cleaning & dedup -------------------------------------------------------

def _config(**overrides) -> DatasetConfig:
    data = {"name": "t", "source_path": "unused.csv", "columns": {"text": ["title", "body"], "rating": "stars",
                                                                 "product_id": "item"}}
    data.update(overrides)
    return DatasetConfig.from_dict(data)


def test_cleaning_counts_every_exclusion():
    records = [
        {"title": "", "body": "Great sound", "stars": "5", "item": "A"},
        {"title": "", "body": "  ", "stars": "5", "item": "A"},            # empty
        {"title": "Five Stars", "body": "", "stars": "5", "item": "A"},    # star title only -> empty
        {"title": "", "body": "Bad", "stars": "x", "item": "A"},           # invalid rating
        {"title": "", "body": "Okay item", "stars": "3", "item": ""},      # missing product
        {"title": "", "body": "great sound", "stars": "4", "item": "B"},   # duplicate (same label)
    ]
    rows, report = clean_records(iter(records), _config())
    assert [r["text"] for r in rows] == ["Great sound"]
    assert report.input_rows == 6
    assert report.excluded == {"empty_or_too_short_text": 2, "missing_or_invalid_rating": 1,
                               "missing_product_id": 1, "duplicate_text": 1, "conflicting_label_duplicates": 0}
    assert report.missing_values["body"] == 2 and report.missing_values["title"] == 5


def test_conflicting_duplicates_are_all_dropped():
    rows = [{"text": "It is a phone.", "label": "positive", "source_row": 0},
            {"text": "it is a  PHONE.", "label": "negative", "source_row": 1},
            {"text": "Nice", "label": "positive", "source_row": 2}]
    kept, stats = deduplicate(rows)
    assert [r["text"] for r in kept] == ["Nice"]
    assert stats == {"duplicate_text": 0, "conflicting_label_duplicates": 2}


def test_explicit_label_column_mapping():
    config = DatasetConfig.from_dict({"name": "t", "source_path": "x.csv", "label_source": "column",
                                      "columns": {"text": "review", "label": "sentiment"}})
    rows, report = clean_records(iter([{"review": "fine", "sentiment": "Neutral"},
                                       {"review": "bad", "sentiment": "angry"}]), config)
    assert [r["label"] for r in rows] == ["neutral"]
    assert report.excluded["missing_or_invalid_label"] == 1


@pytest.mark.parametrize("bad", [
    {"columns": {}},
    {"columns": {"text": "t"}, "label_source": "rating"},
    {"columns": {"text": "t", "rating": "r"}, "split": {"strategy": "group"}},
    {"columns": {"text": "t", "rating": "r"}, "split": {"val_size": 0.6, "test_size": 0.5}},
    {"columns": {"text": "t", "rating": "r"}, "unknown_key": 1},
])
def test_invalid_configs_rejected(bad):
    with pytest.raises(DatasetConfigError):
        DatasetConfig.from_dict({"name": "t", "source_path": "x.csv", **bad})


# --- splitting --------------------------------------------------------------

def _rows(n=60):
    labels = ["negative", "neutral", "positive"]
    return [{"text": f"text {i}", "label": labels[i % 3], "source_row": i, "product_id": f"p{i % 7}",
             "_group": f"u{i // 2}"} for i in range(n)]


def test_split_is_reproducible_and_disjoint():
    a = split_rows(_rows(), SplitConfig(seed=7))
    b = split_rows(_rows(), SplitConfig(seed=7))
    c = split_rows(_rows(), SplitConfig(seed=8))
    ids = {k: [r["source_row"] for r in v] for k, v in a.items()}
    assert ids == {k: [r["source_row"] for r in v] for k, v in b.items()}
    assert ids != {k: [r["source_row"] for r in v] for k, v in c.items()}
    assert sum(len(v) for v in ids.values()) == 60
    assert not (set(ids["train"]) & set(ids["val"]) | set(ids["train"]) & set(ids["test"]) | set(ids["val"]) & set(ids["test"]))


def test_stratified_split_preserves_class_balance():
    splits = split_rows(_rows(90), SplitConfig(val_size=0.2, test_size=0.2))
    for rows in splits.values():
        counts = Counter(r["label"] for r in rows)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_group_split_keeps_groups_together():
    splits = split_rows(_rows(), SplitConfig(strategy="group", seed=1))
    groups = {name: {r["_group"] for r in rows} for name, rows in splits.items()}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])


def test_split_rejects_tiny_classes():
    rows = _rows(30)
    rows[0]["label"] = "rare"
    with pytest.raises(DatasetConfigError, match="per class"):
        split_rows(rows, SplitConfig())


# --- end to end + manifest --------------------------------------------------

def test_prepare_fixture_end_to_end(fixture_data):
    manifest = json.loads((fixture_data / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["is_synthetic"] is True
    assert manifest["counts"]["input_rows"] == 73
    assert manifest["counts"]["rows_after_cleaning"] == 64
    assert sum(manifest["split_sizes"].values()) == 64
    assert manifest["label_mapping"] == {"negative": [1, 2], "neutral": [3], "positive": [4, 5]}
    assert all(v == 0 for k, v in manifest["leakage_checks"].items() if k.startswith("text_overlap"))
    assert len(manifest["source"]["sha256"]) == 64
    assert "packages" in manifest["environment"]
    train = read_processed_csv(fixture_data / "train.csv")
    assert set(train[0]) >= {"review_id", "product_id", "text", "rating", "label"}
    assert "_group" not in train[0]  # user ids are never written


def test_prepare_from_jsonl_with_group_split(tmp_path):
    src = tmp_path / "raw.jsonl"
    with src.open("w", encoding="utf-8") as f:
        for i in range(40):
            f.write(json.dumps({"title": f"t{i}", "text": f"review number {i}", "rating": [1, 3, 5][i % 3],
                                "parent_asin": f"P{i % 4}", "user_id": f"U{i % 10}"}) + "\n")
    config = DatasetConfig.from_dict({"name": "j", "source_path": str(src), "columns": {
        "text": ["title", "text"], "rating": "rating", "product_id": "parent_asin", "group": "user_id"},
        "split": {"strategy": "group", "val_size": 0.2, "test_size": 0.2}})
    result = prepare_dataset(config, tmp_path / "out")
    assert sum(result.manifest["split_sizes"].values()) == 40


def test_missing_input_file(tmp_path):
    config = DatasetConfig.from_dict({"name": "m", "source_path": str(tmp_path / "nope.csv"),
                                      "columns": {"text": "t", "rating": "r"}})
    with pytest.raises(DatasetConfigError, match="not found"):
        prepare_dataset(config, tmp_path / "out")


def test_user_csv_with_custom_columns(tmp_path):
    src = tmp_path / "mine.csv"
    with src.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opinion", "score", "sku"])
        for i in range(30):
            w.writerow([f"opinion text {i}", [1, 3, 5][i % 3], "S1"])
    config = DatasetConfig.from_dict({"name": "c", "source_path": str(src), "columns": {
        "text": "opinion", "rating": "score", "product_id": "sku"}})
    result = prepare_dataset(config, tmp_path / "out")
    assert result.manifest["class_distribution"]["all"]["neutral"]["count"] == 10


# --- Amazon streaming helpers (local gzip, no network) ----------------------

def _write_gz(path, n):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"rating": 5.0, "title": "t", "text": f"r{i}", "images": [], "asin": f"A{i}",
                                "parent_asin": f"P{i % 5}", "user_id": f"U{i}", "timestamp": i,
                                "verified_purchase": True, "helpful_vote": 0}) + "\n")


def test_reservoir_sample_is_deterministic_and_counts_products(tmp_path):
    src = tmp_path / "r.jsonl.gz"
    _write_gz(src, 1000)
    stream, _ = amazon.open_jsonl_gz(src)
    with stream:
        a = amazon.reservoir_sample_pass(amazon.iter_lines(stream), 50, seed=1)
    stream, _ = amazon.open_jsonl_gz(src)
    with stream:
        b = amazon.reservoir_sample_pass(amazon.iter_lines(stream), 50, seed=1)
    assert [i for i, _ in a.sample] == [i for i, _ in b.sample]
    assert len(a.sample) == 50 and a.total_lines == 1000
    assert a.product_counts == {f"P{k}": 200 for k in range(5)}
    # Uniformity sanity check: sampled positions are spread over the file, not just the head.
    assert max(i for i, _ in a.sample) > 500


def test_product_pass_caps_and_excludes(tmp_path):
    src = tmp_path / "r.jsonl.gz"
    _write_gz(src, 100)
    stream, _ = amazon.open_jsonl_gz(src)
    with stream:
        kept = amazon.product_reviews_pass(amazon.iter_lines(stream), {"P0", "P1"}, per_product_cap=5, seed=0,
                                           exclude_line_indices={0})
    assert set(kept) == {"P0", "P1"} and all(len(v) == 5 for v in kept.values())
    assert 0 not in {i for i, _ in kept["P0"]}
    row = amazon.parse_review_line(kept["P1"][0][1], kept["P1"][0][0])
    assert "images" not in row and row["source_line"] == kept["P1"][0][0]


def test_stream_resumes_when_server_closes_connection_early(tmp_path):
    """Regression: a server that ends a response early (clean close) must trigger a Range resume, not a fake EOF."""
    import http.server
    import threading

    payload = gzip.compress(b"".join(json.dumps({"parent_asin": f"P{i}", "text": "x" * 50}).encode() + b"\n"
                                     for i in range(2000)))
    requests_seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            range_header = self.headers.get("Range")
            requests_seen.append(range_header)
            if range_header:
                start = int(range_header.split("=")[1].rstrip("-"))
                self.send_response(206)
                body = payload[start:]
            else:
                self.send_response(200)
                body = payload[: len(payload) // 3]  # promise everything, deliver a third, then close
            self.send_header("Content-Length", str(len(payload) - (start if range_header else 0)))
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        stream, _ = amazon.open_jsonl_gz(f"http://127.0.0.1:{server.server_address[1]}/f.jsonl.gz")
        with stream:
            lines = list(amazon.iter_lines(stream))
    finally:
        server.shutdown()
    assert len(lines) == 2000
    assert requests_seen[0] is None and requests_seen[1] == f"bytes={len(payload) // 3}-"
