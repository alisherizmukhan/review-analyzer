"""The review corpus that analytics and search run over.

Two on-disk formats are accepted:

* ``.json`` - the original v1 format: a list of
  ``{"id", "product", "text", "sentiment"}`` objects, optionally with
  ``"rating"`` and ``"product_name"`` (e.g. review-analyzer-mcp/reviews.json).
* ``.csv``  - a processed file written by ``scripts/prepare_data.py``
  (columns review_id, product_id, product_name, text, rating, label).

Both are converted to the same validated ``CorpusReview`` dicts. Loaded
corpora are cached by (path, mtime, size); treat the returned list as
read-only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from review_intel.data.labels import parse_rating

Sentiment = Literal["positive", "negative", "neutral"]
SENTIMENTS: tuple[Sentiment, ...] = ("positive", "negative", "neutral")


class ReviewDataError(Exception):
    """The review dataset is missing, is not valid JSON/CSV, or has malformed records."""


class CorpusReview(TypedDict):
    id: int
    product: str
    text: str
    sentiment: Sentiment
    rating: NotRequired[float | None]
    product_name: NotRequired[str]


def validate_reviews(data: Any) -> list[CorpusReview]:
    """Check that ``data`` is a list of well-formed review records (v1 rules + optional rating)."""
    if not isinstance(data, list):
        raise ReviewDataError(f"Dataset must be a JSON list of reviews, got {type(data).__name__}.")

    for index, record in enumerate(data):
        where = f"Review at index {index}"
        if not isinstance(record, dict):
            raise ReviewDataError(f"{where} must be a JSON object, got {type(record).__name__}.")
        missing = [key for key in ("id", "product", "text", "sentiment") if key not in record]
        if missing:
            raise ReviewDataError(f"{where} is missing required field(s): {', '.join(missing)}.")
        # bool is a subclass of int in Python, so reject it explicitly.
        if not isinstance(record["id"], int) or isinstance(record["id"], bool):
            raise ReviewDataError(f"{where}: 'id' must be an integer.")
        if not isinstance(record["product"], str) or not record["product"].strip():
            raise ReviewDataError(f"{where}: 'product' must be a non-empty string.")
        if not isinstance(record["text"], str):
            raise ReviewDataError(f"{where}: 'text' must be a string.")
        if record["sentiment"] not in SENTIMENTS:
            raise ReviewDataError(
                f"{where}: 'sentiment' must be one of {', '.join(SENTIMENTS)}, got {record['sentiment']!r}."
            )
        rating = record.get("rating")
        if rating is not None and (isinstance(rating, bool) or not isinstance(rating, (int, float))
                                   or not 1 <= rating <= 5):
            raise ReviewDataError(f"{where}: 'rating' must be a number from 1 to 5 or null.")
    return data


def _load_json(path: Path) -> list[CorpusReview]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ReviewDataError(f"Review dataset not found: {path.name}") from None
    except OSError as exc:
        raise ReviewDataError(f"Could not read review dataset {path.name}: {exc.strerror}") from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReviewDataError(
            f"Review dataset {path.name} is not valid JSON (line {exc.lineno}, column {exc.colno}: {exc.msg})."
        ) from None
    return validate_reviews(data)


_CSV_REQUIRED = ("review_id", "product_id", "text", "label")


def _load_csv(path: Path) -> list[CorpusReview]:
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except FileNotFoundError:
        raise ReviewDataError(f"Review dataset not found: {path.name}") from None
    reviews: list[CorpusReview] = []
    with handle:
        reader = csv.DictReader(handle)
        missing = [c for c in _CSV_REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ReviewDataError(f"CSV corpus {path.name} is missing column(s): {', '.join(missing)}.")
        for line, row in enumerate(reader, start=2):
            if not str(row["review_id"]).isdigit():
                raise ReviewDataError(f"{path.name} line {line}: 'review_id' must be an integer.")
            if row["label"] not in SENTIMENTS:
                raise ReviewDataError(f"{path.name} line {line}: 'label' must be one of {', '.join(SENTIMENTS)}.")
            if not row["product_id"].strip():
                raise ReviewDataError(f"{path.name} line {line}: 'product_id' must be non-empty.")
            review: CorpusReview = {"id": int(row["review_id"]), "product": row["product_id"], "text": row["text"],
                                    "sentiment": row["label"], "rating": parse_rating(row.get("rating"))}  # type: ignore[typeddict-item]
            if row.get("product_name"):
                review["product_name"] = row["product_name"]
            reviews.append(review)
    return reviews


_CACHE: dict[Path, tuple[tuple[int, int], list[CorpusReview]]] = {}


def load_reviews(path: Path) -> list[CorpusReview]:
    """Load and validate a corpus (.json or .csv). Cached until the file changes."""
    path = Path(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        raise ReviewDataError(f"Review dataset not found: {path.name}") from None
    key = (stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(path)
    if cached and cached[0] == key:
        return cached[1]
    if path.suffix.lower() == ".csv":
        reviews = _load_csv(path)
    else:
        reviews = _load_json(path)
    _CACHE[path] = (key, reviews)
    return reviews


def corpus_fingerprint(path: Path) -> tuple[str, int, int]:
    stat = Path(path).stat()
    return (str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size)
