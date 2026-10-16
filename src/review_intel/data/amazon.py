"""Streaming access to the Amazon Reviews 2023 dataset (McAuley Lab).

Official files (https://amazon-reviews-2023.github.io/):
    reviews:  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/<Category>.jsonl.gz
    metadata: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_<Category>.jsonl.gz

The Electronics review file is ~6.5 GB compressed (43.9M reviews) and is
ordered by *user*, so "the first N lines" is a biased sample of a few
thousand users. Instead we stream the file once without storing it and:

* draw a uniform random sample of reviews with seeded reservoir sampling
  (every review has the same probability of selection), and
* count reviews per product, so a second pass can collect the reviews of
  the most-reviewed products for product-level analytics.

Only the sampled lines are written to disk.
"""

from __future__ import annotations

import gzip
import http.client
import io
import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"

# Fields kept from each review (images are dropped: large and unused).
REVIEW_FIELDS = (
    "rating", "title", "text", "asin", "parent_asin", "user_id",
    "timestamp", "verified_purchase", "helpful_vote",
)
_PARENT_ASIN_RE = re.compile(rb'"parent_asin":\s*"([^"]+)"')


def review_url(category: str) -> str:
    return f"{BASE_URL}/review_categories/{category}.jsonl.gz"


def meta_url(category: str) -> str:
    return f"{BASE_URL}/meta_categories/meta_{category}.jsonl.gz"


HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"


def hf_review_url(category: str, revision: str = "main") -> str:
    """The same review file on the official Hugging Face mirror, *uncompressed* (supports byte-range resume)."""
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/{revision}/raw/review_categories/{category}.jsonl"


# HTTP statuses that will not change on retry: fail at once with a clear message.
FATAL_HTTP_CODES = frozenset({400, 401, 403, 404, 410})


class SourceAccessError(OSError):
    """The source refused access or does not exist (e.g. HTTP 403/404); retrying will not help."""


class ResumableHTTPStream(io.RawIOBase):
    """Read-only byte stream over HTTP that reconnects with a Range header on errors.

    The gzip decompressor reading from this stream keeps its state, so a
    dropped connection in the middle of a multi-GB file only costs a reconnect.
    ``start`` opens the stream at a byte offset (used for chunked, resumable
    reads of uncompressed files). HTTP 429/503 honour ``Retry-After``; the
    statuses in ``FATAL_HTTP_CODES`` raise ``SourceAccessError`` without retrying.
    """

    def __init__(self, url: str, max_retries: int = 8, timeout: float = 60.0, start: int = 0):
        self.url = url
        self.max_retries = max_retries
        self.timeout = timeout
        self.position = start
        self.total_length: int | None = None
        self.headers: dict[str, str] = {}
        self._retry_after = 0.0
        self._response = None
        for attempt in range(self.max_retries + 1):
            try:
                self._response = self._open()
                break
            except SourceAccessError:
                raise
            except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
                if attempt == self.max_retries:
                    raise
                self._wait(attempt, exc)

    def _open(self):
        headers = {"User-Agent": "review-intel/0.2 (research sampling)"}
        if self.position:
            headers["Range"] = f"bytes={self.position}-"
        try:
            response = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in FATAL_HTTP_CODES:
                raise SourceAccessError(f"{self.url} returned HTTP {exc.code} ({exc.reason}); check the URL, "
                                        "category and access rights. Not retried.") from None
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            self._retry_after = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 0.0
            raise
        if self.position and response.status != 206:
            raise OSError(f"Server ignored Range request (HTTP {response.status}); cannot resume.")
        if not self.headers:
            self.headers = {k.lower(): v for k, v in response.headers.items()}
            content_range = self.headers.get("content-range", "")
            length = self.headers.get("content-length")
            if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit():
                self.total_length = int(content_range.rsplit("/", 1)[1])
            elif length and length.isdigit():
                self.total_length = self.position + int(length)
        return response

    def _wait(self, attempt: int, exc: Exception) -> None:
        wait = max(min(2**attempt, 60), self._retry_after)
        self._retry_after = 0.0
        logger.warning("Stream error at byte %d (%s); reconnecting in %ss", self.position, exc, wait)
        time.sleep(wait)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        for attempt in range(self.max_retries + 1):
            try:
                n = self._response.readinto(buffer)
                if n == 0 and self.total_length is not None and self.position < self.total_length:
                    # The server closed the connection early: not a real EOF.
                    raise http.client.IncompleteRead(b"", self.total_length - self.position)
                self.position += n
                return n
            except SourceAccessError:
                raise
            except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
                if attempt == self.max_retries:
                    raise
                self._wait(attempt, exc)
                try:
                    self._response.close()
                except OSError:
                    pass
                try:
                    self._response = self._open()
                except SourceAccessError:
                    raise
                except (OSError, urllib.error.URLError, http.client.HTTPException) as reopen_exc:
                    logger.warning("Reconnect failed: %s", reopen_exc)
        return 0

    def close(self) -> None:
        try:
            if self._response is not None:
                self._response.close()
        finally:
            super().close()


def open_jsonl_gz(source: str | Path) -> tuple[BinaryIO, dict[str, str]]:
    """Open a local path or URL to a .jsonl.gz file as a binary line stream."""
    if isinstance(source, Path) or not str(source).startswith(("http://", "https://")):
        return gzip.open(source, "rb"), {}  # type: ignore[return-value]
    raw = ResumableHTTPStream(str(source))
    stream = gzip.GzipFile(fileobj=io.BufferedReader(raw, buffer_size=1 << 20))
    return io.BufferedReader(stream, buffer_size=1 << 20), raw.headers  # type: ignore[arg-type]


def _progress(i: int, every: int, started: float) -> None:
    if i and i % every == 0:
        logger.info("  %d lines read (%.0f lines/s)", i, i / max(time.time() - started, 1e-9))


@dataclass
class SamplePassResult:
    sample: list[tuple[int, bytes]]  # (line index, raw line)
    product_counts: Counter[str] = field(default_factory=Counter)
    total_lines: int = 0
    source_headers: dict[str, str] = field(default_factory=dict)


def reservoir_sample_pass(
    lines: Iterable[bytes], sample_size: int, seed: int, count_products: bool = True
) -> SamplePassResult:
    """Uniform reservoir sample (Algorithm R) of ``sample_size`` lines, plus per-product counts.

    After the pass, every line had probability sample_size / total_lines of
    being kept, independent of its position in the file.
    """
    rng = random.Random(seed)
    reservoir: list[tuple[int, bytes]] = []
    counts: Counter[str] = Counter()
    started = time.time()
    i = -1
    for i, line in enumerate(lines):
        if count_products:
            match = _PARENT_ASIN_RE.search(line)
            if match:
                counts[match.group(1).decode()] += 1
        if i < sample_size:
            reservoir.append((i, line))
        else:
            j = rng.randrange(i + 1)
            if j < sample_size:
                reservoir[j] = (i, line)
        _progress(i, 2_000_000, started)
    reservoir.sort()
    return SamplePassResult(sample=reservoir, product_counts=counts, total_lines=i + 1)


def product_reviews_pass(
    lines: Iterable[bytes],
    products: set[str],
    per_product_cap: int,
    seed: int,
    exclude_line_indices: set[int] = frozenset(),  # type: ignore[assignment]
) -> dict[str, list[tuple[int, bytes]]]:
    """Collect up to ``per_product_cap`` reviews per product (per-product reservoir sampling)."""
    rng = random.Random(seed)
    seen: Counter[str] = Counter()
    kept: dict[str, list[tuple[int, bytes]]] = {p: [] for p in products}
    started = time.time()
    for i, line in enumerate(lines):
        _progress(i, 2_000_000, started)
        match = _PARENT_ASIN_RE.search(line)
        if not match or i in exclude_line_indices:
            continue
        asin = match.group(1).decode()
        if asin not in products:
            continue
        n = seen[asin]
        seen[asin] += 1
        if n < per_product_cap:
            kept[asin].append((i, line))
        else:
            j = rng.randrange(n + 1)
            if j < per_product_cap:
                kept[asin][j] = (i, line)
    return {p: sorted(rows) for p, rows in kept.items()}


def metadata_pass(lines: Iterable[bytes], products: set[str]) -> dict[str, dict]:
    """Return title/category/rating metadata for the requested parent ASINs."""
    found: dict[str, dict] = {}
    for line in lines:
        match = _PARENT_ASIN_RE.search(line)
        if not match or match.group(1).decode() not in products:
            continue
        record = json.loads(line)
        found[record["parent_asin"]] = {
            "parent_asin": record["parent_asin"],
            "title": record.get("title"),
            "main_category": record.get("main_category"),
            "average_rating": record.get("average_rating"),
            "rating_number": record.get("rating_number"),
        }
        if len(found) == len(products):
            break
    return found


def parse_review_line(line: bytes, line_index: int) -> dict:
    """Parse one raw review line, keeping only ``REVIEW_FIELDS`` plus provenance."""
    record = json.loads(line)
    row = {key: record.get(key) for key in REVIEW_FIELDS}
    row["source_line"] = line_index
    return row


def iter_lines(stream: BinaryIO, max_lines: int | None = None) -> Iterator[bytes]:
    for i, line in enumerate(stream):
        if max_lines is not None and i >= max_lines:
            return
        if line.strip():
            yield line
