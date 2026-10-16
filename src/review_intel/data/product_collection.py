"""Per-product review collection from the Amazon Reviews 2023 review file (analytics corpus, not training data).

Why a second method next to ``amazon.product_reviews_pass``
-----------------------------------------------------------
The original pass streams the *gzip* file once and keeps a per-product reservoir in memory, so an interrupted
run has to start over (a gzip stream cannot be resumed mid-file). The official Hugging Face mirror also serves
the same file *uncompressed*, with HTTP range requests. That allows:

* **chunks**: the file is split into byte ranges, processed independently (in parallel threads), and every
  finished chunk is saved under ``<out>/chunks/``. A re-run skips finished chunks, so an interrupted collection
  resumes where it stopped;
* **order-independent sampling (bottom-k)**: each review gets a pseudo-random priority
  ``sha1(seed | review key)``, and for each product the ``cap`` reviews with the smallest priorities are kept.
  This is a uniform random sample without replacement of the product's reviews; it depends only on the
  reviews, not on chunking, thread scheduling or resumes, so every run with the same seed gives the same set;
* **stable de-duplication**: the review key ``user_id|parent_asin|asin|timestamp`` identifies a review (the
  dataset has no review id). Identical keys collapse to one review.

Reviews in the research sample (train/val/test) are excluded by the same key, so the analytics corpus never
contains a review the models were trained, tuned or tested on.
"""

from __future__ import annotations

import hashlib
import heapq
import io
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from review_intel.data.amazon import (
    _PARENT_ASIN_RE,
    REVIEW_FIELDS,
    ResumableHTTPStream,
    SourceAccessError,
)
from review_intel.utils import read_json, write_json

logger = logging.getLogger(__name__)

ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
KEY_FIELDS = ("user_id", "parent_asin", "asin", "timestamp")
DEFAULT_CHUNK_BYTES = 256 * 1024 * 1024
DEFAULT_PER_PRODUCT_CAP = 1000

# Opens ``source`` positioned at byte ``start``; returns a binary stream.
RangeOpener = Callable[[str, int], BinaryIO]


class CollectionError(RuntimeError):
    """The collection could not be completed (some chunks failed); finished chunks are kept for a resume."""


# ---------------------------------------------------------------------------
# Product selection
# ---------------------------------------------------------------------------


def validate_product_ids(product_ids: Iterable[str]) -> list[str]:
    """Return de-duplicated product ids in order; reject anything that is not a 10-character ASIN."""
    ids: list[str] = []
    invalid = []
    for raw in product_ids:
        pid = str(raw).strip()
        if not ASIN_RE.match(pid):
            invalid.append(raw)
        elif pid not in ids:
            ids.append(pid)
    if invalid:
        raise ValueError(f"Invalid product id(s) {invalid[:10]!r}: expected 10 upper-case letters/digits "
                         "(an Amazon parent_asin such as 'B075X8471B').")
    if not ids:
        raise ValueError("No product ids given.")
    return ids


def rank_products_in_sample(sample_path: Path, top: int) -> list[dict[str, Any]]:
    """Rank products by their review count in the uniform research sample (an unbiased popularity estimate).

    Selecting from the sample avoids a separate full counting pass over the 43.9M-review file.
    """
    counts: Counter[str] = Counter()
    with Path(sample_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                counts[json.loads(line)["parent_asin"]] += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [{"rank": i + 1, "parent_asin": asin, "reviews_in_research_sample": n}
            for i, (asin, n) in enumerate(ranked)]


def load_exclusion_keys(sample_path: Path | None) -> set[str]:
    """Keys of the reviews in the research sample (excluded from the analytics corpus)."""
    if sample_path is None:
        return set()
    keys = set()
    with Path(sample_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                keys.add(review_key(json.loads(line)))
    return keys


# ---------------------------------------------------------------------------
# Bottom-k per-product sampling
# ---------------------------------------------------------------------------


def review_key(record: dict[str, Any]) -> str:
    return "|".join("" if record.get(f) is None else str(record.get(f)) for f in KEY_FIELDS)


def sample_priority(key: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha1(f"{seed}|{key}".encode()).digest()[:8], "big")


@dataclass(order=True)
class _Entry:
    neg_priority: int
    key: str = field(compare=False)
    offset: int = field(compare=False)
    row: dict[str, Any] = field(compare=False)


class BottomKCollector:
    """Keep, per product, the ``cap`` reviews with the smallest ``sample_priority`` (see module docstring)."""

    def __init__(self, products: Iterable[str], cap: int, seed: int, exclude_keys: set[str] | frozenset = frozenset()):
        if cap < 1:
            raise ValueError("cap must be >= 1")
        self.cap = cap
        self.seed = seed
        self.exclude_keys = exclude_keys
        self.heaps: dict[str, list[_Entry]] = {p: [] for p in products}
        self.kept_keys: dict[str, set[str]] = {p: set() for p in self.heaps}
        self.reviews_in_source: Counter[str] = Counter()  # every line of the product, before exclusions
        self.excluded: Counter[str] = Counter()
        self.duplicates: Counter[str] = Counter()
        self.stats: Counter[str] = Counter()

    def offer_line(self, offset: int, line: bytes) -> None:
        match = _PARENT_ASIN_RE.search(line)
        if not match:
            return
        asin = match.group(1).decode()
        if asin not in self.heaps:
            return
        self.reviews_in_source[asin] += 1
        try:
            record = json.loads(line)
        except ValueError:
            self.stats["unparseable_lines"] += 1
            return
        row = {k: record.get(k) for k in REVIEW_FIELDS}
        row["source_offset"] = offset
        self.offer_row(asin, row)

    def offer_row(self, asin: str, row: dict[str, Any]) -> None:
        key = review_key(row)
        if key in self.exclude_keys:
            self.excluded[asin] += 1
            return
        if key in self.kept_keys[asin]:
            self.duplicates[asin] += 1
            return
        entry = _Entry(-sample_priority(key, self.seed), key, row["source_offset"], {**row, "review_key": key})
        heap = self.heaps[asin]
        if len(heap) < self.cap:
            heapq.heappush(heap, entry)
            self.kept_keys[asin].add(key)
        elif entry.neg_priority > heap[0].neg_priority:  # smaller priority than the current worst kept review
            dropped = heapq.heapreplace(heap, entry)
            self.kept_keys[asin].discard(dropped.key)
            self.kept_keys[asin].add(key)

    def merge(self, other: BottomKCollector) -> None:
        """Fold in another collector (e.g. one chunk). The result equals collecting everything at once."""
        for counter in ("reviews_in_source", "excluded", "duplicates", "stats"):
            getattr(self, counter).update(getattr(other, counter))
        for asin, heap in other.heaps.items():
            for entry in heap:
                self.offer_row(asin, entry.row)

    def rows(self) -> list[dict[str, Any]]:
        """Kept reviews, ordered by product then source offset (deterministic)."""
        out = [entry.row for asin in sorted(self.heaps) for entry in self.heaps[asin]]
        return sorted(out, key=lambda r: (r["parent_asin"], r["source_offset"]))

    # Chunk persistence: kept rows + counters, enough to rebuild the collector exactly.
    def to_state(self) -> dict[str, Any]:
        return {"rows": [e.row for heap in self.heaps.values() for e in heap],
                "reviews_in_source": dict(self.reviews_in_source), "excluded": dict(self.excluded),
                "duplicates": dict(self.duplicates), "stats": dict(self.stats)}

    @classmethod
    def from_state(cls, state: dict[str, Any], products: Iterable[str], cap: int, seed: int) -> BottomKCollector:
        collector = cls(products, cap, seed)
        for row in state["rows"]:
            collector.offer_row(row["parent_asin"], {k: v for k, v in row.items() if k != "review_key"})
        for counter in ("reviews_in_source", "excluded", "duplicates", "stats"):
            getattr(collector, counter).update(state[counter])
        return collector


# ---------------------------------------------------------------------------
# Chunked, resumable reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int


def plan_chunks(total_bytes: int, chunk_bytes: int) -> list[Chunk]:
    if total_bytes <= 0 or chunk_bytes <= 0:
        raise ValueError("total_bytes and chunk_bytes must be positive")
    starts = range(0, total_bytes, chunk_bytes)
    return [Chunk(i, s, min(s + chunk_bytes, total_bytes)) for i, s in enumerate(starts)]


def is_http(source: str) -> bool:
    return str(source).startswith(("http://", "https://"))


def default_opener(source: str, start: int) -> BinaryIO:
    if is_http(source):
        return io.BufferedReader(ResumableHTTPStream(source, start=start), buffer_size=1 << 20)  # type: ignore[return-value]
    handle = open(source, "rb")  # noqa: SIM115 - closed by the caller
    handle.seek(start)
    return handle


def source_size(source: str) -> tuple[int, dict[str, str]]:
    """Total size in bytes (and response headers) of a local file or an HTTP resource supporting ranges."""
    if not is_http(source):
        return os.path.getsize(source), {}
    stream = ResumableHTTPStream(source, start=1)  # a range request reports the full size in Content-Range
    try:
        if stream.total_length is None:
            raise SourceAccessError(f"{source} did not report its size; byte-range collection is not possible.")
        return stream.total_length, dict(stream.headers)
    finally:
        stream.close()


def iter_chunk_lines(stream: BinaryIO, start: int, end: int) -> Iterator[tuple[int, bytes]]:
    """Yield (offset, line) for every line that *starts* in [start, end).

    ``stream`` must be positioned at ``max(start - 1, 0)``. A line that straddles ``start`` belongs to the
    previous chunk; a line that starts before ``end`` is read to its end even past ``end``.
    """
    offset = max(start - 1, 0)
    if start > 0:
        offset += len(stream.readline())  # finish the line that contains byte start-1
    while offset < end:
        line = stream.readline()
        if not line:
            break
        yield offset, line
        offset += len(line)


def collect_chunk(source: str, chunk: Chunk, products: list[str], cap: int, seed: int, exclude_keys: set[str],
                  opener: RangeOpener = default_opener) -> tuple[BottomKCollector, int]:
    collector = BottomKCollector(products, cap, seed, exclude_keys)
    lines = 0
    stream = opener(source, max(chunk.start - 1, 0))
    try:
        for offset, line in iter_chunk_lines(stream, chunk.start, chunk.end):
            if line.strip():
                lines += 1
                collector.offer_line(offset, line)
    finally:
        stream.close()
    return collector, lines


@dataclass
class CollectionResult:
    collector: BottomKCollector
    total_lines: int
    chunks_total: int
    chunks_resumed: int
    seconds: float


def collect_products(source: str, products: list[str], cap: int, seed: int, state_dir: Path,
                     exclude_keys: set[str] | None = None, chunk_bytes: int = DEFAULT_CHUNK_BYTES, workers: int = 4,
                     opener: RangeOpener = default_opener, total_bytes: int | None = None,
                     chunk_retries: int = 2) -> CollectionResult:
    """Collect up to ``cap`` reviews per product from an *uncompressed* JSONL source, resumably.

    Finished chunks are stored in ``state_dir`` and skipped on a re-run with identical parameters (checked
    against ``state_dir/plan.json``). Raises ``CollectionError`` if chunks still fail after ``chunk_retries``;
    the finished ones stay on disk, so running again resumes.
    """
    products = validate_product_ids(products)
    exclude_keys = exclude_keys or set()
    if total_bytes is None:
        total_bytes, _ = source_size(source)
    chunks = plan_chunks(total_bytes, chunk_bytes)
    plan = {"source": source, "total_bytes": total_bytes, "chunk_bytes": chunk_bytes, "products": products,
            "cap": cap, "seed": seed, "excluded_keys": len(exclude_keys),
            "excluded_keys_sha256": hashlib.sha256("\n".join(sorted(exclude_keys)).encode()).hexdigest()}
    state_dir = Path(state_dir)
    plan_path = state_dir / "plan.json"
    if plan_path.exists():
        previous = read_json(plan_path)
        if previous != plan:
            changed = sorted(k for k in plan if previous.get(k) != plan[k])
            raise CollectionError(f"{state_dir} holds a collection with different parameters ({', '.join(changed)}). "
                                  "Use the same arguments to resume, or a new --out-dir to start over.")
    else:
        write_json(plan_path, plan)

    def chunk_path(chunk: Chunk) -> Path:
        return state_dir / f"chunk_{chunk.index:05d}.json"

    done = {c.index for c in chunks if chunk_path(c).exists()}
    todo = [c for c in chunks if c.index not in done]
    logger.info("Collection: %d chunks of %d MB, %d already done (resumed), %d to read with %d workers",
                len(chunks), chunk_bytes >> 20, len(done), len(todo), workers)
    started = time.time()
    lock = threading.Lock()
    finished = [len(done)]

    def run(chunk: Chunk) -> None:
        for attempt in range(chunk_retries + 1):
            try:
                collector, lines = collect_chunk(source, chunk, products, cap, seed, exclude_keys, opener)
                break
            except SourceAccessError:
                raise
            except Exception as exc:  # noqa: BLE001 - network/IO errors: retry the whole chunk
                if attempt == chunk_retries:
                    raise
                logger.warning("Chunk %d failed (%s); retrying (%d/%d)", chunk.index, exc, attempt + 1, chunk_retries)
                time.sleep(min(2**attempt, 30))
        tmp = chunk_path(chunk).with_suffix(".tmp")
        write_json(tmp, {"chunk": chunk.index, "start": chunk.start, "end": chunk.end, "lines": lines,
                         **collector.to_state()})
        os.replace(tmp, chunk_path(chunk))  # atomic: a chunk file exists only when complete
        with lock:
            finished[0] += 1
            elapsed = time.time() - started
            logger.info("  chunk %d done (%d/%d, %.0f s elapsed)", chunk.index, finished[0], len(chunks), elapsed)

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(run, c): c for c in todo}
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                failures.append(f"chunk {futures[future].index}: {exc}")
                if isinstance(exc, SourceAccessError):
                    for other in futures:
                        other.cancel()
    if failures:
        raise CollectionError(f"{len(failures)} chunk(s) failed; {len(chunks) - len(failures)} are saved in "
                              f"{state_dir}. Run the same command again to resume. First failure: {failures[0]}")

    merged = BottomKCollector(products, cap, seed)
    total_lines = 0
    for chunk in chunks:
        state = read_json(chunk_path(chunk))
        total_lines += state["lines"]
        merged.merge(BottomKCollector.from_state(state, products, cap, seed))
    return CollectionResult(merged, total_lines, len(chunks), len(done), round(time.time() - started, 1))


def collection_report(collector: BottomKCollector, selected: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-product counts for the manifest (all values are counted, nothing is estimated)."""
    kept = Counter(r["parent_asin"] for r in collector.rows())
    per_product = []
    for item in selected:
        asin = item["parent_asin"]
        per_product.append({**item, "reviews_in_source": collector.reviews_in_source.get(asin, 0),
                            "excluded_research_sample": collector.excluded.get(asin, 0),
                            "duplicate_keys_dropped": collector.duplicates.get(asin, 0),
                            "collected": kept.get(asin, 0)})
    return {"products": per_product,
            "products_not_found": [p["parent_asin"] for p in per_product if p["reviews_in_source"] == 0],
            "rows": sum(kept.values()),
            "excluded_research_sample": sum(collector.excluded.values()),
            "duplicate_keys_dropped": sum(collector.duplicates.values()),
            "unparseable_lines": collector.stats.get("unparseable_lines", 0)}
