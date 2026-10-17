"""What the model has actually seen: the ground truth that citations in the final report are checked against.

Only the *bounded* tool results (exactly what was sent to the model) are indexed, so a review that was
truncated away cannot be cited. A record counts as a review when it carries an integer ``id`` or
``review_id`` together with ``text`` or ``excerpt``; this covers every review-bearing field of the server's
tools (``search_reviews.reviews``, ``search_similar_reviews.results``, aspect ``negative_examples``,
``representative_reviews``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from review_intel.agent.guard import TRUNCATION_MARK

_ELLIPSIS = re.compile(r"\s*(?:…|\.\.\.)\s*")
_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace(TRUNCATION_MARK, " ").casefold()
    text = text.strip().strip("\"'“”‘’").strip()
    return _SPACE.sub(" ", text)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass
class SeenReview:
    review_id: int
    products: set[str] = field(default_factory=set)
    texts: list[str] = field(default_factory=list)  # normalized
    labels: set[str] = field(default_factory=set)  # stored star-rating labels and/or model predictions
    tools: set[str] = field(default_factory=set)


class EvidenceIndex:
    def __init__(self) -> None:
        self.reviews: dict[int, SeenReview] = {}
        self.numbers: list[float] = []
        self.products: set[str] = set()  # casefolded ids and names

    # -- indexing ------------------------------------------------------------

    def add(self, tool: str, data: Any) -> None:
        self._walk(tool, data)

    def _walk(self, tool: str, value: Any) -> None:
        if isinstance(value, dict):
            self._record(tool, value)
            for item in value.values():
                self._walk(tool, item)
        elif isinstance(value, list):
            for item in value:
                self._walk(tool, item)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            self.numbers.append(float(value))

    def _record(self, tool: str, record: dict[str, Any]) -> None:
        for key in ("product", "product_a", "product_b", "product_name"):
            if isinstance(record.get(key), str) and record[key].strip():
                self.products.add(record[key].strip().casefold())
        review_id = record.get("review_id", record.get("id"))
        text = record.get("text", record.get("excerpt"))
        if not _is_int(review_id) or not isinstance(text, str):
            return
        seen = self.reviews.setdefault(review_id, SeenReview(review_id))
        seen.texts.append(normalize_text(text))
        seen.tools.add(tool)
        if isinstance(record.get("product"), str):
            seen.products.add(record["product"])
        for key in ("sentiment", "predicted_sentiment"):
            if isinstance(record.get(key), str):
                seen.labels.add(record[key])

    # -- checks --------------------------------------------------------------

    def has_review(self, review_id: int) -> bool:
        return review_id in self.reviews

    def excerpt_matches(self, review_id: int, excerpt: str) -> bool:
        """True if every fragment of ``excerpt`` (split on ellipses) occurs in a text seen for that review."""
        seen = self.reviews.get(review_id)
        if seen is None:
            return False
        fragments = [f for f in (normalize_text(p) for p in _ELLIPSIS.split(excerpt)) if f]
        if not fragments or sum(len(f) for f in fragments) < 8:
            return False
        return any(all(f in text for f in fragments) for text in seen.texts)

    def number_seen(self, value: float) -> bool:
        """True if ``value`` matches a number in any tool result (rounding tolerance; fraction <-> percent)."""
        for seen in self.numbers:
            for candidate in (seen, seen * 100, seen / 100):
                if abs(value - candidate) <= max(0.051, 0.005 * abs(candidate)):
                    return True
        return False

    def product_seen(self, product: str) -> bool:
        wanted = product.strip().casefold()
        return not wanted or wanted in ("all", "corpus") or wanted in self.products

    @property
    def review_ids(self) -> set[int]:
        return set(self.reviews)
