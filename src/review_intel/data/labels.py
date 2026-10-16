"""Sentiment label set and the documented star-rating -> label rule."""

from __future__ import annotations

import math
from typing import Literal

Label = Literal["negative", "neutral", "positive"]

# Fixed order used everywhere (confusion matrices, model outputs, reports).
LABELS: tuple[Label, ...] = ("negative", "neutral", "positive")

# Documented 3-class rule: 1-2 stars negative, 3 neutral, 4-5 positive.
RATING_TO_LABEL: dict[int, Label] = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
LABEL_MAPPING_DOC = {"negative": [1, 2], "neutral": [3], "positive": [4, 5]}


def parse_rating(value: object) -> float | None:
    """Return a numeric rating, or None if the value is missing/unparseable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        rating = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            rating = float(text)
        except ValueError:
            return None
    return None if math.isnan(rating) else rating


def rating_to_label(rating: float | None) -> Label | None:
    """Map a whole-star rating in [1, 5] to a label; anything else -> None (excluded).

    Fractional ratings (e.g. 3.5) are rejected rather than rounded, so the
    documented rule is applied exactly.
    """
    if rating is None or not float(rating).is_integer():
        return None
    return RATING_TO_LABEL.get(int(rating))


def normalize_label(value: object) -> Label | None:
    """Accept an explicit label column value (case-insensitive)."""
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    return label if label in LABELS else None  # type: ignore[return-value]
