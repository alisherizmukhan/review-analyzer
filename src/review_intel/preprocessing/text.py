"""Deterministic, stateless text preprocessing.

Nothing here is *learned* from data, so applying it to every split cannot
leak information. All learned preprocessing (TF-IDF vocabulary and IDF
weights, tokenizer fine-tuning) lives inside the model pipelines and is fit
on the training split only.
"""

from __future__ import annotations

import html
import re
import unicodedata

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
# Amazon auto-generates titles such as "Five Stars" / "One Star" from the rating.
# Keeping them would leak the label into the input text.
STAR_TITLE_RE = re.compile(r"^\s*(one|two|three|four|five|[1-5])\s+stars?\s*\.?\s*$", re.IGNORECASE)
# Sentence boundaries: ., !, ? (optionally repeated) or newlines, followed by space.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def clean_text(text: str) -> str:
    """Normalize a raw review string for modelling.

    * Unicode NFKC normalization (e.g. full-width chars, ligatures)
    * HTML entity unescaping and tag removal (Amazon text contains ``<br />``)
    * URLs replaced by a placeholder token
    * whitespace collapsed

    Case is preserved: vectorizers lowercase internally and transformer
    tokenizers handle case themselves.
    """
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" URL ", text)
    return _WS_RE.sub(" ", text).strip()


def is_star_title(title: str) -> bool:
    return bool(STAR_TITLE_RE.match(title))


def combine_title_and_body(title: str, body: str, separator: str = ". ") -> str:
    """Join title and body, dropping auto-generated star titles and empty parts."""
    title = clean_text(title or "")
    body = clean_text(body or "")
    if is_star_title(title):
        title = ""
    if title and body:
        title = title.rstrip(".!? ")
        return f"{title}{separator}{body}"
    return title or body


def dedup_key(text: str) -> str:
    """Key for exact-duplicate detection: case- and whitespace-insensitive."""
    return _WS_RE.sub(" ", text.casefold()).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a simple punctuation rule (no model needed)."""
    return [s.strip() for s in _SENTENCE_RE.split(clean_text_keep_newlines(text)) if s and s.strip()]


def clean_text_keep_newlines(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub(" ", text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()
