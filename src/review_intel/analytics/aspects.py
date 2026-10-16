"""Aspect-level review analysis, version 1: lexicon extraction + sentence-level sentiment.

Method
------
1. Split each review into sentences (punctuation rule).
2. A sentence *mentions* an aspect if it contains one of the aspect's lexicon
   terms (``configs/aspects.json``), case-insensitive, on word boundaries.
3. The sentiment of each mentioning sentence is predicted by the configured
   sentiment model (the same model as ``analyze_sentiment``).
4. Counts are aggregated per aspect; the highest-scoring sentences per
   sentiment are returned as evidence.

This is NOT a trained aspect-based sentiment analysis (ABSA) model. Known
limitations are listed in ``METHOD_LIMITATIONS`` and returned with every result.
``AspectAnalyzer`` only depends on an ``AspectExtractor`` and a prediction
function, so a learned extractor or ABSA model can replace either part.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from review_intel.analytics.corpus import CorpusReview
from review_intel.models.base import Prediction
from review_intel.preprocessing.text import split_sentences
from review_intel.utils import CONFIGS_DIR, read_json

DEFAULT_LEXICON_PATH = CONFIGS_DIR / "aspects.json"
MAX_EXCERPT_CHARS = 300

METHOD_DESCRIPTION = (
    "Dictionary-based aspect extraction (configurable lexicon, word-boundary matching) on sentence level; "
    "each mentioning sentence is classified by the configured review-level sentiment model."
)
METHOD_LIMITATIONS = [
    "Lexicon matching, not a trained aspect extractor: synonyms outside the lexicon are missed and ambiguous "
    "words can produce false matches (e.g. 'value' as a number, 'range' in other senses).",
    "Sentiment is predicted for the whole sentence, so a sentence with two aspects ('great screen but poor "
    "battery') gives both aspects the same label.",
    "The sentiment model was trained on whole reviews labelled by star ratings, then applied to single "
    "sentences (a domain shift); sentence-level accuracy has not been measured.",
    "Counts describe how often an aspect is mentioned with a predicted sentiment; they do not explain why.",
]


class AspectExtractor(Protocol):
    version: str

    def find(self, sentence: str) -> list[tuple[str, str]]:
        """Return (aspect, matched_term) pairs found in the sentence."""


@dataclass
class LexiconAspectExtractor:
    lexicon: dict[str, list[str]]
    version: str = "custom"

    def __post_init__(self) -> None:
        if not self.lexicon:
            raise ValueError("Aspect lexicon is empty.")
        self._patterns: dict[str, re.Pattern[str]] = {}
        for aspect, terms in self.lexicon.items():
            if not terms:
                raise ValueError(f"Aspect {aspect!r} has no terms.")
            # Longest terms first so "battery life" wins over "battery" in the match text.
            alternation = "|".join(re.escape(t.lower()) for t in sorted(terms, key=len, reverse=True))
            self._patterns[aspect] = re.compile(rf"(?<![\w-])(?:{alternation})(?![\w-])", re.IGNORECASE)

    @classmethod
    def from_file(cls, path: Path = DEFAULT_LEXICON_PATH) -> LexiconAspectExtractor:
        data = read_json(path)
        return cls(lexicon=data["aspects"], version=data.get("version", path.stem))

    @property
    def aspects(self) -> list[str]:
        return list(self.lexicon)

    def find(self, sentence: str) -> list[tuple[str, str]]:
        found = []
        for aspect, pattern in self._patterns.items():
            match = pattern.search(sentence)
            if match:
                found.append((aspect, match.group(0).lower()))
        return found


@dataclass
class Mention:
    review_id: int
    product: str
    aspect: str
    term: str
    sentence: str


def extract_mentions(reviews: Sequence[CorpusReview], extractor: AspectExtractor) -> list[Mention]:
    mentions = []
    for review in reviews:
        for sentence in split_sentences(review["text"]):
            for aspect, term in extractor.find(sentence):
                mentions.append(Mention(review["id"], review["product"], aspect, term, sentence))
    return mentions


def _excerpt(sentence: str) -> str:
    return sentence if len(sentence) <= MAX_EXCERPT_CHARS else sentence[: MAX_EXCERPT_CHARS - 1].rstrip() + "…"


class AspectAnalyzer:
    def __init__(self, extractor: AspectExtractor, predict: Callable[[Sequence[str]], list[Prediction]],
                 model_identity: dict[str, Any]):
        self.extractor = extractor
        self.predict = predict
        self.model_identity = model_identity

    def analyze(self, reviews: Sequence[CorpusReview], examples_per_sentiment: int = 2) -> dict[str, Any]:
        """Per-aspect mention counts, predicted sentence sentiment counts and evidence excerpts."""
        mentions = extract_mentions(reviews, self.extractor)
        predictions = self.predict([m.sentence for m in mentions]) if mentions else []
        by_aspect: dict[str, list[tuple[Mention, Prediction]]] = defaultdict(list)
        for mention, prediction in zip(mentions, predictions):
            by_aspect[mention.aspect].append((mention, prediction))

        aspects = {}
        for aspect, items in by_aspect.items():
            counts = {"positive": 0, "neutral": 0, "negative": 0}
            for _, p in items:
                counts[p.label] += 1
            total = len(items)
            examples: dict[str, list[dict[str, Any]]] = {}
            for label in ("negative", "positive", "neutral"):
                ranked = sorted(((m, p) for m, p in items if p.label == label),
                                key=lambda mp: (-mp[1].scores.get(label, 0.0), mp[0].review_id))
                seen, picked = set(), []
                for m, p in ranked:
                    if m.sentence in seen:
                        continue
                    seen.add(m.sentence)
                    picked.append({"review_id": m.review_id, "product": m.product, "excerpt": _excerpt(m.sentence),
                                   "matched_term": m.term, "score": p.scores.get(label)})
                    if len(picked) == examples_per_sentiment:
                        break
                examples[label] = picked
            aspects[aspect] = {
                "aspect": aspect,
                "reviews_mentioning": len({m.review_id for m, _ in items}),
                "mentions": total,
                "sentiment_counts": counts,
                "negative_share": round(counts["negative"] / total, 4) if total else 0.0,
                "examples": examples,
            }
        return {
            "reviews_analyzed": len(reviews),
            "aspects": aspects,
            "method": {
                "description": METHOD_DESCRIPTION,
                "lexicon_version": self.extractor.version,
                "sentiment_model": self.model_identity,
                "limitations": METHOD_LIMITATIONS,
            },
        }


def top_aspects(analysis: dict[str, Any], limit: int, by: str = "mentions") -> list[dict[str, Any]]:
    """Rank aspects by total mentions or by negative mentions (ties: negative share, then name)."""
    rows = list(analysis["aspects"].values())
    if by == "negative":
        rows = [r for r in rows if r["sentiment_counts"]["negative"] > 0]
        rows.sort(key=lambda r: (-r["sentiment_counts"]["negative"], -r["negative_share"], r["aspect"]))
    else:
        rows.sort(key=lambda r: (-r["mentions"], r["aspect"]))
    return rows[:limit]
