"""Generate data/fixtures/synthetic_reviews.csv (SYNTHETIC test data, not real reviews).

Deterministic: rerunning produces the identical file. The column names
(``body``, ``stars``, ``item``) deliberately differ from Amazon's so tests
exercise the configurable column mapping. A few rows are intentionally
invalid (empty text, bad ratings, duplicates) to test the cleaning step.
"""

from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "synthetic_reviews.csv"

PRODUCTS = ["Nova X1 Phone", "Pulse Earbuds", "Orbit Laptop"]
PHRASES = {
    "positive": [
        "The battery life is excellent and lasts all day",
        "Great camera, the photos are sharp and vivid",
        "The screen is bright and beautiful",
        "Fast performance, apps open instantly",
        "Good value for the price, highly recommend",
        "Love the sleek design and solid build",
        "Delivery was quick and the box arrived intact",
    ],
    "negative": [
        "The battery drains way too fast, very disappointing",
        "The camera is blurry and terrible in low light",
        "The screen scratches easily and looks dim",
        "Performance is slow and it lags constantly",
        "Way overpriced for what you get, a waste of money",
        "The design feels cheap and flimsy, it broke quickly",
        "Delivery was late and the package was damaged",
    ],
    "neutral": [
        "The battery is okay, nothing special either way",
        "The camera is average for this class of device",
        "The screen is fine, it does the job",
        "Performance is acceptable for basic tasks",
        "The price is about what I expected",
        "The design is plain, neither good nor bad",
        "Delivery took the usual amount of time",
    ],
}
STARS = {"positive": ["5", "4"], "negative": ["1", "2"], "neutral": ["3"]}


def build_rows() -> list[dict[str, str]]:
    rows = []
    for p_index, product in enumerate(PRODUCTS):
        for sentiment, phrases in PHRASES.items():
            for a_index, phrase in enumerate(phrases):
                stars = STARS[sentiment][(p_index + a_index) % len(STARS[sentiment])]
                rows.append({"title": "", "body": f"{phrase} on the {product}.", "stars": stars, "item": product})
    # Edge cases exercised by tests.
    rows += [
        {"title": "", "body": "   ", "stars": "5", "item": "Pulse Earbuds"},                 # empty text
        {"title": "Five Stars", "body": "", "stars": "5", "item": "Pulse Earbuds"},           # only a star title
        {"title": "Nice", "body": "Works as described.", "stars": "abc", "item": "Orbit Laptop"},  # bad rating
        {"title": "", "body": "Works as described, really.", "stars": "0", "item": "Orbit Laptop"},  # out of range
        {"title": "", "body": "Works as described, honestly.", "stars": "3.5", "item": "Orbit Laptop"},  # fractional
        {"title": "", "body": "The battery life is excellent and lasts all day on the Nova X1 Phone.",
         "stars": "5", "item": "Nova X1 Phone"},                                             # exact duplicate
        {"title": "", "body": "It is a phone.", "stars": "5", "item": "Nova X1 Phone"},       # conflicting dup 1
        {"title": "", "body": "it is a  PHONE.", "stars": "1", "item": "Nova X1 Phone"},      # conflicting dup 2
        {"title": "Five Stars", "body": "Love it, great sound!", "stars": "5", "item": "Pulse Earbuds"},  # star title dropped
        {"title": "", "body": "Fine earbuds.", "stars": "4", "item": ""},                     # missing product
    ]
    for i, row in enumerate(rows, start=1):
        row["id"] = str(i)
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "item", "title", "body", "stars"])
        writer.writeheader()
        writer.writerows(build_rows())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
