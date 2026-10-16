"""Validate a per-product review collection and its processed analytics corpus; write a JSON report.

Checks (all counted, nothing estimated):
* collected rows, products, per-product distribution, products with >= 30/50/100/500 reviews;
* duplicates (review keys) and missing values for the important fields;
* product association: each raw row's review_key embeds its own parent_asin; each processed row maps back to
  its raw row (review_id = source_offset) with the same product; optionally, a seeded random subset is
  re-read from the source at its recorded byte offset and compared field by field (--verify-source N);
* overlap with the research sample: review keys (must be 0) and identical texts (reported, not removed).

Usage (PowerShell):
    python scripts/validate_product_corpus.py --raw-dir data/raw/amazon_electronics_products `
        --corpus data/processed/amazon_electronics_products/corpus.csv --verify-source 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_intel.analytics.corpus import load_reviews  # noqa: E402
from review_intel.data import product_collection as pc  # noqa: E402
from review_intel.data.pipeline import read_processed_csv  # noqa: E402
from review_intel.preprocessing.text import dedup_key  # noqa: E402
from review_intel.utils import DATA_DIR, read_json, relative_to_root, setup_logging, utc_now, write_json  # noqa: E402

THRESHOLDS = (30, 50, 100, 500, 1000)
IMPORTANT_FIELDS = ("rating", "title", "text", "parent_asin", "user_id", "timestamp")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def distribution(counts: Counter) -> dict:
    values = sorted(counts.values())
    if not values:
        return {}
    return {"products": len(values), "min": values[0], "median": values[len(values) // 2], "max": values[-1],
            **{f"products_with_at_least_{t}": sum(v >= t for v in values) for t in THRESHOLDS}}


def verify_against_source(source: str, rows: list[dict], n: int, seed: int) -> dict:
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    mismatches = []
    for row in picked:
        with pc.default_opener(source, row["source_offset"]) as stream:
            original = json.loads(stream.readline())
        for field in ("parent_asin", "user_id", "timestamp", "rating", "text"):
            if original.get(field) != row.get(field):
                mismatches.append({"source_offset": row["source_offset"], "field": field})
                break
    return {"checked": len(picked), "seed": seed, "mismatches": len(mismatches), "examples": mismatches[:10]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=DATA_DIR / "raw" / "amazon_electronics_products")
    parser.add_argument("--corpus", type=Path, default=DATA_DIR / "processed" / "amazon_electronics_products" / "corpus.csv")
    parser.add_argument("--research-dir", type=Path, default=DATA_DIR / "processed" / "amazon_electronics",
                        help="Processed research splits (train/val/test.csv) for the text-overlap check")
    parser.add_argument("--verify-source", type=int, default=0, help="Re-read N random rows from the source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    setup_logging()

    manifest = read_json(args.raw_dir / "download_manifest.json")
    selected = [p["parent_asin"] for p in read_json(args.raw_dir / "selected_products.json")["products"]]
    rows = read_jsonl(args.raw_dir / "product_reviews.jsonl")
    keys = Counter(r["review_key"] for r in rows)
    raw_counts = Counter(r["parent_asin"] for r in rows)
    report: dict = {
        "created_utc": utc_now(),
        "raw": {
            "file": relative_to_root(args.raw_dir / "product_reviews.jsonl"),
            "rows": len(rows),
            "selected_products": len(selected),
            "products_with_reviews": len(raw_counts),
            "products_not_found": [p for p in selected if p not in raw_counts],
            "rows_outside_selected_products": sum(1 for r in rows if r["parent_asin"] not in set(selected)),
            "duplicate_review_keys_remaining": sum(v - 1 for v in keys.values() if v > 1),
            "duplicate_keys_dropped_during_collection": manifest.get("duplicate_keys_dropped"),
            "excluded_research_sample_reviews": manifest.get("excluded_research_sample"),
            "rows_whose_key_names_another_product": sum(1 for r in rows if r["review_key"].split("|")[1]
                                                        != r["parent_asin"]),
            "missing_rate": {f: round(sum(r.get(f) in (None, "") for r in rows) / len(rows), 5) for f in IMPORTANT_FIELDS},
            "distribution": distribution(raw_counts),
        },
    }

    sample_path = DATA_DIR / "raw" / "amazon_electronics" / "reviews_sample.jsonl"
    if sample_path.exists():
        report["raw"]["review_keys_shared_with_research_sample"] = len(set(keys) & pc.load_exclusion_keys(sample_path))

    if args.corpus.exists():
        processed = read_processed_csv(args.corpus)
        by_offset = {r["source_offset"]: r for r in rows}
        wrong = sum(1 for r in processed if by_offset.get(r["review_id"], {}).get("parent_asin") != r["product_id"])
        corpus = load_reviews(args.corpus)
        corpus_counts = Counter(r["product"] for r in corpus)
        named = {r["product"] for r in corpus if r.get("product_name")}
        report["corpus"] = {
            "file": relative_to_root(args.corpus),
            "rows": len(corpus),
            "rows_without_matching_raw_row_or_product": wrong,
            "products_with_name": len(named),
            "label_distribution": dict(Counter(r["sentiment"] for r in corpus)),
            "distribution": distribution(corpus_counts),
            "per_product": [{"product": p, "product_name": next((r.get("product_name", "") for r in corpus
                                                                  if r["product"] == p), ""), "reviews": n}
                            for p, n in corpus_counts.most_common()],
        }
        research_texts = set()
        for split in ("train", "val", "test"):
            path = args.research_dir / f"{split}.csv"
            if path.exists():
                research_texts |= {dedup_key(r["text"]) for r in read_processed_csv(path)}
        if research_texts:
            report["corpus"]["texts_identical_to_a_research_review"] = sum(
                dedup_key(r["text"]) in research_texts for r in corpus)

    if args.verify_source:
        report["source_verification"] = verify_against_source(manifest["review_source"], rows, args.verify_source,
                                                               args.seed)

    out = args.out or args.corpus.parent / "validation_report.json"
    write_json(out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "corpus"} | (
        {"corpus": {k: v for k, v in report["corpus"].items() if k != "per_product"}} if "corpus" in report else {}),
        indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
