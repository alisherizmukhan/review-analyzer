"""Prepare a processed dataset (cleaned, labelled, deduplicated, split) from a raw CSV/JSONL file.

Usage (PowerShell):
    python scripts/prepare_data.py --config configs/data/amazon_electronics.json
    python scripts/prepare_data.py --config configs/data/fixture.json

    # Your own CSV without writing a config file:
    python scripts/prepare_data.py --csv my_reviews.csv --name my_data `
        --text-col review_text --rating-col stars --product-col product

Writes <output_dir>/{train,val,test}.csv (or corpus.csv) and manifest.json,
and prints a short summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_intel.data.pipeline import DatasetConfig, prepare_dataset  # noqa: E402
from review_intel.utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="JSON dataset config (configs/data/*.json)")
    parser.add_argument("--csv", type=Path, help="Raw CSV (alternative to --config)")
    parser.add_argument("--name", default="custom_csv")
    parser.add_argument("--text-col", action="append", help="Text column; repeat for title + body")
    parser.add_argument("--rating-col")
    parser.add_argument("--label-col", help="Use an explicit positive/neutral/negative column instead of ratings")
    parser.add_argument("--product-col")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    setup_logging()

    if args.config:
        config = DatasetConfig.from_file(args.config)
    elif args.csv:
        if not args.text_col:
            parser.error("--csv requires at least one --text-col")
        config = DatasetConfig.from_dict({
            "name": args.name,
            "origin": f"user-provided CSV {args.csv.name}",
            "source_path": str(args.csv.resolve()),
            "source_format": "csv",
            "columns": {"text": args.text_col, "rating": args.rating_col, "label": args.label_col,
                        "product_id": args.product_col},
            "label_source": "column" if args.label_col else "rating",
            "split": {"strategy": "stratified", "seed": args.seed},
        })
    else:
        parser.error("give --config or --csv")

    result = prepare_dataset(config, args.output_dir)
    m = result.manifest
    summary = {
        "name": m["name"],
        "is_synthetic": m["is_synthetic"],
        "output_dir": m["output_dir"],
        "input_rows": m["counts"]["input_rows"],
        "excluded": m["counts"]["excluded"],
        "rows_after_cleaning": m["counts"]["rows_after_cleaning"],
        "split_sizes": m["split_sizes"],
        "class_distribution_all": m["class_distribution"]["all"],
        "leakage_checks": m["leakage_checks"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
