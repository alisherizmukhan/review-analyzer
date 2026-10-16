"""Look up product titles for given parent ASINs in the official Amazon Reviews 2023 metadata.

Reads only the `parent_asin` and `title` columns of the Hugging Face Parquet
shards (raw_meta_<Category>) with HTTP range requests, so a few hundred MB are
transferred instead of the full ~2 GB. Needs: pip install -e .[data]

Usage (PowerShell):
    python scripts/fetch_product_titles.py --from-csv data/processed/amazon_electronics/test.csv --top 25 `
        --out data/raw/amazon_electronics/product_metadata.json
    python scripts/fetch_product_titles.py --asin-file data/raw/amazon_electronics_products/selected_products.json `
        --out data/raw/amazon_electronics_products/product_metadata.json

Products whose title is not found stay unnamed (listed under "not_found"); no title is ever guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_intel.utils import read_json, setup_logging, write_json  # noqa: E402

logger = logging.getLogger("fetch_product_titles")
REPO = "datasets/McAuley-Lab/Amazon-Reviews-2023"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="Electronics")
    parser.add_argument("--from-csv", type=Path, help="Processed CSV; look up its most-reviewed products")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--asin", action="append", default=[])
    parser.add_argument("--asin-file", type=Path,
                        help="JSON with product ids: a list, or selected_products.json from download_amazon.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    setup_logging()
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        print(f"error: {exc}. Install with: pip install -e .[data]", file=sys.stderr)
        return 2

    wanted = set(args.asin)
    if args.asin_file:
        data = read_json(args.asin_file)
        items = data["products"] if isinstance(data, dict) else data
        wanted |= {item["parent_asin"] if isinstance(item, dict) else str(item) for item in items}
    if args.from_csv:
        with args.from_csv.open(encoding="utf-8", newline="") as f:
            counts = Counter(row["product_id"] for row in csv.DictReader(f))
        wanted |= {asin for asin, _ in counts.most_common(args.top)}
    found: dict[str, dict] = read_json(args.out) if args.out.exists() else {}
    wanted -= set(found)
    logger.info("Looking up %d products", len(wanted))

    fs = HfFileSystem()
    shards = sorted(fs.glob(f"{REPO}/raw_meta_{args.category}/*.parquet"))
    for shard in shards:
        if not wanted:
            break
        with fs.open(shard, "rb", block_size=4 << 20) as handle:
            table = pq.read_table(handle, columns=["parent_asin", "title", "main_category"])
        hits = table.filter(pc.is_in(table["parent_asin"], value_set=__import__("pyarrow").array(sorted(wanted))))
        for row in hits.to_pylist():
            found[row["parent_asin"]] = {"parent_asin": row["parent_asin"], "title": row["title"],
                                         "main_category": row["main_category"],
                                         "source": f"hf://{REPO}/raw_meta_{args.category}"}
            wanted.discard(row["parent_asin"])
        logger.info("%s: %d found so far, %d remaining", shard.rsplit("/", 1)[-1], len(found), len(wanted))

    write_json(args.out, found)
    print(json.dumps({"found": len(found), "not_found": sorted(wanted)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
