"""Build an analytics corpus (for the MCP tools) from a processed split, adding product names.

Default use: the held-out TEST split of the research dataset. It was never used
for training or model selection, so the sentiment model does not predict on
texts it has seen. (The dedicated per-product corpus from
`download_amazon.py` pass 2 is the larger alternative.)

Usage (PowerShell):
    python scripts/build_corpus.py --split data/processed/amazon_electronics/test.csv `
        --names data/raw/amazon_electronics/product_metadata.json `
        --out data/processed/amazon_electronics/analytics_corpus.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_intel.data.pipeline import load_product_names, read_processed_csv, write_rows_csv  # noqa: E402
from review_intel.utils import relative_to_root, sha256_file, utc_now, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--names", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = read_processed_csv(args.split)
    names = load_product_names(args.names)
    for row in rows:
        row["product_name"] = names.get(row["product_id"], row.get("product_name", ""))
    write_rows_csv(args.out, rows)
    info = {"source_split": relative_to_root(args.split), "source_sha256": sha256_file(args.split),
            "product_names": relative_to_root(args.names) if args.names else None,
            "rows": len(rows), "products_with_names": len({r["product_id"] for r in rows if r["product_name"]}),
            "created_utc": utc_now(),
            "note": "Analytics corpus for MCP tools. Built from a held-out split; not used for training/selection."}
    write_json(args.out.with_suffix(".json"), info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
