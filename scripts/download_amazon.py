"""Stream Amazon Reviews 2023 and save (1) a uniform random sample and (2) a product corpus.

Nothing but the sampled lines is written to disk; the multi-GB source is
streamed and discarded. See src/review_intel/data/amazon.py for the method.

Example (PowerShell):
    python scripts/download_amazon.py --category Electronics --sample-size 100000 `
        --top-products 10 --per-product-cap 1000 --seed 42

Outputs (data/raw/ is git-ignored):
    data/raw/amazon_<category>/reviews_sample.jsonl      uniform sample (research data)
    data/raw/amazon_<category>/product_corpus.jsonl      reviews of the top-N products (analytics)
    data/raw/amazon_<category>/product_metadata.json     titles of those products
    data/raw/amazon_<category>/download_manifest.json    source URLs, headers, seeds, counts

Per-product collection only (--products-only; resumable, see src/review_intel/data/product_collection.py).
Reads the *uncompressed* file from the official Hugging Face mirror in byte-range chunks; a re-run with the
same arguments resumes. Products come from --products or are the --top-from-sample most-reviewed products of
the research sample, whose reviews are excluded:

    python scripts/download_amazon.py --products-only --category Electronics `
        --top-from-sample 50 --per-product-cap 1000 --seed 42

Outputs (default data/raw/amazon_<category>_products/):
    selected_products.json  product_reviews.jsonl  download_manifest.json  chunks/ (resume state)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_intel.data import amazon  # noqa: E402
from review_intel.data import product_collection as pc  # noqa: E402
from review_intel.utils import (  # noqa: E402
    DATA_DIR, environment_info, relative_to_root, setup_logging, sha256_file, utc_now, write_json,
)  # noqa: E402

logger = logging.getLogger("download_amazon")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="Electronics")
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--top-products", type=int, default=10, help="0 skips the product-corpus pass")
    parser.add_argument("--per-product-cap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Debug only: stop after N lines (sample is then NOT uniform over the category)")
    parser.add_argument("--source", default=None, help="Local .jsonl.gz instead of the official URL")
    parser.add_argument("--out-dir", type=Path, default=None)
    collect = parser.add_argument_group("per-product collection (--products-only)")
    collect.add_argument("--products-only", action="store_true",
                         help="Skip the sample pass; collect reviews for selected products (resumable)")
    collect.add_argument("--products", nargs="+", default=None, help="parent_asin ids to collect")
    collect.add_argument("--top-from-sample", type=int, default=50,
                         help="Without --products: the N most-reviewed products of the research sample")
    collect.add_argument("--sample-path", type=Path, default=None,
                         help="Research sample (selection + exclusion); default "
                              "data/raw/amazon_<category>/reviews_sample.jsonl")
    collect.add_argument("--hf-revision", default="main", help="Hugging Face mirror revision (pinned to a commit)")
    collect.add_argument("--chunk-mb", type=int, default=256)
    collect.add_argument("--workers", type=int, default=4)
    collect.add_argument("--keep-chunks", action="store_true", help="Keep chunks/ after a successful run")
    args = parser.parse_args()
    setup_logging()
    if args.products_only:
        sys.exit(collect_products_main(args))

    out_dir = args.out_dir or DATA_DIR / "raw" / f"amazon_{args.category.lower()}"
    source = args.source or amazon.review_url(args.category)
    manifest: dict = {
        "dataset": "Amazon Reviews 2023 (McAuley Lab)",
        "citation": "Hou et al., Bridging Language and Items for Retrieval and Recommendation, arXiv:2403.03952, 2024",
        "category": args.category,
        "review_source": source,
        "seed": args.seed,
        "sample_size_requested": args.sample_size,
        "max_lines": args.max_lines,
        "sampling_method": "reservoir sampling (Algorithm R) over every review line of the streamed file",
        "started_utc": utc_now(),
        "environment": environment_info(),
    }

    # Pass 1: uniform sample + product counts.
    logger.info("Pass 1/3: streaming %s", source)
    started = time.time()
    stream, headers = amazon.open_jsonl_gz(source)
    with stream:
        result = amazon.reservoir_sample_pass(amazon.iter_lines(stream, args.max_lines), args.sample_size, args.seed)
    manifest["source_headers"] = {k: headers.get(k) for k in ("content-length", "last-modified", "etag")}
    manifest["total_review_lines_streamed"] = result.total_lines
    manifest["distinct_products_streamed"] = len(result.product_counts)
    manifest["pass1_seconds"] = round(time.time() - started, 1)
    sample_rows = [amazon.parse_review_line(line, i) for i, line in result.sample]
    sample_path = out_dir / "reviews_sample.jsonl"
    write_jsonl(sample_path, sample_rows)
    manifest["sample_file"] = relative_to_root(sample_path)
    manifest["sample_rows_written"] = len(sample_rows)
    manifest["sample_sha256"] = sha256_file(sample_path)
    write_json(out_dir / "product_counts_top1000.json", dict(result.product_counts.most_common(1000)))
    write_json(out_dir / "download_manifest.json", {**manifest, "status": "pass 1 complete"})
    logger.info("Pass 1 done: %d lines streamed, %d sampled", result.total_lines, len(sample_rows))

    if args.top_products > 0:
        top = result.product_counts.most_common(args.top_products)
        products = {asin for asin, _ in top}
        manifest["top_products"] = [{"parent_asin": a, "reviews_in_category": n} for a, n in top]

        # Pass 2: reviews of the most-reviewed products, excluding the research sample.
        logger.info("Pass 2/3: collecting reviews for %d products", len(products))
        started = time.time()
        sampled_indices = {i for i, _ in result.sample}
        stream, _ = amazon.open_jsonl_gz(source)
        with stream:
            per_product = amazon.product_reviews_pass(
                amazon.iter_lines(stream, args.max_lines), products, args.per_product_cap, args.seed, sampled_indices
            )
        corpus_rows = [amazon.parse_review_line(line, i) for rows in per_product.values() for i, line in rows]
        corpus_path = out_dir / "product_corpus.jsonl"
        write_jsonl(corpus_path, corpus_rows)
        manifest["pass2_seconds"] = round(time.time() - started, 1)
        manifest["product_corpus_file"] = relative_to_root(corpus_path)
        manifest["product_corpus_rows_written"] = len(corpus_rows)
        manifest["product_corpus_excludes_research_sample"] = True
        write_json(out_dir / "download_manifest.json", {**manifest, "status": "pass 2 complete"})

        # Pass 3: product titles from the metadata file.
        logger.info("Pass 3/3: product metadata")
        started = time.time()
        meta_source = amazon.meta_url(args.category) if not args.source else None
        if meta_source:
            stream, _ = amazon.open_jsonl_gz(meta_source)
            with stream:
                meta = amazon.metadata_pass(amazon.iter_lines(stream), products)
            write_json(out_dir / "product_metadata.json", meta)
            manifest["metadata_source"] = meta_source
            manifest["metadata_products_found"] = len(meta)
            manifest["pass3_seconds"] = round(time.time() - started, 1)

    manifest["status"] = "complete"
    manifest["finished_utc"] = utc_now()
    write_json(out_dir / "download_manifest.json", manifest)
    logger.info("Done. Manifest: %s", relative_to_root(out_dir / "download_manifest.json"))


def resolve_hf_revision(revision: str) -> tuple[str, str | None]:
    """Pin the mirror to a commit for reproducibility (needs huggingface_hub, the [data] extra)."""
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(amazon.HF_REPO, revision=revision).sha, None
    except Exception as exc:  # noqa: BLE001 - optional: fall back to the named revision, and say so
        return revision, f"could not resolve revision {revision!r} to a commit ({exc}); using it unpinned"


def collect_products_main(args: argparse.Namespace) -> int:
    category = args.category
    out_dir = args.out_dir or DATA_DIR / "raw" / f"amazon_{category.lower()}_products"
    sample_path = args.sample_path or DATA_DIR / "raw" / f"amazon_{category.lower()}" / "reviews_sample.jsonl"
    if args.per_product_cap < 1:
        logger.error("--per-product-cap must be >= 1")
        return 2
    if args.products:
        try:
            ids = pc.validate_product_ids(args.products)
        except ValueError as exc:
            logger.error("%s", exc)
            return 2
        selected = [{"rank": i + 1, "parent_asin": a} for i, a in enumerate(ids)]
        selection_rule = "explicit --products list"
    else:
        if not sample_path.exists():
            logger.error("Research sample not found: %s (run the sample pass first, or pass --products)",
                         relative_to_root(sample_path))
            return 2
        selected = pc.rank_products_in_sample(sample_path, args.top_from_sample)
        selection_rule = (f"the {args.top_from_sample} products with the most reviews in the uniform research sample "
                          f"{relative_to_root(sample_path)} (ties: product id)")
    exclude = pc.load_exclusion_keys(sample_path) if sample_path.exists() else set()

    pin_note = None
    if args.source:
        source = args.source
        if source.endswith(".gz"):
            logger.error("--products-only needs an uncompressed .jsonl source (byte ranges); got %s", source)
            return 2
    else:
        revision, pin_note = resolve_hf_revision(args.hf_revision)
        source = amazon.hf_review_url(category, revision)
        if pin_note:
            logger.warning(pin_note)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "selected_products.json", {"selection_rule": selection_rule, "products": selected})
    manifest: dict = {
        "dataset": "Amazon Reviews 2023 (McAuley Lab)",
        "citation": "Hou et al., Bridging Language and Items for Retrieval and Recommendation, arXiv:2403.03952, 2024",
        "licence_note": "No licence is stated on the dataset website or the Hugging Face card; used for "
                        "non-commercial research with citation. Raw data stays local (data/raw/ is git-ignored) "
                        "and is not redistributed.",
        "category": category, "review_source": source, "revision_note": pin_note,
        "selection_rule": selection_rule, "per_product_cap": args.per_product_cap, "seed": args.seed,
        "sampling_method": "per-product bottom-k: the cap reviews with the smallest sha1(seed|review_key) "
                           "(uniform without replacement, independent of file order and resumes)",
        "review_key": "|".join(pc.KEY_FIELDS),
        "excluded": (f"reviews of the research sample {relative_to_root(sample_path)} (matched by review_key)"
                     if exclude else "nothing (research sample not found)"),
        "excluded_keys": len(exclude), "chunk_mb": args.chunk_mb, "workers": args.workers,
        "command": " ".join(sys.argv), "started_utc": utc_now(), "environment": environment_info(),
    }
    try:
        total_bytes, headers = pc.source_size(source)
        manifest["source_bytes"] = total_bytes
        manifest["source_headers"] = {k: headers.get(k) for k in ("etag", "last-modified", "x-repo-commit")}
        write_json(out_dir / "download_manifest.json", {**manifest, "status": "collecting"})
        result = pc.collect_products(source, [p["parent_asin"] for p in selected], args.per_product_cap, args.seed,
                                     out_dir / "chunks", exclude, args.chunk_mb << 20, args.workers,
                                     total_bytes=total_bytes)
    except (pc.CollectionError, amazon.SourceAccessError, ValueError) as exc:
        logger.error("%s", exc)
        write_json(out_dir / "download_manifest.json",
                   {**manifest, "status": f"failed: {exc}", "failed_utc": utc_now()})
        return 1

    rows = result.collector.rows()
    reviews_path = out_dir / "product_reviews.jsonl"
    write_jsonl(reviews_path, rows)
    report = pc.collection_report(result.collector, selected)
    missing = {f: sum(1 for r in rows if r.get(f) in (None, "")) for f in (*amazon.REVIEW_FIELDS, "source_offset")}
    manifest.update({
        "status": "complete", "finished_utc": utc_now(), "seconds": result.seconds,
        "total_review_lines_in_source": result.total_lines, "chunks": result.chunks_total,
        "chunks_reused_from_previous_run": result.chunks_resumed,
        "product_reviews_file": relative_to_root(reviews_path), "product_reviews_sha256": sha256_file(reviews_path),
        "missing_values_in_collected_rows": missing, **report,
    })
    write_json(out_dir / "download_manifest.json", manifest)
    if not args.keep_chunks:
        import shutil
        shutil.rmtree(out_dir / "chunks", ignore_errors=True)
    logger.info("Done: %d reviews for %d products (%d not found). Manifest: %s", report["rows"], len(selected),
                len(report["products_not_found"]), relative_to_root(out_dir / "download_manifest.json"))
    return 0


if __name__ == "__main__":
    main()
