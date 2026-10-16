"""Reproducible dataset preparation: load -> clean -> label -> deduplicate -> split -> manifest.

Driven by a JSON config (see ``configs/data/*.json``) so the exact same
processed dataset can be rebuilt from the same raw file and seed.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from review_intel.data.labels import LABEL_MAPPING_DOC, LABELS, normalize_label, parse_rating, rating_to_label
from review_intel.preprocessing.text import clean_text, combine_title_and_body, dedup_key
from review_intel.utils import PROJECT_ROOT, environment_info, read_json, relative_to_root, sha256_file, utc_now, write_json

logger = logging.getLogger(__name__)

PROCESSED_COLUMNS = ("review_id", "product_id", "product_name", "text", "rating", "label", "source_row")
SPLIT_NAMES = ("train", "val", "test")


class DatasetConfigError(ValueError):
    """The dataset config or input file is unusable."""


@dataclass
class ColumnMapping:
    """Which input columns hold which fields. ``text`` may list several columns (e.g. title + body)."""

    text: list[str]
    product_id: str | None = None
    rating: str | None = None
    label: str | None = None
    review_id: str | None = None
    group: str | None = None  # used only by the "group" split strategy


@dataclass
class SplitConfig:
    strategy: str = "stratified"  # "stratified" | "group" | "none"
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42


@dataclass
class DatasetConfig:
    name: str
    source_path: str
    columns: ColumnMapping
    source_format: str = "auto"  # "csv" | "jsonl" | "auto"
    origin: str = ""
    is_synthetic: bool = False
    label_source: str = "rating"  # "rating" | "column"
    text_join: str = ". "
    drop_star_titles: bool = True
    min_text_chars: int = 3
    deduplicate: bool = True
    product_names_path: str | None = None  # JSON {product_id: {"title": ...}} or {product_id: "name"}
    output_dir: str = ""
    split: SplitConfig = field(default_factory=SplitConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        data = dict(data)
        columns = data.pop("columns", None)
        if not columns or not columns.get("text"):
            raise DatasetConfigError("Config must define columns.text (one column name or a list).")
        text_cols = columns["text"] if isinstance(columns["text"], list) else [columns["text"]]
        mapping = ColumnMapping(**{**columns, "text": text_cols})
        split = SplitConfig(**data.pop("split", {}))
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise DatasetConfigError(f"Unknown config keys: {sorted(unknown)}")
        config = cls(columns=mapping, split=split, **data)
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: Path) -> DatasetConfig:
        return cls.from_dict(read_json(path))

    def validate(self) -> None:
        if self.label_source == "rating" and not self.columns.rating:
            raise DatasetConfigError("label_source 'rating' requires columns.rating.")
        if self.label_source == "column" and not self.columns.label:
            raise DatasetConfigError("label_source 'column' requires columns.label.")
        if self.label_source not in ("rating", "column"):
            raise DatasetConfigError("label_source must be 'rating' or 'column'.")
        if self.split.strategy not in ("stratified", "group", "none"):
            raise DatasetConfigError("split.strategy must be 'stratified', 'group' or 'none'.")
        if self.split.strategy == "group" and not self.columns.group:
            raise DatasetConfigError("split.strategy 'group' requires columns.group.")
        if self.split.strategy != "none" and not (0 < self.split.val_size and 0 < self.split.test_size
                                                   and self.split.val_size + self.split.test_size < 1):
            raise DatasetConfigError("val_size and test_size must be > 0 and sum to < 1.")

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Reading raw records
# ---------------------------------------------------------------------------


def detect_format(path: Path, declared: str = "auto") -> str:
    if declared != "auto":
        return declared
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    raise DatasetConfigError(f"Cannot infer format of {path.name}; set source_format to 'csv' or 'jsonl'.")


def iter_raw_records(path: Path, fmt: str) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise DatasetConfigError(f"Input file not found: {relative_to_root(path)}")
    if fmt == "csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
    elif fmt == "jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise DatasetConfigError(f"Invalid JSON on line {line_no} of {path.name}: {exc.msg}") from None
    else:
        raise DatasetConfigError(f"Unsupported format: {fmt}")


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def load_product_names(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    data = read_json(path)
    return {k: (v.get("title") if isinstance(v, dict) else v) or "" for k, v in data.items()}


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


@dataclass
class CleaningReport:
    input_rows: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    excluded: dict[str, int] = field(default_factory=lambda: Counter())  # type: ignore[assignment]
    output_rows: int = 0


def clean_records(records: Iterator[dict[str, Any]], config: DatasetConfig,
                  product_names: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], CleaningReport]:
    """Turn raw records into validated rows with a label; count every exclusion reason."""
    cols = config.columns
    product_names = product_names or {}
    report = CleaningReport()
    mapped = {c for c in [*cols.text, cols.product_id, cols.rating, cols.label, cols.review_id, cols.group] if c}
    missing: Counter[str] = Counter({c: 0 for c in mapped})
    excluded: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for source_row, record in enumerate(records):
        report.input_rows += 1
        for column in mapped:
            if _is_missing(record.get(column)):
                missing[column] += 1

        # Text
        parts = [str(record.get(c) or "") for c in cols.text]
        if len(parts) >= 2 and config.drop_star_titles:
            text = combine_title_and_body(parts[0], " ".join(parts[1:]), config.text_join)
        else:
            text = clean_text(config.text_join.join(p for p in parts if p.strip()))
        if len(text.replace(" ", "")) < config.min_text_chars:
            excluded["empty_or_too_short_text"] += 1
            continue

        # Rating and label
        rating = parse_rating(record.get(cols.rating)) if cols.rating else None
        if config.label_source == "rating":
            label = rating_to_label(rating)
            if label is None:
                excluded["missing_or_invalid_rating"] += 1
                continue
        else:
            label = normalize_label(record.get(cols.label))
            if label is None:
                excluded["missing_or_invalid_label"] += 1
                continue

        product_id = str(record.get(cols.product_id) or "").strip() if cols.product_id else ""
        if cols.product_id and not product_id:
            excluded["missing_product_id"] += 1
            continue

        raw_id = record.get(cols.review_id) if cols.review_id else None
        rows.append({
            "review_id": int(raw_id) if raw_id not in (None, "") and str(raw_id).isdigit() else source_row,
            "product_id": product_id,
            "product_name": product_names.get(product_id, ""),
            "text": text,
            "rating": rating,
            "label": label,
            "source_row": source_row,
            "_group": str(record.get(cols.group) or "") if cols.group else "",
        })

    if config.deduplicate:
        rows, dup_stats = deduplicate(rows)
        excluded.update(dup_stats)

    report.missing_values = dict(missing)
    report.excluded = dict(excluded)
    report.output_rows = len(rows)
    return rows, report


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Remove exact (case/whitespace-insensitive) duplicate texts.

    * Duplicates with the same label: keep the first occurrence.
    * Duplicates with conflicting labels: drop every copy (the text is ambiguous).

    Done *before* splitting, so the same text can never appear in two splits.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(dedup_key(row["text"]), []).append(row)
    kept, duplicates, conflicting = [], 0, 0
    for group in groups.values():
        if len({r["label"] for r in group}) > 1:
            conflicting += len(group)
            continue
        kept.append(group[0])
        duplicates += len(group) - 1
    kept.sort(key=lambda r: r["source_row"])
    return kept, {"duplicate_text": duplicates, "conflicting_label_duplicates": conflicting}


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_rows(rows: list[dict[str, Any]], split: SplitConfig) -> dict[str, list[dict[str, Any]]]:
    """Seeded train/val/test split. Stratified by label, or grouped (no group spans two splits)."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    if split.strategy == "none":
        return {"corpus": rows}
    if len(rows) < 10:
        raise DatasetConfigError(f"Need at least 10 rows to split, got {len(rows)}.")
    indices = list(range(len(rows)))
    labels = [r["label"] for r in rows]
    holdout = split.val_size + split.test_size
    test_share_of_holdout = split.test_size / holdout

    if split.strategy == "stratified":
        counts = Counter(labels)
        if min(counts.values()) < 3:
            raise DatasetConfigError(f"Stratified split needs >= 3 rows per class, got {dict(counts)}.")
        train_idx, rest_idx = train_test_split(indices, test_size=holdout, random_state=split.seed, stratify=labels)
        rest_labels = [labels[i] for i in rest_idx]
        val_idx, test_idx = train_test_split(rest_idx, test_size=test_share_of_holdout, random_state=split.seed,
                                             stratify=rest_labels)
    else:
        groups = [r["_group"] or f"row-{i}" for i, r in enumerate(rows)]
        gss = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=split.seed)
        train_idx, rest_idx = next(gss.split(indices, groups=groups))
        rest_groups = [groups[i] for i in rest_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=test_share_of_holdout, random_state=split.seed)
        val_pos, test_pos = next(gss2.split(rest_idx, groups=rest_groups))
        val_idx = [rest_idx[i] for i in val_pos]
        test_idx = [rest_idx[i] for i in test_pos]

    return {name: [rows[i] for i in sorted(idx)] for name, idx in zip(SPLIT_NAMES, (train_idx, val_idx, test_idx))}


def leakage_checks(splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Count overlaps between splits (text duplicates must be 0 after deduplication)."""
    keys = {name: {dedup_key(r["text"]) for r in rows} for name, rows in splits.items()}
    products = {name: {r["product_id"] for r in rows if r["product_id"]} for name, rows in splits.items()}
    checks = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in keys and b in keys:
            checks[f"text_overlap_{a}_{b}"] = len(keys[a] & keys[b])
            checks[f"product_overlap_{a}_{b}"] = len(products[a] & products[b])
    return checks


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROCESSED_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_processed_csv(path: Path) -> list[dict[str, Any]]:
    """Read a processed split written by ``write_rows_csv``."""
    if not path.exists():
        raise DatasetConfigError(f"Processed split not found: {relative_to_root(path)}. Run scripts/prepare_data.py first.")
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["review_id"] = int(row["review_id"])
            row["source_row"] = int(row["source_row"])
            row["rating"] = parse_rating(row.get("rating"))
            rows.append(row)
    return rows


def load_split(processed_dir: Path, split: str) -> tuple[list[str], list[str]]:
    """Return (texts, labels) for one split of a processed dataset."""
    rows = read_processed_csv(processed_dir / f"{split}.csv")
    return [r["text"] for r in rows], [r["label"] for r in rows]


def class_distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    counts = Counter(r["label"] for r in rows)
    total = len(rows)
    return {label: {"count": counts.get(label, 0),
                    "percent": round(100 * counts.get(label, 0) / total, 2) if total else 0.0}
            for label in LABELS}


@dataclass
class PreparedDataset:
    splits: dict[str, list[dict[str, Any]]]
    manifest: dict[str, Any]


def prepare_dataset(config: DatasetConfig, output_dir: Path | None = None) -> PreparedDataset:
    """Run the full pipeline and write split CSVs + ``manifest.json``."""
    source = config.resolve(config.source_path)
    fmt = detect_format(source, config.source_format)
    names_path = config.resolve(config.product_names_path) if config.product_names_path else None
    rows, report = clean_records(iter_raw_records(source, fmt), config, load_product_names(names_path))
    if not rows:
        raise DatasetConfigError("No valid rows remain after cleaning.")
    splits = split_rows(rows, config.split)

    out = output_dir or config.resolve(config.output_dir or f"data/processed/{config.name}")
    for name, split_rows_ in splits.items():
        write_rows_csv(out / f"{name}.csv", split_rows_)

    manifest = {
        "name": config.name,
        "is_synthetic": config.is_synthetic,
        "created_utc": utc_now(),
        "source": {
            "path": relative_to_root(source),
            "format": fmt,
            "sha256": sha256_file(source),
            "origin": config.origin,
        },
        "config": asdict(config),
        "label_mapping": LABEL_MAPPING_DOC if config.label_source == "rating" else "explicit label column",
        "counts": {
            "input_rows": report.input_rows,
            "missing_values_in_mapped_columns": report.missing_values,
            "excluded": report.excluded,
            "rows_after_cleaning": report.output_rows,
        },
        "split_sizes": {name: len(r) for name, r in splits.items()},
        "class_distribution": {"all": class_distribution(rows),
                               **{name: class_distribution(r) for name, r in splits.items()}},
        "leakage_checks": leakage_checks(splits),
        "output_dir": relative_to_root(out),
        "environment": environment_info(),
    }
    write_json(out / "manifest.json", manifest)
    return PreparedDataset(splits=splits, manifest=manifest)
