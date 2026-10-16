"""Cross-family model selection on VALIDATION data, then a single held-out TEST evaluation.

Selection rule (documented in docs/research_methodology.md):
    1. Highest validation macro-F1.
    2. Tie (equal to 4 decimals) -> fixed preference order TIE_BREAK_ORDER
       (cheaper models first, probability-producing before decision-score
       models). Deterministic, unlike a latency tie-break, which is timing noise.
The test split is used only after the selection has been written to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from review_intel.data.pipeline import load_split
from review_intel.evaluation.metrics import evaluate_model, measure_latency
from review_intel.models.artifacts import load_model, write_selected_pointer
from review_intel.utils import environment_info, read_json, relative_to_root, utc_now, write_json

logger = logging.getLogger(__name__)

TIE_BREAK_ORDER = ("tfidf_logreg", "tfidf_linearsvm", "distilbert")
SELECTION_RULE = ("Highest validation macro-F1; ties (equal to 4 decimals) broken by the fixed preference order "
                  f"{list(TIE_BREAK_ORDER)} (cheaper model first). The test split is not used for selection.")


class SelectionError(RuntimeError):
    pass


def load_validation_reports(reports_dir: Path) -> list[dict[str, Any]]:
    files = sorted((reports_dir / "validation").glob("*.json"))
    if not files:
        raise SelectionError(f"No validation reports in {relative_to_root(reports_dir / 'validation')}. "
                             "Run scripts/train_baselines.py (and optionally scripts/train_transformer.py) first.")
    return [read_json(f) for f in files]


def rank_candidates(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(r: dict[str, Any]):
        family = r["family"]
        preference = TIE_BREAK_ORDER.index(family) if family in TIE_BREAK_ORDER else len(TIE_BREAK_ORDER)
        return (-round(r["validation"]["macro_f1"], 4), preference, family)
    return sorted(reports, key=key)


def select_model(reports_dir: Path, models_dir: Path, activate_pointer: Path | None = None) -> dict[str, Any]:
    ranked = rank_candidates(load_validation_reports(reports_dir))
    winner = ranked[0]
    model_dir = models_dir / winner["family"]
    selection = {
        "selected_family": winner["family"],
        "model_version": winner["model_version"],
        "selection_rule": SELECTION_RULE,
        "validation_macro_f1": winner["validation"]["macro_f1"],
        "ranking": [{"family": r["family"], "val_macro_f1": r["validation"]["macro_f1"],
                     "val_accuracy": r["validation"]["accuracy"],
                     "median_latency_ms": (r.get("latency") or {}).get("single_text_median_ms")} for r in ranked],
        "dataset": winner["dataset"]["name"],
    }
    write_selected_pointer(model_dir, selection, models_dir / "selected.json")
    if activate_pointer is not None:
        write_selected_pointer(model_dir, selection, activate_pointer)
    write_json(reports_dir / "model_selection.json", {**selection, "created_utc": utc_now()})
    (reports_dir / "validation_comparison.md").write_text(render_comparison(ranked, selection), encoding="utf-8")
    return selection


def evaluate_on_test(processed_dir: Path, reports_dir: Path, models_dir: Path,
                     include_non_selected: bool = False) -> dict[str, Any]:
    selection_path = reports_dir / "model_selection.json"
    if not selection_path.exists():
        raise SelectionError("No model selection recorded. Run `scripts/evaluate_models.py compare` first; "
                             "the test split may only be used after selection.")
    selection = read_json(selection_path)
    x_test, y_test = load_split(processed_dir, "test")
    families = [r["family"] for r in selection["ranking"]] if include_non_selected else [selection["selected_family"]]
    validation = {r["family"]: r for r in load_validation_reports(reports_dir)}

    results = {}
    for family in families:
        model = load_model(models_dir / family)
        metrics, _ = evaluate_model(model, x_test, y_test)
        results[family] = {
            "role": "selected" if family == selection["selected_family"] else "post-selection comparison only",
            "model_version": model.version,
            "score_type": model.score_type,
            "test": metrics,
            "latency_on_test_texts": measure_latency(model, x_test),
            "training_time_seconds": validation[family].get("training_time_seconds"),
        }
        logger.info("%s test macro-F1 %.4f", family, metrics["macro_f1"])

    report = {"dataset": selection["dataset"], "test_size": len(y_test), "selection": selection,
              "results": results, "created_utc": utc_now(), "environment": environment_info()}
    write_json(reports_dir / "test_results.json", report)
    (reports_dir / "test_report.md").write_text(render_test_report(report), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _row(name: str, m: dict[str, Any], latency: dict | None, train_s, score_type: str) -> str:
    pc = m["per_class"]
    lat = latency or {}
    return (f"| {name} | {m['accuracy']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | "
            f"**{m['macro_f1']:.4f}** | {pc['negative']['f1']:.4f} | {pc['neutral']['f1']:.4f} | "
            f"{pc['positive']['f1']:.4f} | {train_s if train_s is not None else '-'} | "
            f"{lat.get('single_text_median_ms', '-')} / {lat.get('single_text_p95_ms', '-')} | "
            f"{lat.get('batch_throughput_texts_per_s', '-')} | {score_type} |")


_HEADER = ("| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | F1 neg | F1 neu | F1 pos | Train time (s) | "
           "Latency median / p95 (ms) | Batch texts/s | Score type |\n|---|---|---|---|---|---|---|---|---|---|---|---|")


def _confusion(m: dict[str, Any]) -> str:
    labels = m["confusion_matrix"]["labels"]
    lines = ["| true \\ predicted | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for label, row in zip(labels, m["confusion_matrix"]["matrix"]):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def render_comparison(ranked: list[dict[str, Any]], selection: dict[str, Any]) -> str:
    dataset = ranked[0]["dataset"]
    out = [f"# Validation comparison: {dataset['name']}", "",
           f"Synthetic data: **{dataset['is_synthetic']}**. Split sizes: {dataset['split_sizes']}.", "",
           f"Selection rule: {SELECTION_RULE}", "", f"**Selected: `{selection['selected_family']}`**", "",
           "All numbers below are on the VALIDATION split.", "", _HEADER]
    for r in ranked:
        out.append(_row(r["family"], r["validation"], r.get("latency"), r.get("training_time_seconds"), r["score_type"]))
    for r in ranked:
        out += ["", f"## {r['family']}: confusion matrix (validation)", "", _confusion(r["validation"]),
                "", f"Best parameters: `{r['best_params']}`"]
        if "calibration" in r["validation"]:
            out.append(f"Calibration (top-label ECE, 15 bins): {r['validation']['calibration']}")
    if ranked and ranked[0].get("latency"):
        out += ["", "## Latency method", "", ranked[0]["latency"]["method"]]
    return "\n".join(out) + "\n"


def render_test_report(report: dict[str, Any]) -> str:
    out = [f"# Held-out test results: {report['dataset']}", "",
           f"Test size: {report['test_size']}. Selected model: **`{report['selection']['selected_family']}`** "
           f"(selected on validation macro-F1 = {report['selection']['validation_macro_f1']}).", "",
           "Rows marked 'post-selection' were evaluated on test only after the selection was frozen; "
           "they were not used for any decision.", "", _HEADER]
    for family, r in report["results"].items():
        name = family if r["role"] == "selected" else f"{family} (post-selection)"
        out.append(_row(name, r["test"], r["latency_on_test_texts"], r["training_time_seconds"], r["score_type"]))
    for family, r in report["results"].items():
        out += ["", f"## {family}: confusion matrix (test)", "", _confusion(r["test"])]
        if "calibration" in r["test"]:
            out.append(f"\nCalibration (top-label ECE, 15 bins): {r['test']['calibration']}")
    env = report["environment"]
    out += ["", "## Environment", "", f"`{env.get('packages')}`", "", f"Platform: {env.get('platform')}; "
            f"CPU: {env.get('processor')}; GPU: {env.get('gpu', 'n/a')}"]
    return "\n".join(out) + "\n"
