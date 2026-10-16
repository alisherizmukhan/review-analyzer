"""Training + validation-based hyperparameter search for the classical baselines.

Protocol (no test data is touched here):
1. Fit each feature configuration's TF-IDF on the TRAIN split only.
2. For each classifier hyperparameter combination, fit on TRAIN and score
   macro-F1 on the VALIDATION split.
3. Per model family keep the configuration with the best validation
   macro-F1, save it as an artifact and record its metrics, training time and
   latency. Cross-family selection happens later in scripts/evaluate_models.py.
"""

from __future__ import annotations

import itertools
import logging
import time
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from review_intel.data.pipeline import load_split
from review_intel.evaluation.metrics import classification_metrics, evaluate_model, measure_latency
from review_intel.models.artifacts import SklearnSentimentModel, save_sklearn_model
from review_intel.models.sklearn_models import build_classifier, build_features
from review_intel.utils import environment_info, read_json, relative_to_root, utc_now, write_json

logger = logging.getLogger(__name__)


def _grid(classifier_grid: dict[str, list]) -> list[dict[str, Any]]:
    keys = sorted(classifier_grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(classifier_grid[k] for k in keys))]


def train_baselines(processed_dir: Path, config: dict[str, Any], models_dir: Path, reports_dir: Path,
                    families: list[str] | None = None, measure_speed: bool = True) -> dict[str, Any]:
    seed = config.get("seed", 42)
    manifest = read_json(processed_dir / "manifest.json")
    dataset = {"name": manifest["name"], "is_synthetic": manifest["is_synthetic"],
               "processed_dir": relative_to_root(processed_dir), "source_sha256": manifest["source"]["sha256"],
               "split_sizes": manifest["split_sizes"]}
    x_train, y_train = load_split(processed_dir, "train")
    x_val, y_val = load_split(processed_dir, "val")
    logger.info("Loaded %s: train=%d val=%d", dataset["name"], len(x_train), len(x_val))

    selected_families = {k: v for k, v in config["families"].items() if not families or k in families}
    feature_grid = config["feature_grid"]

    # Vectorize once per feature configuration; share matrices across families.
    candidates: dict[str, list[dict[str, Any]]] = {f: [] for f in selected_families}
    best: dict[str, dict[str, Any]] = {}
    for feat in feature_grid:
        logger.info("Fitting features %s on train", feat["id"])
        start = time.perf_counter()
        features = build_features(feat["word"], feat.get("char"))
        xt = features.fit_transform(x_train)
        feature_seconds = time.perf_counter() - start
        xv = features.transform(x_val)
        logger.info("  %d features in %.1fs", xt.shape[1], feature_seconds)

        for family, spec in selected_families.items():
            for params in _grid(spec["classifier_grid"]):
                clf = build_classifier(spec["model_type"], params, seed)
                start = time.perf_counter()
                clf.fit(xt, y_train)
                clf_seconds = time.perf_counter() - start
                metrics = classification_metrics(y_val, clf.predict(xv))
                row = {"features": feat["id"], "classifier_params": params,
                       "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"],
                       "fit_seconds_features": round(feature_seconds, 2), "fit_seconds_classifier": round(clf_seconds, 2)}
                candidates[family].append(row)
                logger.info("  %s %s %s -> val macro-F1 %.4f (%.1fs)", family, feat["id"], params,
                            metrics["macro_f1"], clf_seconds)
                if family not in best or metrics["macro_f1"] > best[family]["row"]["val_macro_f1"]:
                    best[family] = {"row": row, "features": features, "clf": clf, "feature_spec": feat,
                                    "params": params}

    results: dict[str, Any] = {}
    for family, spec in selected_families.items():
        chosen = best[family]
        pipeline = Pipeline([("features", chosen["features"]), ("clf", chosen["clf"])])
        metadata = save_sklearn_model(models_dir / family, pipeline, {
            "name": family,
            "algorithm": {"logreg": "TF-IDF + LogisticRegression", "linear_svm": "TF-IDF + LinearSVC"}[spec["model_type"]],
            "dataset": dataset,
            "seed": seed,
            "params": {"features": chosen["feature_spec"], "classifier": {"model_type": spec["model_type"], **chosen["params"]}},
            "training_time_seconds": round(chosen["row"]["fit_seconds_features"] + chosen["row"]["fit_seconds_classifier"], 2),
        })
        model = SklearnSentimentModel(pipeline, metadata)
        val_metrics, _ = evaluate_model(model, x_val, y_val)
        latency = measure_latency(model, x_val, seed=seed) if measure_speed else None
        metadata["validation_metrics"] = {"macro_f1": val_metrics["macro_f1"], "accuracy": val_metrics["accuracy"]}
        write_json(models_dir / family / "metadata.json", metadata)

        family_report = {
            "family": family,
            "model_dir": relative_to_root(models_dir / family),
            "model_version": metadata["version"],
            "algorithm": metadata["algorithm"],
            "score_type": metadata["score_type"],
            "dataset": dataset,
            "selection_within_family": "highest validation macro-F1 over the grid below",
            "best_params": metadata["params"],
            "training_time_seconds": metadata["training_time_seconds"],
            "validation": val_metrics,
            "latency": latency,
            "grid": candidates[family],
            "created_utc": utc_now(),
            "environment": environment_info(),
        }
        write_json(reports_dir / "validation" / f"{family}.json", family_report)
        results[family] = family_report
    return results
