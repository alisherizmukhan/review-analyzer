"""Model A (TF-IDF + Logistic Regression) and Model B (TF-IDF + Linear SVM).

Each model is one scikit-learn ``Pipeline(features -> classifier)``, so the
fitted vectorizer and classifier are always saved and loaded together. The
vectorizer (vocabulary, IDF weights) is fitted on the training split only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from review_intel.models.base import Prediction

MODEL_TYPES = ("logreg", "linear_svm")


def build_features(word: dict[str, Any], char: dict[str, Any] | None = None) -> FeatureUnion:
    """Word n-gram TF-IDF, optionally concatenated with character n-gram TF-IDF.

    ``word``/``char`` are TfidfVectorizer kwargs; lists for ngram_range are converted to tuples.
    """
    def vec(params: dict[str, Any], analyzer: str) -> TfidfVectorizer:
        params = dict(params)
        if "ngram_range" in params:
            params["ngram_range"] = tuple(params["ngram_range"])
        params.setdefault("sublinear_tf", True)
        params.setdefault("lowercase", True)
        params.setdefault("dtype", np.float32)
        return TfidfVectorizer(analyzer=analyzer, **params)

    parts = [("word", vec(word, "word"))]
    if char:
        parts.append(("char", vec(char, "char_wb")))
    return FeatureUnion(parts)


def build_classifier(model_type: str, params: dict[str, Any], seed: int):
    if model_type == "logreg":
        return LogisticRegression(
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=params.get("max_iter", 2000),
            solver=params.get("solver", "lbfgs"),
            random_state=seed,
        )
    if model_type == "linear_svm":
        return LinearSVC(
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=params.get("max_iter", 5000),
            random_state=seed,
        )
    raise ValueError(f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")


def build_pipeline(model_type: str, features: dict[str, Any], clf_params: dict[str, Any], seed: int) -> Pipeline:
    return Pipeline([
        ("features", build_features(features["word"], features.get("char"))),
        ("clf", build_classifier(model_type, clf_params, seed)),
    ])


def score_type_for(pipeline: Pipeline) -> str:
    """Probabilities only when the classifier really produces them."""
    return "probability" if hasattr(pipeline.named_steps["clf"], "predict_proba") else "decision_score"


def predict_with_scores(pipeline: Pipeline, texts: Sequence[str]) -> list[Prediction]:
    clf = pipeline.named_steps["clf"]
    classes = [str(c) for c in clf.classes_]
    features = pipeline.named_steps["features"].transform(list(texts))
    if hasattr(clf, "predict_proba"):
        matrix = clf.predict_proba(features)
    else:
        matrix = clf.decision_function(features)
        if matrix.ndim == 1:  # binary case: single margin for classes[1]
            matrix = np.column_stack([-matrix, matrix])
    best = matrix.argmax(axis=1)
    return [
        Prediction(label=classes[b], scores={c: round(float(v), 6) for c, v in zip(classes, row)})
        for b, row in zip(best, matrix)
    ]
