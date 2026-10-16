"""Evaluation metrics, calibration and the documented latency measurement."""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from review_intel.data.labels import LABELS
from review_intel.models.base import Prediction, SentimentModel


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str],
                           labels: Sequence[str] = LABELS) -> dict[str, Any]:
    """Accuracy, macro P/R/F1, per-class P/R/F1/support and the confusion matrix.

    ``zero_division=0`` makes a never-predicted class score 0 rather than
    raising, which is the conservative choice for a report.
    """
    labels = list(labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(mp), 4),
        "macro_recall": round(float(mr), 4),
        "macro_f1": round(float(mf), 4),
        "per_class": {
            label: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                    "f1": round(float(f[i]), 4), "support": int(s[i])}
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {"labels": labels, "rows_are": "true label", "columns_are": "predicted label",
                             "matrix": cm.tolist()},
    }


def expected_calibration_error(predictions: Sequence[Prediction], y_true: Sequence[str], bins: int = 15) -> dict:
    """Top-label ECE: |accuracy - mean confidence| averaged over equal-width confidence bins.

    Only meaningful when scores are probabilities; 0 = perfectly calibrated.
    """
    confidences = np.array([max(p.scores.values()) for p in predictions])
    correct = np.array([p.label == t for p, t in zip(predictions, y_true)], dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return {"ece": round(float(ece), 4), "bins": bins, "mean_confidence": round(float(confidences.mean()), 4),
            "accuracy": round(float(correct.mean()), 4)}


def evaluate_model(model: SentimentModel, texts: Sequence[str], y_true: Sequence[str],
                   batch_size: int = 512) -> tuple[dict[str, Any], list[Prediction]]:
    predictions: list[Prediction] = []
    for start in range(0, len(texts), batch_size):
        predictions.extend(model.predict(texts[start:start + batch_size]))
    metrics = classification_metrics(y_true, [p.label for p in predictions], model.labels)
    if model.score_type in ("probability", "softmax_probability"):
        metrics["calibration"] = expected_calibration_error(predictions, y_true)
    return metrics, predictions


LATENCY_METHOD = (
    "Single-text latency: {n_single} texts drawn with a fixed seed from the evaluation split are predicted one "
    "at a time (batch size 1) after {warmup} warm-up calls; we report the median and 95th percentile wall-clock "
    "time per call (time.perf_counter), including vectorization/tokenization. Batch throughput: the same texts "
    "are predicted in batches of {batch_size}; we report texts per second. Measured in-process on the machine "
    "listed under 'environment'; no network or disk I/O is included."
)


def measure_latency(model: SentimentModel, texts: Sequence[str], n_single: int = 300, warmup: int = 20,
                    batch_size: int = 256, seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    sample = rng.sample(list(texts), min(n_single, len(texts)))
    for text in sample[:warmup]:
        model.predict([text])
    timings = []
    for text in sample:
        start = time.perf_counter()
        model.predict([text])
        timings.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    for i in range(0, len(sample), batch_size):
        model.predict(sample[i:i + batch_size])
    batch_seconds = time.perf_counter() - start
    timings.sort()
    return {
        "single_text_median_ms": round(statistics.median(timings), 3),
        "single_text_p95_ms": round(timings[min(len(timings) - 1, int(0.95 * len(timings)))], 3),
        "batch_throughput_texts_per_s": round(len(sample) / batch_seconds, 1) if batch_seconds else None,
        "n_texts": len(sample),
        "method": LATENCY_METHOD.format(n_single=len(sample), warmup=warmup, batch_size=batch_size),
    }
