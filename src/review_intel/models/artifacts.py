"""Saving and loading trained model artifacts, and the "selected model" pointer.

Layout::

    artifacts/models/<model_name>/
        model.joblib        # sklearn: fitted Pipeline(vectorizer + classifier)
        (transformer: config.json, model.safetensors, tokenizer files)
        metadata.json       # name, version, labels, score type, params, data manifest, versions, sha256
    artifacts/models/selected.json   # which model the inference service loads by default

Security note: ``model.joblib`` is a pickle. Only load artifacts you trained
yourself or fully trust; the sha256 check detects corruption, not malice.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib

from review_intel.data.labels import LABELS
from review_intel.models.base import SCORE_TYPES, ModelNotAvailableError, Prediction, SentimentModel
from review_intel.models.sklearn_models import predict_with_scores, score_type_for
from review_intel.utils import ARTIFACTS_DIR, package_versions, read_json, relative_to_root, sha256_file, utc_now, write_json

logger = logging.getLogger(__name__)

MODELS_DIR = ARTIFACTS_DIR / "models"
SELECTED_POINTER = MODELS_DIR / "selected.json"

TRAIN_HINT = (
    "Train one with: python scripts/train_baselines.py --data data/processed/amazon_electronics "
    "then python scripts/evaluate_models.py compare --data data/processed/amazon_electronics. "
    "Or explicitly choose the demo baseline with REVIEW_ANALYZER_SENTIMENT_BACKEND=keyword."
)


class SklearnSentimentModel:
    """A fitted scikit-learn pipeline behind the common ``SentimentModel`` interface."""

    model_type = "sklearn"
    is_trained_model = True

    def __init__(self, pipeline, metadata: dict[str, Any]):
        self.pipeline = pipeline
        self.metadata = metadata
        self.name = metadata["name"]
        self.version = metadata["version"]
        self.labels = tuple(metadata.get("labels", LABELS))
        self.score_type = score_type_for(pipeline)

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        return predict_with_scores(self.pipeline, texts)

    def info(self) -> dict[str, Any]:
        return model_info(self.metadata, self.score_type)


def model_info(metadata: dict[str, Any], score_type: str) -> dict[str, Any]:
    keys = ("name", "version", "model_type", "algorithm", "labels", "dataset", "params", "created_utc",
            "validation_metrics", "training_time_seconds")
    info = {k: metadata[k] for k in keys if k in metadata}
    info["is_trained_model"] = True
    info["score_type"] = score_type
    info["score_description"] = SCORE_TYPES[score_type]
    return info


def make_version(name: str, dataset_name: str, file_sha256: str) -> str:
    return f"{name}@{dataset_name}-{utc_now()[:10].replace('-', '')}-{file_sha256[:8]}"


def save_sklearn_model(directory: Path, pipeline, metadata: dict[str, Any]) -> dict[str, Any]:
    """Persist pipeline + metadata. Returns the completed metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    digest = sha256_file(model_path)
    metadata = {
        **metadata,
        "model_type": "sklearn",
        "labels": list(LABELS),
        "score_type": score_type_for(pipeline),
        "model_file": "model.joblib",
        "model_sha256": digest,
        "created_utc": utc_now(),
        "package_versions": package_versions(),
    }
    metadata["version"] = make_version(metadata["name"], metadata.get("dataset", {}).get("name", "data"), digest)
    write_json(directory / "metadata.json", metadata)
    return metadata


def load_model(directory: Path) -> SentimentModel:
    """Load any trained artifact (sklearn or transformer) from its directory."""
    meta_path = directory / "metadata.json"
    if not meta_path.exists():
        raise ModelNotAvailableError(f"No trained model found at {relative_to_root(directory)}. {TRAIN_HINT}")
    metadata = read_json(meta_path)
    model_type = metadata.get("model_type")
    if model_type == "sklearn":
        model_path = directory / metadata.get("model_file", "model.joblib")
        if not model_path.exists():
            raise ModelNotAvailableError(f"Model file missing: {relative_to_root(model_path)}. {TRAIN_HINT}")
        if metadata.get("model_sha256") and sha256_file(model_path) != metadata["model_sha256"]:
            raise ModelNotAvailableError(f"Checksum mismatch for {relative_to_root(model_path)}; retrain the model.")
        saved = metadata.get("package_versions", {}).get("scikit-learn")
        current = package_versions().get("scikit-learn")
        if saved and current and saved != current:
            logger.warning("Model was saved with scikit-learn %s but %s is installed", saved, current)
        return SklearnSentimentModel(joblib.load(model_path), metadata)
    if model_type == "transformer":
        try:
            from review_intel.models.transformer import TransformerSentimentModel
        except ImportError as exc:
            raise ModelNotAvailableError(
                f"Model {metadata.get('name')} needs the optional transformer dependencies ({exc}). "
                "Install with: pip install -e .[transformer]"
            ) from None
        return TransformerSentimentModel.load(directory)
    raise ModelNotAvailableError(f"Unknown model_type {model_type!r} in {relative_to_root(meta_path)}")


def write_selected_pointer(model_dir: Path, selection: dict[str, Any], pointer: Path = SELECTED_POINTER) -> None:
    # Stored relative to the pointer file, so the repo can be moved or cloned.
    relative = Path(os.path.relpath(model_dir.resolve(), pointer.parent.resolve())).as_posix()
    write_json(pointer, {"model_dir": relative, **selection, "selected_utc": utc_now()})


def resolve_selected_model_dir(pointer: Path = SELECTED_POINTER) -> Path:
    if not pointer.exists():
        raise ModelNotAvailableError(f"No model has been selected yet ({relative_to_root(pointer)} missing). {TRAIN_HINT}")
    path = Path(read_json(pointer)["model_dir"])
    return path if path.is_absolute() else (pointer.parent / path).resolve()
