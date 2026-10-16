"""Helpers that build small, fully offline fixtures for tests (synthetic data only)."""

from __future__ import annotations

from pathlib import Path

from review_intel.data.pipeline import DatasetConfig, prepare_dataset
from review_intel.utils import CONFIGS_DIR, read_json

FIXTURE_CONFIG = CONFIGS_DIR / "data" / "fixture.json"
FIXTURE_MODELS_CONFIG = CONFIGS_DIR / "models" / "baselines_fixture.json"


def prepare_fixture(out_dir: Path, split_strategy: str = "stratified") -> Path:
    """Prepare the synthetic fixture into ``out_dir`` (train/val/test or corpus.csv). Returns out_dir."""
    data = read_json(FIXTURE_CONFIG)
    data["split"] = {**data["split"], "strategy": split_strategy}
    prepare_dataset(DatasetConfig.from_dict(data), out_dir)
    return out_dir


def train_fixture_models(processed_dir: Path, models_dir: Path, reports_dir: Path) -> dict:
    from review_intel.models.training import train_baselines

    return train_baselines(processed_dir, read_json(FIXTURE_MODELS_CONFIG), models_dir, reports_dir,
                           measure_speed=False)
