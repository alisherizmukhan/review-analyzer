"""Shared fixtures. Everything is built from the SYNTHETIC fixture: offline, no GPU, no downloads."""

from __future__ import annotations

import pytest

from review_intel.testing import prepare_fixture, train_fixture_models


@pytest.fixture(scope="session")
def fixture_data(tmp_path_factory):
    """Processed synthetic dataset (train/val/test CSV + manifest)."""
    return prepare_fixture(tmp_path_factory.mktemp("processed"))


@pytest.fixture(scope="session")
def fixture_corpus(tmp_path_factory):
    """The synthetic fixture as a single corpus.csv (split strategy 'none')."""
    return prepare_fixture(tmp_path_factory.mktemp("corpus"), split_strategy="none") / "corpus.csv"


@pytest.fixture(scope="session")
def trained(fixture_data, tmp_path_factory):
    """Both baseline families trained on the fixture. Returns (models_dir, reports_dir, results)."""
    models_dir = tmp_path_factory.mktemp("models")
    reports_dir = tmp_path_factory.mktemp("reports")
    results = train_fixture_models(fixture_data, models_dir, reports_dir)
    return models_dir, reports_dir, results
