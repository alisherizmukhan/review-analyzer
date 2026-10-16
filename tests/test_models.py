"""Classical models: training on the fixture, artifacts, score semantics, metrics, selection protocol."""

from __future__ import annotations

import json

import pytest

from review_intel.data.pipeline import load_split
from review_intel.evaluation.metrics import classification_metrics, expected_calibration_error, measure_latency
from review_intel.evaluation.selection import SelectionError, evaluate_on_test, rank_candidates, select_model
from review_intel.models.artifacts import load_model, resolve_selected_model_dir, save_sklearn_model
from review_intel.models.base import ModelNotAvailableError, Prediction, SentimentModel
from review_intel.models.keyword import KeywordSentimentModel
from review_intel.models.sklearn_models import build_pipeline


def test_training_on_fixture_produces_artifacts_and_reports(trained):
    models_dir, reports_dir, results = trained
    assert set(results) == {"tfidf_logreg", "tfidf_linearsvm"}
    for family, report in results.items():
        assert (models_dir / family / "model.joblib").exists()
        meta = json.loads((models_dir / family / "metadata.json").read_text(encoding="utf-8"))
        assert meta["dataset"]["is_synthetic"] is True
        assert meta["version"].startswith(f"{family}@synthetic_fixture-")
        assert "scikit-learn" in meta["package_versions"]
        assert report["validation"]["n"] == 13
        assert len(report["grid"]) >= 1
        assert (reports_dir / "validation" / f"{family}.json").exists()


def test_training_never_reads_test_split(fixture_data, tmp_path, monkeypatch):
    import review_intel.models.training as training

    seen = []
    original = training.load_split
    monkeypatch.setattr(training, "load_split", lambda d, split: seen.append(split) or original(d, split))
    from review_intel.testing import train_fixture_models
    train_fixture_models(fixture_data, tmp_path / "m", tmp_path / "r")
    assert sorted(set(seen)) == ["train", "val"]


def test_save_load_round_trip_gives_identical_predictions(trained, fixture_data):
    models_dir, _, _ = trained
    texts, _ = load_split(fixture_data, "test")
    for family in ("tfidf_logreg", "tfidf_linearsvm"):
        model = load_model(models_dir / family)
        again = load_model(models_dir / family)
        assert isinstance(model, SentimentModel)
        assert [p.label for p in model.predict(texts)] == [p.label for p in again.predict(texts)]


def test_logreg_returns_probabilities_svm_returns_decision_scores(trained):
    models_dir, _, _ = trained
    lr = load_model(models_dir / "tfidf_logreg")
    svm = load_model(models_dir / "tfidf_linearsvm")
    p = lr.predict(["The battery drains way too fast"])[0]
    assert lr.score_type == "probability"
    assert set(p.scores) == {"negative", "neutral", "positive"}
    assert sum(p.scores.values()) == pytest.approx(1.0, abs=1e-4)
    assert p.label == max(p.scores, key=p.scores.get)
    s = svm.predict(["The battery drains way too fast"])[0]
    assert svm.score_type == "decision_score"
    assert s.label == max(s.scores, key=s.scores.get)


def test_missing_artifact_gives_actionable_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="train_baselines.py"):
        load_model(tmp_path / "nothing")
    with pytest.raises(ModelNotAvailableError, match="No model has been selected"):
        resolve_selected_model_dir(tmp_path / "selected.json")


def test_corrupted_artifact_detected(trained, tmp_path):
    import shutil

    models_dir, _, _ = trained
    copy = tmp_path / "copy"
    shutil.copytree(models_dir / "tfidf_logreg", copy)
    (copy / "model.joblib").write_bytes((copy / "model.joblib").read_bytes()[:-10] + b"0123456789")
    with pytest.raises(ModelNotAvailableError, match="Checksum mismatch"):
        load_model(copy)


def test_save_sklearn_model_records_metadata(fixture_data, tmp_path):
    texts, labels = load_split(fixture_data, "train")
    pipe = build_pipeline("logreg", {"word": {"ngram_range": [1, 1]}}, {"C": 1.0}, seed=0).fit(texts, labels)
    meta = save_sklearn_model(tmp_path / "m", pipe, {"name": "x", "dataset": {"name": "d"}})
    assert meta["score_type"] == "probability" and len(meta["model_sha256"]) == 64
    assert load_model(tmp_path / "m").version == meta["version"]


# --- metrics ----------------------------------------------------------------

def test_classification_metrics_hand_checked():
    y_true = ["positive", "positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "negative", "positive"]
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.5
    assert m["per_class"]["positive"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    assert m["per_class"]["negative"]["precision"] == 0.5 and m["per_class"]["negative"]["recall"] == 1.0
    assert m["per_class"]["neutral"]["f1"] == 0.0
    assert m["macro_f1"] == round((0.5 + 2 / 3 + 0.0) / 3, 4)
    assert m["confusion_matrix"]["matrix"] == [[1, 0, 0], [0, 0, 1], [1, 0, 1]]


def test_expected_calibration_error():
    preds = [Prediction("positive", {"positive": 0.9, "negative": 0.1})] * 10
    truth = ["positive"] * 9 + ["negative"]
    assert expected_calibration_error(preds, truth)["ece"] == pytest.approx(0.0, abs=1e-6)
    overconfident = expected_calibration_error(preds, ["positive"] * 5 + ["negative"] * 5)
    assert overconfident["ece"] == pytest.approx(0.4, abs=1e-6)


def test_latency_measurement_reports_documented_method():
    result = measure_latency(KeywordSentimentModel(), ["good"] * 30, n_single=20, warmup=2)
    assert result["n_texts"] == 20 and result["single_text_median_ms"] >= 0
    assert "warm-up" in result["method"]


# --- selection protocol -----------------------------------------------------

def test_rank_by_validation_macro_f1_with_deterministic_tie_break():
    reports = [{"family": "tfidf_linearsvm", "validation": {"macro_f1": 0.8}},
               {"family": "tfidf_logreg", "validation": {"macro_f1": 0.8}},
               {"family": "distilbert", "validation": {"macro_f1": 0.79}}]
    assert [r["family"] for r in rank_candidates(reports)] == ["tfidf_logreg", "tfidf_linearsvm", "distilbert"]
    reports[2]["validation"]["macro_f1"] = 0.81
    assert rank_candidates(reports)[0]["family"] == "distilbert"


def test_test_evaluation_requires_prior_selection(trained, fixture_data, tmp_path):
    models_dir, reports_dir, _ = trained
    import shutil

    reports_copy = tmp_path / "reports"
    shutil.copytree(reports_dir, reports_copy)
    with pytest.raises(SelectionError, match="compare"):
        evaluate_on_test(fixture_data, reports_copy, models_dir)
    selection = select_model(reports_copy, models_dir, activate_pointer=tmp_path / "selected.json")
    assert resolve_selected_model_dir(tmp_path / "selected.json") == (models_dir / selection["selected_family"]).resolve()
    report = evaluate_on_test(fixture_data, reports_copy, models_dir, include_non_selected=True)
    roles = {f: r["role"] for f, r in report["results"].items()}
    assert roles[selection["selected_family"]] == "selected"
    assert sorted(roles.values()) == ["post-selection comparison only", "selected"]
    assert (reports_copy / "test_report.md").read_text(encoding="utf-8").startswith("# Held-out test results")
