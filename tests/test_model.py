"""
tests/test_model.py
-------------------
Tests unitaires pour src/models/train.py et src/models/evaluate.py
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.train    import get_candidates, save_model, load_model
from src.models.evaluate import compute_metrics


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_data():
    """Dataset minimal pour tester les modeles."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(np.random.randn(n, 25),
                     columns=[f"feat_{i}" for i in range(25)])
    y = pd.Series(np.random.randint(0, 2, n), name="valve_optimal")
    return X, y


@pytest.fixture
def trained_pipeline(dummy_data):
    """Pipeline RandomForest entraine sur donnees factices."""
    X, y = dummy_data
    candidates = get_candidates()
    pipeline = candidates["Random Forest"]
    pipeline.fit(X, y)
    return pipeline


# ── Tests get_candidates ─────────────────────────────────────────────────────

class TestGetCandidates:

    def test_returns_dict(self):
        candidates = get_candidates()
        assert isinstance(candidates, dict)

    def test_expected_models(self):
        candidates = get_candidates()
        expected = [
            "Logistic Regression", "Decision Tree", "Random Forest",
            "Extra Trees", "Gradient Boosting", "SVM", "KNN"
        ]
        for name in expected:
            assert name in candidates

    def test_all_have_scaler(self):
        candidates = get_candidates()
        for name, pipeline in candidates.items():
            assert "scaler" in pipeline.named_steps, f"{name} n'a pas de scaler"

    def test_all_have_clf(self):
        candidates = get_candidates()
        for name, pipeline in candidates.items():
            assert "clf" in pipeline.named_steps, f"{name} n'a pas de clf"


# ── Tests save_model / load_model ─────────────────────────────────────────────

class TestSaveLoadModel:

    def test_save_creates_file(self, trained_pipeline):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(trained_pipeline, tmpdir, "test_model.pkl")
            assert os.path.exists(os.path.join(tmpdir, "test_model.pkl"))

    def test_load_returns_pipeline(self, trained_pipeline):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_model(trained_pipeline, tmpdir, "test_model.pkl")
            loaded = load_model(path)
            assert hasattr(loaded, "predict")
            assert hasattr(loaded, "predict_proba")

    def test_loaded_model_predicts(self, trained_pipeline, dummy_data):
        X, y = dummy_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_model(trained_pipeline, tmpdir, "test_model.pkl")
            loaded = load_model(path)
            preds = loaded.predict(X)
            assert len(preds) == len(X)
            assert set(preds).issubset({0, 1})

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("models/inexistant.pkl")

    def test_predictions_consistent_after_reload(self, trained_pipeline, dummy_data):
        X, _ = dummy_data
        preds_before = trained_pipeline.predict(X)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_model(trained_pipeline, tmpdir)
            loaded = load_model(path)
            preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)


# ── Tests compute_metrics ─────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_perfect_predictions(self):
        y_true  = pd.Series([0, 1, 0, 1, 1])
        y_pred  = pd.Series([0, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.1, 0.9, 0.9])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert metrics["accuracy"]  == 1.0
        assert metrics["f1"]        == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"]    == 1.0
        assert metrics["roc_auc"]   == 1.0

    def test_worst_predictions(self):
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = pd.Series([1, 1, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_without_proba(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([0, 1, 1, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert "roc_auc" not in metrics
        assert "accuracy" in metrics

    def test_metrics_between_0_and_1(self):
        np.random.seed(42)
        y_true  = pd.Series(np.random.randint(0, 2, 50))
        y_pred  = pd.Series(np.random.randint(0, 2, 50))
        y_proba = np.random.rand(50)
        metrics = compute_metrics(y_true, y_pred, y_proba)
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key} hors de [0,1] : {val}"

    def test_keys_present(self):
        y_true  = pd.Series([0, 1, 0, 1])
        y_pred  = pd.Series([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        for key in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
            assert key in metrics
