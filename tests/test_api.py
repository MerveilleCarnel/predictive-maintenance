"""
tests/test_api.py
-----------------
Tests unitaires pour src/api/app.py
Utilise httpx + TestClient de FastAPI (pas besoin de lancer le serveur)
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Cree un client de test FastAPI avec un modele et des donnees factices.
    Le modele est entraine sur des donnees aleatoires pour les tests.
    """
    from fastapi.testclient import TestClient
    from unittest.mock import patch, MagicMock
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Modele factice entraine
    np.random.seed(42)
    n = 100
    X_fake = pd.DataFrame(
        np.random.randn(n, 25),
        columns=[f"feat_{i}" for i in range(25)]
    )
    y_fake = pd.Series(np.random.randint(0, 2, n))

    fake_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    fake_pipeline.fit(X_fake, y_fake)

    # Patch des chemins fichiers pour ne pas dependre du filesystem
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "valve_model.pkl")
        joblib.dump(fake_pipeline, model_path)

        with patch("src.api.app.MODELS_PATH", model_path),              patch("src.api.app.PROCESSED_PATH", tmpdir):

            # Sauvegarder X factice
            X_fake.to_csv(os.path.join(tmpdir, "X_train.csv"), index=False)
            pd.DataFrame(np.random.randn(10, 25),
                         columns=[f"feat_{i}" for i in range(25)]).to_csv(
                os.path.join(tmpdir, "X_test.csv"), index=False
            )

            from src.api.app import app
            with TestClient(app) as c:
                yield c


# ── Tests /health ─────────────────────────────────────────────────────────────

class TestHealth:

    def test_status_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_has_model_field(self, client):
        data = client.get("/health").json()
        assert "model" in data
        assert isinstance(data["model"], str)

    def test_has_total_cycles(self, client):
        data = client.get("/health").json()
        assert "total_cycles" in data
        assert data["total_cycles"] > 0

    def test_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


# ── Tests POST /predict ───────────────────────────────────────────────────────

class TestPredictPost:

    def test_status_200(self, client):
        response = client.post("/predict", json={"cycle_index": 0})
        assert response.status_code == 200

    def test_response_fields(self, client):
        data = client.post("/predict", json={"cycle_index": 0}).json()
        for field in ["cycle_index", "valve_optimal", "label", "probability", "confidence"]:
            assert field in data

    def test_cycle_index_returned(self, client):
        data = client.post("/predict", json={"cycle_index": 5}).json()
        assert data["cycle_index"] == 5

    def test_valve_optimal_binary(self, client):
        data = client.post("/predict", json={"cycle_index": 0}).json()
        assert data["valve_optimal"] in [0, 1]

    def test_label_coherent(self, client):
        data = client.post("/predict", json={"cycle_index": 0}).json()
        if data["valve_optimal"] == 1:
            assert data["label"] == "Optimal"
        else:
            assert data["label"] == "Non optimal"

    def test_probability_between_0_and_1(self, client):
        data = client.post("/predict", json={"cycle_index": 0}).json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_confidence_valid(self, client):
        data = client.post("/predict", json={"cycle_index": 0}).json()
        assert data["confidence"] in ["Haute", "Moyenne", "Faible"]

    def test_invalid_cycle_index(self, client):
        response = client.post("/predict", json={"cycle_index": 99999})
        assert response.status_code == 422

    def test_negative_cycle_index(self, client):
        response = client.post("/predict", json={"cycle_index": -1})
        assert response.status_code == 422

    def test_missing_cycle_index(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422


# ── Tests GET /predict/{cycle_index} ─────────────────────────────────────────

class TestPredictGet:

    def test_status_200(self, client):
        response = client.get("/predict/0")
        assert response.status_code == 200

    def test_same_result_as_post(self, client):
        get_data  = client.get("/predict/3").json()
        post_data = client.post("/predict", json={"cycle_index": 3}).json()
        assert get_data["valve_optimal"] == post_data["valve_optimal"]
        assert get_data["probability"]   == post_data["probability"]

    def test_invalid_index(self, client):
        response = client.get("/predict/99999")
        assert response.status_code == 422


# ── Tests /cycles/info ────────────────────────────────────────────────────────

class TestCyclesInfo:

    def test_status_200(self, client):
        response = client.get("/cycles/info")
        assert response.status_code == 200

    def test_has_required_fields(self, client):
        data = client.get("/cycles/info").json()
        for field in ["total_cycles", "train_cycles", "test_cycles", "n_features", "features"]:
            assert field in data

    def test_features_is_list(self, client):
        data = client.get("/cycles/info").json()
        assert isinstance(data["features"], list)
        assert len(data["features"]) > 0
