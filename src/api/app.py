"""
src/api/app.py
--------------
API FastAPI — Predictive Maintenance
Expose un endpoint /predict pour predire la condition de la valve
a partir d'un numero de cycle donne.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Chemins ─────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parents[2]
MODELS_PATH    = os.path.join(ROOT_DIR, "models", "valve_model.pkl")
PROCESSED_PATH = os.path.join(ROOT_DIR, "data", "processed")

sys.path.append(str(ROOT_DIR))
from src.data.load_data  import load_raw_data, make_binary_target
from src.data.preprocess import build_feature_matrix


# ── Lifespan : chargement du modele au demarrage ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modele et les donnees une seule fois au demarrage."""
    print("[INFO] Chargement du modele...")
    app.state.model = joblib.load(MODELS_PATH)
    print(f"[INFO] Modele charge : {type(app.state.model.named_steps['clf']).__name__}")

    print("[INFO] Chargement des features pre-calculees...")
    app.state.X = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train.csv"))
    X_test       = pd.read_csv(os.path.join(PROCESSED_PATH, "X_test.csv"))
    
    app.state.X  = pd.concat([app.state.X, X_test], ignore_index=True)
    print(f"[INFO] {len(app.state.X)} cycles charges en memoire")

    yield
    print("[INFO] Arret de l'API")


# ── Application ──────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Predictive Maintenance API",
    description = "Predit si la condition de la valve est optimale (100%) ou non.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── Monitoring Prometheus ────────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app)

# ── Schemas Pydantic ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    cycle_index: int = Field(
        ...,
        ge=0,
        description="Index du cycle de production (0-base, max 2204)"
    )


class PredictResponse(BaseModel):
    cycle_index     : int
    valve_optimal   : int          # 0 ou 1
    label           : str          # "Optimal" ou "Non optimal"
    probability     : float        # probabilite classe positive
    confidence      : str          # "Haute" / "Moyenne" / "Faible"


class HealthResponse(BaseModel):
    status          : str
    model           : str
    total_cycles    : int
    version         : str


# ── Utilitaire confiance ─────────────────────────────────────────────────────
def get_confidence(proba: float) -> str:
    if proba >= 0.85 or proba <= 0.15:
        return "Haute"
    elif proba >= 0.70 or proba <= 0.30:
        return "Moyenne"
    else:
        return "Faible"


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return {"message": "Predictive Maintenance API — voir /docs pour la documentation"}


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Verifie que l'API et le modele sont operationnels."""
    return HealthResponse(
        status       = "ok",
        model        = type(app.state.model.named_steps["clf"]).__name__,
        total_cycles = len(app.state.X),
        version      = "1.0.0",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predit la condition de la valve pour un cycle donne.

    - **cycle_index** : numero du cycle (0 a 2204)

    Retourne :
    - **valve_optimal** : 1 = optimal (100%), 0 = non optimal
    - **label**         : "Optimal" ou "Non optimal"
    - **probability**   : probabilite que la valve soit optimale
    - **confidence**    : Haute / Moyenne / Faible
    """
    X = app.state.X

    # Validation de l'index
    if request.cycle_index >= len(X):
        raise HTTPException(
            status_code=422,
            detail=f"cycle_index {request.cycle_index} invalide. "
                   f"Valeur max : {len(X) - 1}"
        )

    # Extraction du cycle
    cycle_features = X.iloc[[request.cycle_index]]

    # Prediction
    prediction = int(app.state.model.predict(cycle_features)[0])
    proba      = float(app.state.model.predict_proba(cycle_features)[0][1])

    return PredictResponse(
        cycle_index   = request.cycle_index,
        valve_optimal = prediction,
        label         = "Optimal" if prediction == 1 else "Non optimal",
        probability   = round(proba, 4),
        confidence    = get_confidence(proba),
    )


@app.get("/predict/{cycle_index}", response_model=PredictResponse, tags=["Prediction"])
def predict_get(cycle_index: int):
    """
    Version GET de /predict — permet de tester directement depuis le navigateur.

    Exemple : GET /predict/42
    """
    return predict(PredictRequest(cycle_index=cycle_index))


@app.get("/cycles/info", tags=["Data"])
def cycles_info():
    """Retourne des informations sur les cycles disponibles."""
    return {
        "total_cycles" : len(app.state.X),
        "train_cycles" : 2000,
        "test_cycles"  : len(app.state.X) - 2000,
        "n_features"   : len(app.state.X.columns),
        "features"     : app.state.X.columns.tolist(),
    }
