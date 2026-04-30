"""
src/models/train.py
-------------------
Entraînement et sauvegarde du modèle de prédiction de valve condition.
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ── Catalogue des modèles candidats ─────────────────────────────────────────

def get_candidates() -> dict:
    """Retourne le dictionnaire des pipelines candidats."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
        "Extra Trees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_jobs=-1)),
        ]),
    }


# ── Grilles d'hyperparamètres ────────────────────────────────────────────────

PARAM_GRIDS = {
    "Random Forest": {
        "clf__n_estimators"     : [100, 200, 300],
        "clf__max_depth"        : [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__max_features"     : ["sqrt", "log2"],
    },
    "Extra Trees": {
        "clf__n_estimators"     : [100, 200, 300],
        "clf__max_depth"        : [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__max_features"     : ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "clf__n_estimators" : [100, 200],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth"    : [3, 5],
        "clf__subsample"    : [0.8, 1.0],
    },
    "Logistic Regression": {
        "clf__C"      : [0.01, 0.1, 1, 10],
        "clf__penalty": ["l1", "l2"],
        "clf__solver" : ["liblinear"],
    },
    "SVM": {
        "clf__C"     : [0.1, 1, 10],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma" : ["scale", "auto"],
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 11],
        "clf__weights"    : ["uniform", "distance"],
        "clf__metric"     : ["euclidean", "manhattan"],
    },
    "Decision Tree": {
        "clf__max_depth"        : [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__criterion"        : ["gini", "entropy"],
    },
}


# ── Fonctions principales ────────────────────────────────────────────────────

def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5
) -> pd.DataFrame:
    """
    Compare plusieurs modèles par cross-validation stratifiée.

    Returns:
        DataFrame trié par F1 Score décroissant
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    candidates = get_candidates()
    results = {}

    print(f"{'Modèle':<25} {'Accuracy':>10} {'F1':>10} {'ROC AUC':>10}")
    print("─" * 60)

    for name, pipeline in candidates.items():
        cv_res = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=["accuracy", "f1", "roc_auc"],
            n_jobs=-1
        )
        results[name] = {
            "accuracy": cv_res["test_accuracy"].mean(),
            "f1"      : cv_res["test_f1"].mean(),
            "roc_auc" : cv_res["test_roc_auc"].mean(),
            "acc_std" : cv_res["test_accuracy"].std(),
            "f1_std"  : cv_res["test_f1"].std(),
        }
        r = results[name]
        print(f"{name:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")

    df = pd.DataFrame(results).T.sort_values("f1", ascending=False)
    return df


def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_model_name: str,
    n_splits: int = 5
) -> GridSearchCV:
    """
    Optimise les hyperparamètres du meilleur modèle via GridSearchCV.

    Returns:
        GridSearchCV fitté
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    candidates  = get_candidates()
    pipeline    = candidates[best_model_name]
    param_grid  = PARAM_GRIDS[best_model_name]

    grid_search = GridSearchCV(
        estimator  = pipeline,
        param_grid = param_grid,
        cv         = cv,
        scoring    = "f1",
        n_jobs     = -1,
        verbose    = 1
    )
    grid_search.fit(X_train, y_train)

    print(f"\nMeilleurs hyperparamètres ({best_model_name}) :")
    for param, val in grid_search.best_params_.items():
        print(f"  {param} : {val}")
    print(f"Meilleur F1 (CV) : {grid_search.best_score_:.4f}")

    return grid_search


def save_model(pipeline, models_path: str, filename: str = "valve_model.pkl") -> str:
    """
    Sauvegarde le pipeline entraîné avec joblib.

    Returns:
        Chemin complet du fichier sauvegardé
    """
    os.makedirs(models_path, exist_ok=True)
    model_path = os.path.join(models_path, filename)
    joblib.dump(pipeline, model_path)
    size = os.path.getsize(model_path) / 1024
    print(f"Modèle sauvegardé : {model_path}  ({size:.1f} KB)")
    return model_path


def load_model(model_path: str):
    """Charge un modèle depuis un fichier .pkl."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    return joblib.load(model_path)
