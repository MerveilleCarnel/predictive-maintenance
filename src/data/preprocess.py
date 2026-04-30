"""
src/data/preprocess.py
----------------------
Feature engineering : transformation des séries temporelles brutes
en features statistiques par cycle.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats


def extract_features(signal_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Extrait 12 features statistiques pour chaque cycle (ligne).

    Features extraites :
        mean, std, min, max, median, q25, q75, iqr,
        skew, kurtosis, rms, peak_to_peak

    Args:
        signal_df : DataFrame (n_cycles x n_timesteps)
        prefix    : préfixe pour nommer les colonnes ('ps2' ou 'fs1')

    Returns:
        DataFrame (n_cycles x 12)
    """
    arr = signal_df.values  # numpy array (n_cycles x n_timesteps)

    features = {
        f"{prefix}_mean"        : arr.mean(axis=1),
        f"{prefix}_std"         : arr.std(axis=1),
        f"{prefix}_min"         : arr.min(axis=1),
        f"{prefix}_max"         : arr.max(axis=1),
        f"{prefix}_median"      : np.median(arr, axis=1),
        f"{prefix}_q25"         : np.percentile(arr, 25, axis=1),
        f"{prefix}_q75"         : np.percentile(arr, 75, axis=1),
        f"{prefix}_iqr"         : np.percentile(arr, 75, axis=1) - np.percentile(arr, 25, axis=1),
        f"{prefix}_skew"        : stats.skew(arr, axis=1),
        f"{prefix}_kurtosis"    : stats.kurtosis(arr, axis=1),
        f"{prefix}_rms"         : np.sqrt((arr ** 2).mean(axis=1)),
        f"{prefix}_peak_to_peak": arr.max(axis=1) - arr.min(axis=1),
    }

    return pd.DataFrame(features)


def build_feature_matrix(
    ps2: pd.DataFrame,
    fs1: pd.DataFrame,
    profile: pd.DataFrame
) -> pd.DataFrame:
    """
    Construit le dataset final X en combinant :
    - 12 features PS2
    - 12 features FS1
    - 1 stable flag

    Returns:
        X : DataFrame (n_cycles x 25)
    """
    features_ps2 = extract_features(ps2, prefix="ps2")
    features_fs1 = extract_features(fs1, prefix="fs1")

    X = pd.concat([
        features_ps2.reset_index(drop=True),
        features_fs1.reset_index(drop=True),
        profile[["stable"]].reset_index(drop=True),
    ], axis=1)

    return X


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split temporel : les N premiers cycles = train, le reste = test.
    Pas de shuffle (contrainte projet).

    Args:
        X          : features (n_cycles x n_features)
        y          : cible binaire (n_cycles,)
        train_size : nombre de cycles pour l'entraînement (défaut: 2000)

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train = X.iloc[:train_size].reset_index(drop=True)
    X_test  = X.iloc[train_size:].reset_index(drop=True)
    y_train = y.iloc[:train_size].reset_index(drop=True)
    y_test  = y.iloc[train_size:].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_path: str
) -> None:
    """
    Sauvegarde les datasets préparés dans data/processed/.

    Args:
        output_path : chemin vers le dossier data/processed/
    """
    os.makedirs(output_path, exist_ok=True)

    X_train.to_csv(os.path.join(output_path, "X_train.csv"), index=False)
    X_test.to_csv( os.path.join(output_path, "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(output_path, "y_train.csv"), index=False)
    y_test.to_csv( os.path.join(output_path, "y_test.csv"),  index=False)

    print(f"Données sauvegardées dans : {output_path}")
    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        size = os.path.getsize(os.path.join(output_path, fname)) / 1024
        print(f"  {fname:20s} → {size:.1f} KB")
