"""
src/data/load_data.py
---------------------
Chargement des fichiers bruts du dataset hydraulique.
"""

import os
import pandas as pd


def load_raw_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les 3 fichiers bruts : PS2, FS1, profile.

    Args:
        data_path : chemin vers le dossier data/raw/

    Returns:
        ps2     : DataFrame (n_cycles x 6000) — pression 100 Hz
        fs1     : DataFrame (n_cycles x 600)  — débit 10 Hz
        profile : DataFrame (n_cycles x 5)    — variables cibles
    """
    ps2_path     = os.path.join(data_path, "PS2.txt")
    fs1_path     = os.path.join(data_path, "FS1.txt")
    profile_path = os.path.join(data_path, "profile.txt")

    # Vérification existence des fichiers
    for path in [ps2_path, fs1_path, profile_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    ps2 = pd.read_csv(ps2_path, sep="\t", header=None)
    fs1 = pd.read_csv(fs1_path, sep="\t", header=None)
    profile = pd.read_csv(
        profile_path, sep="\t", header=None,
        names=["cooler", "valve", "pump", "accumulator", "stable"]
    )

    return ps2, fs1, profile


def make_binary_target(profile: pd.DataFrame) -> pd.Series:
    """
    Crée la cible binaire à partir de la colonne 'valve'.

    Returns:
        pd.Series : 1 si valve == 100 (optimal), 0 sinon
    """
    return (profile["valve"] == 100).astype(int).rename("valve_optimal")
