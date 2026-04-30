"""
src/models/evaluate.py
----------------------
Evaluation du modèle sur l'echantillon de test final.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Calcule toutes les metriques d'evaluation.

    Args:
        y_true  : vraies etiquettes
        y_pred  : predictions binaires
        y_proba : probabilites de la classe positive (optionnel)

    Returns:
        dict des metriques
    """
    metrics = {
        "accuracy"  : accuracy_score(y_true, y_pred),
        "f1"        : f1_score(y_true, y_pred),
        "precision" : precision_score(y_true, y_pred),
        "recall"    : recall_score(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["avg_precision"] = average_precision_score(y_true, y_proba)

    return metrics


def print_evaluation_report(y_true, y_pred, y_proba=None, dataset_name="Test") -> dict:
    """
    Affiche un rapport complet d'evaluation.

    Returns:
        dict des metriques
    """
    metrics = compute_metrics(y_true, y_pred, y_proba)

    print(f"{'='*50}")
    print(f"  Evaluation sur : {dataset_name}")
    print(f"{'='*50}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  F1 Score       : {metrics['f1']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    if y_proba is not None:
        print(f"  ROC AUC        : {metrics['roc_auc']:.4f}")
        print(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
    print()
    print("  Classification Report :")
    print(classification_report(
        y_true, y_pred,
        target_names=["Non optimal (0)", "Optimal (1)"],
        digits=4
    ))

    return metrics


def plot_confusion_matrix(y_true, y_pred, ax=None, title="Matrice de confusion"):
    """Trace la matrice de confusion normalisee et brute."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm, annot=False, fmt=".2f", cmap="RdYlGn",
        ax=ax, linewidths=1, linecolor="white",
        xticklabels=["Non optimal", "Optimal"],
        yticklabels=["Non optimal", "Optimal"],
        vmin=0, vmax=1
    )

    # Annotations manuelles : valeur brute + pourcentage
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5, i + 0.5,
                f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if cm_norm[i, j] < 0.5 else "black"
            )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Prediction", fontsize=11)
    ax.set_ylabel("Vraie classe", fontsize=11)

    tn, fp, fn, tp = cm.ravel()
    return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}


def plot_roc_curve(y_true, y_proba, ax=None, label="Modele"):
    """Trace la courbe ROC."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    ax.plot(fpr, tpr, color="#3b82f6", linewidth=2.5,
            label=f"{label} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#ef4444", linestyle="--",
            linewidth=1.5, label="Baseline aleatoire")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3b82f6")
    ax.set_title("Courbe ROC", fontsize=12, fontweight="bold")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])


def plot_precision_recall_curve(y_true, y_proba, ax=None, label="Modele"):
    """Trace la courbe Precision-Recall."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()

    ax.plot(recall, precision, color="#10b981", linewidth=2.5,
            label=f"{label} (AP = {ap:.4f})")
    ax.axhline(y=baseline, color="#ef4444", linestyle="--",
               linewidth=1.5, label=f"Baseline ({baseline:.2f})")
    ax.fill_between(recall, precision, alpha=0.1, color="#10b981")
    ax.set_title("Courbe Precision-Recall", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])


def plot_full_evaluation(y_true, y_pred, y_proba, model_name="Modele", save_path=None):
    """
    Genere un rapport visuel complet : 
    confusion matrix + ROC + PR curve + distribution des probabilites.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Rapport d'evaluation — {model_name}\nEchantillon de test final",
        fontsize=16, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ─ Matrice de confusion ─
    ax1 = fig.add_subplot(gs[0, 0])
    cm_stats = plot_confusion_matrix(y_true, y_pred, ax=ax1)

    # ─ Courbe ROC ─
    ax2 = fig.add_subplot(gs[0, 1])
    plot_roc_curve(y_true, y_proba, ax=ax2, label=model_name)

    # ─ Courbe Precision-Recall ─
    ax3 = fig.add_subplot(gs[0, 2])
    plot_precision_recall_curve(y_true, y_proba, ax=ax3, label=model_name)

    # ─ Distribution des probabilites par classe ─
    ax4 = fig.add_subplot(gs[1, :2])
    mask_opt  = y_true == 1
    mask_nopt = y_true == 0
    ax4.hist(y_proba[mask_opt],  bins=40, color="#10b981", alpha=0.6,
             label="Optimal (1)", density=True)
    ax4.hist(y_proba[mask_nopt], bins=40, color="#ef4444", alpha=0.6,
             label="Non optimal (0)", density=True)
    ax4.axvline(0.5, color="black", linestyle="--", linewidth=2, label="Seuil 0.5")
    ax4.set_title("Distribution des probabilites predites par classe",
                  fontsize=12, fontweight="bold")
    ax4.set_xlabel("Probabilite predite (classe 1 = optimal)")
    ax4.set_ylabel("Densite")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # ─ Tableau recapitulatif ─
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    metrics = compute_metrics(y_true, y_pred, y_proba)
    table_data = [
        ["Metrique", "Valeur"],
        ["Accuracy",      f"{metrics['accuracy']:.4f}"],
        ["F1 Score",       f"{metrics['f1']:.4f}"],
        ["Precision",      f"{metrics['precision']:.4f}"],
        ["Recall",         f"{metrics['recall']:.4f}"],
        ["ROC AUC",        f"{metrics['roc_auc']:.4f}"],
        ["Avg Precision",  f"{metrics['avg_precision']:.4f}"],
        ["TP",  str(cm_stats["TP"])],
        ["TN",  str(cm_stats["TN"])],
        ["FP",  str(cm_stats["FP"])],
        ["FN",  str(cm_stats["FN"])],
    ]
    table = ax5.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f9")
    ax5.set_title("Metriques", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Rapport sauvegarde : {save_path}")
    plt.show()
