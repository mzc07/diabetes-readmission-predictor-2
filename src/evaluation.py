"""Métricas, gráficas e importancia de variables."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, proba, threshold: float) -> dict:
    """ROC-AUC y PR-AUC (independientes del umbral) + Recall/Precision/F1 en ``threshold``."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "recall": recall_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred),
        "threshold": threshold,
    }


def plot_roc_pr(y_true, proba, path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax[0], plot_chance_level=True)
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax[1], plot_chance_level=True)
    ax[0].set_title("Curva ROC (test)")
    ax[1].set_title("Curva Precision-Recall (test)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_confusion(y_true, proba, threshold, path=None):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        (np.asarray(proba) >= threshold).astype(int),
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title(f"Matriz de confusión (umbral={threshold:.2f})")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def global_importance(pipe, X, y, n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """Importancia por permutación (caída de ROC-AUC) sobre las columnas ORIGINALES."""
    r = permutation_importance(
        pipe, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": r.importances_mean,
                "std": r.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
