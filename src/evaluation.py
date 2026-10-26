"""Métricas, curvas e interpretabilidad (HU-07, HU-08)."""

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
    PrecisionRecallDisplay.from_predictions(
        y_true, proba, ax=ax[1], plot_chance_level=True
    )
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


# ---------------------------------------------------------------- interpretabilidad
def shap_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False


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


def local_occlusion(pipe, row: pd.DataFrame, background: pd.DataFrame) -> pd.Series:
    """Contribución de cada variable a la predicción de UN paciente (log-odds).

    Para cada columna, se sustituye su valor por los de una muestra de referencia y se mide
    cuánto cambia el log-odds en promedio. Es la aproximación de primer orden de un valor de
    Shapley (sin interacciones); se usa cuando ``shap`` no está disponible.
    """

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    base = logit(pipe.predict_proba(row)[:, 1])[0]
    bg_mean = logit(pipe.predict_proba(background)[:, 1]).mean()
    out = {}
    for col in row.columns:
        perturbed = background.copy()
        perturbed[col] = row[col].iloc[0]
        out[col] = logit(pipe.predict_proba(perturbed)[:, 1]).mean() - bg_mean
    s = pd.Series(out, name="contribucion")
    s.attrs["base"] = base
    return s.sort_values(key=np.abs, ascending=False)


def shap_explanation(pipe, X, max_rows: int = 2000, seed: int = 42):
    """Explicación SHAP (TreeExplainer) sobre una muestra de ``X``; requiere ``shap``.

    El explainer se aplica al último paso (modelo) con datos ya transformados:
    ``pipe[:-1].transform(X)`` y ``pipe[-1]``.
    """
    import shap

    sample = X.sample(min(max_rows, len(X)), random_state=seed)
    Xt = pd.DataFrame(
        pipe[:-1].transform(sample),
        columns=pipe[:-1].get_feature_names_out(),
        index=sample.index,
    )
    explainer = shap.TreeExplainer(pipe[-1])
    return explainer(Xt)
