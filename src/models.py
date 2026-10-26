"""Los 3 modelos y la validación cruzada."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .features import build_preprocessor

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def get_models() -> dict[str, Pipeline]:
    """Modelos a comparar. Todos con ``class_weight`` para el desbalance (~9 % positivos)."""
    return {
        "LogReg": Pipeline(
            [
                ("prep", build_preprocessor(scale=True)),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1),
                ),
            ]
        ),
        "Árbol": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=6,
                        min_samples_leaf=100,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=30,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Umbral que maximiza F1 (se calcula con predicciones fuera de muestra del train)."""
    grid = np.linspace(0.05, 0.95, 91)
    f1 = [_f1(y_true, proba >= t) for t in grid]
    return float(grid[int(np.argmax(f1))])


def _f1(y_true, pred) -> float:
    y_true = np.asarray(y_true)
    tp = np.sum((pred == 1) & (y_true == 1))
    denom = 2 * tp + np.sum((pred == 1) & (y_true == 0)) + np.sum((pred == 0) & (y_true == 1))
    return 2 * tp / denom if denom else 0.0


def cv_evaluate(model: Pipeline, X, y, cv=CV) -> tuple[pd.DataFrame, np.ndarray]:
    """CV estratificada SOLO sobre train. Devuelve métricas por fold y probabilidades OOF."""
    oof = np.zeros(len(y))
    rows = []
    for k, (tr, va) in enumerate(cv.split(X, y)):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        rows.append(
            {
                "fold": k,
                "roc_auc": roc_auc_score(y.iloc[va], p),
                "pr_auc": average_precision_score(y.iloc[va], p),
            }
        )
    return pd.DataFrame(rows), oof
