"""Los 4 modelos y la validación cruzada (HU-05, HU-06, HU-07)."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .features import build_preprocessor

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

try:  # LightGBM es el modelo previsto en el plan
    from lightgbm import LGBMClassifier

    GBM_BACKEND = "lightgbm"
except ImportError:  # respaldo equivalente de scikit-learn (boosting por histogramas)
    LGBMClassifier = None
    GBM_BACKEND = "sklearn-hist-gradient-boosting"


def make_gbm(**params):
    """LightGBM si está instalado; si no, HistGradientBoosting (misma familia de algoritmo)."""
    if GBM_BACKEND == "lightgbm":
        base = {
            "class_weight": "balanced",
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        if "subsample" in params:
            base["subsample_freq"] = 1
        return LGBMClassifier(**{**base, **params})
    return HistGradientBoostingClassifier(
        class_weight="balanced", random_state=SEED, **params
    )


def gbm_param_space() -> dict:
    """Espacio de búsqueda (prefijo ``model__`` porque el modelo es un paso del Pipeline)."""
    if GBM_BACKEND == "lightgbm":
        space = {
            "n_estimators": [200, 400, 600],
            "learning_rate": [0.02, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "min_child_samples": [20, 50, 100],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.5, 0.7, 1.0],
            "reg_lambda": [0, 1, 10],
        }
    else:
        space = {
            "max_iter": [100, 200, 300, 500],
            "learning_rate": [0.02, 0.05, 0.1],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [20, 50, 100],
            "l2_regularization": [0, 1, 10],
            "max_features": [0.5, 0.7, 1.0],
        }
    return {f"model__{k}": v for k, v in space.items()}


def get_models() -> dict[str, Pipeline]:
    """Escalera de modelos. Todos con ``class_weight`` para el desbalance (~9 % positivos)."""
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
        "GBM": Pipeline([("prep", build_preprocessor()), ("model", make_gbm())]),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Umbral que maximiza F1 (se calcula con predicciones fuera de muestra del train)."""
    grid = np.linspace(0.05, 0.95, 91)
    f1 = [_f1(y_true, proba >= t) for t in grid]
    return float(grid[int(np.argmax(f1))])


def _f1(y_true, pred) -> float:
    y_true = np.asarray(y_true)
    tp = np.sum((pred == 1) & (y_true == 1))
    denom = (
        2 * tp
        + np.sum((pred == 1) & (y_true == 0))
        + np.sum((pred == 0) & (y_true == 1))
    )
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


def tune_gbm(X, y, n_iter: int = 15, cv_splits: int = 3) -> RandomizedSearchCV:
    """RandomizedSearchCV (<= 20 iteraciones) sobre train."""
    pipe = Pipeline([("prep", build_preprocessor()), ("model", make_gbm())])
    search = RandomizedSearchCV(
        pipe,
        gbm_param_space(),
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(cv_splits, shuffle=True, random_state=SEED),
        random_state=SEED,
        n_jobs=1,
        refit=False,
        verbose=1,
    )
    return search.fit(X, y)
