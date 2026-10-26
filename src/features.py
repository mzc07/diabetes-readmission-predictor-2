"""Ingeniería de variables y preprocesador sin fuga de datos.

``engineer`` es determinista (no aprende nada de los datos), por lo que puede
aplicarse antes del split. Todo lo que *aprende* (imputación, categorías, escalado)
vive en ``build_preprocessor`` y se ajusta únicamente con train dentro de un ``Pipeline``.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SPECIALTY = "medical_specialty"
ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
DIAG_COLS = ["diag_1", "diag_2", "diag_3"]
N_TOP_SPECIALTIES = 10  # + "Unknown" + bucket de infrecuentes ("Otra")


def icd9_to_group(code) -> str:
    """Agrupa un código ICD-9 en las categorías clínicas de Strack et al. (2014).

    Los códigos con prefijo ``V`` o ``E`` van explícitamente a "Other"; NaN -> "Missing".
    """
    if code is None or (isinstance(code, float) and np.isnan(code)) or pd.isna(code):
        return "Missing"
    code = str(code).strip()
    if code[:1].upper() in ("V", "E"):
        return "Other"
    try:
        n = float(code)
    except ValueError:
        return "Other"
    if 250 <= n < 251:
        return "Diabetes"
    if 390 <= n <= 459 or int(n) == 785:
        return "Circulatory"
    if 460 <= n <= 519 or int(n) == 786:
        return "Respiratory"
    if 520 <= n <= 579 or int(n) == 787:
        return "Digestive"
    if 580 <= n <= 629 or int(n) == 788:
        return "Genitourinary"
    if 800 <= n <= 999:
        return "Injury"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if 140 <= n <= 239:
        return "Neoplasms"
    return "Other"


def _age_midpoint(age: str) -> float:
    """'[40-50)' -> 45.0"""
    lo, hi = age.strip("[)").split("-")
    return (int(lo) + int(hi)) / 2


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Transformaciones deterministas: diag -> grupo clínico, edad -> numérica, ids -> texto."""
    X = X.copy()
    for col in DIAG_COLS:
        X[f"{col}_group"] = X[col].map(icd9_to_group)
    X = X.drop(columns=DIAG_COLS)
    X["age"] = X["age"].map(_age_midpoint)
    for col in ID_COLS:  # son categorías, no números
        X[col] = X[col].astype(str)
    return X


def _numeric_cols(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include="number").columns.tolist()


def _categorical_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.select_dtypes(exclude="number").columns if c != SPECIALTY]


def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    """Imputación + encoding. ``scale=True`` solo para modelos lineales."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    specialty = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=N_TOP_SPECIALTIES + 2,  # top-10 + Unknown + "Otra"
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), _numeric_cols),
            ("cat", cat, _categorical_cols),
            ("spec", specialty, [SPECIALTY]),
        ],
        verbose_feature_names_out=False,
    )
