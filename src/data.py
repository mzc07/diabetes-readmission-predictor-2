"""Carga, filtros y construcción del target.

Se usa el primer encuentro por paciente, se quitan las altas por fallecimiento/hospicio y el
target es ``readmitted == "<30"``.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "diabetic_data.csv"
RANDOM_STATE = 42

# Altas por fallecimiento u hospicio: el paciente no puede reingresar.
EXCLUDED_DISCHARGE_IDS = (11, 13, 14, 19, 20, 21)

# Identificadores, columnas casi vacías y constantes.
DROP_COLS = [
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
    "examide",
    "citoglipton",
]

_STR_COLS = {"diag_1": str, "diag_2": str, "diag_3": str}


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Lee el CSV. Los faltantes vienen como ``"?"``.

    ``keep_default_na=False`` evita que pandas convierta el texto ``"None"`` de
    ``A1Cresult`` / ``max_glu_serum`` en NaN: ahí "None" significa "no medido" (información).
    """
    return pd.read_csv(
        path, na_values="?", keep_default_na=False, dtype=_STR_COLS, low_memory=False
    )


def filter_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """Primer encuentro por paciente y sin altas por fallecimiento/hospicio."""
    out = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    out = out[~out["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)]
    return out.reset_index(drop=True)


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa el target binario (reingreso < 30 días) y elimina columnas inútiles."""
    y = (df["readmitted"] == "<30").astype(int).rename("target")
    X = df.drop(columns=["readmitted", *DROP_COLS])
    return X, y


def load_dataset(path: Path | str = RAW_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Pipeline completo de carga: CSV -> filtros -> (X, y)."""
    return build_xy(filter_encounters(load_raw(path)))


def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Split estratificado 80/20 (se hace ANTES de imputar/codificar)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
