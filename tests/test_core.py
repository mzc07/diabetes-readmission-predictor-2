"""Los 3 tests clave del plan."""

import numpy as np
import pandas as pd

from src.data import EXCLUDED_DISCHARGE_IDS, build_xy, filter_encounters
from src.features import icd9_to_group


def _toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": [5, 1, 2, 3, 4, 6],
            "patient_nbr": [10, 10, 20, 30, 40, 50],
            "discharge_disposition_id": [1, 1, 11, 1, 13, 1],
            "readmitted": ["<30", ">30", "NO", "<30", "NO", ">30"],
            "weight": np.nan,
            "payer_code": "MC",
            "examide": "No",
            "citoglipton": "No",
        }
    )


def test_no_repeated_patients_and_first_encounter_kept():
    out = filter_encounters(_toy())
    assert out["patient_nbr"].is_unique
    # el paciente 10 tiene encuentros 5 y 1: se conserva el primero (id menor)
    assert out.loc[out["patient_nbr"] == 10, "encounter_id"].item() == 1


def test_icd9_mapping_known_examples():
    assert icd9_to_group("250.83") == "Diabetes"
    assert icd9_to_group("428") == "Circulatory"
    assert icd9_to_group("785") == "Circulatory"
    assert icd9_to_group("486") == "Respiratory"
    assert icd9_to_group("715") == "Musculoskeletal"
    assert icd9_to_group("V27") == "Other"
    assert icd9_to_group("E909") == "Other"
    assert icd9_to_group(np.nan) == "Missing"


def test_filter_removes_expired_hospice_and_target_is_binary():
    out = filter_encounters(_toy())
    assert not out["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS).any()
    X, y = build_xy(out)
    assert set(y.unique()) <= {0, 1}
    # sólo "<30" es positivo
    assert y.sum() == (out["readmitted"] == "<30").sum()
    assert "readmitted" not in X and "patient_nbr" not in X and "weight" not in X
