"""App de Streamlit: ``uv run streamlit run app.py``.

Usa los archivos que generan los notebooks 03 y 04 (``artifacts/`` y ``reports/figures/``).
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
ART, FIG = ROOT / "artifacts", ROOT / "reports" / "figures"

st.set_page_config(page_title="Reingreso en diabetes", page_icon="🏥", layout="wide")
st.title("🏥 Predicción de reingreso <30 días en pacientes diabéticos")
st.caption("UCI «Diabetes 130-US Hospitals» · Proyecto de Aprendizaje de Máquina (UIS)")

if not (ART / "model.joblib").exists():
    st.error("Faltan los archivos del modelo. Ejecuta primero los notebooks 03 y 04.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(ART / "model.joblib")


@st.cache_data
def load_tables():
    sample = pd.read_csv(ART / "test_sample.csv", index_col=0)
    winner = json.loads((ART / "winner.json").read_text(encoding="utf-8"))
    return sample, winner


pipe = load_model()
sample, winner = load_tables()
threshold = winner["umbral"]

tab_models, tab_global, tab_patient = st.tabs(["Modelos", "Importancia global", "Paciente"])

with tab_models:
    st.subheader("Comparación con validación cruzada (solo train)")
    st.dataframe(pd.read_csv(ART / "cv_results.csv").round(4), hide_index=True)
    if (FIG / "comparacion_modelos_cv.png").exists():
        st.image(str(FIG / "comparacion_modelos_cv.png"))
    st.subheader(f"Resultado final en test — {winner['modelo']}")
    st.dataframe(pd.read_csv(ART / "metrics.csv").round(4), hide_index=True)
    if (FIG / "roc_pr_test.png").exists():
        st.image(str(FIG / "roc_pr_test.png"))

with tab_global:
    for name in ["importancia_global.png", "dependencia_parcial.png"]:
        if (FIG / name).exists():
            st.image(str(FIG / name))

with tab_patient:
    st.write(f"Umbral de decisión (elegido en train): **{threshold:.2f}**")
    pid = st.selectbox("Paciente del conjunto de test (muestra)", sample.index)
    row = sample.loc[[pid]].drop(columns="target")
    p = float(pipe.predict_proba(row)[0, 1])
    c1, c2, c3 = st.columns(3)
    c1.metric("Puntaje de riesgo", f"{p:.2f}")
    c2.metric("Decisión", "Alto riesgo" if p >= threshold else "Riesgo bajo")
    c3.metric("Reingresó realmente", "Sí" if sample.loc[pid, "target"] == 1 else "No")

    st.write("Datos del paciente")
    st.dataframe(row.T.astype(str))
