"""App Streamlit (1 archivo, 3 pestañas): ``uv run streamlit run app.py``.

Requiere haber ejecutado los notebooks 03 y 04 (generan ``artifacts/`` y ``reports/figures/``).
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.evaluation import local_occlusion, shap_available

ROOT = Path(__file__).parent
ART, FIG = ROOT / "artifacts", ROOT / "reports" / "figures"

st.set_page_config(page_title="Reingreso en diabetes", page_icon="🏥", layout="wide")
st.title("🏥 Predicción de reingreso <30 días en pacientes diabéticos")
st.caption("UCI «Diabetes 130-US Hospitals» · Proyecto de Aprendizaje de Máquina (UIS)")

if not (ART / "model.joblib").exists():
    st.error("Faltan los artefactos. Ejecuta primero los notebooks 03 y 04.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(ART / "model.joblib")


@st.cache_data
def load_tables():
    sample = pd.read_csv(ART / "test_sample.csv", index_col=0)
    background = pd.read_csv(ART / "background.csv", index_col=0)
    winner = json.loads((ART / "winner.json").read_text(encoding="utf-8"))
    return sample, background, winner


pipe = load_model()
sample, background, winner = load_tables()
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
    for name in ["shap_summary.png", "importancia_global.png", "dependencia_parcial.png"]:
        if (FIG / name).exists():
            st.image(str(FIG / name))

with tab_patient:
    st.write(f"Umbral de decisión (elegido en train): **{threshold:.2f}**")
    pid = st.selectbox("Paciente del conjunto de test (muestra)", sample.index)
    row = sample.loc[[pid]].drop(columns="target")
    p = float(pipe.predict_proba(row)[0, 1])
    c1, c2, c3 = st.columns(3)
    c1.metric("Puntaje de riesgo (no calibrado)", f"{p:.2f}")
    c2.metric("Decisión", "Alto riesgo" if p >= threshold else "Riesgo bajo")
    c3.metric("Reingresó realmente", "Sí" if sample.loc[pid, "target"] == 1 else "No")

    fig, ax = plt.subplots(figsize=(7, 4))
    use_shap = (
        shap_available() and winner.get("gbm_backend") == "lightgbm" and "GBM" in winner["modelo"]
    )
    if use_shap:  # waterfall SHAP de UNA fila (rápido con TreeExplainer)
        import shap

        from src.evaluation import shap_explanation

        shap.plots.waterfall(shap_explanation(pipe, row, max_rows=1)[0], max_display=10, show=False)
        fig = plt.gcf()
    else:  # respaldo: contribución por oclusión
        contrib = local_occlusion(pipe, row, background).head(10).iloc[::-1]
        ax.barh(
            [f"{k} = {row[k].iloc[0]}" for k in contrib.index],
            contrib.values,
            color=["#c44e52" if v > 0 else "#4c72b0" for v in contrib.values],
        )
        ax.axvline(0, c="k", lw=0.8)
        ax.set_xlabel("Contribución al log-odds (rojo: sube el riesgo)")
    st.pyplot(fig)
    with st.expander("Datos del paciente"):
        st.dataframe(row.T)
