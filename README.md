# Diabetes Readmission Predictor

![Imagen ML](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRuCI42BNejNN0PX3pC1k7_eX1-c-RQ_LCCPxMRDb-cisdGsVyBU_fyuAY&s=10)

Predicción de reingreso hospitalario **en menos de 30 días** en pacientes diabéticos, usando el dataset de UCI "Diabetes 130-US Hospitals" (licencia CC BY 4.0).

## Autores
- [Martín Ariza (2251516)](https://github.com/mzc07)
- [Ivanna Álvarez (2251805)](https://github.com/Ivannaalvarezs)
- [María Rueda (2251801)](https://github.com/julianarp8-droid)

## Índice

| Índice | Notebook | Descripción |
| --- | --- | --- |
| [1. Datos](data/README.md) | [Análisis exploratorio](notebooks/01_eda.ipynb) | Origen del dataset, valores faltantes, variable objetivo y primeras relaciones con el reingreso. |
| 2. Preprocesamiento | [Preprocesamiento](notebooks/02_preprocessing.ipynb) | Limpieza, agrupación de diagnósticos, split 80/20 y `Pipeline` de imputación + one-hot. |
| 3. Modelado | [Modelado](notebooks/03_modeling.ipynb) | Comparación de Regresión logística, Árbol y Random Forest con validación cruzada. |
| 4. Evaluación | [Evaluación](notebooks/04_evaluation.ipynb) | Resultado en test, importancia de variables y dependencia parcial. |
| 5. App | [app.py](app.py) | App en Streamlit para ver los resultados y probar pacientes. |

## Resultados

| Modelo (CV 5 folds sobre train) | ROC-AUC | PR-AUC |
|---|---|---|
| Regresión logística | 0.648 | 0.165 |
| Árbol de decisión | 0.622 | 0.149 |
| **Random Forest (ganador)** | **0.651** | **0.165** |

En test el Random Forest obtiene ROC-AUC 0.649 y PR-AUC 0.172 (tasa base 0.09), con recall 0.47 y
precisión 0.15 usando el umbral 0.51. Las variables más influyentes son el destino al alta
(`discharge_disposition_id`) y las hospitalizaciones previas (`number_inpatient`).

## Cómo ejecutarlo
Requiere [uv](https://docs.astral.sh/uv/) y los CSV en `data/raw/` (`diabetic_data.csv`, `IDS_mapping.csv`).

    git clone <url> && cd diabetes-readmission-predictor
    uv sync
    uv run jupyter lab notebooks/      # ejecutar los notebooks en orden (01 → 04)
    uv run streamlit run app.py

## Docker
Primero hay que ejecutar los notebooks 03 y 04 para que exista la carpeta `artifacts/`.

    docker build -t diabetes .
    docker run -p 8501:8501 diabetes   # app en http://localhost:8501

## Estructura
    data/raw/          CSV originales
    notebooks/         01_eda · 02_preprocessing · 03_modeling · 04_evaluation
    src/               data.py · features.py · models.py · evaluation.py
    artifacts/         modelo entrenado y resultados (los generan los notebooks 03 y 04)
    reports/figures/   gráficas de los notebooks
    app.py             app de Streamlit
