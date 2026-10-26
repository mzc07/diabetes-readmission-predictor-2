# Diabetes Readmission Predictor

![Imagen ML](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRuCI42BNejNN0PX3pC1k7_eX1-c-RQ_LCCPxMRDb-cisdGsVyBU_fyuAY&s=10)

Predicción de reingreso hospitalario **<30 días** en pacientes diabéticos (UCI "Diabetes 130-US Hospitals", licencia CC BY 4.0) e interpretación del modelo.

## Autores
- [Martín Ariza (2251516)](https://github.com/mzc07)
- [Ivanna Álvarez (2251805)](https://github.com/Ivannaalvarezs)
- [María Rueda (2251801)](https://github.com/julianarp8-droid)

| Índice | Notebook | Descripción |
| --- | --- | --- |
| [1. Datos](data/README.md) | [Análisis exploratorio](notebooks/01_eda.ipynb)| Marco historico del dataset y una introducción al problema. |
| [2. Preprocesamiento]() | []() |  |
| [3. Modelado]() | [Modelado de datos](notebooks/03_modeling.ipynb) | |


## Resultado en una mirada

| Modelo (CV 5 folds, solo train) | ROC-AUC | PR-AUC |
|---|---|---|
| Regresión logística | 0.648 ± 0.007 | 0.165 |
| Árbol de decisión | 0.622 ± 0.006 | 0.149 |
| Random Forest | 0.651 ± 0.005 | 0.165 |
| GBM por defecto | 0.647 ± 0.004 | 0.169 |
| **GBM ajustado (ganador)** | **0.653 ± 0.007** | **0.171** |

**Test (usado una sola vez):** ROC-AUC 0.653 (IC 95 % 0.638-0.670), PR-AUC 0.184 (base 0.090),
recall 0.40, precision 0.18, F1 0.25 con umbral 0.56. El 10 % de pacientes con mayor puntaje
reingresa el 22.9 % de las veces (base 9.0 %, lift 2.5x).
Variables más influyentes: destino al alta (`discharge_disposition_id`) y hospitalizaciones
previas (`number_inpatient`). Detalle y salvedades en [`docs/INFORME.md`](docs/INFORME.md).

> ⚠️ Los resultados incluidos se generaron con `HistGradientBoostingClassifier` (respaldo de
> scikit-learn) y sin `shap`, porque en el entorno de ejecución no se podían instalar LightGBM/SHAP.
> Con `uv sync` el código usa LightGBM y SHAP automáticamente; ver la sección 8 del informe.

## Qué hace
- Limpieza sin fuga de datos (primer encuentro por paciente, `Pipeline` de sklearn)
- Comparación de 4 modelos: Logística, Árbol, Random Forest, GBM/LightGBM
- Interpretabilidad (SHAP si está disponible; permutación + dependencia parcial + oclusión si no)
- App en Streamlit para explorar resultados

## Inicio
Requiere [uv](https://docs.astral.sh/uv/) y los CSV en `data/raw/` (`diabetic_data.csv`, `IDS_mapping.csv`).
```
    git clone <url> && cd diabetes-readmission-predictor
    uv sync
    uv run jupyter lab notebooks/
    uv run streamlit run app.py
```

## Docker (ejecutar en cualquier equipo)
Solo requiere Docker y los CSV en `data/raw/`.

    docker compose run --rm train      # 1ª vez: entrena y genera artifacts/ y reports/figures/
    docker compose up app              # app en http://localhost:8501


## Calidad
    uv run ruff check . && uv run ruff format --check .
    uv run pytest

## Estructura
    data/raw/          CSV originales (en .gitignore)
    notebooks/         01_eda · 02_preprocessing · 03_modeling · 04_evaluation_shap
    src/               data.py · features.py · models.py · evaluation.py
    tests/             test_core.py
    artifacts/         model.joblib, cv_results.csv, metrics.csv, winner.json, muestras para la app
    reports/figures/   gráficos generados por los notebooks
    docs/              
    app.py             Streamlit (3 pestañas)