#!/usr/bin/env bash
set -euo pipefail

run_nb() {  # ejecuta un notebook en su sitio (cwd = notebooks/, como en Jupyter)
  echo ">>> Ejecutando $1"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "notebooks/$1.ipynb"
}

case "${1:-app}" in
  app)
    if [ ! -f artifacts/model.joblib ]; then
      echo "No hay artifacts/model.joblib. Ejecuta primero:  docker compose run --rm train" >&2
      exit 1
    fi
    exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
    ;;
  train)  # modelado + evaluación (genera artifacts/ y reports/figures/)
    run_nb 03_modeling; run_nb 04_evaluation_shap ;;
  all)    # pipeline completo 01 -> 04
    for nb in 01_eda 02_preprocessing 03_modeling 04_evaluation_shap; do run_nb "$nb"; done ;;
  lab)
    exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' ;;
  test)
    exec pytest -q ;;
  *)
    exec "$@" ;;
esac
