# Imagen reproducible del proyecto: notebooks, tests y app Streamlit.
FROM python:3.11-slim

# libgomp1: requerido por LightGBM / scikit-learn (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    MPLBACKEND=Agg

RUN pip install --no-cache-dir uv

WORKDIR /app

# 1) Dependencias (capa cacheada: solo se reconstruye si cambia pyproject/uv.lock)
COPY pyproject.toml README.md .python-version ./
COPY uv.lock* ./
RUN uv sync --no-install-project

# 2) Código del proyecto (instala `src` en modo editable)
COPY src ./src
COPY tests ./tests
COPY notebooks ./notebooks
COPY app.py ./
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN uv sync && chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p data/raw artifacts reports/figures

EXPOSE 8501 8888
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["app"]
