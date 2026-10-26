FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . .
RUN uv sync

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]
