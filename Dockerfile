# BMS Digital Twin — live API service.
# Build:  docker build -t bms-twin .
# Run:    docker run -p 8000:8000 bms-twin
# Then:   curl localhost:8000/health   ·   docs at localhost:8000/docs
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY bms ./bms

RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000
# --factory: create_app() builds the FastAPI app (default chemistry nmc).
CMD ["uvicorn", "--factory", "bms.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
