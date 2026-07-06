FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt /app/requirements-api.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements-api.txt

COPY . /app

EXPOSE 8000 8001

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
