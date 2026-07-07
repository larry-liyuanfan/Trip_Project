# OTA Multimodal Search Foundation

VLM-based OTA image understanding and data preparation foundation.

This branch contains the reviewed current-week engineering scope:

```text
Image / Text
-> VLM Multimodal Understanding
-> Structured Information Extraction
-> Evaluation & Experiment Tracking
```

## Project Overview

The current-week scope covers:

- image-to-structured-info extraction from travel-related images;
- reproducible Docker and vLLM serving;
- FastAPI smoke endpoints for health and image understanding;
- Yelp Open Dataset preparation into project-local JSONL files;
- experiment logging templates and smoke-test records.

## Motivation

OTA users often search with vague intent and visual references: a cafe photo, a hotel room screenshot, a restaurant dish, or a scenic street. The current branch focuses on turning those images into structured fields that can be reviewed and tested.

## System Architecture

```text
Client
  -> FastAPI business API
  -> vLLM OpenAI-compatible VLM service
  -> Structured extraction
  -> Experiment records
```

## Features

- Dockerized API and vLLM serving layout.
- Qwen3-VL primary model config with Qwen2.5-VL fallback.
- DeepSeek-VL2 config file for model selection review.
- `/health`, `/v1/image-understanding`.
- Deterministic fallback responses when live vLLM is not configured.
- Sample POI catalog and review snippets.
- Experiment log and results CSV templates.

## Quick Start

Create a local environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Test health:

```bash
python scripts/test_health.py
```

Test image understanding:

```bash
python scripts/test_image_understanding.py
```

## Docker Setup

Build and run both API and vLLM containers:

```bash
cd docker
docker compose up --build
```

The API image uses `requirements-api.txt` and stays lightweight. The compose file expects NVIDIA GPU container support for the separate vLLM service. The default compose runtime pins `vllm/vllm-openai:v0.8.5` with `Qwen/Qwen2-VL-2B-Instruct` because this is compatible with local CUDA 12.x drivers and 8GB VRAM smoke tests; use Qwen2.5-VL or Qwen3-VL only after confirming the vLLM image, NVIDIA driver, and GPU memory support them.

## vLLM Serving

Run vLLM directly:

```bash
MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct \
SERVED_MODEL_NAME=Qwen3-VL-2B-Instruct \
PORT=8001 \
bash scripts/run_vllm_server.sh
```

If Qwen3-VL is unavailable in the local environment, use the Qwen2.5-VL or smaller Qwen2-VL fallback config and record the change in `experiments/experiment_log.md`. On the current local Docker path, the compose default uses the smaller Qwen2-VL 2B model for service validation.

## API Usage

Health:

```bash
curl http://localhost:8000/health
```

Image understanding:

```bash
curl -X POST http://localhost:8000/v1/image-understanding \
  -H "Content-Type: application/json" \
  -d '{"image_urls":["file://data/samples/images/cafe_001.jpg"],"user_text":"这张图可能适合什么旅行场景？","language":"zh"}'
```

## Experiment Tracking

Each experiment must record:

- date and Git commit;
- model name and size;
- inference backend and serving command;
- prompt version and generation parameters;
- dataset version and task type;
- metrics, summary, failure cases, and next action.

Use:

- `experiments/experiment_log.md` for human-readable notes;
- `experiments/results.csv` for tabular comparison;
- `experiments/failure_cases.md` for error analysis.

## Data

Week 1 uses a small mock sample catalog in `data/samples/`. Yelp Open Dataset can be integrated by placing the official raw JSON files under `data/yelp/raw/` and running:

```bash
python scripts/prepare_yelp_subset.py --raw-dir data/yelp/raw --output-dir data/yelp/processed/ota_subset_v1
```

See `docs/yelp_dataset.md` for the expected raw files and generated schemas.

## Evaluation

Initial metrics:

- JSON parse success rate;
- structured field accuracy;
- API response availability;
- vLLM service availability.

## Weekly Progress

- Week 1: Docker/vLLM serving, image-understanding API, sample data, Yelp data preparation, and experiment records.
