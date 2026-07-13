# OTA Multimodal Search and Travel Planning System

VLM-based OTA multimodal intelligent search and travel planning system.

This repository is not a generic chatbot demo. It is structured as an AI Search / Multimodal Search application for OTA scenarios:

```text
Image / Text / Reviews / Preferences
-> VLM Multimodal Understanding
-> Structured Information Extraction
-> Visual / Keyword / Hybrid Retrieval
-> Candidate POI / Product Ranking
-> Travel Planning
-> Evaluation & Experiment Tracking
```

## Project Overview

The project builds a minimal but extensible OTA pipeline for:

- image-to-structured-info extraction from travel-related images;
- visual search over restaurants, cafes, hotels, attractions, and products;
- multimodal travel planning from images, reviews, and user preferences;
- reproducible vLLM serving and experiment tracking.

## Motivation

OTA users often search with vague intent and visual references: a cafe photo, a hotel room screenshot, a restaurant dish, or a scenic street. The system turns those multimodal signals into searchable structured fields and planning inputs.

## System Architecture

```text
Client
  -> FastAPI business API
  -> vLLM OpenAI-compatible VLM service
  -> Structured extraction
  -> Retrieval baseline
  -> Travel planner
  -> Experiment records
```

## Repository Layout

```text
src/          reusable API, inference, retrieval, planning, data, and evaluation code
scripts/      repository-root command entry points; no business logic duplication
configs/      model, inference, and data-pipeline configuration
data/         checked-in samples plus ignored Yelp raw/interim/processed layers
docker/       API, vLLM, and one-off CLIP runtime definitions
docs/         durable requirements, decisions, weekly delivery, and technical references
reports/      generated mentor-facing report artifacts
experiments/  reproducible experiment logs, metrics, and failure cases
tests/        unittest behavior and data-pipeline contract coverage
```

`docs/weekly_delivery.md` is the single complete Week 1/Week 2 delivery record;
`docs/weekly_log.md` is only the concise timeline. Generated reports stay under
`reports/`, and agent plans or personal internship notes remain ignored.

## Features

- Dockerized API and vLLM serving layout.
- Qwen3-VL primary model config with Qwen2.5-VL fallback.
- DeepSeek-VL2 config for later comparison.
- `/health`, `/v1/image-understanding`, `/v1/visual-search`, `/v1/travel-planning`.
- Deterministic fallback responses when live vLLM is not configured.
- Sample POI catalog and review snippets.
- Experiment log and results CSV templates.

## Quick Start

Create a local API environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
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

## Dependency Sets

Install only the dependency group needed for the task:

```bash
pip install -r requirements-api.txt
pip install -r requirements-data.txt
pip install -r requirements-llm.txt
```

- `requirements-api.txt`: FastAPI service and smoke-test dependencies.
- `requirements-data.txt`: Week 2 Yelp data processing dependencies only.
- `requirements-llm.txt`: vLLM and Qwen-VL utilities for live model serving.
- `requirements-clip.txt`: table dependencies added to the dedicated CUDA CLIP container.
- `requirements.txt`: safe default aggregate for API + data dependencies. It intentionally does not install vLLM.

For the Week 2 data pipeline, use only:

```bash
pip install -r requirements-data.txt
```

Do not install `vllm` in a native Windows Python environment unless live model serving is explicitly needed. Prefer Docker or WSL2 for `requirements-llm.txt`, because vLLM/GPU/CUDA compatibility is much easier to control there.

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

### Week 1: Engineering Baseline

Week 1 established Docker/vLLM serving, FastAPI image understanding,
deterministic fallback behavior, experiment tracking, and a small Yelp sample.
The sample workflow remains available and is not replaced by Week 2:

```bash
python scripts/prepare_yelp_subset.py --raw-dir data/yelp/raw --output-dir data/yelp/processed/ota_subset_v1
```

See `docs/yelp_dataset.md` for the expected raw files and generated schemas.

### Week 2: Full Yelp Data Pipeline

Week 2 consumes the previously downloaded Yelp archives and adds a reusable
full-data processing pipeline configured by `configs/data_processing.yaml`.
Expected local layout:

```text
data/yelp/
├── raw/
├── interim/
├── processed/
├── logs/
└── validation/
reports/
└── figures/
```

Extract the official Yelp archives into the normalized raw directory when starting from zip files:

```bash
python scripts/extract_yelp_archives.py \
  --json-zip data/Yelp-JSON.zip \
  --photos-zip data/Yelp-Photos.zip \
  --raw-dir data/yelp/raw \
  --include-photo-files
```

This extracts the 5 core JSON files, `photos.json`, the `photos/` image directory, and official documentation/ToS files under `data/yelp/raw/docs/`.

Run the full offline data-processing flow:

```bash
pip install -r requirements-data.txt
python scripts/parse_yelp_json.py --config configs/data_processing.yaml
python scripts/build_yelp_alignment.py --config configs/data_processing.yaml
docker compose -f docker/docker-compose.yml stop vllm
docker compose -f docker/docker-compose.yml --profile data run --rm clip-denoising
python scripts/generate_yelp_report.py --config configs/data_processing.yaml
```

Outputs include interim business/review/photo tables, image validation summaries, strong image-caption pairs, medium image-business pairs, bounded weak business-level image-review groups, dataset statistics, optional denoising status, and `reports/yelp_multimodal_data_processing_report_part1.md`. Install `pyarrow` for true Parquet output; without a local Parquet engine, the scripts keep running with a CSV fallback at the configured table path.

The default config uses `processing_limits.max_reviews: null` for full review parsing and writes review rows in chunks to avoid holding the full review table in memory.

### CLIP Denoising Runtime

`clip-denoising` is a one-off GPU Docker task, separate from the API and vLLM service. It mounts the project root at `/workspace` and `models/` at `/models`, so it reads `data/yelp/processed/business_level_weak_pairs.parquet`, caches `openai/clip-vit-base-patch32`, and writes `weak_pairs_denoised.parquet` plus `clip_denoising_summary.json` back to the host.

Stop `vllm` before running CLIP on the local 8GB GPU. The CLIP task needs GPU memory for model inference; it must not share the GPU with the running Qwen service. Restart `vllm` afterwards with `docker compose -f docker/docker-compose.yml start vllm`.

Week 2 mentor-facing report:

- `reports/yelp_multimodal_data_processing_report_part1.md`
- `docs/weekly_delivery.md` contains the completed Week 1 and Week 2 checklists and measured results.

## Evaluation

Initial metrics:

- JSON parse success rate;
- structured field accuracy;
- Top-K hit rate for retrieval;
- Recall@K;
- planning relevance and route reasonability by human review.

## Weekly Progress

- Week 1: Docker/vLLM, API, live single-image inference, Yelp sample preparation, and experiment records completed.
- Week 2: Full Yelp parsing, image validation, multimodal alignment, CLIP denoising, output validation, and report completed.
