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

### Week 3: Zero-Shot Evaluation Framework

Week 3 is `READY / COMPLETED`. The immutable v1 manifests and runs remain historical
evidence. After the mentor requested removal of low-quality images for a fair
baseline/comparison set, the active work moved to a separately versioned v2
dataset; v1 files and runs are never overwritten.

| Scenario | target_count | candidate_count | annotated_count | validated_count | tested_count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product understanding | 200 | 200 | 200 | 200 | 200 |
| After-sales issue recognition | 150 | 150 | 150 | 150 | 150 |
| Itinerary constraint understanding | 100 | 100 | 100 | 100 | 100 |

These counts are bound to the completed v2 baseline run
`week3_v2_baseline_full_20260724_001`. The 70 low-evidence after-sales rows
were replaced with visually reviewed evidence and annotated by the existing
human annotator. The final v2 set retains both public Yelp and
business-synthetic sources and the 38/38/37/37 candidate strata.

One human annotator is sufficient. Model outputs and deterministic suggestions
must not replace the human labels. Unknown values are allowed when evidence is
insufficient and must be reported as a data limitation rather than guessed.

Product labels are reused unchanged, including evidence-supported `unknown`.
The approved v2 labeling scope covered 70 replacement after-sales rows and
100 itinerary style supplements whose field was not exposed reliably in the
historical tool; both are complete. All original v1 itinerary text, hard/soft
constraints, and required elements were inherited server-side rather than
re-entered.

The core local commands are:

```bash
python scripts/prepare_week3_evaluation.py init
python scripts/build_week3_candidate_manifests.py --config configs/evaluation_week3.yaml
python scripts/validate_week3_evaluation.py
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3_v2.yaml
python scripts/run_week3_evaluation.py --config configs/evaluation_week3_v2.yaml --run-id <run-id> --mode live --run-scope full --prompt-version baseline_minimal_v1
python scripts/score_week3_evaluation.py --config configs/evaluation_week3_v2.yaml --run-id <run-id>
python scripts/score_week3_evaluation.py --config configs/evaluation_week3_v2.yaml --run-id week3_v2_baseline_full_20260724_001 --semantic-coding-config configs/evaluation/baseline_semantic_coding_v1.json --score-id week3_v2_baseline_full_20260724_001__baseline_semantic_coding_v1
```

Prepare the separately versioned candidates with
`python scripts/prepare_week3_v2_dataset.py --config configs/evaluation_week3_v2.yaml`.
The command refuses to overwrite an existing v2 dataset. Standardized v2 uses
JSON-object response mode plus the full versioned Schema contract and an
explicit type skeleton; the bounded itinerary v2 Schema leaves v1 unchanged.

See `reports/week3_zero_shot_baseline_report.md` for the status report and
`docs/evaluation_data_contracts.md`, `docs/prompt_architecture.md`,
`docs/evaluation_framework.md`, `docs/evaluation_metrics.md`, and
`docs/week3_annotation_guidelines.md` for the
technical contracts.

## Evaluation

The Week 3 evaluation framework defines scenario-specific structured metrics.
The completed v2 baseline measured format compliance and latency over 450
records. Its natural-language business metrics are now measured independently
by the fixed, gold-independent `baseline_semantic_coding_v1` lexical track.
The same-set standardized v2 run retains its strict structured-business
metrics. The report keeps the two scoring tracks separate and preserves the
baseline's measured 0% JSON and Schema compliance.

Framework metric groups include:

- product label accuracy, completeness, and format compliance;
- after-sales issue/severity accuracy, key-information F1, and OCR recall;
- itinerary constraint accuracy, element completeness, and format compliance;
- inference latency and representative error cases.

## Weekly Progress

- Week 1: Docker/vLLM, API, live single-image inference, Yelp sample preparation, and experiment records completed.
- Week 2: Full Yelp parsing, image validation, multimodal alignment, CLIP denoising, output validation, and report completed.
- Week 3: `READY / COMPLETED`. Data, human gold, real baseline, deterministic baseline semantic scoring, standardized v2, and reporting are complete and traceable.

### Week 4：Prompt 优化与 Milvus

Week 4 保持全部 Week 3 产物不变。Few-Shot 示例来自独立的
`week4_demo_dev_v1` development 人工金标池；固定 pilot 和全量测试仍使用
`week3_evaluation_v2`。两个池按样本、来源、图片哈希和来源组隔离。格式
兜底仅移除可选 Markdown 围栏、解析 JSON 并执行现有场景 Schema 校验，
不修复模型内容。

当前有效 Few-Shot 版本为 `fewshot_4_v2` 和 `fewshot_7_v2`。旧 v1
行程请求因超过 4096-token 上下文而返回 HTTP 400，仅保留为失败证据；
runner 和统一验证器会拒绝包含模型请求错误的运行。独立池重跑后，商品由
`fewshot_4_v2` 胜出，售后和行程由 `standardized_v2` 胜出；450 条混合
winner 全量结果单独版本化。baseline 与 winner 的业务指标通过共同语义
轨道成对比较，不覆盖 Week 3 原评分。

```bash
python scripts/build_week3_candidate_manifests.py --config configs/evaluation_week4_demo_dev_v1.yaml
python scripts/manage_week3_annotations.py --config configs/evaluation_week4_demo_dev_v1.yaml --scenario <scenario> export --include-suggestions --output <packet.jsonl>
python scripts/manage_week3_annotations.py --config configs/evaluation_week4_demo_dev_v1.yaml --scenario <scenario> apply --input <completed-packet.jsonl>
python scripts/run_week4_prompt_evaluation.py --config configs/evaluation_week4.yaml --run-id <run-id> --stage pilot --variant <standardized_v2|fewshot_4_v2|fewshot_7_v2>
python scripts/analyze_week4_prompts.py --config configs/evaluation_week4.yaml --pilot-run-id <run-id> --pilot-run-id <run-id> --pilot-run-id <run-id>
python scripts/compare_week4_common_semantics.py --winner-run-id week4_winners_full_20260726_002 --output-dir outputs/week4/common_semantic/week4_common_semantic_coding_v1_20260726_003
python scripts/validate_week4_output.py --scenario image_product_search --raw-output-file <raw-output-file>
python scripts/validate_week4_delivery.py --config configs/evaluation_week4.yaml
```

Milvus 客户端使用独立依赖组，不向 API、data 或 vLLM 环境添加依赖。
standalone Compose 固定 Milvus、etcd 和 MinIO 版本，并包含健康检查、
持久化、本地端口和资源限制。凭据只从本机环境文件读取；仓库只提交脱敏示例。

```bash
python -m pip install -r requirements-milvus.txt
Copy-Item docker/milvus/.env.example docker/milvus/.env
# 编辑 docker/milvus/.env，替换两个 MinIO 占位值
docker compose --env-file docker/milvus/.env -f docker/milvus/docker-compose.yml config
docker compose --env-file docker/milvus/.env -f docker/milvus/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml stop vllm
python scripts/build_week4_clip_vectors.py --config configs/milvus_week4.yaml
python scripts/benchmark_week4_milvus.py --config configs/milvus_week4.yaml
```

本地 8 GB GPU 不得同时运行 vLLM 和 CLIP。生成的 Week 4 运行、向量、
性能输出和 Milvus volumes 均保持忽略。详细说明见
`reports/week4_prompt_optimization_report.md`,
`reports/week4_bad_cases.md`,
`docs/milvus_collection_design.md`, and
`reports/week4_milvus_deployment_performance_report.md`.

### Week 5：数据集标注与质检

Week 5 使用 `configs/week5_dataset.json` 从本地 Yelp OTA 数据构建三场景候选池，
并同时调用 Week 3 v1/v2 exclusion manifest。候选、合成凭证、预标注、人工包、
质检记录和对话输出均位于忽略目录 `outputs/week5/`。模型预标注永远不计为人工完成。

```bash
python scripts/manage_week5_dataset.py build-pools
python scripts/manage_week5_dataset.py validate-pools
python scripts/manage_week5_dataset.py preannotate --scenario <image_product_search|after_sales|itinerary_planning>
python scripts/manage_week5_dataset.py preannotate-all
python scripts/manage_week5_dataset.py export-annotations --scenario <scenario> --output <packet.jsonl>
python scripts/manage_week5_dataset.py apply-human --scenario <scenario> --input <completed.jsonl>
python scripts/manage_week5_dataset.py apply-quality --scenario <scenario> --input <quality.jsonl>
python scripts/manage_week5_dataset.py generate-dialogues
python scripts/manage_week5_dataset.py apply-dialogue-quality --input <dialogue-quality.jsonl>
python scripts/manage_week5_dataset.py report
```

真实候选池为商品 50,000、售后 20,000、行程 10,000，隔离验证通过；当前人工
修正、三级质检和合格对话均为 0。字段口径见
`docs/week5_annotation_guidelines.md`，实测数量见
`reports/week5_dataset_quality_report.md`。

## Aliyun Runtime

The cloud runtime uses Alibaba Cloud Model Studio `qwen3.7-plus` through the
Singapore workspace-specific OpenAI-compatible endpoint. The ECS deployment
does not run local vLLM. FastAPI and Milvus bind to loopback by default; use an
SSH tunnel for operator access. See `docs/aliyun_deployment.md` for the exact
deployment and secret-handling workflow.

The versioned Qwen3.7 evaluation configs are
`configs/evaluation_week3_qwen37_plus_aliyun.yaml` and
`configs/evaluation_week4_qwen37_plus_aliyun.yaml`. The measured rerun is
summarized in `reports/qwen37_previous_weeks_rerun_report.md`; generated runs
and scores remain local and ignored.

The Qwen3.7 itinerary repair uses the versioned `standardized_v4` prompt and a
scenario-only evaluation config. With `MODEL_API_BASE_URL` and
`MODEL_API_KEY_FILE` set locally, reproduce the live run and score it with:

```bash
python scripts/run_week3_evaluation.py --config configs/evaluation_itinerary_qwen37_repair.yaml --run-id <new-run-id> --mode live --run-scope full --prompt-version standardized_v4 --scenario itinerary_planning
python scripts/score_week3_evaluation.py --config configs/evaluation_itinerary_qwen37_repair.yaml --run-id <new-run-id>
```

Measured repair results are in `reports/qwen37_itinerary_repair_report.md`.

## Reports

The consolidated Week 1-4 report after migration to Alibaba Cloud
`qwen3.7-plus` is `reports/week1_to_week4_qwen37_overall_report.md`.
Use `reports/README.md` as the report index; historical process snapshots are
kept separate from current conclusions.
