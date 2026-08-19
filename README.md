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
当前采用单人预算内抽样验收：三场景各固定选择 100 条，并固定包含每场景 10 条盲
二次复核候选和 3 条核心抽检候选；另从 10,000 条自动对话候选中固定抽取 100 条
人工验收，总操作上限 439 次。未进入队列的有效预标注保持
`silver`。人工修正提交必须包含真实 `annotator`、`corrected_at`、
`review_session_id` 和 `self_review_confirmed=true`，保存动作同时记录自审。商品仅有
确定性选中的 0.2%/0.05% 样本需要盲二次复核/核心抽检，售后和行程为
0.5%/0.1%；同一人
可以执行后续阶段，但必须换用新的 `review_session_id`，不得声称独立审核。
人工修订、质检和报告命令使用
`--config configs/week5_dataset_qwen3_vl_4b_single_operator.json`。正在运行的全量
预标注继续使用其 manifest 已绑定的 `configs/week5_dataset_qwen3_vl_4b_gpu.json`，
不得为修改质检规则而改变该活动 run 的配置哈希。

本地标注台仅显示上述确定性队列：

```bash
python scripts/run_week5_annotation_station.py --host 127.0.0.1 --port 8095
```

```bash
python scripts/manage_week5_dataset.py build-pools
python scripts/manage_week5_dataset.py validate-pools
python scripts/manage_week5_dataset.py preannotate --scenario <image_product_search|after_sales|itinerary_planning>
python scripts/manage_week5_dataset.py preannotate-all --run-id <unique-run-id>
python scripts/manage_week5_dataset.py export-annotations --scenario <scenario> --output <packet.jsonl>
python scripts/manage_week5_dataset.py apply-human --scenario <scenario> --input <completed.jsonl>
python scripts/manage_week5_dataset.py export-quality --scenario <scenario> --stage cross_review --output <packet.jsonl>
python scripts/manage_week5_dataset.py export-quality --scenario <scenario> --stage core_audit --output <packet.jsonl>
python scripts/manage_week5_dataset.py apply-quality --scenario <scenario> --input <quality.jsonl>
python scripts/manage_week5_dataset.py generate-dialogues
python scripts/manage_week5_dataset.py generate-dialogues --resume
python scripts/manage_week5_dataset.py apply-dialogue-quality --input <dialogue-quality.jsonl>
python scripts/manage_week5_dataset.py report --dialogue-run-id <authoritative-run-id>
python scripts/export_week5_localized_annotations.py
```

`export_week5_localized_annotations.py` 生成仅供快速复核的中文展示镜像：稳定字段名和
已知枚举值转换为中文，并附带 canonical annotation SHA-256；自由文本及 canonical
人工标注保持原样。该镜像不得作为训练数据或覆盖正式标注。对话生成首次运行会写入
不可变 run identity；中断后只能使用相同 run ID、配置哈希和合格样本集合配合
`--resume` 续跑。

Windows 本地执行长时间 GPU 预标注时，使用守护脚本维持 SSH 隧道并在连接中断后按
原 manifest 自动续跑。主流程不会重复成功样本，也不会在每次重连时反复请求已知
失败；全池结束后只执行一次失败清理。日志和状态写入对应 run 的 `supervisor/` 目录。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/supervise_week5_preannotation.ps1 `
  -SshHost <current-ecs-public-ip>
```

当候选、引用图片和同一 run 的断点已通过哈希校验同步到 GPU ECS 后，可由服务器
本机 systemd 服务运行，并设置
`WEEK5_MODEL_BASE_URL_OVERRIDE=http://127.0.0.1:8001/v1`。该变量只覆盖部署端点，
不修改已冻结配置或 run identity。服务器模式启用后不得同时启动 Windows supervisor；
具体实例路径和服务名只记录在忽略的 `.agents/server.local.md`。

真实候选池为商品 50,000、售后 20,000、行程 10,000，隔离验证通过。预算内人工
验收已完成三场景各 100 条修订与内联自审、各 10 条盲二次复核及各 3 条核心抽检。
权威多轮对话 run `week5_dialogues_merged_10000_20260816_522b4af` 已合并并严格验证
10,000 个唯一候选；本人随后完成固定 100 条人工验收队列，100 条决定均为 `pass`。
因此只计 100 条人工 accepted；其余 9,900 条仍是未人工验收候选，不得改写为人工
accepted。字段口径见 `docs/week5_annotation_guidelines.md`，实测数量见
`reports/week5_dataset_quality_report.md`。

#### Spartan 迁移

欠费停止的 A10 不再承担活动计算。Spartan 迁移不续写历史 run，而是从一个只读
恢复快照生成 100 条 benchmark、确定性互斥分片和新的合并产物：

```bash
python scripts/manage_spartan_migration.py prepare \
  --source-run-dir outputs/week5_qwen3_vl_4b/runs/<source-run> \
  --output-dir outputs/week5_qwen3_vl_4b/spartan/<migration-id> \
  --migration-id <migration-id> --shard-count 4 --benchmark-count 100
bash scripts/spartan/inspect_gpu_queue.sh
bash scripts/spartan/submit_week5.sh benchmark <account> <h100|a100|l40s> <migration-dir>
python scripts/manage_spartan_migration.py status --migration-dir <migration-dir>
```

benchmark 通过且只选择一个分区后，才可提交 `shards`。Spartan project ID、quota、
scratch 路径和提交身份必须在提交前核验。当前获批登录身份为 `yzhang3504`；密码不得
落盘。远端必须新建 Trip_Project 专属目录，且只管理本项目 job ID，不得操作账户内其他
文件、作业或进程。迁移产物继续位于忽略的 `outputs/`。

Spartan 的 Trip 文件、缓存和虚拟环境统一放在 project GPFS 的版本目录，禁止写入
已满的 home。当前部署根目录为
`/data/gpfs/projects/punim2936/Trip_Project_yzhang3504/20260812a`，仓库位于
`project/repo`，Python 3.11 环境位于 `envs/trip-week5-week6-py311`。基础和训练依赖
由 CPU Slurm 作业安装；Week 5 vLLM 仍由固定 Apptainer 镜像提供：

```bash
sbatch --account=punim2936 \
  --export=ALL,TRIP_DEPLOY_ROOT=<deploy-root>,TRIP_PROJECT_ROOT=<repo-root>,TRIP_VENV=<venv-path> \
  scripts/spartan/setup_trip_venv.sbatch
```

### Week 6：Qwen3-VL-8B QLoRA 小样本链路

Week 6 当前活动配置为
`configs/week6/qwen3_vl_8b_qlora_final300_v4.json`，绑定最终三场景各 100 条
人工修订；旧 `qwen3_vl_8b_qlora.json` 与早期数据锁只保留为历史证据。Spartan
Week 6 不复用上述未固定的历史 venv；使用 CUDA 12.8 版本化环境
`envs/trip-week6-py311-cu128-v1`。该环境仅安装
`requirements-training-spartan-cu128.txt`，不安装 API/data 聚合依赖，并关闭 pip
下载缓存，避免共享 GPFS inode 被 Pandas/PyArrow 等训练无关文件占满：

固定训练链 `29312210`、`29312212`、`29312214`、`29312215`、`29312217`
均已 `COMPLETED 0:0`，并绑定提交 `3d6bc81df8c4afd496e1e78d41c6b4bfa07c7bf4`。
三场景 adapter-only 保存和磁盘回载均通过。行程的低 `eval_loss` 后续经业务结构审计
发现不能代表约束遵循；专项优化必须先在派生锁的同一固定 validation 子集上评估现有
adapter，候选只有在全通过样本增加且各核心检查不回退时才能晋级，禁止只凭 loss 继续加
epoch。

行程专项候选已通过固定 64 条同集非回退门禁，获胜 adapter SHA-256 为
`7ab168a0f7073f2fad3369c028f744585362a0668f77c024098d9b27d92c9a6a`。参数锁定后，
冻结 `week3_evaluation_v2` 的一次性最终评测完成 200/150/100 条：商品 JSON/Schema
100%/100%，售后 100%/96.67%，行程 95%/85%。该结果不用于继续调参；业务指标、
支持数和局限见 `reports/week6_qlora_quality_report.md`。
训练后完成度判断、前沿方法映射和下一版本的非污染实验门禁见
`reports/week6_post_training_improvement_review.md`。现有冻结结果不得用于继续调参；
没有新的 development/test 身份锁时，不得声称业务指标已经上升。

```bash
sbatch --account=punim2936 --partition=sapphire \
  --export=ALL,TRIP_DEPLOY_ROOT=<deploy-root>,TRIP_PROJECT_ROOT=<repo-root>,TRIP_VENV=<new-cu128-venv> \
  scripts/spartan/setup_week6_cuda128_venv.sbatch
python scripts/prepare_week6_data.py \
  --config configs/week6/qwen3_vl_8b_qlora_final300_v4.json
python scripts/train_week6_qlora.py \
  --config configs/week6/qwen3_vl_8b_qlora_final300_v4.json check-environment
python scripts/train_week6_qlora.py \
  --config configs/week6/qwen3_vl_8b_qlora_final300_v4.json validate-data \
  --scenario <scenario> \
  --input outputs/week6/locked_data/<dataset-version>/<scenario>/train.jsonl
python scripts/evaluate_week6_adapter.py run \
  --config configs/week6/qwen3_vl_8b_qlora_itinerary_refinement_v1.json \
  --eval-input <refinement-validation-jsonl> \
  --adapter-dir <verified-adapter-dir> \
  --output-dir <new-evaluation-output-dir>
python scripts/evaluate_week6_adapter.py compare \
  --baseline-summary <baseline-summary.json> \
  --candidate-summary <candidate-summary.json> \
  --output <new-comparison.json>
```

正式 `train-pilot` 必须显式提供锁定的数据版本、manifest/split 哈希和
`--confirm-dataset-lock`。当前框架使用 8B、NF4 double quant、bf16、LoRA
`r=16/alpha=32/dropout=0.05` 和等效 batch 16；冻结 Week 3 评测集不作为 validation
或调参数据。数据锁定采用 sample ID SHA-256 的确定性 95%/5% 训练/验证切分；模型
预标注保持 `model_preannotation` 且权重为 0.5，只有真实人工修订使用 1.0。

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

包月 CPU ECS `trip-api-sg` 的结果展示使用
`docker/aliyun/docker-compose.display.yml`。它只挂载版本化 `status.json` 和静态报告，
默认监听 `127.0.0.1:8010`，提供 `/v1/project-status` 与 `/reports/`；不安装 CUDA、vLLM、模型权重或实时 LoRA
推理服务。

2026-08-12 已在 `/opt/trip-display/20260812a` 完成独立部署，容器名为
`ota-trip-display-api`。服务器内可用的只读检查为：

```bash
curl -fsS http://127.0.0.1:8010/health
curl -fsS http://127.0.0.1:8010/v1/project-status
curl -fsSI http://127.0.0.1:8010/reports/week5_dataset_quality_report.md
```

原有 `ota-trip-api` 继续监听 `127.0.0.1:8000`，本次部署未开放公网端口，也未覆盖
其 Compose、容器或数据卷。

## Reports

The consolidated Week 1-4 report after migration to Alibaba Cloud
`qwen3.7-plus` is `reports/week1_to_week4_qwen37_overall_report.md`.
Use `reports/README.md` as the report index; historical process snapshots are
kept separate from current conclusions.
