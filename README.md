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
- reproducible model serving and experiment tracking.

## Motivation

OTA users often search with vague intent and visual references: a cafe photo, a hotel room screenshot, a restaurant dish, or a scenic street. The system turns those multimodal signals into searchable structured fields and planning inputs.

## System Architecture

```text
Client
  -> FastAPI business API
  -> Qwen3-VL + PEFT release runtime (Transformers)
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

- 当前正式模型为 Qwen3-VL-8B + system-repair PEFT adapter；生产任务使用 `/v1/tasks/*`。
- `/health` 与 readiness、独立 CLIP/Milvus 图片检索、版本化 Prompt/Schema 和实验记录。
- 生产依赖不可用时返回错误，不用固定示例冒充模型结果。对话 beta 的确定性分支只保证状态
  更新与三键契约，不代表已经生成推荐或理解新图片。
- 旧 vLLM 端点和 Docker 配置保留；Qwen2.5/DeepSeek 配置不是当前发布模型或自动兜底。
- 示例 POI planner `/v1/travel-planning` 仅限非生产环境；生产行程抽取使用
  `/v1/tasks/itinerary-planning`，并不等同于完整检索推荐链路。

商品效果口径：Week 8 的 caption/商家 metadata `silver` 匹配分不等于图像事实正确率。
2026-08-27 复审确认固定 development 的 60/60 条混有非视觉 metadata，详见
`reports/week8_product_understanding_optimization_report.md`。历史结果不覆盖，也不将这些
target 用作新的视觉专项 SFT 真值。新增工作不要求任何人工标注、复核或验收。

商品复审命令（从仓库根目录运行，输出目录必须尚不存在）：

```bash
python scripts/review_week8_product.py --audit-only --output-dir outputs/week8/review/audit_new_run
python scripts/review_week8_product.py --output-dir outputs/week8/review/product_new_run
python scripts/review_week8_product.py --rescore-dir outputs/week8/review/product_new_run --output-dir outputs/week8/review/rescored_new_run
python scripts/review_week8_product.py --config configs/week8/product_review_v2.json --output-dir outputs/week8/review/decoder_new_run
```

第一条需要既有 Spartan 数据锁；第二条还需 GPU、既有模型缓存与 `TRIP_ADAPTER_DIR`。
第三条校验原始输出哈希后重新计分，不再次推理。
第四条是在相同 256-token 证据预算下取消解码器约束的诊断，仍执行完整 Schema 后校验。
GPU 作业入口为
`scripts/spartan/week8_product_review.sbatch`；修复后 10 条对话及固定图片重复基准用
`scripts/spartan/week8_review_regression.sbatch`。两者沿用项目运行环境变量，不能与其他
独占 GPU 任务重叠；命令不访问已消费 final，也不自动替换 release。

回归脚本使用 `configs/week8/runtime_review_v3.json` 的 533×400 真实 development 图片，
必须设置 `TRIP_SMOKE_IMAGE` 与 `TRIP_SMOKE_IMAGE_SHA256` 为该配置绑定的路径和 SHA。
图片不匹配会在加载模型前失败。默认示例 `data/samples/images/cafe_001.jpg` 实际是
64×64 图形占位图，只适合接口连通性检查，不能用作真实商品质量或照片延迟基准。
最近真实照片测量中 512/384 输出上限均约 3.9 s，输出一致但仍有不可见设施猜测；
因此没有宣称提速或更换已选商品 Prompt/adapter。

## Quick Start

Create a local API environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
```

Run the API:

仅安装 API 依赖可运行健康检查；真实模型任务还需后文的 GPU runtime、release 配置和
adapter。生产环境即使误设 `MODEL_FALLBACK_ENABLED=true` 也不允许返回固定模型示例。

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

## Week 7 Multitask Context

会议汇报优先阅读 `reports/week7_multitask_context_report.md`。当前准确结论是：三核心场景
正式 test 通过，corrected-dialogue fix2 一次性 test 未通过，模型不进入 `stg`。

Week 7 当前修复身份固定使用
`configs/week7/qwen3_vl_8b_multitask_context_v4_fix2.json`；fix1、此前
`qwen3_vl_8b_multitask_context_v4.json` 与 v3 均只作不可改写历史证据。v3 因对话
assistant/user 错序失效，首版 v4 与 fix1 均未通过 development 门禁。fix2 数据构建必须显式
指向保留 Week 5/6 历史
产物的只读来源项目；默认验证只读 train/development，corrected-dialogue test
只能在 development 自动门禁选定 checkpoint 后消费一次。v4 锁与 v3 完整
identity manifest 执行五维零重叠审计，不仅排除旧 test，也排除 v3 的训练和
development 来源。fix2 还排除 fix1 全量身份，并把嵌套值改为叶子级评分、把 protocol
coverage 与 semantic accuracy 分开；训练 early stopping 使用与 selector 一致的
`eval_gate_selection_score`，不降低原门禁阈值。

```bash
python scripts/manage_week7.py --config configs/week7/qwen3_vl_8b_multitask_context_v4_fix2.json build-lock --source-project-root <read-only-week5-week6-project>
python scripts/manage_week7.py --config configs/week7/qwen3_vl_8b_multitask_context_v4_fix2.json validate-lock
python scripts/run_week7.py --config configs/week7/qwen3_vl_8b_multitask_context_v4_fix2.json check-environment
python scripts/run_week7.py --config configs/week7/qwen3_vl_8b_multitask_context_v4_fix2.json train-multitask --help
python scripts/run_week7.py --config configs/week7/qwen3_vl_8b_multitask_context_v4_fix2.json select-v4-checkpoint --help
python scripts/run_week7.py --config configs/week7/qwen3_vl_8b_multitask_context_v4_fix2.json v4-dialogue-test --help
python scripts/run_week7.py adversarial-audit
```

v4 对话对所有 assistant span 计算 SFT loss，不再只训练最后 JSON；development
和 test 依次生成每个 assistant 轮，将模型自己的前序回复放回上下文，并统计
逐轮覆盖率、逐轮失败率、上下文值准确率及任务结果键/值准确率。新评测不
要求人工输入；机器结果不标记为人工验收。对应 Spartan 作业为
`week7_v4_multitask_train.sbatch` 和 `week7_v4_dialogue_test.sbatch`。

最新 fix1 锁定身份训练由 Spartan job `29526965` 在 clean commit `6bb5322` 上完成，
运行 02:43:24，并在 step 226 按 patience=2 早停。六个 development checkpoint
仍为 0/6 通过全部自动门禁；最佳 step 151 的 weighted composite 为 0.764049，
唯一失败项为 sequential coverage 0.725585 < 0.75。不可覆盖 selector 已写出
`BLOCKED_NO_ELIGIBLE_CHECKPOINT` 证据（文件 SHA-256 `782e92ab...cc104`），没有
selected checkpoint；corrected-dialogue fix1 test 保持 `LOCKED_UNCONSUMED`。该终态
不允许绕过门禁提交 test，也不把 development 最优分解释为一次性 test 结果；完整实证
见 `reports/week7_multitask_context_report.md`。

fix2 本地锁 `week7_corrected_multitask_context_20260824_v4_fix2` 已生成并验证：
train/development/test=3000/114/114，五维跨分区碰撞为 0，canonical lock SHA-256 为
`86a4360142c2517e46460cefc575131940989aa8129eca236c68eaaf71e5b14b`。Spartan 训练 job
`29540085` 完成并在 step 301 早停；不可覆盖 selector 通过 development 自动门禁，在
5 个合格 checkpoint 中锁定 step 226（adapter SHA-256
`ccc6062f7e451b9265c571c0df397903cbbc707a6bf2e894039079175e5f24ee`）。唯一 one-shot
test job `29544969` 已消费 24 条 corrected dialogue，最终门禁为 `FAIL`，不得重跑：
multitask/Week 6 routed/zero-shot 自动综合分分别为 0.793399/0.152144/0.174505；multitask
格式、上下文召回、失败率分别为 0.916667/0.750000/0.041667。相对基线的提升只作描述性
结果，不能覆盖 10 项绝对阈值失败，也不宣称 Week 7 fix2 已验收。完整证据见
`reports/week7_multitask_context_report.md`。

以下标注台命令仅用于已完成的历史 v3/corrected-development 真人证据；
v4 自动闭环不再等待新人工输入：

```bash
python scripts/run_week7_dialogue_review.py
```

默认只监听 `127.0.0.1:8097`，并校验 v3 数据锁、固定队列、development 和
checkpoint-151 protocol-v5 raw output SHA-256。标注台还会检查 assistant 回复是否在
对应 user 问题之前出现；命中时前后端均禁止保存四维分数，并列出错位内容作为标注辅助，
不能通过页面重排伪装修复。只有上下文完整性通过后，才允许真实操作者按历史图片指代、
需求迭代、上下文承接和逻辑连贯/OTA 专业性四维评分，并确认逐项自审。结果以 revision 追加到忽略目录
`outputs/week7/human_review/week7_dialogue_human_scores_v1.jsonl`，不会修改数据锁、raw
output 或既有评分。默认 raw 文件的固定 SHA-256 为
`aee27cf1cab1d97d26f9ba81c1319d3fe5532e8328b6738c59416c78bfa37090`。

训练配置固定为 Qwen3-VL-8B、NF4 4bit、LoRA `r=16/alpha=32/dropout=0.08`、
学习率 `1.5e-4`、weight decay `0.03`、gradient clipping `1.0` 和 gradient
checkpointing。完整 development 每约 10% 更新步生成评测一次，按三场景加权综合分
选择 checkpoint，连续两次无提升早停。Schema 对照必须使用独立 run，并只报告格式、
Schema、延迟和失败回退；对话人工队列只能由真实用户填写。生产作业分别使用
`week7_development_eval.sbatch`、`week7_dialogue_development.sbatch`、
`week7_schema_experiment.sbatch`、`week7_multitask_train.sbatch` 和
`week7_latency_protocol_v5.sbatch`、`week7_final_test.sbatch`。checkpoint 的 development metrics/raw、Schema comparison/raw、
参数锁和最终 test 状态均以 SHA-256 逐级绑定；final-test 在 Slurm 中断后只能依据原
job、锁和 test 身份执行受审计恢复。

数据和训练身份仍为 v3；独立 `evaluation_protocol_v5` 只锁定公平推理口径，不是新模型
或新数据锁。它在同一 Slurm allocation 中统一 BF16、static KV cache、Transformers
compile、32-token warm-up、CUDA 同步计时、结构感知截断和 gold-evaluable 支持集合，
串行重测三套 Week 6 adapters、四个多任务 checkpoint 与 zero-shot。完整作业
`29456896` 为 `COMPLETED 0:0`，selector 选择 checkpoint-151：development 综合分
0.740904、全局延迟比 0.8809、失败率 0。

参数锁随后完整绑定 v5 runtime；唯一 final-test 作业 `29459265` 为 `COMPLETED 0:0`，
marker 和 7 个结果 artifact 的 SHA-256 验证通过，全部自动非回退门禁通过。test 上统一
模型商品/售后/行程 composite 为 0.153846/1.000000/0.996667，平均延迟 7173.16 ms、
失败率 0，三个核心场景自动门禁通过。后续人工界面审计确认 v3 对话构造把 assistant
回复放在对应 user 问题之前；历史格式 1.0/字符串包含式上下文召回 0.878472 保留为原始
输出，但状态为 `BLOCKED_INVALID_SOURCE_CONTEXT`，不能作为真实多轮能力结论或进入人工
评分。修复没有改写 v3：独立 development-only 身份
`week7_dialogue_review_20260821_v2` 将首轮任务和每个 follow-up 都改为
user→具体 assistant 回答，24 条按 5/6/7/8 轮各 6 条，图片仍只在首次用户轮。
checkpoint-151 新 raw 由 Spartan job `29479822` 生成，24/24 成功，SHA-256 为
`9cb8cafc148aa8fd7fc2c9e656c59a8c6ce24e237b995df6557d22381dbcd162`。本地标注台用以下
显式身份启动，评分仍只能由真实用户填写：

```bash
python scripts/run_week7.py build-dialogue-review-v2 --review-config configs/week7/dialogue_review_v2.json
python scripts/run_week7_dialogue_review.py --dataset-dir outputs/week7/dialogue_review_v2/week7_dialogue_review_20260821_v2 --raw-outputs outputs/week7/dialogue_review_v2/runs/run_3e5e767_a100/raw_outputs.jsonl --output-dir outputs/week7/human_review/dialogue_review_v2 --raw-sha256 9cb8cafc148aa8fd7fc2c9e656c59a8c6ce24e237b995df6557d22381dbcd162 --dataset-version week7_dialogue_review_20260821_v2 --run-id week7_dev_dialogue_repair_checkpoint151_20260821_v2 --model-name multitask_checkpoint_151_dialogue_review_v2
```

该 v2 只恢复 development 人工四维评估，不能修复、重开或改写已消费的 v3 final-test。
真实单人操作者已在一个会话中完成 24/24 条并逐条确认本人自审，最终决定均为 `pass`；
历史图片指代/需求迭代/上下文承接/逻辑连贯均分为
4.5417/4.6250/4.5000/4.7083，四维未加权均值 4.59375。结果文件 SHA-256 为
`bdec2d18...af932`，原始人工记录继续位于忽略目录，Git 仅保存聚合与哈希证据。
DPO 最初因 0 条真实审核偏好对保持 `SKIPPED`。两组真实四维评分完成后，按锁定规则派生
16 个非平局偏好对，并通过 JSON、视觉证据、来源身份及反转探针审计；其中 10 对训练、
6 对隔离 validation。唯一一次 mDPO-style job `29491859` 已完成，但 validation 准确率
0.3333、平均 policy-reference margin -0.00981，未通过 0.5/>0 门禁，因此新 adapter
不被选用，checkpoint-151 保持生产选择，test 未重跑。

为补齐 corrected development 上与纯单任务模型的人工对比，锁定的三个 Week 6 adapter
按父场景 8/8/8 路由生成另一组 24 条输出。运行命令为：

```bash
python scripts/run_week7.py dialogue-week6-baseline-v1 \
  --comparison-config configs/week7/dialogue_comparison_v1.json \
  --dataset-dir outputs/week7/dialogue_review_v2/week7_dialogue_review_20260821_v2 \
  --product-adapter <week6-product-adapter> \
  --after-sales-adapter <week6-after-sales-adapter> \
  --itinerary-adapter <week6-itinerary-adapter> \
  --output-dir <new-development-output>
```

Spartan job `29491047` 已完成 24/24、失败 0；真实单人操作者随后完成 Week 6 routed
输出 24/24 四维评分。multitask/Week 6 四维总均值为 4.59375/4.56250，配对差
+0.03125，按样本为 10 胜/7 平/7 负；这是描述性小差异，不作显著提升结论。加入该
runner 与 mDPO 审计实现加入后，完整 `unittest` 为 428/428。终态再执行机器优先的
`adversarial-audit`：基线替换、跨分区碰撞、采样漂移、Schema 语义洗白、test 重跑、
支持数删除、对话缺陷洗白、repair 读取 test、Agent 冒充人工、失败 DPO 晋级和 DPO
读取 test 共 11/11 个反事实均被拒绝。完整 `unittest` 更新为 431/431。审计允许代码
进入 `dev` 集成，但不允许把已知 v3 test 对话缺陷写成完全通过，也不允许进入 `stg`。

## Unified Qwen3-VL System Runtime

当前统一系统入口使用 `configs/releases/qwen3_vl_system_v1.json`，默认基座为
`Qwen/Qwen3-VL-8B-Instruct`，adapter 必须通过文件 SHA-256 核验。生产模式没有模型、
Schema 或检索静默回退。

当前 release candidate 已绑定 system-repair checkpoint-87，adapter SHA-256 为
`c2fbb5c7...eaa2a`。唯一一次 120 条 fresh test 已完成，三场景 JSON/Schema 均为
1.0、请求失败率为 0，对话 Beta 综合为 0.973330；不可覆盖 final gate 为 `PASS`。

```bash
python scripts/tripctl.py doctor
python scripts/tripctl.py validate
python scripts/tripctl.py serve
python scripts/tripctl.py smoke --base-url http://127.0.0.1:8000
```

`TRIP_RELEASE_CONFIG` 或全局参数 `--release-config` 显式选择同一配置；参数优先于环境变量。
`validate` 返回实际文件路径及 SHA-256，缺失或损坏配置会失败。Compose 使用同一入口：

```bash
python scripts/tripctl.py --release-config configs/releases/qwen3_vl_system_v1.json compose config --quiet
python scripts/tripctl.py --release-config configs/releases/qwen3_vl_system_v1.json compose up -d
```

该入口把解析后的绝对路径只读挂到 `/run/trip/release.json`；直接调用 Docker Compose 时，
`TRIP_RELEASE_CONFIG` 必须为主机绝对路径，以免 Compose 与仓库根目录的相对路径基准不同。

`smoke` 会依次检查 `/health`、`/ready`、三场景任务、多轮对话和视觉检索；部署主机上的
样例图片路径可通过 `--image-path` 覆盖。任一模型、Schema 或检索请求失败都会使 smoke
失败，不使用 mock 或 keyword fallback 冒充真实系统验证。

统一 Compose 还要求将 `RETRIEVAL_HOST_DIR` 指向已解压且通过 manifest 校验的
`retrieval/` 目录。启动时一次性 `retrieval-init` 会拒绝部分入库状态，仅在集合为空时
写入固定 1,000 条向量；集合已完整时幂等通过。API 必须等待该初始化成功后才能启动。

统一接口：

- `POST /v1/tasks/image-product-search`
- `POST /v1/tasks/after-sales`
- `POST /v1/tasks/itinerary-planning`
- `POST /v1/dialogue`
- `POST /v1/visual-search`
- `GET /health`
- `GET /ready`

`/health` 只表示 API 进程存活；`/ready` 会实际核验 adapter、模型后端、Prompt、Schema、
CLIP、Milvus 和 release identity。Milvus 基准配置位于
`docker/system/milvus_system.yaml`，统一 Compose 位于 `docker/system/docker-compose.yml`。
没有实际 adapter 时 `tripctl doctor` 返回 `not_ready`，这是预期的 fail-closed 行为。
Spartan job `29571134` 已用发布配置和 checkpoint-87 adapter 完成真实生产模型 smoke：
三类任务均 Schema-valid，对话经一次模型级纠错后达到 `DIALOGUE_BETA`。

系统收敛修复的真实进度、Milvus 实测和发布封装门禁见
`reports/system_consolidation_repair_report.md`。

### Local Model Handoff

导师最新口径不要求 Spartan 或 OSS 留存。当前只保留一份 Git 外本地交接包：

`outputs/releases/trip-qwen3-vl-8b-system-repair-v1-rc1-final-v3`

交接前执行：

```bash
python scripts/verify_model_handoff.py \
  outputs/releases/trip-qwen3-vl-8b-system-repair-v1-rc1-final-v3
```

该命令核验四层归档、release config、adapter、final gate、真实模型 smoke 和 Milvus
基准，不连接 Spartan 或 OSS。模型二进制不进入 Git，因此移交仓库时必须同时移交这一
目录。基座模型按 release config 中固定的 Hugging Face model/revision 下载。完整接手
步骤见 `docs/model_handoff.md`。

可再生成的本地资产可先预览再清理：

```powershell
.\scripts\cleanup_local_assets.ps1
.\scripts\cleanup_local_assets.ps1 -Apply
```

脚本会先复验唯一交接包并保护凭据；当前清理已释放约 66.8 GiB，只留下轻量样例和交接包。

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

## Week 8 全自动商品、对话、延迟与检索候选

Week 8 仍为 **PARTIAL，不晋级当前候选**。以下为保留的历史候选配置，不代表新的发布批准：

- 商品 fresh 实验与数据锁：`configs/week8/product_understanding_v7.json`
- 商品可观察证据与 continuation SFT：`configs/week8/product_two_stage_v1.json`
- 商品剩余 Prompt 的 development-only 诊断：
  `configs/week8/product_prompt_refinement_v8_development.json`
- 未消费商品 silver/OCR 可行性审计：`configs/week8/product_silver_source_audit_v8.json`
- 对话/真实图片延迟基准：`configs/week8/runtime_optimization_v7.json`
- 商品 prepared-input cache 负实验：`configs/week8/runtime_optimization_v8.json`
- Milvus Lite 混合检索：`configs/week8/retrieval_relevance_v3.json`
- Milvus Lite 有界 metadata LRU：`configs/week8/retrieval_latency_v5.json`
- 合并后的 release candidate：`configs/releases/qwen3_vl_system_week8_v7.json`

商品 fresh source 从官方 Yelp Photos ZIP 重建，v7 只绑定 post-hash 已完成的 v3 source
manifest；train/development/test 按五维身份隔离。新增标签、review 和 acceptance 全部为
`programmatic_silver`，人工计数固定为 `0`。常用验证命令：

```bash
python scripts/manage_week8_product.py \
  --config configs/week8/product_understanding_v7.json validate-lock
python scripts/manage_week8_product_two_stage.py \
  --config configs/week8/product_two_stage_v1.json validate-hard-slice
python scripts/manage_week8_retrieval.py validate-lock \
  --lock-dir outputs/week8/retrieval/week8_retrieval_query_index_20260827_v3
python scripts/audit_week8_product_silver_source_v8.py \
  --config configs/week8/product_silver_source_audit_v8.json --help
python scripts/manage_week8_retrieval.py evaluate-latency-development \
  --config configs/week8/retrieval_latency_v5.json --help
python -m unittest tests.test_week8_product tests.test_week8_product_fresh_sources \
  tests.test_week8_product_two_stage tests.test_week8_product_sft \
  tests.test_week8_product_silver_source_v8 -v
python -m unittest tests.test_week8_runtime_optimization tests.test_week8_retrieval \
  tests.test_processor_cache tests.test_system_runtime -v
python -m unittest discover -s tests -v
```

完整身份、真实指标、失败尝试和未解决项见
`reports/week8_product_understanding_optimization_report.md`。正式 release manifest 未被候选
配置覆盖；晋级或合并不属于本分支交付。剩余优化中，两个额外商品 Prompt 和
prepared-input cache 均经 development 实测后拒绝；检索 LRU512 在质量完全一致时将真实
Milvus Lite 稳态 P95 从 `9.6339` 降到 `8.4247 ms`，同时记录 `1.60 s` 预计算和内存成本。

### c01b732 审查后的修复入口（2026-08-28）

历史检索 NDCG 使用了查询参考 metadata，不是独立图像理解证据；上述速度数据也不能
替代相关性验收。现在排序只接收 `query_inputs.source=user/model_prediction`，生产
`/v1/visual-search` 接通 `keyword/embedding/hybrid` 与显式城市/业态/价位条件。
尚不能处理的“安静”等文本会列在 `unapplied_query_text`，不能声称满足全部条件。

对话返回 `task_status=COMPLETED/STATE_UPDATED/NOT_COMPLETED` 和实际 `task_result`；
状态更新成功与推荐任务完成分开。图片可放到每条 user 消息的 `image_urls`；兼容的顶层
图片默认绑定最新 user 轮。商品仅接受一图；行程需要非空文字约束，非法请求返回 422。
行程的天数、日序、明确约束和占位文本不合格时进行一次纠错，仍失败返回明确错误。

以下命令需要现有 Spartan 项目数据/运行环境，输出目录必须为新身份；不读取最终 test 标签：

```bash
python scripts/audit_week8_labels.py --config configs/week8/audit_repair_v1.json
python scripts/verify_week8_retrieval_routing.py --config configs/week8/audit_repair_v1.json
python scripts/verify_week8_runtime_repairs.py --config configs/week8/audit_repair_v1.json
python -m unittest tests.test_week8_audit_fixes -v
```

新标签协议 `caption_evidence_v2` 将商家 metadata 与 caption 标签分开，全部仍为 silver；
旧 60 条 development 全部保留，但可靠视觉指标仍无法据此得出。缺少视觉参考或存在
标签矛盾时，选优返回 `DIAGNOSTIC_ONLY_INVALID_REFERENCES`，不得锁定 Prompt。
真实复测技术 smoke PASS、业务 smoke FAIL；商品仍猜测停车设施，不晋级。完整证据见商品报告第 13 节。

新的契约消融仅访问 development，使用独立 Prompt 与不可覆盖的输出身份：

```bash
python scripts/review_week8_contracts.py --config configs/week8/contract_ablation_v1.json
python -m unittest tests.test_week8_contract_ablation -v
```

此命令对比旧 adapter、新 Prompt + adapter 和新 Prompt + base；业务/语法通过不代表
视觉准确率通过，不会写入正式 release 或消费 final。

第二轮使用逐标签可见证据协议并检查实际活动中的截止时间、交通和必去/禁去地点：

```bash
python scripts/review_week8_contracts.py --config configs/week8/contract_ablation_v2.json
python scripts/collect_week8_visual_silver.py --config configs/week8/visual_teacher_v3.json
```

教师命令需要本地配置 `MODEL_API_KEY_FILE`，可通过 `MODEL_API_BASE_URL` 指定既有兼容
端点。它只发送 development 图片与观察协议，不发送历史标签、商家 metadata 或候选答案；
输出保留 `model_generated_silver`，不构成人工真值或自动发布授权。

完整的同口径 development 对比（仅在独立教师产物已校验时评分）：

```bash
python scripts/review_week8_contracts.py --config configs/week8/contract_ablation_v3.json
python scripts/score_week8_visual_silver.py --config configs/week8/contract_ablation_v3.json --output outputs/week8/review/week8_contract_comparison_20260828_v3
```

`DEVELOPMENT_CANDIDATE` 仅表示固定图像银标口径下的开发候选，仍禁止据此直接晋级。

实际服务支持版本化 `product_pipeline=visual_observation` 与逐场景 adapter 开关；商品
观察结果按确定性规则映射到原商品 Schema，缺乏价位比较口径时返回 unknown。售后可保留
正式 adapter，商品/行程/对话独立关闭 adapter，异常时恢复状态。配置哈希不匹配直接失败。

已消费的 `visual_final_v2.json` 因无效参考失败，原锁定协议对应 `40a0f34`；随后仅在
固定 development 上形成实质改进的 v9，再通过完全隔离的 `visual_final_v3.json` 单次
最终验收。两版 final 均不能重跑。以下是本轮已执行的 development 入口，现有产物目录
不可覆盖，不要为了获得 PASS 更换旧最终参考或反复重新抽样。

```bash
python scripts/collect_week8_teacher_reliability.py --config configs/week8/visual_teacher_v4.json
python scripts/review_week8_contracts.py --config configs/week8/contract_ablation_v4.json
python scripts/score_week8_visual_silver.py --config configs/week8/contract_ablation_v4.json --output outputs/week8/review/week8_contract_comparison_20260828_v4
python scripts/compare_week8_development_revision.py --previous outputs/week8/review/week8_contract_comparison_20260828_v3/comparison.json --current outputs/week8/review/week8_contract_comparison_20260828_v4/comparison.json --output outputs/week8/review/week8_development_revision_20260828_v1.json
```

`seal` 需要已存在且通过的探针和候选 release；缺失时禁止启动最终推理。原生图片模板
身份为 null/N/A，不人为制造模板隔离。最终结果仅用于锁定候选的验收，不能传入 development
选择器。运行层打包后应调用 `scripts.build_release_bundle.verify_runtime_archive`，隔离导入
API 与实际 release 配置；文件哈希正确不代表依赖完整或业务通过。

### Week 8 v9 当前候选交接（2026-08-28）

`trip-qwen3-vl-8b-week8-visual-silver-v9` 已完整通过自动 silver 候选验收；默认正式
release 没有替换，也未合并长期分支或打标签。商品使用 `product_visual_observation_v3`
与底座，售后继续使用正式 adapter。相同 100 条最终参考下 composite
0.429365→0.736721，JSON/Schema 100%/100%、请求失败 0；价位 N/A，平均延迟增加
22.15%。这是独立图像教师一致性，不是人工视觉准确率。

四层交接包和验证记录在 `outputs/releases/trip-qwen3-vl-8b-week8-visual-silver-v9-rc1`；
原始运行、错误切片、支持数、失败历史和剩余限制见
`reports/week8_product_understanding_optimization_report.md` 第 14 节。以下命令只校验
候选配置或已生成交接包，不调用模型、不重新消费 final：

交接验证还会检查七个必需 API 端点是否齐全，不依赖 FastAPI 内部路由对象数量。

```bash
python scripts/tripctl.py --release-config configs/releases/qwen3_vl_system_week8_v9.json validate
python -c "from pathlib import Path; from scripts.verify_week8_candidate_handoff import verify; print(verify(Path('outputs/releases/trip-qwen3-vl-8b-week8-visual-silver-v9-rc1'), 'evidence/week8_visual_holdout_20260828_v3/promotion_acceptance.json'))"
```
