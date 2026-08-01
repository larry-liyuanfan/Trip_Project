# Experiment Log

## EXP-20260802-001

- date: 2026-08-02
- model: `qwen3.7-plus`
- backend: Alibaba Cloud Model Studio OpenAI-compatible API, Singapore
- dataset: `week3_evaluation_v2`, 450 human-gold samples
- baseline_run: `week3_qwen37_baseline_full_20260802_002`, 450/450, request errors 0
- standardized_run: `week3_qwen37_standardized_full_20260802_001`, 450/450, request errors 0
- week4_winner_run: `week4_qwen37_winners_full_20260802_001`, 450/450, request errors 0
- winners: product `fewshot_4_v2`; after-sales `fewshot_4_v2`; itinerary `standardized_v2`
- common_semantic: `qwen37_common_semantic_v1_20260802_001`, 450 pairs, 2,000 bootstrap iterations
- status: completed; itinerary JSON/Schema remains 33%/33%

## EXP-20260706-001

- date: 2026-07-06
- git_commit: dirty-initial-scaffold
- model_name: Qwen/Qwen3-VL-2B-Instruct
- model_size: 2B target
- inference_backend: vLLM
- serving_command: scripts/run_vllm_server.sh
- prompt_version: prompt_image_understanding_v1
- temperature: 0.1
- top_p: 0.9
- max_tokens: 512
- input_type: image_plus_text
- dataset_version: sample_ota_images_v1
- task_type: image_understanding
- metrics: JSON schema availability, API response availability
- result_summary: Initial scaffold supports deterministic fallback until live vLLM is started.
- failure_cases: Real image/model inference not verified yet.
- screenshots_or_samples: data/samples/poi_catalog.jsonl
- next_action: Start vLLM container and replace fallback results with live model output.

## EXP-20260706-002

- date: 2026-07-06
- git_commit: e63a00e
- model_name: Qwen/Qwen2-VL-2B-Instruct
- model_size: 2B
- inference_backend: vLLM 0.8.5 Docker
- serving_command: docker compose -f docker/docker-compose.yml up -d --build
- prompt_version: prompt_image_understanding_v1
- temperature: 0.1
- top_p: 0.9
- max_tokens: 512
- input_type: image_plus_text
- dataset_version: sample_ota_images_v1
- task_type: image_understanding
- metrics: /health OK, /v1/models OK, /v1/image-understanding OK
- result_summary: Docker API and vLLM service started on local RTX 4070 Laptop GPU 8GB. vLLM served Qwen2-VL-2B through OpenAI-compatible API and returned structured image-understanding output.
- failure_cases: vllm/vllm-openai:latest required CUDA >= 13.0; Qwen2.5-VL-3B loaded weights but was too tight on 8GB VRAM during profiling. Local file URLs required conversion to data URLs for vLLM.
- screenshots_or_samples: data/samples/images/cafe_001.jpg
- next_action: Use this 2B compose profile for Week 1 demos; retry Qwen2.5-VL/Qwen3-VL on a larger GPU or newer NVIDIA driver.

## EXP-20260707-001

- date: 2026-07-07
- git_commit: tracked-in-this-commit
- model_name: N/A
- model_size: N/A
- inference_backend: N/A
- serving_command: N/A
- prompt_version: N/A
- temperature: N/A
- top_p: N/A
- max_tokens: N/A
- input_type: yelp_json_reviews_photo_metadata
- dataset_version: yelp_ota_subset_v1
- task_type: data_preparation
- metrics: business_count=200; review_count=1000; multimodal_item_count=581; extracted_photo_count=581
- result_summary: Local Yelp JSON and Photos zip archives were extracted into ignored raw data, converted into a bounded OTA subset, and materialized only the photos referenced by the multimodal subset.
- failure_cases: The official zip files contain gzip-compressed tar payloads with .tar filenames and macOS resource entries; the extractor skips resource entries and reads the gzip tar stream directly.
- screenshots_or_samples: data/yelp/processed/ota_subset_v1/manifest.json; data/yelp/raw/extract_photo_manifest.json
- next_action: Use this processed subset for current Week 1 data validation only.

## EXP-20260708-001

- date: 2026-07-08
- git_commit: tracked-in-this-commit
- model_name: Qwen/Qwen2-VL-2B-Instruct
- model_size: 2B
- inference_backend: vLLM 0.8.5 Docker
- serving_command: docker compose -f docker/docker-compose.yml up -d --build --force-recreate
- prompt_version: prompt_image_understanding_v1
- temperature: 0.1
- top_p: 0.9
- max_tokens: 512
- input_type: two_yelp_subset_images
- dataset_version: yelp_ota_subset_v1
- task_type: multi_image_live_vllm_stretch
- metrics: api_health_ok; vllm_models_ok; live_multi_image_http_ok; fallback_detected=false; parsed_json_ok=false
- result_summary: Stretch live vLLM check sent two Yelp subset images through the FastAPI API into vLLM and received a live model response without API fallback.
- failure_cases: Qwen2-VL-2B generated truncated or malformed fenced JSON for the two-image prompt, so structured parsing fell back to an unstructured response with confidence=0.3.
- screenshots_or_samples: scripts/test_multi_image_understanding.py
- next_action: Stretch item only; not required for Week 1 acceptance. Revisit prompt constraints or max_tokens before treating multi-image structured extraction as accepted.

## EXP-20260714-001

- date: 2026-07-14
- git_commit: 06005fa (dirty Week 3 worktree)
- model_name: Qwen/Qwen2-VL-2B-Instruct
- model_size: 2B
- inference_backend: vLLM HTTP readiness probe
- serving_command: N/A (probe only; service already running)
- prompt_version: PENDING (baseline_minimal_v1 configured but not executed)
- temperature: 0.1 configured; not used
- top_p: 0.9 configured; not used
- max_tokens: 512 configured; not used
- input_type: no Week 3 image request
- dataset_version: week3_evaluation_v1
- task_type: week3_evaluation_readiness
- metrics: product target/candidate/annotated/validated/tested=200/0/0/0/0; after-sales=150/0/0/0/0; itinerary=100/0/0/0/0; all real model metrics=PENDING
- result_summary: Week 3 is PARTIAL. Stage 1–4 engineering implementation and verification completed; Stage 5 real baseline and standardized comparison are PENDING. 2026-07-14 `/v1/models` 探测成功，返回 `Qwen/Qwen2-VL-2B-Instruct`；未发送 Week 3 图片请求，未产生模型输出或延迟指标。 Synthetic/mock framework verification: PASS，不属于真实模型 baseline，不计入 tested_count。
- failure_cases: This historical readiness entry predates candidate construction and human annotation; it contains no real model failure case.
- screenshots_or_samples: N/A
- next_action: Historical entry; superseded by the 2026-07-21 completed annotation and validation state.

## EXP-20260714-002

- date: 2026-07-14
- git_commit: 06005fa (dirty Week 3 worktree)
- model_name: N/A; no model request
- inference_backend: deterministic local candidate builder
- serving_command: N/A
- prompt_version: N/A
- input_type: local Yelp photos and project-owned business-synthetic evidence/constraint recipes
- dataset_version: week3_evaluation_v1
- source_version: yelp-week3:11151bf7f6604e2c30baaf86853b6883fec00c8e904c49ea43262457e9e0358b
- command: `python scripts/build_week3_candidate_manifests.py --config configs/evaluation_week3.yaml`
- metrics: product target/candidate/annotated/validated/tested=200/200/0/0/0; after-sales=150/150/0/0/0; itinerary=100/100/0/0/0; exclusion_count=450; all real model metrics=PENDING
- result_summary: Deterministic candidate construction and configuration validation passed. All rows remain pending human annotation; no model baseline was executed.
- failure_cases: The first build exposed insufficient hotel/attraction coverage because multi-category Yelp businesses were discarded. Category precedence and group-first photo selection were corrected with tests; the successful build met every configured stratum quota without weakening the final near-duplicate threshold.
- screenshots_or_samples: ignored local manifests, sampling logs, images, annotation packets, exclusion registry, and `data/eval/readiness/20260714_models_probe_001.json`
- next_action: Historical candidate-build entry; superseded by the 2026-07-21 completed annotation and validation state.

## EXP-20260721-001

- date: 2026-07-21
- git_commit: 06005fa (dirty Week 3 worktree)
- model_name: Qwen/Qwen2-VL-2B-Instruct
- inference_backend: vLLM OpenAI-compatible HTTP
- prompt_version: baseline_minimal_v1
- generation: temperature=0.1; top_p=0.9; max_tokens=512
- dataset_version: week3_evaluation_v1
- run_id: week3_baseline_full_20260721_003
- command: `python scripts/run_week3_evaluation.py --config configs/evaluation_week3.yaml --run-id week3_baseline_full_20260721_003 --mode live --run-scope full --prompt-version baseline_minimal_v1`
- result_summary: completed/live/full; selected_count=450; record_count=450; all persisted raw outputs and latency values present.
- metrics: JSON compliance and Schema pass are 0% in all three scenarios on the strict business track; mean latency product=3463 ms, after-sales=2139 ms, itinerary=3508 ms.
- failure_cases: All 450 minimal-Prompt responses are natural language and fail strict JSON parsing. This is end-to-end parseability evidence, not a separately measured native-semantic score.
- artifacts: `data/eval/runs/week3_baseline_full_20260721_003/`; `data/eval/scores/week3_baseline_full_20260721_003/`

## EXP-20260721-002

- date: 2026-07-21
- git_commit: 06005fa (dirty Week 3 worktree)
- model_name: Qwen/Qwen2-VL-2B-Instruct
- inference_backend: vLLM OpenAI-compatible HTTP
- prompt_version: standardized_v1
- generation: temperature=0.1; top_p=0.9; max_tokens=512
- dataset_version: week3_evaluation_v1
- run_id: week3_standardized_full_20260721_001
- command: `python scripts/run_week3_evaluation.py --config configs/evaluation_week3.yaml --run-id week3_standardized_full_20260721_001 --mode live --run-scope full --prompt-version standardized_v1`
- result_summary: completed/live/full; selected_count=450; record_count=450; selected-sample hash and non-Prompt assets match the baseline run.
- metrics: JSON compliance product=68.5%, after-sales=98.0%, itinerary=28.0%; Schema pass product=29.5%, after-sales=2.0%, itinerary=0.0%; mean latency 5272/2844/10987 ms.
- failure_cases: 138 JSON parse failures and 250 Schema failures. Dominant issues are evidence type errors, missing confidence/OCR/evidence fields, duplicate enum items, hallucinated after-sales fields, and truncated itinerary output.
- artifacts: `data/eval/runs/week3_standardized_full_20260721_001/`; `data/eval/scores/week3_standardized_full_20260721_001/`

## EXP-20260721-003

- date: 2026-07-21
- comparison_id: week3_prompt_pair_strict_20260721_001
- baseline_run_id: week3_baseline_full_20260721_003
- standardized_run_id: week3_standardized_full_20260721_001
- scoring_track: strict_business
- paired_sample_count: 450
- bootstrap_iterations: 2000
- result_summary: Same-set paired comparison completed with deterministic win/tie/loss, confidence intervals, and representative cases.
- artifacts: `data/eval/comparisons/week3_prompt_pair_strict_20260721_001/`; `data/eval/generated_reports/week3_prompt_pair_strict_20260721_001/report.md`
- interpretation: Superseded. Project Control rejected the test-set coverage; this comparison cannot support Prompt-effect or capability conclusions.

## DATA-20260721-004

- date: 2026-07-21
- git_commit: 06005fa (dirty Week 3 worktree)
- model_request: none
- command: `python scripts/replace_week3_after_sales_candidates.py --config configs/evaluation_week3.yaml --repair-id after_sales_rebuild_v3_20260721_001`
- reason: Human gold showed 0 facility-damage samples, 68 unknown samples, unrelated public weak-pair images, and unresolved PII state.
- result: Backed up the old manifest and registry; generated 150 pending project-owned candidates with exact hygiene/facility/closure/delay quotas `38/38/37/37`; rebuilt the 450-row exclusion registry; exported a non-gold suggestion packet.
- artifact_hash: new after-sales manifest SHA-256 `6a691191f7f776dbceee4ae4ba4c139c520330897085f16b981e07b8d049a94a`.
- current_counts: product `200/200/200/110/0`; after-sales `150/150/0/0/0`; itinerary `100/100/100/0/0`.
- status: `PARTIAL`; corrected live baseline and standardized comparison remain `PENDING` until human recuration passes the new gold-coverage gate.

The recuration route above was superseded by Project Control's approved
frozen-label execution plan.

## DATA-20260722-005

- date: 2026-07-22
- git_commit: `2a76cd6` plus dirty v2 worktree
- model_request: none
- command: `python scripts/prepare_week3_v2_dataset.py --config configs/evaluation_week3_v2.yaml`
- dataset_version: `week3_evaluation_v2`
- result_summary: Preserved v1 and created separate v2 manifests, annotation packets, preparation log, and a 450-row exclusion registry. Product reuses 200 completed labels. After-sales retains 80 supported labels and exposes 70 pending replacements with candidate strata `38/38/37/37`. Itinerary uses style-only supplementation with all prior v1 constraint fields inherited.
- counts: product `200/200/200/200/0`; after-sales `150/150/80/80/0`; itinerary `100/100/4/4/0`
- status: `PARTIAL`; no full v2 run or score exists.

## EXP-20260722-006

- date: 2026-07-22
- git_commit: `2a76cd6` plus dirty v2 worktree
- model_name: Qwen/Qwen2-VL-2B-Instruct
- inference_backend: vLLM OpenAI-compatible HTTP
- prompt_version: standardized_v2
- generation: temperature=0.1; top_p=0.9; repetition_penalty=1.05; max_tokens=1280
- dataset_version: week3_evaluation_v2
- run_scope: representative format probes only; not a baseline and not counted in `tested_count`
- result_summary: Product and after-sales representative requests passed JSON and Schema validation. Early itinerary probes exposed nullable-type rejection, strict-guided timeout, repeated arrays, missing fields, and truncation. After adding a bounded itinerary v2 Schema and final type skeleton, probes `week3_v2_standardized_schema_probe_20260722_013` and `_014` both returned JSON- and Schema-valid outputs in approximately 5.3 and 5.6 seconds.
- semantic_limit: The four-day probe returned one itinerary day. Schema pass therefore does not establish constraint satisfaction or itinerary completeness.
- rejected_configuration: strict `json_schema` itinerary request `_011` timed out after 180 seconds and is not used for full runs.
- artifacts: ignored local directories under `data/eval/runs/week3_v2_standardized_schema_probe_20260722_*`

## DATA-20260722-007

- date: 2026-07-22
- model_request: none
- trigger: Human inspection found the pending hygiene/facility replacement images were abstract synthetic diagrams, and the itinerary workflow unnecessarily requested re-entry of existing constraints.
- correction: Paused after-sales annotation; invalidated three early diagram submissions while retaining their audit hashes. Verified all 100 v1 itinerary annotations remain intact; changed the v2 station to collect style only and merge v1 fields server-side. Reconciled four early itinerary submissions to exact v1 non-style fields while preserving their selected styles.
- current_counts: product `200/200/200/200/0`; after-sales `150/150/80/80/0`; itinerary `100/100/100/100/0`
- status: `PARTIAL`; after-sales replacement-image quality and 96 itinerary style supplements remain incomplete.

## DATA-20260724-010

- date: 2026-07-24
- model_request: image generation only; no evaluation inference
- result: Completed all 100 itinerary style supplements. Audit verification found 96 direct payload-hash matches and 4 recorded reconciliation-hash matches, with no unexplained mismatch; one empty style array is evidence-supported and valid.
- after_sales_pilot: Generated eight photorealistic customer-evidence candidates (four hygiene stains and four facility-damage scenes) without text, labels, common card framing, or watermark. Saved under ignored `data/eval/images/after_sales_v2_pilot/` for visual approval only.
- current_counts: product `200/200/200/200/0`; after-sales `150/150/80/80/0`; itinerary `100/100/100/100/0`
- status: `PARTIAL`; pilot images have not entered the manifest or gold labels.

## DATA-20260721-005

- date: 2026-07-21
- git_commit: `06005fa` with dirty Week 3 worktree
- task_type: frozen Week 3 dataset restoration and real-run revalidation
- command: restore the hash-verified after-sales backup; rebuild exclusions; validate both named runs; recompute scores from immutable raw outputs
- dataset_version: `week3_evaluation_v1`
- manifest_hashes: product `cd85ce2926b3c9adee85c95dc166edd3b9905a844d4b9dd8fe76c224e133dd15`; after-sales `e1fdfc1b77db6519b311a6f846f4ff02df336e34661d841c1a5a42c725dc8a6e`; itinerary `584e2725459a88d48925077fe28239c77860f64b039fd410ed9199a0c6909fa8`
- exclusion_hash: `1430478f2af28c63025d017a806c3e8924900a168b39ca756eac8b0d776465c3`
- source_mix: after-sales public Yelp 76 / business synthetic 74
- audit_binding: 150/150 after-sales annotation payload hashes and annotators match retained audit records
- counts: product `200/200/200/200/200`; after-sales `150/150/150/150/150`; itinerary `100/100/100/100/100`, with tested counts bound to `week3_baseline_full_20260721_003`
- run_validation: baseline and standardized completed runs both `status=ok`, 450 selected and 450 persisted
- baseline_metrics: JSON/Schema 0% in all scenarios; semantic task metrics `PENDING` with support 0; mean latency product/after-sales/itinerary 3463/2139/3508 ms
- standardized_metrics: JSON 68.5%/98.0%/28.0%; Schema 29.5%/2.0%/0.0%; known-gold scalar support product category 110, price 100, after-sales issue/severity 82
- limitations: frozen product category unknown 90 and price unknown 100 are valid no-direct-evidence labels rather than automatic omissions; product `visible_facilities` is non-empty for 128 and empty for 72; after-sales issue unknown 68 and facility-damage gold 0; itinerary style preferences empty 100, with audit/backup evidence indicating a probable historical UI field exposure or serialization defect
- itinerary_style_ui_diagnostic: all 100 current payload hashes match their audits and every retained backup already has an empty style array; all submissions accepted the same five deterministic suggestion fields but not `style_preferences`; the final recoverable 15-option vocabulary was compiled after annotation, while the contemporaneous frontend assets were not retained, so absence from the page cannot be proven conclusively
- project_control_final_route: freeze v1; do not create `week3_gold_v2`, repair/reopen the annotation UI, request supplemental annotation, or perform v2 rescoring; keep itinerary style, after-sales facility-damage, and baseline natural-language semantic metrics `PENDING`
- status: `PARTIAL`; no relabeling, new human work, repeat live inference, commit, push, or promotion performed

## EXP-20260724-011

- date: 2026-07-24
- git_commit: `2a76cd6` plus dirty Week 3 v2 worktree
- dataset_version: `week3_evaluation_v2`
- model_name: `Qwen/Qwen2-VL-2B-Instruct`
- inference_backend: vLLM 0.8.5 OpenAI-compatible HTTP on `localhost:8001`
- generation: temperature=0.1; top_p=0.9; repetition_penalty=1.05; max_tokens=1280
- baseline_run_id: `week3_v2_baseline_full_20260724_001`
- baseline_prompt: `baseline_minimal_v1`
- standardized_run_id: `week3_v2_standardized_full_20260724_001`
- standardized_prompt: `standardized_v2`
- comparison_id: `week3_v2_prompt_pair_20260724_001`
- counts: product `200/200/200/200/200`; after-sales `150/150/150/150/150`; itinerary `100/100/100/100/100`; exclusion count 450
- run_identity: both runs are completed/live/full, selected_count=record_count=450, request errors=0, selected-sample SHA-256 `3e900e64bb345df35343c8f14bfb1f8310ae597a57e4a4d9585bc01173ad648c`
- baseline_metrics: JSON/Schema 0% in every scenario; mean latency product/after-sales/itinerary 4456.84/3616.71/7136.77 ms; natural-language semantic metrics `PENDING`
- standardized_metrics: JSON product/after-sales/itinerary 79.00%/96.67%/90.00%; Schema 75.00%/96.00%/88.00%; product category accuracy 60.00% support 110 and price accuracy 17.00% support 100; after-sales issue accuracy 71.33%, severity accuracy 29.33%, OCR recall 1.33% support 75; itinerary constraint recognition 0.41% and element completeness 20.00%
- error_summary: baseline has 450 expected JSON parse failures; standardized has 57 JSON parse failures and 11 Schema validation failures
- comparison: 450 paired rows, identical sample order, 2,000 bootstrap iterations; format/Schema/latency only because baseline semantic output is not deterministically parsed
- artifacts: `data/eval/runs/week3_v2_baseline_full_20260724_001/`; `data/eval/runs/week3_v2_standardized_full_20260724_001/`; `data/eval/scores/week3_v2_baseline_full_20260724_001/`; `data/eval/scores/week3_v2_standardized_full_20260724_001/`; `data/eval/comparisons/week3_v2_prompt_pair_20260724_001/`
- status: `PARTIAL`; data, live runs, scoring, and comparison complete; baseline native semantic metrics remain unsupported and are not reported as zero

## EXP-20260725-012

- date: 2026-07-25
- git_baseline: `04fc0be` on `dev`
- task_type: deterministic lexical coding of the existing minimal-baseline natural-language output
- dataset_version: `week3_evaluation_v2`
- source_run_id: `week3_v2_baseline_full_20260724_001`
- score_id: `week3_v2_baseline_full_20260724_001__baseline_semantic_coding_v1`
- coding_version: `baseline_semantic_coding_v1`
- codebook_sha256: `563dc0747f92b6ccaa37466045cb0e74229787824013d59a5f6f26261bb033a6`
- prediction_boundary: encoder inputs are scenario, raw output, fixed codebook, and normalization only; manifests and human gold load after all predictions are complete
- counts: 450 sample scores; product/after-sales/itinerary 200/150/100
- product_metrics: category accuracy 45.45% support 110; price accuracy 2.00% support 100; style macro/micro F1 28.28%/16.53% support 200; facility macro/micro F1 53.22%/46.41% support 200; label completeness 23.50% support 169
- after_sales_metrics: issue accuracy 60.00% support 150; severity accuracy 0.00% support 150; key-information macro/micro F1 29.67%/21.02% support 150; OCR recall 14.22% support 75
- itinerary_metrics: constraint recognition 0.00% support 100; hard/soft constraint macro F1 0.00%/0.00% support 100; itinerary-element completeness 77.20% support 100; element macro/micro F1 71.09%/71.81% support 100
- format_invariant: baseline JSON/Schema compliance remains 0%/0% in every scenario; raw output, latency, Prompt version, manifests, Schemas, and run directory are unchanged
- quarantine: prior gold-leaking score and incompatible semantic comparison retained unchanged under ignored `data/eval/quarantine/` with hashes and `REJECTED_OUT_OF_SCOPE`
- verification: 226/226 unit tests; standalone and both run-bound v2 validators `status=ok`; 450-row score integrity check passed with 200/150/100 scenario counts and unchanged 0% JSON/Schema rates
- limitation: deterministic lexical coding is not manual free-form semantic review or native structured model output and is not compared causally with the standardized strict track
- status: `READY / COMPLETED`

## EXP-20260725-013

- 日期：2026-07-25
- Git 基线：`dev` 上的 `77b97b8`
- 任务：Week 4 固定 Prompt pilot 与胜出版本选择
- 数据集：`week3_evaluation_v2`
- 模型/后端：`Qwen/Qwen2-VL-2B-Instruct`；vLLM OpenAI-compatible HTTP，
  `localhost:8001`
- 生成参数：temperature=0.1；top_p=0.9；repetition_penalty=1.05；
  max_tokens=1280
- 有效 Pilot：`week4_pilot_standardized_v2_20260725_001`、
  `week4_pilot_fewshot4_v2_20260725_001`、
  `week4_pilot_fewshot7_v2_20260725_001`
- 示例：每场景固定 5 个正例和 2 个边界例；4-shot 使用 3+1，
  7-shot 使用 5+2；pilot 含 5 个不重叠样本
- 选择权重：business 0.55；Schema 0.20；JSON 0.10；token 0.075；
  latency 0.075
- 胜出版本：商品、售后、行程均为 `standardized_v2`，有效分数分别为
  0.3280、0.5967、0.4775
- Pilot 修复：旧 v1 行程候选因上下文超限返回 HTTP 400；v2 压缩重复
  上下文后，4-shot/7-shot 各 15 条均完成且请求错误为 0
- 全量运行：`week4_winners_full_20260725_001`
- 全量结构化 JSON/Schema：商品 77.5%/75.5%；售后
  96.67%/96.67%；行程 90.0%/87.0%
- Baseline 比较：词法业务轨道与结构化业务轨道不可直接比较，不计算业务
  差值；baseline token 未记录，状态为 `PENDING_not_recorded`
- Bad case：分类 86；约束遗漏 100；字段/Schema 7；格式 67；
  严重等级 105
- 状态：完成 450/450；样本哈希
  `3e900e64bb345df35343c8f14bfb1f8310ae597a57e4a4d9585bc01173ad648c`

## EXP-20260725-015

- 日期：2026-07-25
- 任务：修复 Week 4 行程 Few-Shot 有效性和 baseline 比较口径
- Git 状态：`dev` 上未提交修复工作树
- 模型/后端：`Qwen/Qwen2-VL-2B-Instruct`、vLLM
- 数据集：不可变 `week3_evaluation_v2`
- 生成参数：temperature=0.1；top_p=0.9；repetition_penalty=1.05；
  max_tokens=1280
- 根因：v1 的 4-shot 请求为 3011+1280=4291 tokens，7-shot 为
  4369+1280=5649 tokens，超过模型 4096-token 上限
- 修复：保留 4/7 个真实示例及核心人工金标字段，删除示例中重复长行程与
  完整 Schema 文本；最终输出仍按现有 Schema 校验
- 实测：4-shot v2、7-shot v2 均完成 15/15，
  `model_request_error_count=0`
- 结果：新增候选未超过 `standardized_v2`；三个场景继续使用控制组，
  不称为新 Prompt 优化胜出
- 比较口径：删除跨轨道业务差值；同口径记录 JSON、Schema、平均/P95
  延迟；baseline token 明确为 `PENDING_not_recorded`

## INFRA-20260725-014

- 日期：2026-07-25
- 任务：有界本地 Milvus standalone 部署
- 镜像：Milvus 2.6.20；etcd 3.5.18；
  MinIO RELEASE.2024-12-18T13-15-44Z
- SDK：独立 `requirements-milvus.txt` 中的 PyMilvus 2.6.16
- 健康状态：standalone、etcd、MinIO 均 healthy
- 集合：`ota_business_image_vector`，固定十字段，关闭动态字段
- 索引：HNSW/COSINE `M=16`、`efConstruction=128`、查询 `ef=64`；
  8 个 INVERTED 标量索引
- 向量：20 张真实 Yelp 图片，CUDA 生成并归一化的 512 维
  `openai/clip-vit-base-patch32`
- CRUD：批量 19；单条 1；过滤命中 1；删除 1；删除后命中 0；
  剩余可见行 19
- 性能：HNSW 构建 5.6621 s；K=5 查询 10 次；平均/P95
  7.7982/10.7236 ms；Recall@5 1.0000
- 环境：Windows 11 10.0.26200；Python 3.13.13；
  Intel64 Family 6 Model 183
- 审查修复：凭据改为本机环境变量并完成轮换；基准要求空集合，
  实际计数来自 Milvus `count(*)`
- 状态：完成；未使用生成或随机向量替代
