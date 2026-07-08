# Experiment Log

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
- git_commit: e099de5
- model_name: not_applicable_data_preparation
- model_size: not_applicable_data_preparation
- inference_backend: not_applicable_data_preparation
- serving_command: not_applicable_data_preparation
- prompt_version: not_applicable_data_preparation
- temperature: not_applicable_data_preparation
- top_p: not_applicable_data_preparation
- max_tokens: not_applicable_data_preparation
- input_type: yelp_json_reviews_photo_metadata
- dataset_version: yelp_ota_subset_v1
- task_type: data_preparation
- metrics: business_count=200; review_count=1000; multimodal_item_count=581; extracted_photo_count=581
- result_summary: Local Yelp JSON and Photos zip archives were extracted into ignored raw data, converted into a bounded OTA subset, and materialized only the photos referenced by the multimodal subset.
- failure_cases: The official zip files contain gzip-compressed tar payloads with .tar filenames and macOS resource entries; the extractor skips resource entries and reads the gzip tar stream directly.
- screenshots_or_samples: data/yelp/processed/ota_subset_v1/manifest.json; data/yelp/raw/extract_photo_manifest.json
- next_action: Use this processed subset for current Week 1 data validation only.
