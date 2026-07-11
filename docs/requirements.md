# Project Requirements

## Week 1: OTA Multimodal VLM Engineering Foundation

### Goals

- Establish the Python, Docker, FastAPI, and vLLM project foundation.
- Support deterministic fallback tests without requiring a live GPU service.
- Verify a live single-image multimodal request on the local GPU.
- Prepare a small Yelp OTA sample and reproducible experiment records.

### Acceptance Criteria

- API health and image-understanding routes return structured responses.
- Docker serves the selected VLM through an OpenAI-compatible endpoint.
- Sample POIs, reviews, and images are available without committing raw Yelp data.
- Unit tests and experiment logs record the verified Week 1 state.

## Week 2: Yelp Multimodal Dataset Processing Pipeline

### Background
Week 1 established the project scaffold and small Yelp subset workflow. Week 2 builds a reproducible pipeline for parsing, validating, aligning, and reporting Yelp multimodal data.

### Goals
- Normalize Yelp raw, interim, processed, logs, and report directories.
- Consume the Week 1 downloaded/extracted Yelp files from the normalized raw directory.
- Read business, review, and photo JSONL files line by line for the full processing pipeline.
- Validate local photo files and record image metadata.
- Build strong valid-image/non-empty-caption pairs, medium image-business pairs, and weak image-review pairs.
- Provide a reproducible CLIP denoising stage that runs in a dedicated GPU Docker task without adding torch or vLLM to the base data environment.
- Generate `reports/yelp_multimodal_data_processing_report_part1.md`.

### Non-goals
- Do not commit raw Yelp archives, extracted images, or large generated Parquet files.
- Do not require GPU or CLIP for the base pipeline.
- Do not build model training or a UI this week.

### Deliverables
- Config-driven parsing and alignment scripts.
- Reusable `src/data/` modules for archive extraction, JSONL parsing, validation, alignment, statistics, templates, and optional denoising.
- Interim Parquet outputs, processed alignment outputs, statistics JSON, and report draft for the full local Yelp run.
- Tests for parsers, validation, alignment, denoising skip behavior, and report generation.

### Acceptance Criteria
- `python -m unittest discover -s tests -v` passes.
- Week 2 data processing can be installed with `pip install -r requirements-data.txt` without installing `vllm`.
- Parsing script writes business, review, photo, image-index, review-stats, and validation-summary outputs.
- Alignment script writes strong, medium, weak, and dataset-statistics outputs.
- Denoising task writes a row-level denoised table, a summary, and similarity distribution; disabled/dependency-unavailable paths still write an explicit skipped summary.
- Report generator writes the Week 2 report using real statistics or explicit `TODO` markers.

### Risks / Questions
- Official Yelp archives may need manual extraction before parsing unless archive extraction is added.
- The Yelp download/extraction step is treated as a completed prerequisite for this Week 2 processing review.
- Full review/photo data is large, so review output uses chunked table writing to keep memory bounded.
- CLIP requires the `clip-denoising` Docker profile and exclusive GPU access. Stop vLLM before full CLIP inference on the local 8GB GPU.
- Local environments without `pyarrow` will run with a CSV fallback at the configured output path until dependencies are installed.
- The default config sets `processing_limits.max_reviews` to `null` and uses chunked review writes for full-dataset parsing.
- vLLM should not be installed in native Windows Python by default; use Docker or WSL2 for live LLM serving dependencies.
