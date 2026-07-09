# Experiment Notes

## Week 2 Data Processing Baseline

- Dataset source: local Yelp Open Dataset files under `data/yelp/raw/`.
- Config: `configs/data_processing.yaml`.
- Base pipeline does not require GPU, CLIP, live vLLM, or `requirements-llm.txt`.
- Week 2 data-only dependency install: `pip install -r requirements-data.txt`.
- CLIP denoising is disabled by default and records a skipped status in `data/yelp/processed/clip_denoising_summary.json`.
- Generated large outputs remain under ignored `data/yelp/` directories.

Record future full-dataset runs with command, Git commit, raw file availability, output counts, runtime, and any skipped dependency notes.

### Local smoke run on 2026-07-09

- Command sequence: parse, alignment, CLIP denoising, report generation with `configs/data_processing.yaml`.
- Review processing limit: 10000 reviews.
- Parsed rows: 150346 businesses, 10000 reviews, 200100 photo metadata records.
- Image validation: 581 valid images, 199519 missing images, 0 corrupted images.
- Alignment rows: 581 strong image-caption pairs, 581 medium image-business pairs, 38 weak business-level groups.
- Data quality statistics: valid image ratio, photo label distribution, caption length statistics, and denoising before/after counts are recorded in `dataset_statistics.json` and the report.
- CLIP denoising: skipped by config.
- Storage behavior: real Parquet files were written because a pandas Parquet engine is available locally.
- Output validation: `scripts/validate_week2_pipeline.py` confirmed expected files, columns, image paths, report counts, and Parquet format.
- Scale limitation: the smoke run uses a 10,000-review cap; full review parsing should use chunked writes or another bounded-memory output strategy before setting `processing_limits.max_reviews` to `null`.
