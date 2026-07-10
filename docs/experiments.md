# Experiment Notes

## Week 2 Data Processing Baseline

- Dataset source: local Yelp Open Dataset files under `data/yelp/raw/`.
- Config: `configs/data_processing.yaml`.
- Base pipeline does not require GPU, CLIP, live vLLM, or `requirements-llm.txt`.
- Week 2 data-only dependency install: `pip install -r requirements-data.txt`.
- CLIP denoising runs through the dedicated `clip-denoising` Docker profile and records a model, device, candidate, retention, and similarity summary in `data/yelp/processed/clip_denoising_summary.json`.
- Generated large outputs remain under ignored `data/yelp/` directories.

Record future full-dataset runs with command, Git commit, raw file availability, output counts, runtime, and any skipped dependency notes.

### Full local Week 2 run on 2026-07-10

- Command sequence: parse, alignment, CLIP denoising, report generation with `configs/data_processing.yaml`.
- Review processing limit: none.
- Parsed rows: 150346 businesses, 6989830 valid reviews from 6990280 raw review lines, 200100 photo metadata records.
- Raw extraction: 5 core JSON files, `photos.json`, `photos/`, and official documentation/ToS files are present under `data/yelp/raw/`.
- Image validation: 199994 valid images, 0 missing images, 106 corrupted/unreadable images.
- Alignment rows: 96733 strong non-empty-caption image pairs, 199994 medium image-business pairs, 36673 weak business-level groups.
- Data quality statistics: city count, valid image ratio, photo label distribution, caption length statistics, weak-alignment category coverage, and denoising before/after counts are recorded in `dataset_statistics.json` and the report.
- CLIP denoising: completed on 2026-07-10 with `openai/clip-vit-base-patch32` on CUDA (RTX 4070 Laptop GPU). Input: 36,673 weak groups and 555,459 candidates; retained: 131,146 at threshold 0.25; similarity min/mean/max: 0.0226 / 0.2210 / 0.4199.
- Storage behavior: real Parquet files were written because a pyarrow Parquet engine is available locally.
- Output validation: `scripts/validate_week2_pipeline.py` confirmed expected files, columns, image paths, report counts, and Parquet format.
- Full-run robustness: business nested attributes/hours use stable JSON storage for chunk-compatible Parquet schemas; review statistics use running sums/counts; image validation runs in bounded parallel batches.
