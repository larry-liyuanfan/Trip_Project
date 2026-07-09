# Weekly Log

## 2026-07-09: Week 2 Yelp Multimodal Dataset Pipeline

- Added a config-driven Yelp processing pipeline under `configs/data_processing.yaml`.
- Added line-by-line JSONL reading for business, review, and photo metadata in the capped smoke-run pipeline.
- Added image validation for missing, valid, and unreadable local images.
- Added strong, medium, and weak alignment builders with bounded weak grouping.
- Added optional CLIP denoising interface that writes skipped status when disabled or unavailable.
- Added report generation for `reports/yelp_multimodal_data_processing_report_part1.md`.
- Added focused unit tests in `tests/test_yelp_data_pipeline.py`.
- Split dependencies so Week 2 data processing uses `requirements-data.txt` and does not require native Windows vLLM installation.
- Treated Yelp download/archive extraction as a Week 1 prerequisite; Week 2 consumes normalized files under `data/yelp/raw/`.

Verification commands:

```bash
python -m unittest discover -s tests -v
python scripts/parse_yelp_json.py --config configs/data_processing.yaml
python scripts/build_yelp_alignment.py --config configs/data_processing.yaml
python scripts/run_clip_denoising.py --config configs/data_processing.yaml
python scripts/generate_yelp_report.py --config configs/data_processing.yaml
python scripts/validate_week2_pipeline.py --config configs/data_processing.yaml
```

Verification results on 2026-07-09:

- Unit tests: 24 tests passed.
- Parse command: 150346 businesses, 10000 capped reviews, 200100 photo metadata rows, 581 valid local images.
- Alignment command: 581 strong pairs, 581 medium pairs, 38 weak business-level groups.
- Data quality statistics: valid image ratio, label distribution, caption length statistics, and denoising before/after counts are included in dataset statistics and the report.
- CLIP denoising: skipped because `clip_denoising.enabled` is false.
- Report generation: wrote 10 sections to `reports/yelp_multimodal_data_processing_report_part1.md`.
- Output validation: all expected files, required columns, alignment image paths, report counts, and storage format checks passed.
- Local storage note: `pyarrow` is available in the current environment, so real Parquet files were written.
- Scale note: the current implementation is verified for the capped smoke run; full review parsing should add chunked writes before removing the review cap.
