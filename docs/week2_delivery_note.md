# Week 2 Stable Delivery Note

## Scope

Week 2 delivers a reproducible Yelp multimodal data processing pipeline. The work is limited to dependency cleanup, data parsing, image validation, multimodal alignment generation, output validation, report generation, and documentation.

## Delivered Artifacts

- Dependency split: `requirements-api.txt`, `requirements-data.txt`, `requirements-llm.txt`, and aggregate `requirements.txt`.
- Data config: `configs/data_processing.yaml`.
- Pipeline scripts: `scripts/parse_yelp_json.py`, `scripts/build_yelp_alignment.py`, `scripts/run_clip_denoising.py`, `scripts/generate_yelp_report.py`, and `scripts/validate_week2_pipeline.py`.
- Reusable modules under `src/data/`.
- Tests in `tests/test_yelp_data_pipeline.py`.
- Mentor-facing report: `reports/yelp_multimodal_data_processing_report_part1.md`.

## Verification Snapshot

- Unit tests: `python -m unittest discover -s tests -v`.
- Pipeline validation: `python scripts/validate_week2_pipeline.py --config configs/data_processing.yaml`.
- Smoke-run counts: 150,346 businesses, 10,000 capped reviews, 200,100 photo metadata rows, 581 valid local images, 581 strong pairs, 581 medium pairs, and 38 weak groups.

## Known Limits

- The default review cap keeps Windows smoke runs fast. Full review parsing requires setting `processing_limits.max_reviews: null`.
- CLIP denoising is disabled and skipped cleanly.
- LLM dependencies are separated and should be installed through Docker or WSL2 when needed, not by default in native Windows Python.
