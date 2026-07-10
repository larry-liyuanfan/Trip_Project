# Weekly Log

## 2026-07-10: Week 2 Yelp Multimodal Dataset Pipeline

- Added a config-driven Yelp processing pipeline under `configs/data_processing.yaml`.
- Added line-by-line JSONL reading for business, review, and photo metadata.
- Added chunked table writing for business, review, photo, and image-index outputs; nested business fields are serialized to a stable Parquet schema.
- Added bounded parallel image validation for missing, valid, and unreadable local images.
- Added strong, medium, and weak alignment builders; strong pairs require both a valid image and non-empty caption.
- Added optional CLIP denoising interface that writes skipped status when disabled or unavailable.
- Added report generation for `reports/yelp_multimodal_data_processing_report_part1.md`.
- Added focused unit tests in `tests/test_yelp_data_pipeline.py`.
- Split dependencies so Week 2 data processing uses `requirements-data.txt` and does not require native Windows vLLM installation.
- Extended Yelp archive extraction to cover the 5 core JSON files, `photos.json`, full photo extraction, and official documentation/ToS files under `data/yelp/raw/docs/`.

Verification commands:

```bash
python -m unittest discover -s tests -v
python scripts/parse_yelp_json.py --config configs/data_processing.yaml
python scripts/build_yelp_alignment.py --config configs/data_processing.yaml
docker compose -f docker/docker-compose.yml stop vllm
docker compose -f docker/docker-compose.yml --profile data run --rm clip-denoising
python scripts/generate_yelp_report.py --config configs/data_processing.yaml
python scripts/validate_week2_pipeline.py --config configs/data_processing.yaml
```

Verification results on 2026-07-09:

- Unit tests: 43 tests passed.
- Parse command: 150346 businesses, 6989830 valid reviews, 200100 photo metadata rows, 199994 valid local images.
- Image validation: 0 missing local images, 106 corrupted/unreadable images.
- Alignment command: 96733 strong caption pairs, 199994 medium pairs, 36673 weak business-level groups.
- Data quality statistics: city count, valid image ratio, label distribution, caption length statistics, weak-alignment category coverage, and denoising before/after counts are included in dataset statistics and the report.
- CLIP denoising: completed in a dedicated CUDA Docker task with `openai/clip-vit-base-patch32`; 555,459 candidates scored and 131,146 retained at threshold 0.25.
- Report generation: wrote 10 sections to `reports/yelp_multimodal_data_processing_report_part1.md`.
- Output validation: all expected files, required columns, alignment image paths, report counts, and storage format checks passed.
- Local storage note: `pyarrow` is available in the current environment, so real Parquet files were written.
- Scale note: the current implementation is verified on the full Yelp business/review/photo metadata files and the fully extracted local photo folder.
