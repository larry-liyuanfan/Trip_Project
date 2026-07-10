# Week 2 Stable Delivery Note

## Scope

Week 2 delivers a reproducible Yelp multimodal data processing pipeline. The work is limited to dependency cleanup, data parsing, image validation, multimodal alignment generation, output validation, report generation, and documentation.

## Delivered Artifacts

- Dependency split: `requirements-api.txt`, `requirements-data.txt`, `requirements-llm.txt`, `requirements-clip.txt`, and aggregate `requirements.txt`.
- Data config: `configs/data_processing.yaml`.
- Pipeline scripts: `scripts/parse_yelp_json.py`, `scripts/build_yelp_alignment.py`, `scripts/run_clip_denoising.py`, `scripts/generate_yelp_report.py`, and `scripts/validate_week2_pipeline.py`.
- Archive extraction support for 5 core Yelp JSON files, photo metadata, full photo files, and official documentation/ToS PDFs.
- Reusable modules under `src/data/`.
- Tests in `tests/test_yelp_data_pipeline.py`.
- Mentor-facing report: `reports/yelp_multimodal_data_processing_report_part1.md`.

## Verification Snapshot

- Unit tests: `python -m unittest discover -s tests -v`.
- Pipeline validation: `python scripts/validate_week2_pipeline.py --config configs/data_processing.yaml`.
- Full-run counts: 150,346 businesses, 6,989,830 valid reviews, 200,100 photo metadata rows, 199,994 valid local images, 1,416 covered cities, 96,733 strong non-empty-caption pairs, 199,994 medium pairs, and 36,673 weak groups.
- CLIP run: dedicated CUDA Docker task using `openai/clip-vit-base-patch32` on RTX 4070. It scored 555,459 bounded image-review candidates and retained 131,146 pairs at threshold 0.25.

## Known Limits

- Full review parsing uses chunked table writes with `processing_limits.max_reviews: null`.
- Business/photo/image-index outputs also use chunked writes, with stable nested-field serialization and bounded parallel image validation.
- Local image validation found 106 corrupted/unreadable files; these are excluded from alignment outputs.
- CLIP runs as `docker compose -f docker/docker-compose.yml --profile data run --rm clip-denoising`; vLLM must be stopped first to release the local 8GB GPU. The output includes row-level similarity scores, model/device metadata, and a distribution summary.
- LLM dependencies are separated and should be installed through Docker or WSL2 when needed, not by default in native Windows Python.
