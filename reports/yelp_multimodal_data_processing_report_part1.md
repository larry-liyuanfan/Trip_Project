# Week 2 Yelp Multimodal Data Processing Report

## 1. Week 2 objective
Build a reproducible Yelp multimodal dataset pipeline that parses raw JSONL files, validates local images, creates strong/medium/weak image-text alignments, and produces a report-ready dataset summary.

## 2. Raw dataset files used
- Business JSONL: `data/yelp/raw/yelp_academic_dataset_business.json`
- Review JSONL: `data/yelp/raw/yelp_academic_dataset_review.json`
- Photo metadata JSONL: `data/yelp/raw/photos.json`
- Local image root: `data/yelp/raw/photos`
- Review cap for this smoke run: 10000
- Download and archive extraction status: completed before Week 2; this pipeline consumes the normalized files under `data/yelp/raw/`.

## 3. Parsing pipeline overview
- `scripts/parse_yelp_json.py` reads business, review, and photo JSONL inputs line by line, then writes interim tables for the configured smoke-run scope.
- Review parsing rejects empty, symbol-only, and too-short text according to `configs/data_processing.yaml`.
- Image validation checks local file existence and readability with Pillow, then records width and height.
- `scripts/build_yelp_alignment.py` creates processed alignment datasets and statistics.

## 4. Extracted business/review/photo fields
- Business fields: `business_id`, name, location, coordinates, stars, review count, categories, attributes, hours, and selected flattened attributes.
- Review fields: `review_id`, `business_id`, user id, stars, useful/funny/cool counts, cleaned text, and date.
- Photo fields: `photo_id`, `business_id`, caption, label, and local `image_path`.
- Image index fields: `photo_id`, `business_id`, `image_path`, validity flag, width, height, and validation error.

## 5. Local image validation result
- Photo metadata entries parsed: 200100
- Valid local images: 581
- Missing local images: 199519
- Corrupted local images: 0

## 6. Multimodal alignment strategy
- Strong alignment: 581 valid image-caption-label pairs keyed by `photo_id`.
- Medium alignment: 581 valid image-business metadata pairs with generated business descriptions.
- Weak alignment: 38 business-level groups containing bounded image lists and selected review texts.
- CLIP denoising: skipped (clip_denoising.enabled is false).

## 7. Output statistics
- Businesses parsed: 150346
- Reviews parsed: 10000
- Photo metadata entries parsed: 200100
- Valid local images: 581
- Strong pairs: 581
- Medium pairs: 581
- Weak groups: 38
- Businesses with capped reviews: 3930
- Businesses with valid images: 73
- Valid image ratio: 0.0029035482258870566
- Photo label distribution: {'inside': 56031, 'outside': 18569, 'drink': 15670, 'food': 108152, 'menu': 1678}
- Caption length statistics: {'caption_count': 96734, 'min_chars': 1, 'mean_chars': 31.470579113858623, 'max_chars': 140}
- Denoising before/after weak pairs: 38 -> 0
- Top categories: [['restaurants', 52268], ['food', 27781], ['shopping', 24395], ['home services', 14356], ['beauty & spas', 14292], ['nightlife', 12281], ['health & medical', 11890], ['local services', 11198], ['bars', 11065], ['automotive', 10773], ['event planning & services', 9895], ['sandwiches', 8366], ['american (traditional)', 8139], ['active life', 7687], ['pizza', 7093], ['coffee & tea', 6703], ['fast food', 6472], ['breakfast & brunch', 6239], ['american (new)', 6097], ['hotels & travel', 5857]]

## 8. Data quality issues and limitations
- Storage behavior: Real Parquet files were written with the available pandas Parquet engine.
- CLIP denoising status: skipped (clip_denoising.enabled is false)
- The local image set is partial: most photo metadata rows point to files that are not currently extracted locally.
- The default config caps reviews at 10,000 for fast Windows smoke validation. A full raw Yelp review pass should first add chunked table writing or another bounded-memory output path.
- Full live model serving is intentionally out of scope for Week 2 and should use Docker or WSL2 later.

## 9. Reproducible commands
```bash
pip install -r requirements-data.txt
python scripts/parse_yelp_json.py --config configs/data_processing.yaml
python scripts/build_yelp_alignment.py --config configs/data_processing.yaml
python scripts/run_clip_denoising.py --config configs/data_processing.yaml
python scripts/generate_yelp_report.py --config configs/data_processing.yaml
python scripts/validate_week2_pipeline.py --config configs/data_processing.yaml
python -m unittest discover -s tests -v
```

## 10. Follow-up TODOs
- Add chunked table writes before uncapping full review parsing.
- Replace the current CLIP skip/status interface with real semantic scoring when GPU or suitable CPU runtime is available.
- Keep GPU-heavy dependency paths in Docker or WSL2.
