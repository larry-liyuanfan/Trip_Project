# Yelp Open Dataset Integration

## Source

Download Yelp Open Dataset from the official Yelp page:

- https://www.yelp.com/dataset/download
- https://github.com/Yelp/dataset-examples

The raw dataset is not committed to this repository. Keep local downloads under:

```text
data/yelp/raw/
```

Expected raw files:

```text
data/yelp/raw/yelp_academic_dataset_business.json
data/yelp/raw/yelp_academic_dataset_review.json
data/yelp/raw/photos.json
data/yelp/raw/photos/<photo_id>.jpg
```

`photos.json` and `photos/` are optional for text-only experiments, but required for multimodal image metadata.

## Prepare OTA Subset

Run:

```bash
python scripts/prepare_yelp_subset.py \
  --raw-dir data/yelp/raw \
  --output-dir data/yelp/processed/ota_subset_v1 \
  --max-businesses 200 \
  --max-reviews-per-business 5
```

Outputs:

```text
data/yelp/processed/ota_subset_v1/poi_catalog.jsonl
data/yelp/processed/ota_subset_v1/reviews.jsonl
data/yelp/processed/ota_subset_v1/multimodal_items.jsonl
data/yelp/processed/ota_subset_v1/manifest.json
```

## Output Schemas

`poi_catalog.jsonl` maps Yelp businesses into a reviewed POI catalog format:

```json
{
  "poi_id": "yelp_<business_id>",
  "source": "yelp_open_dataset",
  "business_id": "<business_id>",
  "name": "Sample Cafe",
  "category": "Cafe",
  "city": "Shanghai",
  "state": "SH",
  "rating": 4.5,
  "review_count": 20,
  "tags": ["cafes", "restaurants", "coffee & tea"],
  "description": "Sample Cafe Cafe Shanghai SH"
}
```

`reviews.jsonl` keeps a bounded number of reviews per selected POI:

```json
{
  "review_id": "<review_id>",
  "poi_id": "yelp_<business_id>",
  "business_id": "<business_id>",
  "rating": 5,
  "text": "Quiet cafe near the museum.",
  "date": "2024-01-01",
  "source": "yelp_open_dataset"
}
```

`multimodal_items.jsonl` links Yelp photo metadata to local image paths:

```json
{
  "item_id": "yelp_photo_<photo_id>",
  "poi_id": "yelp_<business_id>",
  "business_id": "<business_id>",
  "photo_id": "<photo_id>",
  "image_path": "data/yelp/raw/photos/<photo_id>.jpg",
  "caption": "latte and window seat",
  "label": "inside",
  "source": "yelp_open_dataset"
}
```

## Usage In This Project

- Catalog review: inspect `poi_catalog.jsonl` to confirm OTA-relevant business filtering.
- Review text preparation: inspect `reviews.jsonl` as bounded review text attached to selected POIs.
- Multimodal metadata preparation: use `multimodal_items.jsonl` to pair local Yelp photo files with POI metadata.
- Experiment tracking: record output directory, script arguments, Git commit, and metrics in `experiments/`.

## Notes

- The filter keeps OTA-relevant Yelp categories such as restaurants, cafes, hotels, attractions, parks, museums, shopping, nightlife, and local flavor.
- Raw Yelp files and processed large subsets stay under `data/yelp/`, which is ignored by Git.
- Commit only small curated samples under `data/samples/` when needed for smoke tests.
