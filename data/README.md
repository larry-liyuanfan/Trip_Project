# Data

This project starts with a small local sample dataset to validate the pipeline.

Planned data sources:

- `data/samples/`: manually curated examples for Week 1-2.
- `data/yelp/raw/`: local Yelp Open Dataset downloads from the official source.
- `data/yelp/processed/`: generated OTA subsets from `scripts/prepare_yelp_subset.py`.
- `data/raw/`: ignored local raw downloads.

Do not commit large raw datasets or model weights.

See `docs/yelp_dataset.md` for the end-to-end Yelp preparation workflow.
