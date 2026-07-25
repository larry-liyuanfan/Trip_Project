"""Encode a bounded real Yelp OTA image sample with the verified CLIP model."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.yelp_paths import parse_simple_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/milvus_week4.yaml")
    args = parser.parse_args()
    config = parse_simple_yaml(Path(args.config).read_text(encoding="utf-8"))
    benchmark = config["benchmark"]
    count = benchmark["vector_count"]
    import numpy as np
    import pyarrow.parquet as pq
    from src.data.clip_denoising import _TransformersClipScorer

    print(f"selecting {count} real images from bounded parquet batches", flush=True)
    selected = []
    pair_file = pq.ParquetFile(benchmark["source_table"])
    for batch in pair_file.iter_batches(
        batch_size=max(100, count * 5),
        columns=["photo_id", "business_id", "image_path"],
    ):
        for row in batch.to_pylist():
            if (Path.cwd() / row["image_path"]).is_file():
                selected.append(row)
                if len(selected) == count:
                    break
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"expected {count} readable images, found {len(selected)}")

    selected_business_ids = sorted({row["business_id"] for row in selected})
    business_table = pq.read_table(
        "data/yelp/interim/business.parquet",
        columns=[
            "business_id",
            "categories",
            "city",
            "stars",
            "attr_RestaurantsPriceRange2",
        ],
        filters=[("business_id", "in", selected_business_ids)],
    )
    business_by_id = {
        row["business_id"]: row for row in business_table.to_pylist()
    }
    model_name = config["collection"]["embedding_model"]
    print(
        f"loading {model_name} for {len(selected)} real images on cuda",
        flush=True,
    )
    scorer = _TransformersClipScorer(
        {
            "model_id": model_name,
            "device": "cuda",
            "image_batch_size": 20,
            "text_batch_size": 1,
        }
    )
    print(f"loaded {model_name}; encoding images", flush=True)
    image_paths = [row["image_path"] for row in selected]
    encoded = scorer._image_embeddings(image_paths)
    missing = [path for path in image_paths if path not in encoded]
    if missing:
        raise RuntimeError(f"CLIP failed to encode {len(missing)} selected images")
    matrix = np.stack(
        [encoded[path].numpy().astype("float32") for path in image_paths],
        axis=0,
    )
    print(f"encoded {len(selected)}/{len(selected)}", flush=True)

    metadata = []
    for row in selected:
        business = business_by_id[row["business_id"]]
        metadata.append(
            {
                "business_id": row["business_id"],
                "image_id": row["photo_id"],
                "business_category": _business_category(business["categories"]),
                "city": business["city"] or "unknown",
                "star_rating": float(business["stars"]),
                "price_range": _price_range(
                    business["attr_RestaurantsPriceRange2"]
                ),
                "image_type": "business_photo",
                "embedding_model": model_name,
                "source_image_path": row["image_path"],
            }
        )
    vectors_path = Path(benchmark["vectors_path"])
    metadata_path = Path(benchmark["metadata_path"])
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(vectors_path, multimodal_vector=matrix)
    with metadata_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "vector_count": len(metadata),
                "dimension": int(matrix.shape[1]),
                "device": scorer.device,
                "model": model_name,
                "vectors_path": str(vectors_path),
                "metadata_path": str(metadata_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _business_category(categories) -> str:
    values = {str(value).lower() for value in (categories or [])}
    if any("hotel" in value or "lodging" in value for value in values):
        return "hotel"
    if any(
        word in value
        for value in values
        for word in ("attraction", "museum", "park", "landmark", "arts")
    ):
        return "attraction"
    if any(
        word in value
        for value in values
        for word in ("restaurant", "food", "cafe", "bar")
    ):
        return "restaurant"
    return "other"


def _price_range(value) -> str:
    mapping = {1: "budget", 2: "mid_range", 3: "premium", 4: "luxury"}
    try:
        return mapping.get(int(value), "unknown")
    except (TypeError, ValueError):
        return "unknown"


if __name__ == "__main__":
    main()
