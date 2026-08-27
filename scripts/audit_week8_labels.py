"""Rebuild all 60 historical development references as isolated diagnostic silver."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.rebuild_week8_product_labels import rebuild
from scripts.review_week8_product import load_review_inputs
from src.data.product_labels import LABEL_PROTOCOL
from src.training.week7_data import iter_jsonl, sha256_file
from src.training.week8_product import select_prompt, summarize_product_run, _write_json_new


def _write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _matched_parquet(path, photo_ids):
    import pyarrow.parquet as pq
    result = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            if row["photo_id"] in photo_ids:
                if row["photo_id"] in result:
                    raise ValueError("duplicate source photo identity")
                result[row["photo_id"]] = row
    return result


def _support(rows):
    return {"samples": len(rows), "business_category": sum(row["target"]["business_category"] != "unknown" for row in rows),
            "style": sum(bool(row["target"]["style_tags"]) for row in rows),
            "facility": sum(bool(row["target"]["visible_facilities"]) for row in rows),
            "price": sum(row["target"]["price_range"] != "unknown" for row in rows),
            "parking": sum("parking" in row["target"]["visible_facilities"] for row in rows)}


def run(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["final_test_labels_access"] is not False or config["label_protocol"] != LABEL_PROTOCOL:
        raise ValueError("invalid audit policy")
    _, product, old, validation, development_sha = load_review_inputs(ROOT, ROOT / config["source_review_config"])
    output = ROOT / config["output_root"] / "labels"
    output.mkdir(parents=True, exist_ok=False)
    source_paths = product["dataset"]["source_paths"]
    photos = {row["source_id"].removeprefix("yelp-photo:") for row in old}
    captions = _matched_parquet(ROOT / source_paths["strong_pairs"], photos)
    merchant = _matched_parquet(ROOT / source_paths["medium_pairs"], photos)
    if set(captions) != photos:
        raise ValueError("original captions do not cover all development rows")
    sources = []
    for row in old:
        photo = row["source_id"].removeprefix("yelp-photo:")
        if row["group_id"] != "yelp-business:" + captions[photo]["business_id"]:
            raise ValueError("caption/business identity mismatch")
        sources.append({**row, "caption": captions[photo]["caption"],
                        "business_description": merchant.get(photo, {}).get("business_description", "")})
    source = output / "original_caption_sources.jsonl"
    _write_rows(source, sources)
    lock_root = ROOT / product["dataset"]["output_root"] / product["week8"]["dataset_version"]
    exclusions = output / "exclusion_identities.jsonl"
    _write_rows(exclusions, (row for row in iter_jsonl(lock_root / "identity_manifest.jsonl") if row["split"] != "development"))
    manifest = rebuild(source, output / LABEL_PROTOCOL, exclusions)
    repaired = list(iter_jsonl(output / LABEL_PROTOCOL / "diagnostic_silver.jsonl"))
    # 复算旧 development 生成结果，不调用模型、不选择新 Prompt、不读取 final test。
    before, after, raw_hashes = {}, {}, {}
    for role in product["prompts"]:
        raw_path = ROOT / config["source_prompt_outputs"] / role / "raw_outputs.jsonl"
        records = list(iter_jsonl(raw_path))
        raw_hashes[role] = sha256_file(raw_path)
        before[role] = summarize_product_run(ROOT, old, records)
        after[role] = summarize_product_run(ROOT, repaired, records)
    summary = {"status": "DIAGNOSTIC_ONLY", "manifest": manifest,
               "dataset_validation": validation, "development_source_sha256": development_sha,
               "source_parquet_sha256": {key: sha256_file(ROOT / source_paths[key]) for key in ("strong_pairs", "medium_pairs")},
               "raw_output_sha256": raw_hashes, "old_support": _support(old), "new_support": _support(repaired),
               "old_slices": dict(Counter(slice_name for row in old for slice_name in row.get("error_slices", []))),
               "new_slices": dict(Counter(slice_name for row in repaired for slice_name in row.get("error_slices", []))),
               "before": before, "after": after, "selection_before": select_prompt(product, before),
               "selection_after": select_prompt(product, after), "model_calls": 0, "final_test_labels_read": False,
               "interpretation": "Same outputs, corrected silver references; score changes are NOT model improvements."}
    _write_json_new(output / "summary.json", summary)
    return {"summary_sha256": sha256_file(output / "summary.json"), "old_support": summary["old_support"],
            "new_support": summary["new_support"], "selection_after": summary["selection_after"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week8/audit_repair_v1.json")
    print(json.dumps(run(parser.parse_args().config), ensure_ascii=False))
