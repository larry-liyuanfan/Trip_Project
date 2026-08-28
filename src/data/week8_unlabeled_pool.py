"""Build an identity-only source pool from an existing legal Yelp archive."""
from collections import Counter
import hashlib
import heapq
import json
import os
from pathlib import Path
import re

from PIL import Image

from src.data.week8_product_silver_source_v8 import strict_ota_category
from src.data.week8_visual_holdout import load_history, read_json, within, write_json_new, identity_projection
from src.data.yelp_archives import extract_yelp_photo_files
from src.inference.product_observation import canonical_config_sha256
from src.training.week7_data import IDENTITY_FIELDS, iter_jsonl, sha256_file


PROTOCOL = "week8_unlabeled_archive_pool_v1"
ROW_FIELDS = {*IDENTITY_FIELDS, "image_path"}


def identity_rank(seed, value):
    return hashlib.sha256((seed + "\x1f" + value).encode()).hexdigest()


def parquet_rows(path, columns):
    import pyarrow.parquet as pq
    # 只读取身份或业态范围列，不读取caption、图片标签或商家设施。
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


def choose_source_identities(photo_rows, eligible_groups, consumed, limit, seed):
    """Retain one seeded photo identity per group, independent of any label."""
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError("bounded source extraction count required")
    best, counts = {}, Counter()
    for row in photo_rows:
        counts["photo_rows"] += 1
        photo, group = str(row.get("photo_id") or ""), str(row.get("business_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", photo) or not group:
            counts["invalid_identity"] += 1
            continue
        source = "yelp-photo:" + photo
        if group not in eligible_groups:
            counts["outside_ota_scope"] += 1
        elif group in consumed["group_id"] or source in consumed["source_id"]:
            counts["historical_source_or_group"] += 1
        else:
            key = identity_rank(seed, source)
            if group not in best or key < best[group][0]:
                best[group] = (key, {"source_id": source, "group_id": group, "photo_id": photo})
    selected_groups = heapq.nsmallest(limit, best, key=lambda group: identity_rank(seed, "group:" + group))
    counts["eligible_groups"] = len(best)
    counts["requested_images"] = len(selected_groups)
    return [best[group][1] for group in selected_groups], dict(counts)


def validate_pool_images(root, output, candidates, consumed, namespace):
    accepted, rejected, hashes = [], [], set()
    for item in candidates:
        path = output / "raw/photos" / (item["photo_id"] + ".jpg")
        reason = None
        if not path.is_file():
            reason = "missing_archive_image"
        else:
            digest = sha256_file(path)
            if digest in consumed["image_sha256"]:
                reason = "historical_image_hash"
            elif digest in hashes:
                reason = "duplicate_image_hash"
            else:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        image.load()
                except (OSError, ValueError, SyntaxError):
                    reason = "unreadable_image"
        if reason:
            rejected.append({"source_id": item["source_id"], "group_id": item["group_id"], "reason": reason})
            continue
        hashes.add(digest)
        row = {"sample_id": namespace + ":" + item["photo_id"], "source_id": item["source_id"],
               "group_id": item["group_id"], "image_sha256": digest, "constraint_template_id": None,
               "image_path": path.relative_to(root).as_posix()}
        if any(value in consumed[key] for key, value in identity_projection(row).items() if value):
            raise ValueError("pool sample identity collision")
        accepted.append(row)
    return accepted, rejected


def write_jsonl_new(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_hashes(root):
    paths = ("src/data/week8_unlabeled_pool.py", "src/data/week8_visual_holdout.py",
             "src/data/yelp_archives.py", "src/data/week8_product_silver_source_v8.py")
    return {path: hashlib.sha256((root / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in paths}


def build_pool(root, config_path):
    config = read_json(config_path)
    if (config.get("protocol") != PROTOCOL or config.get("labels_generated") is not False
            or config.get("human_annotation_count") != 0):
        raise ValueError("identity-only pool protocol required")
    output = within(root, config["output_root"])
    if output.exists():
        raise FileExistsError("source pool already exists; preserve partial or completed evidence")
    consumed, history, audit, _ = load_history(root, config["history"])
    for entry in (config["photos"], config["business"]):
        if sha256_file(within(root, entry["path"])) != entry["sha256"]:
            raise ValueError("source table identity changed")
    archive = Path(os.environ[config["archive_environment_variable"]]).resolve()
    if archive.stat().st_size != config["archive_size"] or sha256_file(archive) != config["archive_sha256"]:
        raise ValueError("existing legal photo archive identity changed")
    groups = set()
    for business in parquet_rows(within(root, config["business"]["path"]), ["business_id", "categories"]):
        if strict_ota_category(business.get("categories"), audit)[0]:
            groups.add(str(business["business_id"]))
        if len(groups) > config["maximum_business_groups"]:
            raise ValueError("business identity bound exceeded")
    candidates, counts = choose_source_identities(
        parquet_rows(within(root, config["photos"]["path"]), ["photo_id", "business_id"]),
        groups, consumed, config["maximum_candidate_images"], config["seed"])
    # 先记录身份选择，再解压；失败保留目录，禁止自动覆盖或更换种子。
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl_new(output / "requested_identities.jsonl", candidates)
    extract_yelp_photo_files(archive, output / "raw", {item["photo_id"] for item in candidates})
    accepted, rejected = validate_pool_images(root, output, candidates, consumed, config["dataset_version"])
    write_jsonl_new(output / "identity_manifest.jsonl", accepted)
    write_jsonl_new(output / "rejections.jsonl", rejected)
    summary = {"protocol": PROTOCOL, "counts": {**counts, "accepted": len(accepted), "rejected": len(rejected)},
               "rejection_reasons": dict(Counter(item["reason"] for item in rejected)),
               "selection": "seeded_group_then_photo_identity_without_labels",
               "business_metadata_usage": "OTA_domain_membership_only_not_visual_reference",
               "labels_generated": False, "human_annotation_count": 0, "history": history,
               "status": "PASS" if len(accepted) >= config["minimum_accepted_images"] else "INSUFFICIENT_SOURCE"}
    write_json_new(output / "summary.json", summary)
    lock = {"protocol": PROTOCOL, "config_canonical_sha256": canonical_config_sha256(config),
            "source_files_lf_sha256": source_hashes(root), "history_sha256": canonical_config_sha256(history),
            "archive_sha256": config["archive_sha256"], "count": len(accepted), "labels_generated": False,
            "files": {name: sha256_file(output / name) for name in (
                "requested_identities.jsonl", "identity_manifest.jsonl", "rejections.jsonl", "summary.json",
                "raw/extract_photo_manifest.json")}}
    lock["lock_sha256"] = canonical_config_sha256(lock)
    write_json_new(output / "source_pool_lock.json", lock)
    return {"status": summary["status"], "counts": summary["counts"], "lock_sha256": lock["lock_sha256"]}


def verified_pool(root, declaration, history):
    config = read_json(within(root, declaration["config"]))
    output = within(root, config["output_root"])
    lock = read_json(output / "source_pool_lock.json")
    if (lock["protocol"] != PROTOCOL or lock["labels_generated"] is not False
            or lock["lock_sha256"] != declaration["lock_sha256"]
            or lock["lock_sha256"] != canonical_config_sha256({k: v for k, v in lock.items() if k != "lock_sha256"})
            or lock["config_canonical_sha256"] != canonical_config_sha256(config)
            or lock["history_sha256"] != canonical_config_sha256(history)
            or lock["source_files_lf_sha256"] != source_hashes(root)):
        raise ValueError("unlabeled source pool identity changed")
    for name, digest in lock["files"].items():
        if sha256_file(within(output, name)) != digest:
            raise ValueError("unlabeled source pool artifact changed")
    if read_json(output / "summary.json")["status"] != "PASS":
        raise ValueError("unlabeled source pool insufficient")
    rows = list(iter_jsonl(output / "identity_manifest.jsonl"))
    if len(rows) != lock["count"] or any(set(row) != ROW_FIELDS for row in rows):
        raise ValueError("pool must contain identity fields only")
    for field in IDENTITY_FIELDS:
        values = [row[field] for row in rows if row[field] is not None]
        if field == "constraint_template_id" and values:
            raise ValueError("native photos have no template")
        if field != "constraint_template_id" and (len(values) != len(rows) or len(set(values)) != len(rows)):
            raise ValueError("pool identity missing or duplicated")
    return rows, lock
