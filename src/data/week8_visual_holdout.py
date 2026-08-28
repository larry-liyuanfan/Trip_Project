"""Seal untouched image identities without reading or generating final labels."""
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image

from src.data.week8_product_silver_source_v8 import _verify_identity
from src.inference.product_observation import canonical_config_sha256
from src.training.week7_data import IDENTITY_FIELDS, load_consumed_identities, add_superseded_identities, sha256_file, iter_jsonl


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def within(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def identity_projection(row):
    provenance = row.get("provenance") or {}
    identity = {key: row.get(key) or provenance.get(key) for key in IDENTITY_FIELDS}
    identity["constraint_template_id"] = identity["constraint_template_id"] or row.get("template_id")
    # 检索表使用裸 business_id，商品表使用命名空间前缀；不能因此漏掉同商家。
    if identity["group_id"]:
        identity["group_id"] = str(identity["group_id"]).removeprefix("yelp-business:")
    return identity


def choose_untouched(rows, consumed, count, seed, template, namespace="week8-visual-final-v1"):
    if type(count) is not int or count <= 0:
        raise ValueError("positive fixed sample count required")
    eligible, rejected = [], []
    for row in rows:
        item = identity_projection(row)
        collisions = [key for key, value in item.items() if value and value in consumed[key]]
        if collisions:
            rejected.append({"source_id": item["source_id"], "dimensions": collisions})
        else:
            if not all(item[key] for key in ("source_id", "group_id", "image_sha256")):
                raise ValueError("fresh source identity incomplete")
            eligible.append(row)
    eligible.sort(key=lambda row: hashlib.sha256((seed + "\x1f" + row["source_id"]).encode()).hexdigest())
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} untouched images for fixed count {count}")
    chosen = []
    unique = {key: set() for key in ("sample_id", "source_id", "group_id", "image_sha256")}
    for index, row in enumerate(eligible[:count]):
        identity = identity_projection(row)
        if identity["constraint_template_id"] is not None and identity["constraint_template_id"] != template:
            raise ValueError("cannot replace a real source template identity")
        identity.update(sample_id=f"{namespace}-{index:04d}", constraint_template_id=template)
        for key in IDENTITY_FIELDS:
            if identity[key] in consumed[key]:
                raise ValueError(f"historical {key} collision")
            if key in unique:
                if identity[key] in unique[key]:
                    raise ValueError("duplicate source/group/image; do not silently replace selected samples")
                unique[key].add(identity[key])
        chosen.append({**identity, "source_image_path": row["image_path"], "split": "test"})
    return chosen, {"source_count": len(rows), "eligible_count": len(eligible), "selected_count": count,
                    "rejected_count": len(rejected), "rejected_identities": rejected,
                    "selection": "seeded_source_identity_hash_without_labels", "dimensions": list(IDENTITY_FIELDS),
                    "template_identity_status": "not_applicable_untemplated_images" if template is None else "explicit_template",
                    "historical_overlap_counts": {key: 0 for key in IDENTITY_FIELDS}}


def load_history(root, config):
    audit = read_json(root / config["source_audit_config"])
    fresh, _, _ = _verify_identity(root, audit)
    source_config = read_json(root / audit["source"]["product_config_path"])
    compatibility = {"dataset": {"source_paths": source_config["dataset"]["source_paths"]}}
    consumed, evidence = load_consumed_identities(root, compatibility)
    add_superseded_identities(root, compatibility, consumed, evidence)
    expected_history = fresh["historical_exclusion_evidence"]
    if evidence["files"] != expected_history["files"] or evidence["superseded_week7_identity_manifests"] != expected_history["superseded_week7_identity_manifests"]:
        raise ValueError("historical exclusion artifacts changed")
    consumed["group_id"] = {value.removeprefix("yelp-business:") for value in consumed["group_id"]}
    extra = []
    for kind, directories in (("product", config["previous_product_versions"]),
                              ("retrieval", config["previous_retrieval_versions"]),
                              ("training", [config["previous_training_version"]])):
        for relative in directories:
            directory = within(root, relative)
            lock = read_json(directory / "dataset_lock.json")
            names = ["identity_manifest.jsonl"] if kind == "product" else (
                ["development_query", "final_test_query", "index"] if kind == "retrieval"
                else ["train/image_product_search.jsonl", "development/image_product_search.jsonl"])
            for name in names:
                declared = lock["files"][name]
                filename = declared["path"] if kind == "retrieval" else name
                path = within(directory, filename)
                digest = sha256_file(path)
                if digest != declared["sha256"]:
                    raise ValueError("historical identity file differs from lock")
                count = 0
                # 历史 test 只投影身份用于排除，不读取标签参与新样本选择或评分。
                for row in iter_jsonl(path):
                    for field, value in identity_projection(row).items():
                        if value:
                            consumed[field].add(value)
                    count += 1
                if count != declared["count"]:
                    raise ValueError("historical row count mismatch")
                extra.append({"path": path.relative_to(root).as_posix(), "sha256": digest, "count": count,
                              "scope": "identity_fields_only", "lock_sha256": sha256_file(directory / "dataset_lock.json")})
    for previous in config.get("previous_visual_holdouts", []):
        directory = within(root, previous["path"])
        lock = read_json(directory / "dataset_lock.json")
        if (lock["lock_sha256"] != previous["lock_sha256"]
                or lock["lock_sha256"] != canonical_config_sha256({k: v for k, v in lock.items() if k != "lock_sha256"})):
            raise ValueError("previous visual holdout lock changed")
        path = directory / "manifest.jsonl"
        if sha256_file(path) != lock["manifest_sha256"]:
            raise ValueError("previous visual holdout manifest changed")
        count = 0
        for row in iter_jsonl(path):
            # 失败的 final 仍永久消费；仅看五维身份，不读其参考或候选输出。
            for field, value in identity_projection(row).items():
                if value:
                    consumed[field].add(value)
            count += 1
        if count != lock["count"]:
            raise ValueError("previous visual holdout count changed")
        extra.append({"path": path.relative_to(root).as_posix(), "sha256": lock["manifest_sha256"],
                      "count": count, "scope": "identity_fields_only", "lock_sha256": lock["lock_sha256"]})
    evidence["week8_history"] = extra
    return consumed, evidence, audit, fresh


def build_holdout(root, config_path):
    config = read_json(config_path)
    if config["human_annotation_count"] != 0 or config["label_source"] != "model_generated_silver" or config["no_tuning_after_final"] is not True:
        raise ValueError("immutable automatic silver final protocol required")
    output = within(root, config["output_root"])
    if output.exists():
        raise FileExistsError("holdout identity already sealed")
    consumed, history, audit, fresh = load_history(root, config)
    rows = list(iter_jsonl(root / audit["source"]["fresh_identity_path"]))
    pool_lock = None
    if config.get("unlabeled_source_pool"):
        from src.data.week8_unlabeled_pool import verified_pool
        rows, pool_lock = verified_pool(root, config["unlabeled_source_pool"], history)
    if config["template_id"] is not None:
        raise ValueError("native photos have no synthetic template; preserve null instead of inventing one")
    chosen, selection = choose_untouched(rows, consumed, config["sample_count"], config["seed"], config["template_id"], config["dataset_version"])
    # 选定后检查全部图片；任何损坏直接失败，不替换难例或静默减少支持。
    for row in chosen:
        path = within(root, row["source_image_path"])
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError("selected image hash mismatch")
        with Image.open(path) as image:
            image.verify()
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    with (output / "manifest.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in chosen:
            destination = output / "images" / (row["image_sha256"] + ".jpg")
            shutil.copyfile(within(root, row.pop("source_image_path")), destination)
            row["image_path"] = destination.relative_to(root).as_posix()
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json_new(output / "isolation.json", {"status": "PASS", "selection": selection, "history": history,
                                              "human_annotation_count": 0, "final_labels_read": False})
    lock = {"protocol": config["protocol"], "dataset_version": config["dataset_version"], "count": len(chosen),
            "manifest_sha256": sha256_file(output / "manifest.jsonl"), "isolation_sha256": sha256_file(output / "isolation.json"),
            "config_canonical_sha256": canonical_config_sha256(config), "final_labels_generated": False,
            "images_sha256": {row["image_path"]: row["image_sha256"] for row in chosen},
            "source_identity_sha256": fresh["identity_manifest"]["sha256"]}
    if pool_lock:
        lock["source_identity_sha256"] = pool_lock["files"]["identity_manifest.jsonl"]
        lock["unlabeled_source_pool_lock_sha256"] = pool_lock["lock_sha256"]
    lock["lock_sha256"] = canonical_config_sha256(lock)
    write_json_new(output / "dataset_lock.json", lock)
    return {"status": "SEALED_WITHOUT_LABELS", "count": len(chosen), "eligible_count": selection["eligible_count"], "lock_sha256": lock["lock_sha256"]}


def validate_holdout(root, config):
    output = within(root, config["output_root"])
    lock = read_json(output / "dataset_lock.json")
    if (lock["lock_sha256"] != canonical_config_sha256({k: v for k, v in lock.items() if k != "lock_sha256"})
            or lock["config_canonical_sha256"] != canonical_config_sha256(config)
            or lock["manifest_sha256"] != sha256_file(output / "manifest.jsonl")
            or lock["isolation_sha256"] != sha256_file(output / "isolation.json")
            or read_json(output / "isolation.json")["status"] != "PASS"):
        raise ValueError("sealed holdout identity changed")
    rows = list(iter_jsonl(output / "manifest.jsonl"))
    if config.get("unlabeled_source_pool"):
        from src.data.week8_unlabeled_pool import verified_pool
        consumed, history, _, _ = load_history(root, config)
        pool_rows, pool_lock = verified_pool(root, config["unlabeled_source_pool"], history)
        pool_by_source = {row["source_id"]: row for row in pool_rows}
        if (lock.get("unlabeled_source_pool_lock_sha256") != pool_lock["lock_sha256"]
                or lock["source_identity_sha256"] != pool_lock["files"]["identity_manifest.jsonl"]):
            raise ValueError("holdout source pool binding changed")
        for row in rows:
            source = pool_by_source.get(row["source_id"], {})
            if any(row[key] != source.get(key) for key in ("source_id", "group_id", "image_sha256", "constraint_template_id")):
                raise ValueError("holdout image differs from identity-only source")
            if any(value in consumed[key] for key, value in identity_projection(row).items() if value):
                raise ValueError("holdout overlaps historical identities")
    if len(rows) != config["sample_count"] or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("sealed final sample count changed")
    for row in rows:
        if row["split"] != "test" or sha256_file(within(root, row["image_path"])) != row["image_sha256"] or lock["images_sha256"].get(row["image_path"]) != row["image_sha256"]:
            raise ValueError("sealed image identity changed")
    return rows, lock
