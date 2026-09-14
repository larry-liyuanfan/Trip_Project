"""Create an immutable diagnostic silver version from existing train/dev source rows."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.product_labels import LABEL_PROTOCOL, silver_row

IDENTITY_KEYS = ("sample_id", "source_id", "image_sha256", "group_id", "constraint_template_id")


def _rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild(source: Path, output: Path, excluded_identities: Path | None = None):
    source_sha = _sha256(source)
    seen = {key: {} for key in IDENTITY_KEYS}
    excluded = {key: set() for key in IDENTITY_KEYS}
    if excluded_identities:
        # 只读取隔离清单中的身份，不读取最终 test 标签或模型结果。
        for row in _rows(excluded_identities):
            for key in IDENTITY_KEYS:
                if row.get(key):
                    excluded[key].add(row[key])
    count = 0
    splits = {}
    for row in _rows(source):
        split = row.get("split")
        if split not in {"train", "development"} or not isinstance(row.get("caption"), str):
            raise ValueError("only original caption-bearing train/development sources are allowed")
        for key in IDENTITY_KEYS:
            value = row.get(key)
            # 无模板的图片记录使用 null；真实模板身份不得跨 split。
            if not value and key == "constraint_template_id":
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(f"missing source identity: {key}")
            previous = seen[key].get(value)
            if value in excluded[key] or (previous is not None and (key == "sample_id" or previous != split)):
                raise ValueError(f"duplicate/excluded/cross-split identity: {key}")
            seen[key][value] = split
        silver_row(row)  # 创建任何输出前校验所有值，不积累完整样本。
        count += 1
        splits[split] = splits.get(split, 0) + 1
    if not count:
        raise ValueError("source is empty")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "diagnostic_silver.jsonl"
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for row in _rows(source):
            handle.write(json.dumps(silver_row(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    if _sha256(source) != source_sha:
        raise ValueError("source changed during rebuild; incomplete output must not be used")
    manifest = {"label_protocol": LABEL_PROTOCOL, "sample_count": count, "split_counts": splits, "human_count": 0,
                "purpose": "diagnostic_only_not_visual_gold", "source_sha256": source_sha,
                "output_sha256": _sha256(target), "final_test_labels_accessed": False,
                "identity_dimensions": list(IDENTITY_KEYS), "cross_split_overlap_count": 0,
                "excluded_identities_sha256": _sha256(excluded_identities) if excluded_identities else None,
                "external_isolation_verified": excluded_identities is not None}
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--excluded-identities", type=Path, help="identity-only exclusion list; never final-test labels")
    args = parser.parse_args()
    print(json.dumps(rebuild(args.source, args.output_dir, args.excluded_identities)))
