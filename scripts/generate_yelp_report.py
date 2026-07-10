import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.yelp_paths import create_output_directories, load_config, resolve_pipeline_paths


def render_report(config: dict[str, Any], stats: dict[str, Any], validation: dict[str, Any] | None = None, clip: dict[str, Any] | None = None) -> str:
    validation = validation or {}
    clip = clip or {}
    storage_note = _storage_note(validation)

    def value(key: str) -> Any:
        result = stats.get(key)
        return "TODO" if result is None else result

    def nested(section: dict[str, Any], key: str) -> Any:
        result = section.get(key)
        return "TODO" if result is None else result

    lines = [
        "# Yelp多模态数据处理说明报告（第一部分）",
        "",
        "## 1. 整体规模与文件构成",
        "本部分说明 Yelp Open Dataset 的本地解压、结构化解析、图片有效性校验、多粒度图文对齐和报告产出流程。",
        f"- 商家总量: {value('business_count')}",
        f"- 评论总量: {value('review_count')}",
        f"- 图片元数据总量: {value('photo_metadata_count')}",
        f"- 有效本地图片数量: {value('valid_image_count')}",
        f"- 覆盖城市数量: {value('city_count')}",
        f"- 存储格式: {config.get('output', {}).get('format', 'TODO')}",
        "",
        "## 2. 原始数据核心特性",
        "- 原始数据具有 UGC 属性：评论文本、评分、caption 和图片均来自用户贡献内容。",
        "- 原生标注粒度包括 photo caption、photo label、business category、business attributes 和 review stars。",
        "- 基础数据包已提取 5 份 JSON：business、review、user、checkin、tip。",
        "- 图片数据包已提取 `photos.json` 和 `photos/` 图片目录。",
        "- 官方说明文档/ToS 已归档到 `data/yelp/raw/docs/`。",
        f"- Business JSONL: `{config.get('paths', {}).get('business_json', 'TODO')}`",
        f"- Review JSONL: `{config.get('paths', {}).get('review_json', 'TODO')}`",
        f"- Photo metadata JSONL: `{config.get('paths', {}).get('photo_json', 'TODO')}`",
        f"- Local image root: `{config.get('paths', {}).get('image_root', 'TODO')}`",
        f"- Review processing limit: {_review_limit_text(config)}",
        "",
        "## 3. 数据解析与预处理方法",
        "- `scripts/parse_yelp_json.py` reads business, review, and photo JSONL inputs line by line, then writes interim tables.",
        "- Review output uses chunked table writing so full-review parsing does not require holding all review rows in memory.",
        "- 评论清洗规则：过滤空文本、纯符号文本和低于最小长度阈值的文本。",
        f"- Review filter counts: {_review_filter_counts(validation.get('review_filters', {}))}",
        "- 图片校验规则：逐张检查本地文件是否存在、是否可被 Pillow 正常读取，并记录宽高与错误原因。",
        "- 产出位置：`data/yelp/interim/` 保存结构化中间表，`data/yelp/processed/` 保存对齐数据和统计文件。",
        "",
        "## 4. 商家元数据、用户评论文本、图片元数据字段清单",
        "- 商家字段：`business_id`、名称、城市、州、经纬度、平均评分、评论数、品类列表、营业属性、营业时间、扁平化属性字段。",
        "- 评论字段：`review_id`、`business_id`、`user_id`、评分、useful/funny/cool、清洗后评论文本、发布时间。",
        "- 图片字段：`photo_id`、`business_id`、caption 描述、label 场景标签、本地 `image_path`。",
        "- 图片索引字段：`photo_id`、`business_id`、`image_path`、有效性标记、宽、高、校验错误。",
        "",
        "## 5. 清洗规则与图片有效性校验",
        f"- Photo metadata entries parsed: {value('photo_metadata_count')}",
        f"- Valid local images: {nested(validation.get('image_validation', {}), 'valid_images')}",
        f"- Missing local images: {nested(validation.get('image_validation', {}), 'missing_images')}",
        f"- Corrupted local images: {nested(validation.get('image_validation', {}), 'corrupted_images')}",
        f"- Valid image ratio: {value('valid_image_ratio')}",
        "",
        "## 6. 强对齐：图片 + 图片元数据",
        f"- Strong alignment: {value('strong_pairs')} valid image-caption-label pairs keyed by `photo_id`.",
        "- 样本构成：单张有效图片、非空 caption、label、business_id 和本地图片路径。",
        "- 适用任务：图像描述、场景标签识别、VLM 指令微调高质量样本。",
        f"- Photo label distribution: {stats.get('photo_label_distribution', 'TODO')}",
        f"- Caption length statistics: {stats.get('caption_length_stats', 'TODO')}",
        "",
        "## 7. 中粒度对齐：图片 + 商家结构化属性",
        f"- Medium alignment: {value('medium_pairs')} valid image-business metadata pairs with generated business descriptions.",
        "- 对齐逻辑：以 `business_id` 关联图片与商家名称、品类、评分、停车、氛围、营业时间等属性。",
        "- 字段明细：`photo_id`、`business_id`、`image_path`、`business_description`、`attribute_dimension_labels`。",
        "- 适用任务：属性识别、商家知识推理、图像到结构化营业信息理解。",
        "",
        "## 8. 弱对齐：图片集合 + 用户评论集合",
        f"- Weak alignment: {value('weak_pairs')} business-level groups containing bounded image lists and selected review texts.",
        "- 对齐逻辑：以 `business_id` 聚合同商家图片集合和评论集合，避免全量笛卡尔积爆炸。",
        f"- 弱对齐品类覆盖: {stats.get('weak_group_top_categories', 'TODO')}",
        "- 适用任务：弱监督图文检索、商家级多模态表示学习、评论辅助视觉理解。",
        "",
        "## 9. 语义降噪方案与当前状态",
        f"- CLIP denoising: {clip.get('status', 'TODO')} ({clip.get('reason', 'TODO')}).",
        f"- Denoising before/after weak pairs: {clip.get('input_pairs', 'TODO')} -> {clip.get('retained_pairs', 'TODO')}",
        f"- CLIP threshold: {clip.get('threshold', 'TODO')}",
        f"- CLIP runtime: model={clip.get('model_id', 'TODO')}, device={clip.get('device', 'TODO')}, candidate batch size={clip.get('candidate_batch_size', 'TODO')}.",
        f"- CLIP scoring coverage: groups={clip.get('input_groups', 'TODO')}, scored={clip.get('scored_pairs', 'TODO')}, skipped candidates={clip.get('skipped_candidates', 'TODO')}.",
        f"- Similarity distribution: {clip.get('similarity_distribution', 'not_available')}",
        "- CLIP scoring runs in the dedicated GPU Docker task; the base Windows data environment remains independent of torch and vLLM.",
        "",
        "## 10. 最终产出、统计汇总与复现命令",
        f"- Businesses parsed: {value('business_count')}",
        f"- Reviews parsed: {value('review_count')}",
        f"- Photo metadata entries parsed: {value('photo_metadata_count')}",
        f"- Valid local images: {value('valid_image_count')}",
        f"- Strong pairs: {value('strong_pairs')}",
        f"- Medium pairs: {value('medium_pairs')}",
        f"- Weak groups: {value('weak_pairs')}",
        f"- Businesses with parsed reviews: {value('businesses_with_reviews')}",
        f"- Businesses with valid images: {value('businesses_with_valid_images')}",
        f"- 覆盖城市数量: {value('city_count')}",
        f"- Top categories: {stats.get('top_categories', 'TODO')}",
        f"- Storage behavior: {storage_note}",
        f"- Local image validation: missing={nested(validation.get('image_validation', {}), 'missing_images')}, corrupted={nested(validation.get('image_validation', {}), 'corrupted_images')}; invalid images are excluded from alignments.",
        "- Full review parsing is enabled in the current config (`processing_limits.max_reviews: null`) and uses chunked writes.",
        "```bash",
        "pip install -r requirements-data.txt",
        "python scripts/parse_yelp_json.py --config configs/data_processing.yaml",
        "python scripts/build_yelp_alignment.py --config configs/data_processing.yaml",
        "docker compose -f docker/docker-compose.yml stop vllm",
        "docker compose -f docker/docker-compose.yml --profile data run --rm clip-denoising",
        "python scripts/generate_yelp_report.py --config configs/data_processing.yaml",
        "python scripts/validate_week2_pipeline.py --config configs/data_processing.yaml",
        "python -m unittest discover -s tests -v",
        "```",
    ]
    return "\n".join(lines) + "\n"


def run_report(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    create_output_directories(config)
    paths = resolve_pipeline_paths(config)
    stats = _read_json(paths["processed_dir"] / "dataset_statistics.json")
    validation = _read_json(paths["interim_dir"] / "validation_summary.json")
    clip = _read_json(paths["processed_dir"] / "clip_denoising_summary.json")
    report = render_report(config, stats, validation, clip)
    paths["report_path"].write_text(report, encoding="utf-8")
    return {"report_path": str(paths["report_path"]), "sections": 10}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _storage_note(validation: dict[str, Any]) -> str:
    outputs = validation.get("outputs") or []
    actual_formats = {record.get("actual_format") for record in outputs if isinstance(record, dict)}
    if "csv_fallback" in actual_formats:
        return "CSV fallback was used at the configured .parquet paths because no Parquet engine was available for that run."
    if "parquet" in actual_formats:
        return "Real Parquet files were written with the available pyarrow Parquet engine."
    return "TODO"


def _review_limit_text(config: dict[str, Any]) -> str:
    value = config.get("processing_limits", {}).get("max_reviews")
    if value in {None, "null", "None", ""}:
        return "None; full review file parsed"
    return str(value)


def _review_filter_counts(summary: dict[str, Any]) -> str:
    return (
        f"input={summary.get('input_reviews', 'TODO')}, "
        f"valid={summary.get('valid_reviews', 'TODO')}, "
        f"empty={summary.get('filtered_empty', 'TODO')}, "
        f"too_short={summary.get('filtered_too_short', 'TODO')}, "
        f"symbol_only={summary.get('filtered_symbol_only', 'TODO')}, "
        f"missing_identifier={summary.get('filtered_missing_identifier', 'TODO')}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the Week 2 Yelp data processing report.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_processing.yaml"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(run_report(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
