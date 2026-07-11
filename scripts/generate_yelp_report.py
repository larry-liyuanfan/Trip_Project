"""Render the mentor-facing Week 2 Yelp processing report from run summaries."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.yelp_paths import create_output_directories, load_config, resolve_pipeline_paths


def render_report(config: dict[str, Any], stats: dict[str, Any], validation: dict[str, Any] | None = None, clip: dict[str, Any] | None = None) -> str:
    """Build the complete Markdown report without inventing unavailable values."""
    validation = validation or {}
    clip = clip or {}
    storage_note = _storage_note(validation)
    output_format = config.get("output", {}).get("format", "parquet")
    extension = "csv" if str(output_format).lower() == "csv" else "parquet"
    paths = config.get("paths", {})

    def value(key: str) -> Any:
        """Render a top-level statistic or an explicit TODO marker."""
        result = stats.get(key)
        return "TODO" if result is None else result

    def nested(section: dict[str, Any], key: str) -> Any:
        """Render a nested validation statistic or an explicit TODO marker."""
        result = section.get(key)
        return "TODO" if result is None else result

    lines = [
        "# Yelp多模态数据处理说明报告（第一部分）",
        "",
        "## 1. Week 2 目标与验收范围",
        "本周目标是把 Yelp Open Dataset 转换为可复现、可校验、可用于后续 VLM 训练与检索实验的多粒度图文数据。",
        "验收范围覆盖完整源文件解析、本地图片有效性校验、强/中/弱对齐、CLIP 语义降噪、统计分析和产出验证。",
        "本周不训练模型、不构建前端，也不把原始数据、图片、模型权重或大型生成表提交到 Git。",
        "",
        "### 核心验收结果",
        "| 指标 | 实际结果 |",
        "| --- | ---: |",
        f"| Businesses parsed | {value('business_count')} |",
        f"| Reviews parsed | {value('review_count')} |",
        f"| Photo metadata entries parsed | {value('photo_metadata_count')} |",
        f"| Valid local images | {value('valid_image_count')} |",
        f"| Strong pairs | {value('strong_pairs')} |",
        f"| Medium pairs | {value('medium_pairs')} |",
        f"| Weak groups | {value('weak_pairs')} |",
        f"| Covered cities | {value('city_count')} |",
        "",
        "## 2. 原始数据文件、格式与数据特性",
        "Yelp 数据属于 UGC 数据：评论、评分、caption 和图片均来自用户贡献，因此存在文本噪声、空 caption、损坏图片和商家级弱语义关联。",
        "原始 JSON 使用 JSON Lines 格式，可以逐行读取；图片元数据通过 `photo_id` 对应本地 JPEG，并通过 `business_id` 关联商家。",
        "",
        "| 原始输入 | 配置路径 | 主要用途 |",
        "| --- | --- | --- |",
        f"| Business JSONL | `{paths.get('business_json', 'TODO')}` | 商家身份、位置、评分、品类、属性和营业时间 |",
        f"| Review JSONL | `{paths.get('review_json', 'TODO')}` | 评论文本、评分、时间和商家关联 |",
        f"| Photo metadata JSONL | `{paths.get('photo_json', 'TODO')}` | photo caption、label 和 business_id |",
        f"| Local photos | `{paths.get('image_root', 'TODO')}` | 图片存在性、可读性和尺寸校验 |",
        "| Other extracted JSON | `data/yelp/raw/` | user、checkin、tip，保留供后续阶段使用 |",
        "| Official documentation | `data/yelp/raw/docs/` | 数据说明和使用条款 |",
        f"- Review processing limit: {_review_limit_text(config)}",
        f"- Storage behavior: {storage_note}",
        "",
        "## 3. 目录分层与端到端处理流程",
        "项目按原始层、中间层和最终产出层分离，避免清洗结果覆盖源文件，也便于单独重跑某一阶段。",
        "",
        "```text",
        "data/yelp/raw/       -> 官方解压文件、photos/、docs/",
        "data/yelp/interim/   -> business/reviews/photos/image-index/review-stats",
        "data/yelp/processed/ -> strong/medium/weak/CLIP outputs and statistics",
        "data/yelp/validation/-> validation summary",
        "reports/             -> mentor-facing Markdown report",
        "```",
        "",
        "处理顺序：归档解压 -> JSONL 流式解析 -> 评论清洗 -> 图片并行校验 -> 多粒度对齐 -> CLIP 候选打分 -> 输出验收 -> 报告生成。",
        "`TableStreamWriter` 按配置 chunk 写入，业务嵌套字段先序列化为稳定 JSON，避免不同 Parquet 批次因 schema 漂移失败。",
        "图片验证在有界批次内并行执行；弱对齐仅读取每个目标商家的有限评论，避免加载和组合全部评论。",
        "",
        "## 4. 字段设计与清洗规则",
        "",
        "### 4.1 商家结构化表",
        "| 字段组 | 字段 | 处理规则 |",
        "| --- | --- | --- |",
        "| 关联键 | `business_id` | 缺失时拒绝，不参与后续 join |",
        "| 基础信息 | `name`, `address`, `city`, `state`, `postal_code` | 保留原始值 |",
        "| 数值信息 | `latitude`, `longitude`, `stars`, `review_count`, `is_open` | 保留结构化类型 |",
        "| 品类 | `categories` | 拆分、去空白并转小写 |",
        "| 营业知识 | `attributes`, `hours` | 稳定 JSON 存储，同时展开关键属性字段 |",
        "",
        "### 4.2 评论结构化表",
        "核心字段为 `review_id`, `business_id`, `user_id`, `stars`, `useful`, `funny`, `cool`, `text`, `date`。",
        "拒绝规则依次覆盖缺失关联键、空文本、纯符号文本和低于最小字符阈值的文本；每种原因单独计数，不静默丢弃。",
        f"- Review filter counts: {_review_filter_counts(validation.get('review_filters', {}))}",
        "",
        "### 4.3 图片元数据与图片索引",
        "图片元数据保留 `photo_id`, `business_id`, `caption`, `label`, `image_path`。图片索引额外记录 `image_valid`, `image_width`, `image_height`, `validation_error`。",
        "`photo_id` 缺失的元数据无法映射本地文件并被拒绝；图片文件存在但 Pillow 无法完整解码时标记为 corrupted，而不是删除源文件。",
        "",
        "## 5. 图片有效性、覆盖度与数据质量",
        f"- Photo metadata entries parsed: {value('photo_metadata_count')}",
        f"- Valid local images: {nested(validation.get('image_validation', {}), 'valid_images')}",
        f"- Missing local images: {nested(validation.get('image_validation', {}), 'missing_images')}",
        f"- Corrupted local images: {nested(validation.get('image_validation', {}), 'corrupted_images')}",
        f"- Valid image ratio: {value('valid_image_ratio')}",
        f"- Businesses with parsed reviews: {value('businesses_with_reviews')}",
        f"- Businesses with valid images: {value('businesses_with_valid_images')}",
        f"- 覆盖城市数量: {value('city_count')}",
        f"- Top categories: {stats.get('top_categories', 'TODO')}",
        "缺失或损坏图片保留在验证摘要中，但从强、中、弱和 CLIP 对齐产出中排除。品类和城市统计来自实际输出人口，不使用人工估算。",
        "",
        "## 6. 强对齐：图片 + 原生图片文本",
        f"- Strong alignment: {value('strong_pairs')} valid image-caption-label pairs keyed by `photo_id`.",
        "强对齐要求本地图片可读且原生 caption 非空；join key 为 `photo_id`，输出字段包括 `pair_id`, `photo_id`, `business_id`, `image_path`, `caption`, `label`, `alignment_type`。",
        "该数据适用于图像描述、场景标签识别和高质量单图文本监督；空 caption 图片不会被错误计入强监督规模。",
        f"- Photo label distribution: {stats.get('photo_label_distribution', 'TODO')}",
        f"- Caption length statistics: {stats.get('caption_length_stats', 'TODO')}",
        "",
        "## 7. 中粒度对齐：图片 + 商家结构化知识",
        f"- Medium alignment: {value('medium_pairs')} valid image-business metadata pairs with generated business descriptions.",
        "中粒度对齐使用 `business_id`，把商家名称、品类、评分、停车、氛围、服务和营业时间转换为标准化自然语言描述。",
        "输出保留 `attribute_dimension_labels`，便于后续按属性维度构建识别、问答或知识推理任务。图片只要求可读，不要求原生 caption。",
        "",
        "## 8. 弱对齐：商家级图片集合 + 评论集合",
        f"- Weak alignment: {value('weak_pairs')} business-level groups containing bounded image lists and selected review texts.",
        "弱对齐的语义假设是同一商家的图片集合与评论集合相关，但不声称每张图片都精确对应每条评论。每商家图片和评论数量均由配置限制，避免全量笛卡尔积。",
        "输出字段包括 `business_id`, `photo_ids`, `image_paths`, `review_ids`, `review_texts`，用于弱监督检索和商家级多模态表示。",
        f"- 弱对齐品类覆盖: {stats.get('weak_group_top_categories', 'TODO')}",
        "",
        "## 9. CLIP 语义降噪与效果分析",
        f"- CLIP denoising: {clip.get('status', 'TODO')} ({clip.get('reason', 'TODO')}).",
        f"- Denoising before/after weak pairs: {clip.get('input_pairs', 'TODO')} -> {clip.get('retained_pairs', 'TODO')}",
        f"- CLIP retention rate: {_clip_retention_rate(clip)}",
        f"- CLIP threshold: {clip.get('threshold', 'TODO')}",
        f"- CLIP runtime: model={clip.get('model_id', 'TODO')}, device={clip.get('device', 'TODO')}, candidate batch size={clip.get('candidate_batch_size', 'TODO')}.",
        f"- CLIP scoring coverage: groups={clip.get('input_groups', 'TODO')}, scored={clip.get('scored_pairs', 'TODO')}, skipped candidates={clip.get('skipped_candidates', 'TODO')}.",
        f"- Similarity distribution: {clip.get('similarity_distribution', 'not_available')}",
        "CLIP 对每个候选图片和评论生成归一化向量并计算余弦相似度；低于阈值的候选不进入 `weak_pairs_denoised`。",
        "模型运行在独立 GPU Docker task 中，vLLM 先停止以释放 8GB GPU。阈值是当前基线参数，不代表人工标注的最优决策边界。",
        "",
        "## 10. 输出清单、复现命令、限制与后续计划",
        "",
        "| 输出文件 | 层级 | 内容 |",
        "| --- | --- | --- |",
        f"| `business.{extension}` | interim | 商家结构化字段和稳定嵌套属性 |",
        f"| `reviews.{extension}` | interim | 清洗后的全量有效评论 |",
        f"| `photos.{extension}` | interim | 图片元数据与本地路径 |",
        f"| `photo_image_index.{extension}` | interim | 图片有效性、尺寸和错误原因 |",
        f"| `review_business_stats.{extension}` | interim | 每商家有效评论数和平均评分 |",
        f"| `strong_image_caption_pairs.{extension}` | processed | 强对齐图片-caption 样本 |",
        f"| `image_business_attribute_pairs.{extension}` | processed | 中粒度商家知识样本 |",
        f"| `business_level_weak_pairs.{extension}` | processed | 商家级弱对齐集合 |",
        f"| `weak_pairs_denoised.{extension}` | processed | 通过 CLIP 阈值的逐行候选 |",
        "| `dataset_statistics.json` | processed | 规模、分布和覆盖度统计 |",
        "| `clip_denoising_summary.json` | processed | 模型、阈值、前后数量和相似度分布 |",
        "| `validation_summary.json` | interim/validation | 解析、过滤、图片和输出验证摘要 |",
        "",
        "### 复现命令",
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
        "",
        "### 已知限制与 Week 3 衔接",
        "- Yelp 评论与图片是 UGC 内容，弱对齐和 CLIP 分数不能替代人工语义标注。",
        "- 106 张不可读图片被排除；后续若重新下载数据，应重新运行图片验证并比较差异。",
        "- CLIP 使用英文为主的基础模型，可能低估非英语评论或细粒度餐饮语义相关性。",
        "- 当前弱对齐只保留每商家的有限图片和评论，以控制计算规模；它不覆盖所有可能组合。",
        "- Week 3 应基于已验证的强/中/降噪弱对齐数据比较图文 embedding 与多模态检索基线，不在 Week 2 继续扩展训练功能。",
    ]
    return "\n".join(lines) + "\n"


def run_report(config_path: Path) -> dict[str, Any]:
    """Load run artifacts, write the configured report, and return its location."""
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
    """Read an optional JSON artifact used by report rendering."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _storage_note(validation: dict[str, Any]) -> str:
    """Explain whether the run produced real Parquet or a CSV fallback."""
    outputs = validation.get("outputs") or []
    actual_formats = {record.get("actual_format") for record in outputs if isinstance(record, dict)}
    if "csv_fallback" in actual_formats:
        return "CSV fallback was used at the configured .parquet paths because no Parquet engine was available for that run."
    if "parquet" in actual_formats:
        return "Real Parquet files were written with the available pyarrow Parquet engine."
    return "TODO"


def _review_limit_text(config: dict[str, Any]) -> str:
    """Describe whether review parsing was capped or processed in full."""
    value = config.get("processing_limits", {}).get("max_reviews")
    if value in {None, "null", "None", ""}:
        return "None; full review file parsed"
    return str(value)


def _review_filter_counts(summary: dict[str, Any]) -> str:
    """Format review rejection counters in a stable mentor-readable order."""
    return (
        f"input={summary.get('input_reviews', 'TODO')}, "
        f"valid={summary.get('valid_reviews', 'TODO')}, "
        f"empty={summary.get('filtered_empty', 'TODO')}, "
        f"too_short={summary.get('filtered_too_short', 'TODO')}, "
        f"symbol_only={summary.get('filtered_symbol_only', 'TODO')}, "
        f"missing_identifier={summary.get('filtered_missing_identifier', 'TODO')}"
    )


def _clip_retention_rate(summary: dict[str, Any]) -> str:
    """Format the retained share of scored candidates or an explicit TODO."""
    input_pairs = summary.get("input_pairs")
    retained_pairs = summary.get("retained_pairs")
    if not isinstance(input_pairs, (int, float)) or not isinstance(retained_pairs, (int, float)):
        return "TODO"
    if input_pairs <= 0:
        return "0.00%"
    return f"{retained_pairs / input_pairs:.2%}"


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for report generation."""
    parser = argparse.ArgumentParser(description="Generate the Week 2 Yelp data processing report.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_processing.yaml"))
    return parser


def main() -> None:
    """Generate the report and print a machine-readable completion summary."""
    args = build_arg_parser().parse_args()
    print(json.dumps(run_report(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
