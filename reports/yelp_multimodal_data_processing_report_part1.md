# Yelp多模态数据处理说明报告（第一部分）

## 1. Week 2 目标与验收范围
本周目标是把 Yelp Open Dataset 转换为可复现、可校验、可用于后续 VLM 训练与检索实验的多粒度图文数据。
验收范围覆盖完整源文件解析、本地图片有效性校验、强/中/弱对齐、CLIP 语义降噪、统计分析和产出验证。
本周不训练模型、不构建前端，也不把原始数据、图片、模型权重或大型生成表提交到 Git。

### 核心验收结果
| 指标 | 实际结果 |
| --- | ---: |
| Businesses parsed | 150346 |
| Reviews parsed | 6989830 |
| Photo metadata entries parsed | 200100 |
| Valid local images | 199994 |
| Strong pairs | 96733 |
| Medium pairs | 199994 |
| Weak groups | 36673 |
| Covered cities | 1416 |

## 2. 原始数据文件、格式与数据特性
Yelp 数据属于 UGC 数据：评论、评分、caption 和图片均来自用户贡献，因此存在文本噪声、空 caption、损坏图片和商家级弱语义关联。
原始 JSON 使用 JSON Lines 格式，可以逐行读取；图片元数据通过 `photo_id` 对应本地 JPEG，并通过 `business_id` 关联商家。

| 原始输入 | 配置路径 | 主要用途 |
| --- | --- | --- |
| Business JSONL | `data/yelp/raw/yelp_academic_dataset_business.json` | 商家身份、位置、评分、品类、属性和营业时间 |
| Review JSONL | `data/yelp/raw/yelp_academic_dataset_review.json` | 评论文本、评分、时间和商家关联 |
| Photo metadata JSONL | `data/yelp/raw/photos.json` | photo caption、label 和 business_id |
| Local photos | `data/yelp/raw/photos` | 图片存在性、可读性和尺寸校验 |
| Other extracted JSON | `data/yelp/raw/` | user、checkin、tip，保留供后续阶段使用 |
| Official documentation | `data/yelp/raw/docs/` | 数据说明和使用条款 |
- Review processing limit: None; full review file parsed
- Storage behavior: Real Parquet files were written with the available pyarrow Parquet engine.

## 3. 目录分层与端到端处理流程
项目按原始层、中间层和最终产出层分离，避免清洗结果覆盖源文件，也便于单独重跑某一阶段。

```text
data/yelp/raw/       -> 官方解压文件、photos/、docs/
data/yelp/interim/   -> business/reviews/photos/image-index/review-stats
data/yelp/processed/ -> strong/medium/weak/CLIP outputs and statistics
data/yelp/validation/-> validation summary
reports/             -> mentor-facing Markdown report
```

处理顺序：归档解压 -> JSONL 流式解析 -> 评论清洗 -> 图片并行校验 -> 多粒度对齐 -> CLIP 候选打分 -> 输出验收 -> 报告生成。
`TableStreamWriter` 按配置 chunk 写入，业务嵌套字段先序列化为稳定 JSON，避免不同 Parquet 批次因 schema 漂移失败。
图片验证在有界批次内并行执行；弱对齐仅读取每个目标商家的有限评论，避免加载和组合全部评论。

## 4. 字段设计与清洗规则

### 4.1 商家结构化表
| 字段组 | 字段 | 处理规则 |
| --- | --- | --- |
| 关联键 | `business_id` | 缺失时拒绝，不参与后续 join |
| 基础信息 | `name`, `address`, `city`, `state`, `postal_code` | 保留原始值 |
| 数值信息 | `latitude`, `longitude`, `stars`, `review_count`, `is_open` | 保留结构化类型 |
| 品类 | `categories` | 拆分、去空白并转小写 |
| 营业知识 | `attributes`, `hours` | 稳定 JSON 存储，同时展开关键属性字段 |

### 4.2 评论结构化表
核心字段为 `review_id`, `business_id`, `user_id`, `stars`, `useful`, `funny`, `cool`, `text`, `date`。
拒绝规则依次覆盖缺失关联键、空文本、纯符号文本和低于最小字符阈值的文本；每种原因单独计数，不静默丢弃。
- Review filter counts: input=6990280, valid=6989830, empty=0, too_short=419, symbol_only=31, missing_identifier=0

### 4.3 图片元数据与图片索引
图片元数据保留 `photo_id`, `business_id`, `caption`, `label`, `image_path`。图片索引额外记录 `image_valid`, `image_width`, `image_height`, `validation_error`。
`photo_id` 缺失的元数据无法映射本地文件并被拒绝；图片文件存在但 Pillow 无法完整解码时标记为 corrupted，而不是删除源文件。

## 5. 图片有效性、覆盖度与数据质量
- Photo metadata entries parsed: 200100
- Valid local images: 199994
- Missing local images: 0
- Corrupted local images: 106
- Valid image ratio: 0.9994702648675662
- Businesses with parsed reviews: 150346
- Businesses with valid images: 36673
- 覆盖城市数量: 1416
- Top categories: [['restaurants', 52268], ['food', 27781], ['shopping', 24395], ['home services', 14356], ['beauty & spas', 14292], ['nightlife', 12281], ['health & medical', 11890], ['local services', 11198], ['bars', 11065], ['automotive', 10773], ['event planning & services', 9895], ['sandwiches', 8366], ['american (traditional)', 8139], ['active life', 7687], ['pizza', 7093], ['coffee & tea', 6703], ['fast food', 6472], ['breakfast & brunch', 6239], ['american (new)', 6097], ['hotels & travel', 5857]]
缺失或损坏图片保留在验证摘要中，但从强、中、弱和 CLIP 对齐产出中排除。品类和城市统计来自实际输出人口，不使用人工估算。

## 6. 强对齐：图片 + 原生图片文本
- Strong alignment: 96733 valid image-caption-label pairs keyed by `photo_id`.
强对齐要求本地图片可读且原生 caption 非空；join key 为 `photo_id`，输出字段包括 `pair_id`, `photo_id`, `business_id`, `image_path`, `caption`, `label`, `alignment_type`。
该数据适用于图像描述、场景标签识别和高质量单图文本监督；空 caption 图片不会被错误计入强监督规模。
- Photo label distribution: {'inside': 24662, 'drink': 7547, 'food': 55330, 'outside': 8287, 'menu': 907}
- Caption length statistics: {'caption_count': 96733, 'min_chars': 1, 'mean_chars': 31.47060465404774, 'max_chars': 140}

## 7. 中粒度对齐：图片 + 商家结构化知识
- Medium alignment: 199994 valid image-business metadata pairs with generated business descriptions.
中粒度对齐使用 `business_id`，把商家名称、品类、评分、停车、氛围、服务和营业时间转换为标准化自然语言描述。
输出保留 `attribute_dimension_labels`，便于后续按属性维度构建识别、问答或知识推理任务。图片只要求可读，不要求原生 caption。

## 8. 弱对齐：商家级图片集合 + 评论集合
- Weak alignment: 36673 business-level groups containing bounded image lists and selected review texts.
弱对齐的语义假设是同一商家的图片集合与评论集合相关，但不声称每张图片都精确对应每条评论。每商家图片和评论数量均由配置限制，避免全量笛卡尔积。
输出字段包括 `business_id`, `photo_ids`, `image_paths`, `review_ids`, `review_texts`，用于弱监督检索和商家级多模态表示。
- 弱对齐品类覆盖: [['restaurants', 7897], ['food', 3583], ['nightlife', 1393], ['bars', 1163], ['coffee & tea', 1004], ['american (traditional)', 990], ['sandwiches', 887], ['pizza', 862], ['american (new)', 769], ['breakfast & brunch', 749], ['fast food', 716], ['mexican', 714], ['italian', 614], ['burgers', 582], ['seafood', 477], ['bakeries', 395], ['chinese', 391], ['ice cream & frozen yogurt', 373], ['desserts', 372], ['salad', 370]]

## 9. CLIP 语义降噪与效果分析
- CLIP denoising: completed (CLIP scoring completed).
- Denoising before/after weak pairs: 555459 -> 131146
- CLIP retention rate: 23.61%
- CLIP threshold: 0.25
- CLIP runtime: model=openai/clip-vit-base-patch32, device=cuda, candidate batch size=256.
- CLIP scoring coverage: groups=36673, scored=555459, skipped candidates=0.
- Similarity distribution: {'count': 555459, 'min': 0.022613827139139175, 'mean': 0.22104937891548743, 'max': 0.4198649227619171}
CLIP 对每个候选图片和评论生成归一化向量并计算余弦相似度；低于阈值的候选不进入 `weak_pairs_denoised`。
模型运行在独立 GPU Docker task 中，vLLM 先停止以释放 8GB GPU。阈值是当前基线参数，不代表人工标注的最优决策边界。

## 10. 输出清单、复现命令、限制与后续计划

| 输出文件 | 层级 | 内容 |
| --- | --- | --- |
| `business.parquet` | interim | 商家结构化字段和稳定嵌套属性 |
| `reviews.parquet` | interim | 清洗后的全量有效评论 |
| `photos.parquet` | interim | 图片元数据与本地路径 |
| `photo_image_index.parquet` | interim | 图片有效性、尺寸和错误原因 |
| `review_business_stats.parquet` | interim | 每商家有效评论数和平均评分 |
| `strong_image_caption_pairs.parquet` | processed | 强对齐图片-caption 样本 |
| `image_business_attribute_pairs.parquet` | processed | 中粒度商家知识样本 |
| `business_level_weak_pairs.parquet` | processed | 商家级弱对齐集合 |
| `weak_pairs_denoised.parquet` | processed | 通过 CLIP 阈值的逐行候选 |
| `dataset_statistics.json` | processed | 规模、分布和覆盖度统计 |
| `clip_denoising_summary.json` | processed | 模型、阈值、前后数量和相似度分布 |
| `validation_summary.json` | interim/validation | 解析、过滤、图片和输出验证摘要 |

### 复现命令
```bash
pip install -r requirements-data.txt
python scripts/parse_yelp_json.py --config configs/data_processing.yaml
python scripts/build_yelp_alignment.py --config configs/data_processing.yaml
docker compose -f docker/docker-compose.yml stop vllm
docker compose -f docker/docker-compose.yml --profile data run --rm clip-denoising
python scripts/generate_yelp_report.py --config configs/data_processing.yaml
python scripts/validate_week2_pipeline.py --config configs/data_processing.yaml
python -m unittest discover -s tests -v
```

### 已知限制与 Week 3 衔接
- Yelp 评论与图片是 UGC 内容，弱对齐和 CLIP 分数不能替代人工语义标注。
- 106 张不可读图片被排除；后续若重新下载数据，应重新运行图片验证并比较差异。
- CLIP 使用英文为主的基础模型，可能低估非英语评论或细粒度餐饮语义相关性。
- 当前弱对齐只保留每商家的有限图片和评论，以控制计算规模；它不覆盖所有可能组合。
- Week 3 应基于已验证的强/中/降噪弱对齐数据比较图文 embedding 与多模态检索基线，不在 Week 2 继续扩展训练功能。
