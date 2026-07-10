# Yelp多模态数据处理说明报告（第一部分）

## 1. 整体规模与文件构成
本部分说明 Yelp Open Dataset 的本地解压、结构化解析、图片有效性校验、多粒度图文对齐和报告产出流程。
- 商家总量: 150346
- 评论总量: 6989830
- 图片元数据总量: 200100
- 有效本地图片数量: 199994
- 覆盖城市数量: 1416
- 存储格式: parquet

## 2. 原始数据核心特性
- 原始数据具有 UGC 属性：评论文本、评分、caption 和图片均来自用户贡献内容。
- 原生标注粒度包括 photo caption、photo label、business category、business attributes 和 review stars。
- 基础数据包已提取 5 份 JSON：business、review、user、checkin、tip。
- 图片数据包已提取 `photos.json` 和 `photos/` 图片目录。
- 官方说明文档/ToS 已归档到 `data/yelp/raw/docs/`。
- Business JSONL: `data/yelp/raw/yelp_academic_dataset_business.json`
- Review JSONL: `data/yelp/raw/yelp_academic_dataset_review.json`
- Photo metadata JSONL: `data/yelp/raw/photos.json`
- Local image root: `data/yelp/raw/photos`
- Review processing limit: None; full review file parsed

## 3. 数据解析与预处理方法
- `scripts/parse_yelp_json.py` reads business, review, and photo JSONL inputs line by line, then writes interim tables.
- Review output uses chunked table writing so full-review parsing does not require holding all review rows in memory.
- 评论清洗规则：过滤空文本、纯符号文本和低于最小长度阈值的文本。
- Review filter counts: input=6990280, valid=6989830, empty=0, too_short=419, symbol_only=31, missing_identifier=0
- 图片校验规则：逐张检查本地文件是否存在、是否可被 Pillow 正常读取，并记录宽高与错误原因。
- 产出位置：`data/yelp/interim/` 保存结构化中间表，`data/yelp/processed/` 保存对齐数据和统计文件。

## 4. 商家元数据、用户评论文本、图片元数据字段清单
- 商家字段：`business_id`、名称、城市、州、经纬度、平均评分、评论数、品类列表、营业属性、营业时间、扁平化属性字段。
- 评论字段：`review_id`、`business_id`、`user_id`、评分、useful/funny/cool、清洗后评论文本、发布时间。
- 图片字段：`photo_id`、`business_id`、caption 描述、label 场景标签、本地 `image_path`。
- 图片索引字段：`photo_id`、`business_id`、`image_path`、有效性标记、宽、高、校验错误。

## 5. 清洗规则与图片有效性校验
- Photo metadata entries parsed: 200100
- Valid local images: 199994
- Missing local images: 0
- Corrupted local images: 106
- Valid image ratio: 0.9994702648675662

## 6. 强对齐：图片 + 图片元数据
- Strong alignment: 96733 valid image-caption-label pairs keyed by `photo_id`.
- 样本构成：单张有效图片、非空 caption、label、business_id 和本地图片路径。
- 适用任务：图像描述、场景标签识别、VLM 指令微调高质量样本。
- Photo label distribution: {'inside': 24662, 'drink': 7547, 'food': 55330, 'outside': 8287, 'menu': 907}
- Caption length statistics: {'caption_count': 96733, 'min_chars': 1, 'mean_chars': 31.47060465404774, 'max_chars': 140}

## 7. 中粒度对齐：图片 + 商家结构化属性
- Medium alignment: 199994 valid image-business metadata pairs with generated business descriptions.
- 对齐逻辑：以 `business_id` 关联图片与商家名称、品类、评分、停车、氛围、营业时间等属性。
- 字段明细：`photo_id`、`business_id`、`image_path`、`business_description`、`attribute_dimension_labels`。
- 适用任务：属性识别、商家知识推理、图像到结构化营业信息理解。

## 8. 弱对齐：图片集合 + 用户评论集合
- Weak alignment: 36673 business-level groups containing bounded image lists and selected review texts.
- 对齐逻辑：以 `business_id` 聚合同商家图片集合和评论集合，避免全量笛卡尔积爆炸。
- 弱对齐品类覆盖: [['restaurants', 7897], ['food', 3583], ['nightlife', 1393], ['bars', 1163], ['coffee & tea', 1004], ['american (traditional)', 990], ['sandwiches', 887], ['pizza', 862], ['american (new)', 769], ['breakfast & brunch', 749], ['fast food', 716], ['mexican', 714], ['italian', 614], ['burgers', 582], ['seafood', 477], ['bakeries', 395], ['chinese', 391], ['ice cream & frozen yogurt', 373], ['desserts', 372], ['salad', 370]]
- 适用任务：弱监督图文检索、商家级多模态表示学习、评论辅助视觉理解。

## 9. 语义降噪方案与当前状态
- CLIP denoising: completed (CLIP scoring completed).
- Denoising before/after weak pairs: 555459 -> 131146
- CLIP threshold: 0.25
- CLIP runtime: model=openai/clip-vit-base-patch32, device=cuda, candidate batch size=256.
- CLIP scoring coverage: groups=36673, scored=555459, skipped candidates=0.
- Similarity distribution: {'count': 555459, 'min': 0.022613827139139175, 'mean': 0.22104937891548743, 'max': 0.4198649227619171}
- CLIP scoring runs in the dedicated GPU Docker task; the base Windows data environment remains independent of torch and vLLM.

## 10. 最终产出、统计汇总与复现命令
- Businesses parsed: 150346
- Reviews parsed: 6989830
- Photo metadata entries parsed: 200100
- Valid local images: 199994
- Strong pairs: 96733
- Medium pairs: 199994
- Weak groups: 36673
- Businesses with parsed reviews: 150346
- Businesses with valid images: 36673
- 覆盖城市数量: 1416
- Top categories: [['restaurants', 52268], ['food', 27781], ['shopping', 24395], ['home services', 14356], ['beauty & spas', 14292], ['nightlife', 12281], ['health & medical', 11890], ['local services', 11198], ['bars', 11065], ['automotive', 10773], ['event planning & services', 9895], ['sandwiches', 8366], ['american (traditional)', 8139], ['active life', 7687], ['pizza', 7093], ['coffee & tea', 6703], ['fast food', 6472], ['breakfast & brunch', 6239], ['american (new)', 6097], ['hotels & travel', 5857]]
- Storage behavior: Real Parquet files were written with the available pyarrow Parquet engine.
- Local image validation: missing=0, corrupted=106; invalid images are excluded from alignments.
- Full review parsing is enabled in the current config (`processing_limits.max_reviews: null`) and uses chunked writes.
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
