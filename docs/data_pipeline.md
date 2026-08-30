# 数据流水线

## 范围

数据代码位于 `src/data/`，命令入口位于 `scripts/`。Git 只保存代码、轻量配置、Schema
和示例；Yelp 原始压缩包、图片、Parquet、生成 JSONL、模型输出和人工工作文件都保持忽略。

## 目录

```text
data/yelp/raw/        原始 JSON 与照片
data/yelp/interim/    解析、图片校验和中间表
data/yelp/processed/  对齐结果与可复用子集
data/eval/            Git 外评测 manifest、registry、run 和 score
data/samples/         可提交的轻量接口样例
```

配置入口为 `configs/data_processing.yaml`。路径可通过配置覆盖，代码不得写入机器绝对路径。

## 处理流程

1. `extract_yelp_archives.py` 解压 Yelp JSON 与照片。
2. `parse_yelp_json.py` 增量解析 business、review、photo，并记录输入、接受和拒绝数量。
3. 图片校验记录可读、缺失和损坏状态，不静默删除失败记录。
4. `build_yelp_alignment.py` 生成强图片-caption、中等图片-business 属性和有界弱图片-review 对齐。
5. `run_clip_denoising.py` 可选执行 CLIP 语义降噪；未运行时必须记录 skipped。
6. `validate_week2_pipeline.py` 校验列、数量、图片路径和汇总一致性。

常用命令：

```bash
python scripts/extract_yelp_archives.py --raw-dir data/yelp/raw --include-photo-files
python scripts/parse_yelp_json.py --config configs/data_processing.yaml
python scripts/build_yelp_alignment.py --config configs/data_processing.yaml
python scripts/validate_week2_pipeline.py --config configs/data_processing.yaml
```

## 质量与身份

- 流式读取大 JSONL，分块写出，避免一次载入全量数据。
- 图片-caption 强对齐必须共享 `photo_id`、图片可读且 caption 非空。
- 图片-business 使用 `business_id` 和规范化字段；图片-review 仅是有界业务级弱对齐。
- 评测集与训练候选按 `sample_id`、`source_id`、图片 SHA-256、`group_id` 和约束模板隔离。
- 自动预标注始终是 `silver/model_preannotation`；没有真实人工输入不得写 human accepted。

详细历史数据量与运行结果保留在 Git 历史和 `experiments/`，不作为当前运行要求。
