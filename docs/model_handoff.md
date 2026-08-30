# 模型交接说明

## 交接范围

当前交接模型为 `Qwen/Qwen3-VL-8B-Instruct` 加 system-repair checkpoint-87 adapter。Git 保存运行代码、Prompt、Schema、release config、Docker 配置、测试和报告；模型二进制不进入 Git。交接时必须将仓库与以下唯一目录一并移交：

`outputs/releases/trip-qwen3-vl-8b-system-repair-v1-rc1-final-v3`

该目录共约 59.9 MB，包含：

| 归档 | 内容 | SHA-256 |
| --- | --- | --- |
| `runtime.tar.gz` | API、推理、检索、Prompt、Schema、Compose | `ae61fb867482d3f382572ef166e2b520eba69511e83bb72859dcdc83ec520f72` |
| `adapter.tar.gz` | checkpoint-87 PEFT adapter | `f74c078738fa0229574114986c58040bbc280e11ba4ec06558c9a488c2de619d` |
| `retrieval.tar.gz` | 1,000 条 CLIP 向量、metadata、Milvus 基准 | `3cdb98f4d50bc72ae53c4e7e96d823ea5b08af93f41df5d14ff1118d12d1a15b` |
| `evidence.tar.gz` | final gate、真实 smoke、训练与评测摘要 | `3ab0c0249a55ad006eebebaff65d25412567684ff5fd8702516215f83d2af2a7` |

基座模型不重复打包。接手者按 release config 下载固定版本：

- model：`Qwen/Qwen3-VL-8B-Instruct`
- revision：`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- adapter model SHA-256：`c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a`

## 接手验证

从仓库根目录执行：

```bash
python scripts/verify_model_handoff.py \
  outputs/releases/trip-qwen3-vl-8b-system-repair-v1-rc1-final-v3
```

期望状态为 `PASS`，同时显示三个 Schema-valid 业务场景、`DIALOGUE_BETA`、1,000 x 512 检索向量和 Recall@10=1.0。验证器会检查四层归档、嵌入 release config、adapter、final gate、真实模型 smoke 和 Milvus 基准，不需要 Spartan、OSS 或云端凭据。

## 解压与运行

将四个归档分别解压到独立目录。设置 `TRIP_ADAPTER_DIR` 指向解压后的 `adapter/`，`RETRIEVAL_HOST_DIR` 指向 `retrieval/`，并准备固定 revision 的 Hugging Face 基座缓存。随后执行：

```bash
python scripts/tripctl.py validate
python scripts/tripctl.py doctor
docker compose -f docker/system/docker-compose.yml --env-file docker/system/.env config
```

启动后使用 `python scripts/tripctl.py smoke --base-url http://127.0.0.1:8000` 检查 `/health`、`/ready`、三个任务接口、对话和视觉检索。生产模式不允许静默 fallback。

## 不交接内容

不交接 Yelp 原始压缩包、解压图片、各周中间输出、checkpoint、模型缓存、迁移目录、Spartan 工作目录、OSS Bucket 或任何密钥。历史指标和失败结论已压缩进 evidence 层并在 Git 报告中保留，不需要保留逐周全部运行目录。
