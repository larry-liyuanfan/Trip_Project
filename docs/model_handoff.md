# 模型交接说明

## 交接身份

正式交付版本为 `trip-qwen3-vl-8b-week8-final-v1`，配置文件为
`configs/releases/qwen3_vl_system_final_v1.json`。该版本组合：

- `Qwen/Qwen3-VL-8B-Instruct` 固定 revision；
- system-repair checkpoint-87 PEFT adapter；
- Week 8 v12 商品观察链；
- Week 8 v13 行程 v5；
- 当前 fail-closed FastAPI runtime；
- CLIP 512 维与 Milvus 检索资产。

Git 保存代码、Prompt、Schema、release config、Docker、测试和报告。模型二进制、检索向量和紧凑运行证据位于 Git 外唯一目录：

`outputs/releases/trip-qwen3-vl-8b-week8-final-v1`

基座模型不重复打包。接手者按 release config 下载：

- model：`Qwen/Qwen3-VL-8B-Instruct`
- revision：`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- adapter model SHA-256：`c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a`

## 包结构

| 归档 | 内容 |
| --- | --- |
| `runtime.tar.gz` | 当前 API、推理、检索、Prompt、Schema、Compose 与唯一正式 release config |
| `adapter.tar.gz` | checkpoint-87 PEFT adapter |
| `retrieval.tar.gz` | 1,000 条 CLIP 向量、metadata 与 Milvus 基准 |
| `evidence.tar.gz` | v12 商品 compact evidence、v13 行程比较、完整测试日志和最终状态报告 |
| `release_manifest.json` | 四层大小、成员数和 SHA-256 |

当前四层 SHA-256：

| 归档 | SHA-256 |
| --- | --- |
| `runtime.tar.gz` | `29959a7677ccf8ecd059444d9cacf76481b07589d46ecd3acf64013307354ea5` |
| `adapter.tar.gz` | `f74c078738fa0229574114986c58040bbc280e11ba4ec06558c9a488c2de619d` |
| `retrieval.tar.gz` | `3cdb98f4d50bc72ae53c4e7e96d823ea5b08af93f41df5d14ff1118d12d1a15b` |
| `evidence.tar.gz` | `fecdb55b61a69b7fcc5d1f84ff6623542f07f3934690c24dce1a788a9e6d8253` |

## 离线验证

在仓库根目录执行：

```bash
python scripts/verify_final_delivery.py \
  outputs/releases/trip-qwen3-vl-8b-week8-final-v1
```

验证器检查四层哈希、adapter 身份、v12 商品验收、v13 行程派生范围、最终质量边界、完整测试日志和 runtime 隔离导入。当前结果为 `PASS`，完整测试为 948/948。

## 解压与运行

将四个归档分别解压到独立目录。设置 `TRIP_ADAPTER_DIR` 指向解压后的 `adapter/`，`RETRIEVAL_HOST_DIR` 指向 `retrieval/`，并准备固定 revision 的 Hugging Face 基座缓存。随后执行：

```bash
python scripts/tripctl.py validate
python scripts/tripctl.py doctor
docker compose -f docker/system/docker-compose.yml --env-file docker/system/.env config
```

启动后执行 `python scripts/tripctl.py smoke --base-url http://127.0.0.1:8000`。生产模式不允许静默 fallback；模型、Schema 或检索依赖失败必须显式返回错误。

## 已知边界

- 商品验收参考为自动 silver，human=0，不构成人工视觉准确率声明。
- 商品价位正支持为 0，指标保持 `N/A`。
- v18 设施复查只有 development 增益且延迟增加，未进入正式链路。
- 严格对话研究门禁和独立业务检索相关性不属于本次正式交付门禁。

完整的“已优化/待优化”状态见 `reports/final_delivery_status.md`。

## 不交接内容

不交接 Yelp 原始压缩包、解压图片、逐周中间输出、额外 checkpoint、模型缓存、Spartan 工作目录、OSS Bucket 或任何密钥。历史结论保留在 Git 报告中，下一位接手者不依赖云端环境即可验证包身份。
