# Qwen3-VL 2B/4B 阿里云 A10 验证与 Week 1-5 重跑

## 实验身份

- 日期：2026-08-09（Australia/Sydney；服务器观测时间为 UTC）。
- Git 基线：`4bee591`；GPU 专用配置为本次未提交叠加文件。
- 服务器：`trip-gpu-a10`，Ubuntu 24.04.4，单张 NVIDIA A10 23028 MiB。
- 模型：`Qwen/Qwen3-VL-2B-Instruct`。
- 后端：vLLM 0.11.0，镜像 `vllm/vllm-openai:v0.11.0`。
- 配置：BF16、单 GPU、`max_model_len=8192`、
  `gpu_memory_utilization=0.75`、`enforce_eager=true`、每请求最多一张图片。
- 缓存：`/data/huggingface`。

## 启动与故障处理

首次模型下载在镜像默认 `hf_transfer` 路径中失败，错误为下载器运行时错误；
未发生 OOM。GPU 专用 Compose 随后显式设置
`HF_HUB_ENABLE_HF_TRANSFER=0` 与 `HF_HUB_DISABLE_XET=1`，改用标准下载器，
复用已有缓存后成功。权重下载耗时 347.37 秒；模型加载占用 4.2374 GiB，
引擎初始化与 KV cache 预热耗时 15.57 秒。

## 验证结果

- `/health`：HTTP 200。
- `/v1/models`：HTTP 200，返回 `Qwen3-VL-2B-Instruct`，根模型为
  `Qwen/Qwen3-VL-2B-Instruct`，最大上下文 8192。
- 端口：宿主机仅 `127.0.0.1:8001->8000/tcp`。
- 图片：`data/samples/images/cafe_001.jpg`。
- Prompt：`Describe the main visible scene in this image in one concise sentence.`
- 请求：HTTP 200；9,749 ms；prompt/completion/total token 为 87/18/105。
- 输出：`A simple, dark brown, octagonal shape is centered on a light beige background.`
- 显存：初始化后约 14.6 GiB；请求期间采样峰值 14,617 MiB。
- 错误：最终请求无模型请求错误、无 OOM。

该结果只验证服务启动、图片输入和文本输出链路，不代表完整能力评测，也未运行
Week 3/Week 4 冻结数据、CLIP、训练或全量评测。

## Qwen3-VL-4B 正式重跑

### 实验身份与配置

- 日期：2026-08-09。
- Git 基线：`4bee591` 加本次未提交工作区改动。
- 模型/后端：`Qwen/Qwen3-VL-4B-Instruct`、vLLM 0.11.0、BF16、单 A10。
- 权重缓存：`/data/huggingface`；下载权重约 8.591 GiB。
- 服务监听：`127.0.0.1:8001`；客户端通过 SSH 转发至
  `127.0.0.1:18001/v1`。
- 图片上限：Week 3 为每请求 1 张；Week 4 Few-Shot 通过
  `VLLM_LIMIT_IMAGES=8` 支持 7 张示例加 1 张查询图。
- 固定数据集：`week3_evaluation_v2`；450 条样本哈希
  `3e900e64bb345df35343c8f14bfb1f8310ae597a57e4a4d9585bc01173ad648c`。

### 命令与运行

- Week 1：`python scripts/test_image_understanding.py`，真实图片请求通过，
  5.589 秒。
- Week 2：验证冻结的 Yelp 数据处理产物；没有重复执行模型无关的 CLIP 编码。
- Week 3 baseline：`week3_qwen3vl4b_baseline_full_20260809_001`，
  450/450，模型请求错误 0。
- Week 3 standardized：`week3_qwen3vl4b_standardized_full_20260809_001`，
  450/450，模型请求错误 0。
- Week 3 comparison：
  `week3_qwen3vl4b_baseline_vs_standardized_20260809_001`，450 对，
  1,000 次 bootstrap。
- Week 4 pilots：`standardized_v2`、`fewshot_4_v2`、`fewshot_7_v2`
  各 15 条有效样本。
- Week 4 full：`week4_qwen3vl4b_winners_full_20260809_002`，
  450/450，模型请求错误 0。
- Week 4 common semantic：
  `qwen3vl4b_common_semantic_v1_20260809_002`，450 对、38 个指标、
  2,000 次 bootstrap。
- Week 5：从 80,000 条候选池中每场景抽取 10 条执行真实预标注。

### 观测结果

- Week 3 standardized 严格错误为 83 个 JSON parse error、2 个 Schema error；
  Schema 通过 365/450。
- Week 4 full 的商品/售后/行程 JSON 均为 100%；Schema 分别为
  99.5%、86.67%、36.0%。
- Week 4 full 平均延迟分别为 10.070、6.158、24.442 秒；P95 分别为
  13.043、8.525、41.988 秒。
- Week 5 抽样完成数为商品 10/10、售后 8/10、行程 9/10；总耗时分别为
  15.410、18.014、69.186 秒。线性外推 80,000 条约 50.6 小时，按记录的
  CNY 18.3158/小时约为 CNY 927 实例计算费用。

### 失败与限制

- 首次 4-shot pilot 因服务端图片上限为 1 而 15/15 返回 HTTP 400；将图片
  上限参数化并设为 8 后解决。
- 一次 pilot 和第一次 full winner 因 SSH 隧道中断而不完整；失败运行保留，
  后续使用新 run ID 重跑。
- Week 5 新输出目录的合成图片重建在 5,556 张时停止，避免重复模型无关工作；
  候选池改为复制已验证源，并确认三个 JSONL 的源/目标 SHA-256 相同。
- Week 5 全量预标注因成本边界没有启动；人工修订和三级质检也没有模型替代。

### 验证与关机

- `python -m unittest discover -s tests -v`：280/280 通过。
- `python scripts/validate_week3_evaluation.py --config
  configs/evaluation_week3_qwen3_vl_4b_gpu.yaml --run-id
  week3_qwen3vl4b_standardized_full_20260809_001`：`status=ok`。
- `python scripts/validate_week4_delivery.py --config
  configs/evaluation_week4_qwen3_vl_4b_gpu.yaml`：`status=ok`。
- `git diff --check`：通过，仅有 Windows 换行提示。
- vLLM 容器已正常停止并执行 `sync`，停止后无容器或 GPU 进程。
- ECS 控制台最终显示实例 `i-t4n4bp0crxx5pejoikjh` 为“已停止”和
  “节省停机模式”；固定公网 IP 已释放，页面只保留私网 IP。
