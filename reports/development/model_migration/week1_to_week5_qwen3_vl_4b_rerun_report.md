# Qwen3-VL-4B Week 1-5 重跑报告

## 1. 执行摘要

- 日期：2026-08-09
- Git 基线：`4bee591`（`dev`，叠加本次未提交工作区改动）
- 模型：`Qwen/Qwen3-VL-4B-Instruct`
- 推理后端：vLLM 0.11.0，`bfloat16`
- 计算资源：阿里云 ECS `ecs.gn7i-c8g1.2xlarge`，NVIDIA A10 24 GiB，8 vCPU，30 GiB 内存
- 数据盘：`/data`，约 196 GiB；实验结束前约使用 13 GiB、可用 174 GiB
- 服务边界：vLLM 仅监听服务器回环地址，通过 SSH 本地转发访问

本次完成了 Week 1 实图 API 验证、Week 2 冻结数据产物验证、Week 3 的 450 条 baseline/standardized 全量重跑、Week 4 Prompt pilot 与 450 条胜出 Prompt 全量重跑，以及 Week 5 每场景 10 条的真实预标注测量。Week 5 的 80,000 条全量预标注和人工修订/质检没有执行：前者按实测速率约需 50.6 小时和约 CNY 927 的实例计算费用，后者必须由人工完成，不能由模型冒充。

## 2. 基础设施与模型部署

数据盘已挂载到 `/data`，模型缓存位于 `/data/huggingface`。Qwen3-VL-4B 模型权重约 8.591 GiB，vLLM 空载时 GPU 显存占用约 14,690 MiB。为支持 Week 4 的 7-shot 加查询图，服务端图片上限设为 8；默认仍可通过 `VLLM_LIMIT_IMAGES` 配置。

部署与本地配置见：

- `docs/gpu_ecs_deployment.md`
- `docker/gpu/docker-compose.yml`
- `scripts/deploy_gpu_vllm.sh`
- `configs/model_qwen3_vl_4b_gpu.yaml`
- `configs/inference_qwen3_vl_4b_gpu.yaml`

## 3. 分周结果

### Week 1：API 与实图推理

真实图片理解 smoke 通过，单次请求耗时 5.589 秒。Qwen3-VL-4B 会把部分可选标量字段输出为空数组或单元素数组，因此客户端增加了受限归一化：只对已知可选文本字段处理 `[] -> null`、`[value] -> value`，不改变必填字段、枚举或 Schema。

### Week 2：Yelp 数据产物验证

本次没有重复执行模型无关且成本较高的全量数据处理与 CLIP 编码，而是验证冻结产物：

| 项目 | 记录数 |
| --- | ---: |
| business | 150,346 |
| review | 6,989,830 |
| photo | 200,100 |
| strong 对齐 | 96,733 |
| medium 对齐 | 199,994 |
| weak 对齐 | 36,673 |

因此，Week 2 的结论是“既有真实产物通过验证”，不是“使用 4B 重新生成数据或重新运行 CLIP”。

### Week 3：零样本 baseline 与标准化 Prompt

数据集固定为 `week3_evaluation_v2`，450 条样本哈希为 `3e900e64...ad648c`。

| 运行 | 样本 | 请求错误 | 状态 |
| --- | ---: | ---: | --- |
| `week3_qwen3vl4b_baseline_full_20260809_001` | 450/450 | 0 | completed |
| `week3_qwen3vl4b_standardized_full_20260809_001` | 450/450 | 0 | completed |

标准化 Prompt 的严格评分共有 83 个 JSON 解析错误和 2 个 Schema 错误，365/450 通过 Schema。分场景 JSON/Schema 为：售后 100%/100%，商品 100%/99.5%，行程 17%/16%。这说明请求层稳定，但行程长输出经常无法形成完整、严格合规的 JSON。

配对比较 `week3_qwen3vl4b_baseline_vs_standardized_20260809_001` 包含 450 对样本和 1,000 次 bootstrap。baseline 自然语言采用固定词法编码轨道，standardized 采用严格结构化轨道；两条原始业务轨道不直接做因果差值解释。

### Week 4：Prompt 优化与胜出方案

有效 pilot 包含 zero-shot、4-shot 和 7-shot，各 15 条。最终按场景选择：

| 场景 | 胜出 Prompt |
| --- | --- |
| 商品检索 | `standardized_v2` |
| 售后 | `fewshot_4_v2` |
| 行程 | `fewshot_4_v2` |

全量运行 `week4_qwen3vl4b_winners_full_20260809_002` 完成 450/450，请求错误为 0。主要结果如下：

| 场景 | 样本 | JSON | Schema | 平均延迟 | P95 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 商品检索 | 200 | 100% | 99.5% | 10.070 s | 13.043 s |
| 售后 | 150 | 100% | 86.67% | 6.158 s | 8.525 s |
| 行程 | 100 | 100% | 36.0% | 24.442 s | 41.988 s |

共同语义比较 `qwen3vl4b_common_semantic_v1_20260809_002` 对 450 对输出使用同一固定词法编码器、38 个聚合指标、2,000 次 bootstrap。该轨道只支持固定词法编码器下的成对解释，不能替代人工语义评审。

失败过程也保留为证据：一次 4-shot pilot 因服务端图片上限为 1 而 15/15 返回 HTTP 400；另一次 4-shot 和第一次全量 winner 运行因 SSH 隧道中断而不完整。修复图片上限和隧道后重新运行，不覆盖失败产物。

### Week 5：候选池与预标注边界

复用并逐文件校验了既有 80,000 条候选池副本：

| 场景 | 记录数 | SHA-256 |
| --- | ---: | --- |
| 商品检索 | 50,000 | `F5302007DC77DF5F48CDE4C50002402549F1CB4B350855466226488360D2EFE7` |
| 售后 | 20,000 | `78F411750FCE487FCFA4D80BB7E81E659D3C15B43961C121D5AB4D1AFD99B312` |
| 行程 | 10,000 | `4072260173F0B25CF7D5D63AB694F0849B351A483F42E4C39B9A99C5B9A17E75` |

新输出与已经验证的源文件哈希一致。独立逐图重新计算全部图片哈希在超过 10 分钟后超时，因此没有声称完成本轮完整图像重哈希。

每场景抽取 10 条真实样本执行预标注：

| 场景 | 完成 | Schema 失败 | 总耗时 |
| --- | ---: | ---: | ---: |
| 商品检索 | 10 | 0 | 15.410 s |
| 售后 | 8 | 2 | 18.014 s |
| 行程 | 9 | 1 | 69.186 s |

售后失败原因为 `ocr_text` 被输出为非数组；行程失败原因为额外字段 `source_evidence`、`transport`。按这 30 条实测吞吐线性外推，80,000 条约需 50.6 小时。按本次记录的约 CNY 18.3158/小时计算，实例计算费用约 CNY 927，尚不含人工修订和三级质检。因此未在没有新增成本授权的情况下启动全量任务。

本轮为了测试新输出目录曾开始重建模型无关的合成售后图片，生成 5,556 张后主动停止；该部分只是未完成过程产物，不作为新的候选池交付。

## 4. 代码修正

- `src/inference/client.py`：兼容 Qwen3-VL 对已知可选文本字段的空数组/单元素数组输出。
- `src/data/week5_workflow.py`：允许无 API key 的本机回环 OpenAI-compatible endpoint，仍拒绝无密钥的远程端点。
- `src/evaluation/week4_analysis.py`：当 baseline token 状态为 `PENDING_not_recorded` 时，明确写入 `null`，不再产生误导性数值。
- GPU Compose、部署脚本和模型配置新增 4B 白名单、回环绑定、模型缓存和可配置图片上限。

## 5. 限制与结论

Qwen3-VL-4B 在单张图商品和售后任务上的结构化稳定性较好；行程任务仍受到长输出与严格 Schema 的明显限制。Few-Shot 改善了部分场景，但没有消除行程合规问题。Week 5 的完整交付仍依赖两项外部条件：明确批准约 50.6 GPU 小时的全量预标注成本，以及真实人工修订/质检资源。

服务器内的 vLLM 容器已通过正常 `down` 停止并执行 `sync`，停止前确认无残留容器或 GPU 进程。ECS 控制台最终明确显示实例 `i-t4n4bp0crxx5pejoikjh` 为“已停止”和“节省停机模式”；固定公网 IP 已释放，页面只保留私网 IP。计算资源已暂停计费，系统盘和数据盘仍按云盘规则持续计费。
