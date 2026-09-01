# Trip_Project

面向 OTA 场景的多模态理解、视觉检索、智能售后和行程规划系统。项目以
`Qwen/Qwen3-VL-8B-Instruct`、PEFT adapter、CLIP 和 Milvus 为核心，提供版本锁定的
FastAPI 服务、结构化输出校验、失败关闭以及可复验交付包。

## 正式版本

- Release：`trip-qwen3-vl-8b-week8-final-v1`
- 配置：`configs/releases/qwen3_vl_system_final_v1.json`
- 基座：`Qwen/Qwen3-VL-8B-Instruct`
- 推理：Transformers + PEFT，vLLM 仅保留为可选兼容后端
- 商品链路：Week 8 v12 商品观察与字段校验
- 行程链路：Week 8 v13 行程运行时
- 检索：CLIP 512 维 + Milvus HNSW/COSINE
- 本地交付包：`outputs/releases/trip-qwen3-vl-8b-week8-final-v1`

正式版本的证据边界、哈希和待优化项见：

- `reports/final_delivery_status.md`
- `reports/project_summary.md`
- `docs/model_handoff.md`

## 系统能力

```text
图片 / 文本 / 对话历史
        |
        v
Qwen3-VL 场景理解 + Prompt/Schema 约束
        |
        +--> 商品结构化标签
        +--> 售后问题识别与信息抽取
        +--> 多模态行程约束与行程框架
        +--> 有状态对话任务分派（beta）
        |
        v
CLIP 向量 + Milvus + 标量过滤
        |
        v
FastAPI 业务接口与显式错误响应
```

生产模式不使用静默模型 fallback。模型、Schema、adapter 或检索依赖不可用时，接口返回
明确错误，不用示例数据冒充真实结果。

## 目录

```text
src/          API、推理、检索、规划、数据、评测和训练代码
configs/      正式 release、Prompt、Schema 及可复现实验配置
scripts/      tripctl、验证、构建和数据工具
docker/       统一系统、Milvus、API 和 GPU 运行配置
tests/        unittest 测试
data/         轻量示例；原始和生成数据保持 Git 忽略
docs/         当前技术说明、代码导航和模型交接文档
reports/      按分支分层保存最终状态、稳定周报或开发证据
experiments/  保留的机器可读历史证据
```

`main` 不携带逐周过程报告；`stg` 每周保留一份稳定总结；`dev` 额外保留详细报告、bad case
和已接受的轻量权衡证据。聊天记录、会议转录和未来计划不进入任何正式分支。

## 分支职责

| 分支 | 用途 | 晋级规则 |
| --- | --- | --- |
| `dev` | 日常开发、详细实验痕迹和权衡证据 | 功能分支合入后执行完整验证 |
| `stg` | 稳定候选和每周一份的验收总结 | 仅接收已验证的 `dev` 代码，不接收原始运行输出 |
| `main` | 精简最终提交 | 仅保留确认交付的代码、技术文档和最终报告 |

常规流程为 `feature/* -> dev -> stg -> main`。不要直接在 `main` 开发。
核心代码入口和关键不变量见 `docs/code_guide.md`。

## 环境

创建基础环境：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

按用途安装依赖：

```bash
python -m pip install -r requirements-api.txt
python -m pip install -r requirements-test.txt
python -m pip install -r requirements-data.txt
python -m pip install -r requirements-milvus.txt
```

GPU 训练和推理依赖独立维护在 `requirements-llm.txt` 和 `requirements-training.txt`。
不要在原生 Windows Python 中默认安装 vLLM。

## 交付包验证

接手者取得 Git 外交付目录后，先执行：

```bash
python scripts/verify_final_delivery.py outputs/releases/trip-qwen3-vl-8b-week8-final-v1
python scripts/tripctl.py validate
python scripts/tripctl.py doctor
```

`verify_final_delivery.py` 会检查 runtime、adapter、retrieval、evidence 四层归档哈希、release
身份、adapter 身份、测试记录以及隔离运行时导入。

## 本地 API

只验证代码和健康接口：

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

完整模型服务需要：

- 固定 revision 的 Qwen3-VL-8B 本地缓存；
- 解压后的 PEFT adapter，并设置 `TRIP_ADAPTER_DIR`；
- 解压后的检索资产；
- 可用 Milvus；
- 本地 `.env`，不得提交密钥。

统一 Compose：

```bash
cp docker/system/.env.example docker/system/.env
# 修改 adapter、Hugging Face cache、retrieval 路径和 MinIO 本地凭据
docker compose -f docker/system/docker-compose.yml --env-file docker/system/.env config
docker compose -f docker/system/docker-compose.yml --env-file docker/system/.env up --build
```

启动后：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
python scripts/tripctl.py smoke --base-url http://127.0.0.1:8000
```

`/health` 只表示进程存活；`/ready` 才检查 release、模型、adapter、Schema、CLIP 和 Milvus。

## 业务接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 进程存活 |
| `GET` | `/ready` | 完整依赖就绪检查 |
| `POST` | `/v1/tasks/image-product-search` | 单图商品理解 |
| `POST` | `/v1/tasks/after-sales` | 单图售后分析 |
| `POST` | `/v1/tasks/itinerary-planning` | 图片与文字约束行程规划 |
| `POST` | `/v1/dialogue` | 有状态 beta 对话，需 `ENABLE_BETA_DIALOGUE=true` |
| `POST` | `/v1/visual-search` | CLIP/Milvus 视觉及混合检索 |

`/v1/image-understanding` 和 `/v1/travel-planning` 是兼容或示例接口；生产业务应使用
`/v1/tasks/*`。请求与响应字段见 `docs/api_design.md` 和运行时 `/docs` OpenAPI 页面。

## 验证

```bash
python -m unittest discover -s tests -v
python scripts/tripctl.py validate
python scripts/verify_final_delivery.py outputs/releases/trip-qwen3-vl-8b-week8-final-v1
docker compose -f docker/system/docker-compose.yml --env-file docker/system/.env.example config --quiet
git diff --check
```

正式 v1 交付包内保留的封装证据为 948/948；仓库完成交接清理后，当前保留代码的
回归集为 521/521。两者分别描述历史封装时点和当前精简工作树，不应混为同一测试规模。

## 数据与安全

- 不提交 Yelp 原始数据、生成数据集、冻结评测样本、模型缓存、adapter、向量库或运行输出。
- 不提交 `.env`、API Key、SSH 密钥或私有端点。
- 商品最终参考为模型生成 silver，不是人工视觉金标。
- 商品价位没有有效正支持，相关指标保持 `N/A/PENDING`。
- Recall 工程基准不等于人工业务相关性。

数据、评测与检索契约见 `docs/data_pipeline.md`、`docs/evaluation.md` 和
`docs/retrieval.md`；当前不可变决策见 `docs/architecture_decisions.md`。
