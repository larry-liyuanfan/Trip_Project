# API 设计

正式服务由 `src/api/app.py` 创建，路由集中在 `src/api/routes.py`，边界模型位于
`src/inference/schemas.py`。以下为接手所需的稳定接口；完整字段以运行时 `/docs` 生成的
OpenAPI 为准。

## 状态接口

### `GET /health`

只检查 API 进程存活，不能代表模型和检索服务可用。

### `GET /ready`

检查正式 release 配置、基座模型、adapter、Prompt/Schema、CLIP 和 Milvus。任何关键依赖
失败均返回 HTTP 503，并在响应中列出失败检查项。

## 三个业务任务

### `POST /v1/tasks/image-product-search`

### `POST /v1/tasks/after-sales`

两个接口共用单图请求：

```json
{
  "image_urls": ["file:///absolute/path/to/image.jpg"],
  "text_context": null
}
```

只允许一张非空图片，不接受额外文字上下文。

### `POST /v1/tasks/itinerary-planning`

```json
{
  "image_urls": ["file:///absolute/path/to/reference.jpg"],
  "text_context": "上海出发，2天，预算中等，希望包含博物馆和步行街"
}
```

接受 1-8 张图片和非空文字约束。明确天数必须在 1-14 天内，日期和业务约束在模型输出后
再次校验。

三个任务统一返回：

```json
{
  "scenario": "itinerary_planning",
  "result": {},
  "schema_valid": true,
  "business_valid": true,
  "prompt_version": "week8_itinerary_actionable_v5",
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "adapter": "trip-qwen3-vl-8b-system-repair-checkpoint-87-v1",
  "release_id": "trip-qwen3-vl-8b-week8-final-v1",
  "attempts": [],
  "total_latency_ms": 0
}
```

`attempts` 保留每次模型原始输出、错误、延迟和 token 信息。Schema 或业务校验首次失败时
最多进行一次模型级纠错；服务不会用脚本猜测或补写语义字段。

## 对话

### `POST /v1/dialogue`

默认关闭，启用需设置 `ENABLE_BETA_DIALOGUE=true`。

```json
{
  "messages": [
    {"role": "user", "content": "把行程改成两天", "image_urls": []}
  ],
  "image_urls": [],
  "state": {"days": 1},
  "task": "auto"
}
```

支持最多 32 个 turn 和合计最多 8 张图片。图片只能附着在用户 turn。响应明确区分
`COMPLETED`、`NOT_COMPLETED` 和 `STATE_UPDATED`，并包含 `task_result`、`task_error`、
`tool_calls`、合并后的 state 及模型 attempts。质量层固定为 `DIALOGUE_BETA`。

## 视觉检索

### `POST /v1/visual-search`

```json
{
  "image_urls": ["file:///absolute/path/to/image.jpg"],
  "query_text": "上海中等价位餐厅",
  "city": "Shanghai",
  "business_category": "restaurant",
  "price_range": "mid_range",
  "top_k": 5,
  "retrieval_mode": "hybrid"
}
```

生产模式使用 CLIP/Milvus，并对允许的 city、business category 和 price range 做标量过滤。
无法应用的自由文本约束通过 `query_status` 和 `unapplied_query_text` 显式披露。检索依赖失败
返回 503/502，不回退到样例结果。

## 兼容接口

- `/v1/image-understanding`：旧通用图片理解入口。
- `/v1/travel-planning`：基于轻量样例目录的非生产 planner；生产模式返回 404。
- `/v1/project-status`：静态项目状态展示入口。

新业务集成应使用 `/v1/tasks/*` 和 `/v1/visual-search`。
