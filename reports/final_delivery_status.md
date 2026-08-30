# 最终交付状态

## 版本

- 正式 release：`trip-qwen3-vl-8b-week8-final-v1`
- 默认配置：`configs/releases/qwen3_vl_system_final_v1.json`
- 基座：`Qwen/Qwen3-VL-8B-Instruct`
- 基座 revision：`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- adapter：`trip-qwen3-vl-8b-system-repair-checkpoint-87-v1`
- adapter model SHA-256：`c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a`
- 状态：正式交付；优化完成度仍为 `PARTIAL`

用户已确认以 Week 8 后续优化版本作为正式交付版本，Week 7 严格研究门禁不阻止本次
晋级。该决定不改变历史实验结果，也不把自动 silver 改写为人工金标。

## 已完成

| 方向 | 状态 | 交付内容 |
| --- | --- | --- |
| 商品理解 | 已优化 | v12 商品观察链完成独立 final，JSON/Schema 100%、请求失败 0 |
| 主体与业态 | 已优化 | 主体复查、类别校正和证据矛盾约束进入链路 |
| 商品设施 | 部分优化 | development F1 有提升，但额外复查增加约 28.28% 延迟，未启用该高成本路径 |
| 行程规划 | 已优化 | v13 固定直接请求 3/3 首轮通过；固定对话探针由二次生成降为一次 |
| 对话路由 | 已优化 | 状态更新、实际任务分派、tool 状态和失败关闭已实现 |
| API runtime | 已优化 | 场景专属输入、业务后校验、一次模型纠错、严格 `/ready` 和 release 身份 |
| 视觉检索 | 已完成工程链 | CLIP 512 维、Milvus HNSW/COSINE、标量过滤、未应用约束披露 |
| 系统封装 | 已完成 | runtime、adapter、retrieval、evidence 四层包及离线验证器 |

## 待优化

| 方向 | 当前边界 |
| --- | --- |
| 商品价位 | 最终参考没有正支持，指标为 `N/A/PENDING` |
| 商品视觉准确率 | 参考为模型生成 silver，不能声明人工视觉准确率 |
| 商品语义 | 多主体、风格漏识别和部分设施仍有误差 |
| 商品延迟 | 当前缓存无实质提速；高成本设施复查未进入正式链路 |
| 对话研究能力 | 当前为业务 beta 路由，不宣称通过 Week 7 strict research gate |
| 检索业务相关性 | 工程 Recall 不等同于独立人工相关性判断 |

## 交付包

本地唯一 Git 外交付目录：

`outputs/releases/trip-qwen3-vl-8b-week8-final-v1`

| 文件 | SHA-256 |
| --- | --- |
| `runtime.tar.gz` | `29959a7677ccf8ecd059444d9cacf76481b07589d46ecd3acf64013307354ea5` |
| `adapter.tar.gz` | `f74c078738fa0229574114986c58040bbc280e11ba4ec06558c9a488c2de619d` |
| `retrieval.tar.gz` | `3cdb98f4d50bc72ae53c4e7e96d823ea5b08af93f41df5d14ff1118d12d1a15b` |
| `evidence.tar.gz` | `fecdb55b61a69b7fcc5d1f84ff6623542f07f3934690c24dce1a788a9e6d8253` |

验证命令：

```bash
python scripts/verify_final_delivery.py outputs/releases/trip-qwen3-vl-8b-week8-final-v1
python scripts/tripctl.py validate
python -m unittest discover -s tests -v
```

正式封装时完整单元测试为 948/948，包验证和 runtime 隔离导入为 `PASS`。项目整体经历和
各阶段结论见 `reports/project_summary.md`；运行交接见 `docs/model_handoff.md`。
