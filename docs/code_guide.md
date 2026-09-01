# 代码导航与维护边界

## 请求主链路

1. `src/api/routes.py` 负责 Pydantic 请求边界、HTTP 状态码和服务获取，不承载模型语义。
2. `src/inference/system_runtime.py` 读取正式 release、加载 backend、执行场景 Prompt、Schema
   校验及一次模型级纠错。
3. 商品场景先进入 `product_observation.py`，再按配置选择主体、风格或设施复查模块。
4. `business_validation.py` 只做可解释业务后校验，不补写模型未输出的标签。
5. `src/retrieval/visual_search.py` 将 CLIP 查询向量交给 `milvus_vectors.py`，过滤表达式只能由
   固定白名单构造。

## 关键不变量

- `ReleaseSettings.load` 必须绑定 release ID、模型 revision、adapter SHA、Prompt、Schema 和
  商品观察配置哈希；配置失败应阻止 `/ready`。
- 正式运行失败关闭。模型、adapter、Schema 或 Milvus 不可用时返回明确错误，不能用样例
  输出或 keyword 结果冒充成功。
- Schema 失败最多执行一次模型级纠错。纠错消息可以描述验证错误，但不能注入金标、猜测
  缺失字段或修改原始输出。
- 商品观察首先记录可见事实，再映射业务字段；`unknown` 是有效的不确定状态。
- 评测运行目录不可覆盖，`mock`、`dry-run` 和 `live` 结果不能混用；指标必须带支持数。
- 训练与评测数据按 sample、source、图片 SHA-256、group 和 constraint template 隔离。

## 数据与训练

- `src/data/` 负责来源解析、候选构建、人工状态和隔离；模型预标注始终保持 silver 身份。
- `src/evaluation/` 负责不可变运行、Schema、指标与错误导出，不负责修改预测。
- `src/training/` 保留 Week 6-8 可复现训练与评测工具。训练配置必须固定随机种子、数据锁、
  adapter 身份和恢复点；冻结测试结果不能回流为训练标签。

## 修改建议

- API 变化先修改 Pydantic Schema 和失败路径测试，再修改 handler。
- release 行为变化必须创建新版本配置和交付包，不覆盖 final v1。
- 复杂函数注释应解释身份、并发、缓存、数据隔离或失败关闭原因；简单赋值和显然循环不写
  逐行注释。
- 新实验的详细报告和小型权衡证据留在 `dev`；稳定周总结进入 `stg`；`main` 只接收最终
  交付说明。
