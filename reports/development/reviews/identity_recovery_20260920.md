# Trip 受保护身份索引恢复（2026-09-20）

## 结论

本包完成了一个**有限的、可复现的 train/development 身份恢复**，没有解除原 12-scope
门禁。当前总状态仍为 `BLOCKED_PROTECTED_IDENTITY_COVERAGE`，训练授权仍为 `false`，因此
不应重新提交此前中断的 Spartan GPU 训练/推理作业。

基于基线 Git SHA `08a4a567df2e3a36f3679b3ac1e26f74ce9e91aa`，恢复脚本只调用
v4/v5/v7/v9 的 `training` 和 `development` split builder；逐 split 重建出的原始 manifest
同时匹配已提交 lock 的行数、canonical SHA-256 和文件 SHA-256。临时图片及包含 label、prompt、
dialogue 正文的 manifest 在临时目录退出时删除，只保留 Git 忽略目录中的扁平 identity-only
导出。`final`/`test` 没有生成、打开或用于调参。

恢复结果为 10 个安全子 scope、2,328 行（training 1,992，development 336）。10 个子 scope
之间的 `sample_id/source_id/image/query/source_record/dialogue/content` 七类 identity collision
均为 0。这个 PASS 只覆盖这 10 个恢复子 scope，**不是**原 12-scope 全量泄漏验收。

## 恢复产物与规范

- 入口：`scripts/recover_synthetic_train_dev_identities.py`
- 测试：`tests/test_synthetic_identity_recovery.py`
- Git 外产物：`outputs/identity-recovery-20260920/synthetic-train-dev-v1`
- recovery manifest SHA-256：
  `094e91f8d017aefb91477c166c06b2c14f8594da0eea303b21ce0ac9d2a26078`
- 脚本 SHA-256：
  `c4eff66404036497a9e77761e23ce83fd1cab6647c434a38e317c756b5a19435`
- 测试 SHA-256：
  `641b411e480be97de449eda413dcf419d098fa6903615cfdbc66a2dda11798c8`
- JSON canonicalization：UTF-8、key sort、compact separators。
- product `content_sha256`：原图片 bytes SHA-256。
- dialogue `content_sha256`：`canonical_json_sha256({"dialogue": text})`。
- VLM `query_sha256`：对 schema version、scenario、实际 prompt 与 content SHA 的 canonical
  JSON 取 SHA-256；导出不包含 prompt 或正文。
- `source_record_sha256`：来自经过 exact manifest lock 校验的生成行，不用 sample/group/template ID
  伪装。

| 子 scope | split | 行数 | 生成器 | lock | 结果 |
| --- | --- | ---: | --- | --- | --- |
| v4 search | training / development | 24 / 24 | `scripts/build_exploration_pool_v4.py` | `exploration_pool_lock_v4.json` | exact lock PASS |
| v4 VLM | training / development | 288 / 36 | 同上 | 同上 | exact lock PASS |
| v5 VLM | training / development | 528 / 48 | `scripts/build_context_focus_pool_v5.py` | `context_focus_pool_lock_v5.json` | exact lock PASS |
| v7 VLM | training / development | 512 / 96 | `scripts/build_semantic_robustness_pool_v7.py` | `semantic_robustness_pool_lock_v7.json` | exact lock PASS |
| v9 VLM | training / development | 640 / 132 | `scripts/build_semantic_robustness_pool_v9.py` | `semantic_robustness_pool_lock_v9.json` | exact lock PASS |

## 原 12-scope 去向

| 原 scope | 当前分类 | 精确来源/依赖 | 仍缺什么 |
| --- | --- | --- | --- |
| `system_repair_fresh_v2` | metadata-only，混合 | `src/training/system_repair.py`；`configs/system_repair/qwen3_vl_8b_system_repair_v2.json`；`system_repair_identity.jsonl` 2,268 行 | train/development 与 consumed test 混合；缺 query/source-record 内容身份 |
| `evaluation_exclusion_v1` | metadata-only，split unknown | `scripts/prepare_week3_evaluation.py`；`evaluation_exclusion_manifest.jsonl` 450 行 | split 未记录；缺 query/source-record 内容身份 |
| `evaluation_exclusion_v2` | metadata-only，split unknown | 同上；`evaluation_exclusion_manifest_v2.jsonl` 450 行 | 同上 |
| `week6_consumed` | metadata-only，consumed/mixed | `src/training/week6_data.py`；`configs/week6/qwen3_vl_8b_qlora_final300_v4.json`；`week6_identity.jsonl` 79,936 行 | 只有 sample/split membership；validation 已消费，未读取原行 |
| `week7_v3` | metadata-only，混合 | `src/training/week7_data.py`；`configs/week7/qwen3_vl_8b_multitask_context_v3.json`；3,228 行 | train/development/test 混合；缺 query/source-record 内容身份 |
| `week7_v4` | metadata-only，混合 | 同一 builder；`qwen3_vl_8b_multitask_context_v4.json`；3,228 行 | 同上 |
| `week7_v4_fix1` | metadata-only，混合 | 同一 builder；`qwen3_vl_8b_multitask_context_v4_fix1.json`；3,228 行 | 同上 |
| `week7_v4_fix2` | metadata-only，混合 | 同一 builder；`qwen3_vl_8b_multitask_context_v4_fix2.json`；3,228 行 | 同上 |
| `week8_audit_exclusion` | metadata-only，混合 | `src/training/week8_product.py`；`configs/week8/audit_repair_v1.json`；460 行 | train/test 混合且只覆盖 Week 8 子集；缺 query/source-record 内容身份 |
| `synthetic_v4_images` | **部分恢复** | v4 generator + committed v4 lock；恢复全部 264 条 train/dev 图片 identity | 48 条 final 图片保持未生成；原 scope 仍非 complete |
| `synthetic_v5_complete` | **部分恢复** | v5 generator + committed v5 lock；恢复 576 条 train/dev product/dialogue identity | 48 条 final 保持未生成；原 scope 仍非 complete |
| `remaining_protected_pools` | **部分恢复** | v4 dialogue 108、v7 608、v9 772，共 1,488 条 train/dev identity | v10、其他 Week 8、weak pools 及 sealed/unknown 范围仍未定位或认证 |

因此原始 scope 完整覆盖仍为 **0/12**；“部分恢复”不能被换写为 ready。对 mixed/unknown
来源，本包在身份元数据层停止，没有读取 raw sample、label、prediction 或 sealed test。

## 正负结果

正结果：

- 2,328/2,328 恢复行均绑定到逐 split exact committed lock。
- 10/10 子 scope 输出只含扁平 identity metadata，不含图片 bytes、prompt、dialogue、gold、
  messages 或 prediction。
- 10-scope 七维跨 scope collision 均为 0。
- 原 `outputs/price-conflict-20260914/protected` 的 11 个文件 SHA-256 与既有登记一致，未改写。

负结果/保留阻断：

- 0/12 原始 scope 达到完整、经批准的 identity 导出要求。
- 未执行原 12-scope overlap；`overlap_count` 仍不得写 0。
- 未取得 custodian approval、人工标注证明或 coordinator release。
- 未训练、未推理、未重新消费 Fresh Test 120、未产生新的业务质量指标。
- 未调用 Spartan；先前作业中断不构成当前重提依据。

## 验证

实际执行：

```text
python -m unittest tests.test_synthetic_identity_recovery tests.test_protected_identity_inventory -v
# 35 tests PASS（首次 focused regression）

python -m unittest tests.test_synthetic_identity_recovery -v
# 4 tests PASS（加入跨 scope fail-closed 检查后）

python scripts/recover_synthetic_train_dev_identities.py \
  --output-dir outputs/identity-recovery-20260920/synthetic-train-dev-v1
# exit 0；10 scopes / 2,328 rows；cross-scope identity PASS
```

本包不声称 full suite、GPU、Milvus、模型推理或正式 release 验证已运行；这些都不属于本次
身份恢复的风险面。

## 下一门槛

只有在以下条件同时满足后，才考虑由总控提交新的 Spartan 作业：12 个原 scope 的批准
identity-only 导出完整；每个 source manifest/data lock 绑定通过；全量 source/image/query/
content overlap 检查完成；总控明确 release。否则保留 `BLOCKED`，且不因已有 2,328 行部分
恢复而启动训练。
