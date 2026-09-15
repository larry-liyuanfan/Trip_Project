# Identity-only 导入校验补丁

日期：2026-09-15。初始代码提交：`a695406538b30cf3caa03683bbfe9c7f1f1218a6`；
验收补修后的当前源码提交：`c3b784377b181aade555d700e72cbee9b63c827d`。
状态：工具链补齐并验证；真实数据前置状态仍为
`BLOCKED_PROTECTED_IDENTITY_COVERAGE`，没有解除训练阻断。

## 完成内容

现有 `protected_identity_inventory.py` 能检查项目配置已知的本地登记文件，但没有一个
独立导入契约来判断未来保管侧导出是否绑定到事先批准的数据来源。此次在同一模块增加
`validate_identity_only_import()`，并增加单一 CLI：
`scripts/validate_identity_only_import.py`。

CLI 固定要求 12 个 scope，采用 `json_utf8_sort_keys_compact_v1` 规范化版本。输入分成两层：

1. 总控或数据保管流程事先批准的 source registry，由命令行给出其预期文件 SHA-256；
   registry 对每个 scope 固定 source manifest SHA、data-lock SHA、必需身份字段以及图片或
   text-only 策略。
2. 保管侧导出 bundle，其固定入口是 `export_manifest.json`；命令行另给出该 manifest 的
   预期文件 SHA。manifest 必须回绑批准 registry 的 approval ID 和文件 SHA，并为全部
   12 个 scope 绑定唯一 JSONL 路径、真实文件 SHA、canonical rows SHA、行数、source
   manifest SHA 和 data-lock SHA。

程序实际读取并重算 manifest、approval registry 和每个 JSONL 的文件 SHA；重算 JSONL
行数组的 canonical SHA；拒绝目录逃逸、符号链接文件、未知或多余结构、重复 JSON key、
非平面内容、raw prompt/gold/prediction、image bytes 声明、非法/大写/伪造摘要、缺失字段、
scope 内重复 sample ID、跨 scope 同 ID 冲突、未知规范化版本以及少于 12 scope。
导出自报的 source manifest/data-lock SHA 只有与外部哈希绑定的批准 registry 完全一致才接受。

总控验收复现了初始实现的一项 P2：非 required 的 identity 字段会进入跨 scope 身份比较，
却没有先做类型/摘要校验；畸形的 `required_fields` 或 `image_identity_policy` 还可能触发
非稳定异常。补修后，对每个出现的 identity/SHA 字段都执行校验；optional 的 `null`/空串
明确视为未提供并在 identity fingerprint 中统一为 `null`，任何非空 optional identity 必须是
非空字符串，任何非空 optional SHA 必须是小写 64 位摘要。CLI 对畸形结构稳定返回 exit 2，
不打印 traceback、不创建输出。

成功状态刻意命名为 `VALIDATED_IMPORT_NOT_TRAINING_AUTHORIZATION`，并始终输出：

- `model_execution_authorized=false`；
- `coordinator_release_required=true`；
- `custodian_attestation_verified=false`；
- `human_annotation_support=null`；
- `leakage_check_status=NOT_RUN`、`overlap_count=null`。

因此，“字段齐全”只证明包的格式和已批准摘要绑定，不证明摘要确由原始保护内容生成，
不证明真人标注、无泄漏或可以训练。

## 使用入口

在取得保管侧 bundle、外部批准 registry 及两者的预期 SHA 后运行：

```powershell
python scripts/validate_identity_only_import.py `
  --bundle-root <identity-export-bundle> `
  --export-manifest-sha256 <pre-communicated-export-manifest-sha256> `
  --approved-sources <approved-source-registry.json> `
  --approved-sources-sha256 <coordinator-approved-registry-sha256> `
  --output <new-exclusive-validation-report.json>
```

结构与绑定全部通过时 exit code 为 0，但仍不授权训练；格式、哈希、来源身份或覆盖失败时
exit code 为 2，且不创建输出。已有输出不覆盖。

## 实际验证

只使用 `TemporaryDirectory` 中即时构造的 synthetic fixture；fixture 明确命名为 unit test，
未写入项目真实数据、登记索引或报告。没有打开/生成 Fresh Test 或 sealed final 的样本、
标签、预测，也没有 SSH、GPU、模型推理、依赖安装或外部服务。

执行命令：

```powershell
python -m unittest discover -s tests -p test_protected_identity_inventory.py -v
python scripts/validate_identity_only_import.py --help
git diff --check
```

结果：32/32 tests passed，0 failed、0 error、0 skip；测试框架报告 1.076 秒，外层计时
1.242 秒。原有 inventory 18 项全部保留通过，导入校验 14 项通过。CLI fixture 覆盖
成功 exit 0、失败 exit 2、失败不落盘和输出不可覆盖；12-scope 成功 fixture 仍保持
`model_execution_authorized=false`。新增回归把非法 optional `dialogue_text_sha256` 写入后，
重新计算 JSONL file SHA、canonical rows SHA 与 export manifest SHA，API 和 CLI 仍拒绝；
list/dict 类型的畸形 approval 也稳定 exit 2。`git diff --check` 通过。

提交后文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `src/evaluation/protected_identity_inventory.py` | `8fc98ac497f975b80860bb9a63d043e2e980afa73af6604a3805422eb5248a70` |
| `scripts/validate_identity_only_import.py` | `7d7dd123d750218828cb7033e6c3fa7a2ed8b15eacaa0b39a005a3a7ba1ffd6e` |
| `tests/test_protected_identity_inventory.py` | `1805c430bcafaa684df66362398fd32c4120cf0ac40d0de6c7e1386ab8b08162` |

## 仍缺证据与下一步

本轮没有真实 approved-source registry，也没有真实保管侧 bundle，因此没有导入任何一条
真实 identity，真实验证分母为 0。2026-09-14 报告中的 12-scope 缺口、96,788 行历史
登记行数之和、`overlap_count=null` 和所有 `NOT_RUN` 语义保持不变。

解除阻断仍需数据保管侧基于原始 manifest/data lock 生成 identity-only 导出，并由总控通过
独立渠道批准 registry SHA 和 export manifest SHA。保管侧还需证明 query/source/content
摘要的规范化及原始内容派生过程；本工具没有原始保护内容，不能验证这一步，也不把来源方
文字声明或任意自报 hash 当成可信来源。真实导入即使 exit 0，仍须另做新 train/dev 与所有
保护池的重叠检查，并由总控单独放行；不得修改旧锁、重建已消费测试或凭改 ID 获得资格。

正式 release、旧实验 JSON、既有验收证据、API、模型、默认 adapter、README 和证据索引
均未修改。工作包到此结束，不启动训练、后续实验或监控。
