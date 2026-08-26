# Week 8 商品理解与并行优化报告

## 1. 结论

Week 8 在独立 `feature/week8-product-understanding` 分支完成了商品、对话、延迟与检索的
全自动扩展执行。自 2026-08-27 起没有任何人工标注、复核或验收；新增数据与 target 均为
`programmatic_silver`，human annotation/review/acceptance=`0/0/0`。

最终继续选择正式 checkpoint-87 adapter 和 `week8_product_field_check_v1`。fresh v7 的
唯一 final test 将商品综合从同口径正式 Prompt 的 `0.819003` 提高到 `0.857729`
（`+0.038725`）；设施 micro-F1、price unknown 和完整性明显提高，业态与风格的轻微回退
如实保留。JSON/Schema 仍为 `100%/100%`，失败率为 `0`，支持数未改变。可观察证据
两阶段方案和 checkpoint-5 continuation SFT 均经 development 实测后被拒绝，没有用
final test 选择 Prompt、adapter 或 checkpoint。

对话首轮改为代码组装三键契约，固定 5 条样本的格式合规、状态召回/准确与整状态准确均为
`1.0`，纠错、失败和 fallback 均为 `0`。真实图片延迟只选择已通过商品质量 test 的
384 token 上限；图片 cap + processor cache 因 mean 变慢而未进入最终 release。检索最终
选择真实 Milvus Lite `hybrid_weighted`，NDCG@10 `0.125654→0.564459`，代价是 P95
增加约 `2.15 ms`。release v7 四场景真实 smoke 为 `PASS`。

## 2. 初始 v4 执行、历史基线与资产审计

本节至第 9 节保留 2026-08-26 初始 v4 的不可变历史证据；2026-08-27 的全自动扩展结果
见第 10 节，并以第 10 节作为当前交付结论。

### 2.1 历史正式基线

- 底座：`Qwen/Qwen3-VL-8B-Instruct`，revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。
- 正式 adapter：`trip-qwen3-vl-8b-system-repair-checkpoint-87-v1`，
  `adapter_model.safetensors` SHA-256
  `c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a`。
- 正式商品 Prompt：`system_repair_product_compact_v3`。
- 已消费历史 fresh test：商品综合 `0.780639`，风格/设施/价位支持 `25/30/5`，
  JSON/Schema `100%/100%`，失败率 `0`。该集合只作历史描述，未用于 Week 8 选择。
- 历史生产 smoke 商品耗时 `37.63 s` 仅为单条结果，不作为稳定延迟对照。

### 2.2 本地、Spartan 与阿里云审计

- 本地 C/D/E/F 盘、常见缓存/临时/备份目录及 WSL 未找到可直接恢复的官方 Yelp 源归档；
  找到的 system-repair/Week 6/Week 7 锁和 adapter 均按冻结历史保留。
- Spartan 的 system-repair adapter、发布配置、历史锁和检索交接归档可验证；旧 Yelp 图片
  子集只有 65,509 个 medium pairs，不能满足新的未消费商品锁。
- 阿里云历史地址 `43.98.181.94` 的主机身份已变化且当前密钥认证失败；历史地址
  `8.219.248.221` 也未恢复认证。没有可用 CLI/profile，因此无法证明数据盘内容已不存在，
  也没有启动、停止、释放、挂载或修改任何云资源。
- 因合法源缺失，从 Yelp 官方 URL 重建。下载 job `29627585` 完成：JSON ZIP
  `4,345,335,132` bytes，Photos ZIP `7,447,210,067` bytes。重建 job `29627942`
  完成，得到 business `150,346`、photos `200,100`、现有图片 medium pairs `65,509`。
- 官方归档实际是 ZIP 内嵌 gzip-compressed tar；失败尝试及修复均保留 Slurm 日志。重建
  manifest：`outputs/week8/source_rebuild/yelp_open_dataset_20250115_v1.json`。

## 3. 商品数据身份与错误切片

### 3.1 fresh source

正式 source build job `29628987` 为 `COMPLETED 0:0`，耗时 `00:01:21`：

| 项目 | 结果 |
|---|---:|
| 初始候选抽取 | 3,000 |
| 可读且历史哈希未消费 | 851 |
| 正式锁定 source | 800 |
| 历史 `image_sha256` 冲突 | 2,140 |
| 不可读图片 | 6 |
| 候选内部图片哈希重复 | 3 |
| source/group/image 唯一数 | 800/800/800 |
| 与历史 source/group/image 重叠 | 0/0/0 |
| OTA source 分类 | 餐饮 792 / 景点 7 / 酒店 1 |
| native caption / photo-label fallback | 108/692 |

source manifest SHA-256 为
`582f7e4700078f41234082d16043a09c59f248a36f9300995663c705525ce195`，图片聚合
SHA-256 为 `3b8788e3551b38f4381b3da43cc79aad130323f090fcc8fcaf199e6216476ede3`。
所有标签保持 `programmatic_silver`，human 数为 `0`。三次失败构建的 `390 MiB` 可再生
图片副本在逐路径校验后删除；失败 Slurm 日志、正式拒绝清单和成功 manifest 保留。

### 3.2 v4 锁与支持

正式数据锁 job `29629630` 为 `COMPLETED 0:0`：

- 数据版本：`week8_product_understanding_20260826_v4`。
- train/development/test：`400/60/60`，test 初始状态 `LOCKED_UNCONSUMED`。
- 锁 SHA-256：`49d238b03d1ac3fa0ba1e20151ef4b24277cb58db66a731f761becf5b3411e7f`。
- `sample_id/source_id/image_sha256/group_id/constraint_template_id` 五维跨 split 冲突为 `0`。
- dev：酒店/景点/餐饮 `2/2/56`，风格多标签/单标签/空 `6/45/9`，设施非空 `60`，
  价位应 unknown `60`。
- test：酒店/景点/餐饮 `2/1/57`，风格多标签/单标签/空 `8/50/2`，设施非空 `60`，
  价位应 unknown `60`。
- 风格和设施包含 Yelp 商家元数据弱银标，并在 target provenance 中明确声明不是图片直接
  证据；价位元数据没有晋升为视觉标签。v3 中间锁因风格/设施切片支持不足被弃用，v3 test
  保持未消费。

## 4. Prompt development

job `29632502` 在 L40S 上 `COMPLETED 0:0`，耗时 `00:07:21`。三版 Prompt 使用同一
60 条 development、同一模型/adapter/生成参数和完全相同指标支持。

| 指标 | 正式 Prompt | 紧凑字段检查 | 视觉证据约束 |
|---|---:|---:|---:|
| 商品综合 | 0.766765 | **0.815131** | 0.698464 |
| 业态准确率 | 0.883333 | **0.933333** | 0.916667 |
| 风格 P/R/F1（micro） | 0.7000/0.7241/0.7119 | 0.6885/0.7241/0.7059 | 0.4343/0.7414/0.5478 |
| 设施 P/R/F1（micro） | 0.8667/0.5361/0.6624 | **0.8816/0.6907/0.7746** | 0.7121/0.4845/0.5767 |
| label completeness | 0.689444 | **0.778333** | 0.670556 |
| price unknown accuracy（n=60） | 0.083333 | **1.000000** | **1.000000** |
| exact unknown set accuracy | 0.000000 | 0.033333 | 0.033333 |
| JSON / Schema / failure | 1/1/0 | 1/1/0 | 1/1/0 |
| 平均 / P50 / P95 latency (ms) | 2261.78/2267.96/2432.04 | **2218.03/2187.88/2361.62** | 2239.58/2212.44/2469.15 |
| input/output tokens 总数 | 40,319/3,519 | **40,079/3,447** | 43,619/3,508 |

`compact_field_check` 严格提高综合分且格式、失败率和 metric support 不回退，因此 selector
锁定 `week8_product_field_check_v1`。selection SHA-256 为
`db60824a18006a08dc631ddc71d1da53ca2aaef4a2ab3056e5801873aa15c90e`。
Prompt 已满足训练门禁，所以 continuation SFT 为 `SKIPPED_NOT_NEEDED`；没有新 adapter、
checkpoint 或“未运行训练”的虚假结果。

## 5. 商品单次最终 test

唯一 final job `29632815` 为 `COMPLETED 0:0`，耗时 `00:04:45`。消费 marker 已从
`STARTED` 更新为 `COMPLETED`，comparison SHA-256
`2d01ec7a52700ce8913322048466f60acddadcbe0c75a2f4064fdf7fd0ef944a7`，不会重跑。

| 指标 | 同口径正式模型 | Week 8 锁定 Prompt | 变化 |
|---|---:|---:|---:|
| 商品综合 | 0.804239 | **0.861085** | **+0.056846** |
| 业态准确率（n=60） | 0.900000 | **0.933333** | +0.033333 |
| 风格 precision micro | 0.847458 | 0.850000 | +0.002542 |
| 风格 recall micro | 0.735294 | 0.750000 | +0.014706 |
| 风格 F1 micro | 0.787402 | **0.796875** | +0.009473 |
| 设施 precision micro | **0.883333** | 0.869048 | -0.014286 |
| 设施 recall micro | 0.546392 | **0.752577** | +0.206186 |
| 设施 F1 micro | 0.675159 | **0.806630** | +0.131471 |
| label completeness | 0.722540 | **0.819484** | +0.096944 |
| price unknown accuracy（n=60） | 0.050000 | **1.000000** | +0.950000 |
| exact unknown set accuracy | 0.016667 | **0.083333** | +0.066667 |
| JSON / Schema / failure | 1/1/0 | 1/1/0 | 0/0/0 |
| 平均 / P50 / P95 latency (ms) | 2213.03/2155.43/2548.12 | 2175.68/2160.82/2233.66 | -37.35/+5.39/-314.46 |
| input tokens 总数/均值 | 40,875/681.25 | 40,635/677.25 | -240/-4.00 |
| output tokens 总数/均值 | 3,525/58.75 | 3,466/57.77 | -59/-0.98 |

同一 test 的两侧 metric support 都是业态/风格/设施/known-price=`60/60/60/0`；参考标签
非空支持为风格 `58`、设施 `60`、known price `0`，另有 price-unknown `60`。因此本次
Prompt 增益没有通过删除样本或改变支持获得。历史正式 fresh test 的 `25/30/5` 属于另一
已消费数据身份，只作历史基线；v4 没有可信可见价位样本，所以常规 price accuracy 如实为
`N/A (support=0)`，只报告 unknown 边界。

## 6. 对话首轮路由

固定 4 条真实模型 development 中，`week8_dialogue_first_turn_v2` 是三次实现里最优者：

| 指标 | 当前正式路由 | v2 路由 | 变化 |
|---|---:|---:|---:|
| 首轮三键格式合规率 | 0.00 | **0.50** | +0.50 |
| 纠错触发率 | 1.00 | **0.50** | -0.50 |
| 上下文召回 | 0.2727 | **0.8182** | +0.5455 |
| 状态值准确率 | 0.1818 | **0.8182** | +0.6364 |
| 端到端失败率 | 0.75 | **0.25** | -0.50 |
| mean / P50 / P95 latency (ms) | 5302.00/4541.46/6371.95 | **3579.30/1773.12**/6305.90 | -1722.69/-2768.34/-66.05 |

v1 没有改善首轮合规；v3 的严格 constrained schema 导致 4/4 输出在字符串内提前终止，
失败率 `1.0`，被拒绝。v2 已减少对第二次纠错的依赖，但仍有 `1/4` 失败、`2/4` 需要纠错，
所以只标记 `WEEK8_DEVELOPMENT_IMPROVED_PARTIAL`，没有声称彻底修复或改写既有研究门禁。

## 7. 商品延迟与显存

固定 L40S、同一 adapter/Prompt/图片、1 次 warmup + 每 profile 5 次测量：

| 指标 | 当前 512 token cap | 384 token cap | 变化 |
|---|---:|---:|---:|
| mean latency (ms) | 1907.79 | 1903.28 | -4.51 (-0.24%) |
| P50 / P95 (ms) | 1905.43/1918.50 | 1901.15/1914.19 | -4.28/-4.32 |
| output tokens/s | 26.733 | 26.796 | +0.063 |
| input/output tokens | 2885/255 | 2885/255 | 0/0 |
| Schema / failure | 1/0 | 1/0 | 0/0 |
| exact result match | - | 1.0 (5/5) | 无质量变化 |

冷启动 `24,669.27 ms`；峰值 GPU allocated/reserved 为
`6,691,487,232 / 8,432,648,192` bytes。由于固定输入没有触及输出上限，收益只有噪声级，
不宣称获得实质延迟优化。v4 release candidate 保留 384 商品上限作为低风险边界；最终质量
test 已证明其不导致当前 v4 质量回退。

## 8. 检索相关性与业务闭环

- 以正式 1,000 向量归档为输入，从官方 Photos ZIP 补齐 1,000/1,000 原图；原本缺失
  661 张。overlay 图片身份聚合 SHA-256
  `28d7d6944f68ba32e3670d5be425eab9cd097ff6344dcc383a533b31382a5a733`。
- index/development/final=`709/147/144`，五维隔离 PASS，全部为 silver。
- development NDCG@10：纯 CLIP `0.106478`，metadata rerank `0.485187`，因此锁定 rerank。
- 唯一 final job `29628157` 完成。纯 CLIP→rerank：NDCG@10
  `0.125654→0.506740`，NDCG@5 `0.127023→0.574840`，Recall@10
  `0.018090→0.133046`，Recall@5 `0.013827→0.100702`。
- 两侧过滤正确率 `1.0`、无结果率 `0.071813`、失败率 `0`、可追溯引用率 `1.0`；
  relevance support `128/144`。
- mean/P50/P95 从 `1.3768/1.3147/1.4393 ms` 变为
  `1.5697/1.5642/1.6976 ms`，相关性改善伴随约 `0.19 ms` rerank 开销。

该实验是 NumPy 精确向量 + metadata rerank 的离线独立 query/index 基准；没有把它描述成
Milvus 网络延迟，也没有新增人工相关性标注或复杂重排。

## 9. 验证与交付状态

- 定向商品/fresh-source/SFT、对话/延迟、检索与原图抽取测试均通过。
- `python -m unittest discover -s tests -v`：`561/561 PASS`。
- 商品 source 三维历史重叠 `0`；v4 train/dev/test 五维隔离 PASS。
- 商品 Prompt development、单次 final test、对话真实模型固定样本、商品真实模型固定输入
  延迟、检索单次 final 均已实际运行。
- SFT 未执行，因此 adapter 回载验证不适用；正式 adapter 文件 SHA-256 与 release manifest
  均为 `c2fbb5c7...eaa2a`。
- Week 8 release candidate 为 `configs/releases/qwen3_vl_system_week8_v4.json`；正式
  `configs/releases/qwen3_vl_system_v1.json` 未修改。
- `git diff --check`、tracked secret signature scan、tracked large-file scan 均为 `PASS`；
  未把数据、模型、凭据或 Slurm 输出纳入 Git。

已完成：商品数据重建/隔离、错误切片、三 Prompt 比较、Prompt 锁定、商品单次 final、
对话首轮路由实现与实测、固定输入延迟/显存基准、检索独立切分/rerank/单次 final、候选
release、完整代码验证和本报告。

仍待优化及准确原因：

- 对话 v2 仍有 `25%` 固定样本失败率和 `50%` 纠错触发率，说明尾部输出路由未完全稳定；
  v3 constrained decoding 已用真实失败证明当前方案不可用。
- 商品 exact unknown set accuracy 只有 `0.083333`；price unknown 已正确，但模型仍会在
  风格/设施弱银标与视觉可见边界上产生额外或遗漏 unknown 字段。
- v4 known-price support 为 `0`，因为隔离后的合法图片没有可靠可见价格证据；不能虚构
  常规 price accuracy，也不能复用已消费历史 test 的 5 条价位样本。
- 纯商品输出 cap 的平均收益仅 `0.24%`，没有达到实质延迟改善。
- 检索 final 是离线银标基准，尚不能替代真实用户相关性判断或 Milvus 端到端网络测量。
- 阿里云历史主机认证未恢复，只能确认当前不可访问，不能断言历史数据盘为空。

## 10. 2026-08-27 全自动扩展结果

### 10.1 无人工策略、fresh source 与 v7 锁

- 人工 annotation/review/acceptance 数量固定为 `0/0/0`；所有新增标签均为
  `programmatic_silver`，没有人工 gold 或人工接受决定。
- v2 source build job `29637053` 因 post-hash 后酒店仅剩 1 条、低于预设 13 条而
  fail-closed；失败目录保留 `BUILD_INCOMPLETE`，未覆盖或冒充成功。
- 恢复 job `29637170` 完成 source v3：6,000 candidates、1,291 validated、1,000
  selected；历史 consumed/不可读/内部重复拒绝 `4677/12/20`，source/group/image 历史
  重叠 `0/0/0`。
- 实际合法类别为餐饮/景点/酒店 `992/7/1`。27/28 个酒店图片已在历史评测中消费，因而
  没有通过复用历史 test 或伪造类别提高酒店支持。source manifest SHA-256
  `5c5387409e617492370b40ce515f9d99e59c187ede243d721fc95499f4309a6e`。
- v7 数据锁 job `29637462` 完成 train/development/test=`400/60/60`；
  `sample_id/source_id/image_sha256/group_id/constraint_template_id` 五维检查为 `PASS`，
  constraint template 对这些 public 图片为 N/A。内部 lock SHA-256
  `321bea495df6e53813d79caa93fcd3478391ecf0b613f972500f7463224b0301`。
- development/test 业态均为景点/酒店/餐饮 `2/1/57`；dev 风格多标签/空/单标签
  `11/7/42`、test `10/3/47`；两侧设施支持均为 `60`，价位应 unknown 均为 `60`。

### 10.2 fresh v7 Prompt development

job `29637779` 在 A100 20 GB MIG 上 `COMPLETED 0:0`，耗时 `00:14:40`。三版 Prompt
使用同一 60 条 development、正式 adapter 和相同生成参数；业务/风格/设施/known-price
metric support 均为 `60/60/60/0`，price-unknown support 为 `60`。

| 指标 | 正式 Prompt | 紧凑字段检查 | 视觉证据约束 |
|---|---:|---:|---:|
| 商品综合 | 0.782941 | **0.836536** | 0.740866 |
| 业态准确率 | 0.883333 | **0.916667** | 0.900000 |
| 风格 P/R/F1（micro） | 0.750000/0.681818/0.714286 | **0.758065/0.712121/0.734375** | 0.500000/0.772727/0.607143 |
| 设施 P/R/F1（micro） | 0.887097/0.591398/0.709677 | **0.887500/0.763441/0.820809** | 0.680000/0.548387/0.607143 |
| label completeness | 0.714444 | **0.803333** | 0.728611 |
| price unknown accuracy（n=60） | 0.050000 | **1.000000** | **1.000000** |
| exact unknown set accuracy | 0.000000 | 0.066667 | 0.066667 |
| JSON / Schema / failure | 1/1/0 | 1/1/0 | 1/1/0 |
| mean / P50 / P95（ms） | 4733.58/4746.96/5009.13 | **4692.59/4701.44/4862.83** | 4837.44/4842.94/5193.81 |
| input/output tokens | 40,968/3,550 | **40,728/3,459** | 44,268/3,528 |

紧凑字段 Prompt 按 development 锁定；selection SHA-256
`35abf1b6748a7bced987d043ef9bac21979421f2ac813aeb4bbba87415c4fae6`，当时 test 未消费。

### 10.3 两阶段证据与 continuation SFT

新增 `product_observable_evidence_v1` Schema，要求只输出主体清晰度、业态/风格/设施线索、
可见价格文字和短可观察事实，不输出思维链。确定性映射只接受显式 tier 价格词；主体模糊、
多主体或缺证据时保留 unknown。

- hard-slice lock job `29637171` 完成 train/dev=`400/60`，最终 test 未包含且未访问，
  lock SHA-256 `cdd56c66...0401e`。train 的 category/style/facility/price unknown/空支持为
  `385/400/387/400`，dev 为 `58/60/58/60`，说明 caption proxy 正标签严重不足。
- 两阶段 development 的首次短时 job `29637294` 超时且没有结果目录；同 identity 恢复
  job `29637921` `COMPLETED 0:0`，composite `0.352974`、evidence Schema pass
  `0.266667`、failure `0.733333`，因此拒绝该推理管线。
- continuation SFT job `29637514` 从正式 adapter 继续，LR `1e-5`、silver weight `0.5`、
  LoRA r/alpha/dropout=`16/32/0.08`。首个 10% checkpoint-5 composite `0.369804`、
  failure `0.683333`；训练到 step 10、第二次 development 前主动停止，避免继续拟合与商品
  参考支持方向相反的稀疏 silver。
- checkpoint-5 adapter SHA-256 `a94f9f751d929a2dfffec37a371705b6b9295dbe0c8f51bf17916d648e7e2249`；
  adapter-only CPU 回载 `PASS`，292 个 LoRA tensor，结构与正式 adapter 一致。该 adapter
  仅保留为失败证据，不参与 final；最终继续选择正式 checkpoint-87。

### 10.4 v7 单次最终 test

唯一 job `29638144` `COMPLETED 0:0`，耗时 `00:09:58`。marker 从 `STARTED` 原子更新为
`COMPLETED`；comparison SHA-256
`5dc83953a12e6da2526981ce73460c520d87a0cbd15086711013ab33710f3829`。没有重跑或根据
test 继续调参。

| 指标 | 同口径正式模型 | 锁定 Prompt | 变化 |
|---|---:|---:|---:|
| 商品综合 | 0.819003 | **0.857729** | **+0.038725** |
| 业态准确率（n=60） | **0.966667** | 0.950000 | -0.016667 |
| 风格 P/R/F1（micro） | 0.800000/0.716418/0.755906 | 0.777778/0.731343/0.753846 | -0.022222/+0.014925/-0.002059 |
| 风格 P/R/F1（macro） | 0.800000/0.766667/0.744444 | 0.800000/0.783333/0.750000 | 0/+0.016667/+0.005556 |
| 设施 P/R/F1（micro） | 0.835821/0.595745/0.695652 | **0.901235/0.776596/0.834286** | +0.065414/+0.180851/+0.138634 |
| 设施 P/R/F1（macro） | 0.852778/0.650000/0.707778 | **0.908333/0.808333/0.830556** | +0.055556/+0.158333/+0.122778 |
| label completeness | 0.749167 | **0.819722** | +0.070556 |
| known price accuracy/support | N/A / 0 | N/A / 0 | 不虚构 |
| price unknown accuracy/support | 0.033333 / 60 | **1.000000 / 60** | +0.966667 / 0 |
| exact unknown set accuracy/support | 0.000000 / 60 | 0.033333 / 60 | +0.033333 / 0 |
| JSON / Schema / failure | 1/1/0 | 1/1/0 | 0/0/0 |
| mean / P50 / P95（ms） | 4701.94/4713.56/4993.48 | **4609.50/4606.05/4733.06** | -92.44/-107.51/-260.42 |
| input tokens total/mean | 40,215/670.25 | 39,975/666.25 | -240/-4.00 |
| output tokens total/mean | 3,544/59.07 | 3,467/57.78 | -77/-1.28 |

两侧 metric support 均为 business/style/facility/known-price=`60/60/60/0`，price-unknown
support=`60`；参考风格非空 `57`、设施非空 `60`。业态与风格 micro-F1 的轻微回退没有
隐藏；综合、设施、unknown、完整性、token 和延迟的总体改善也没有通过删样本获得。

### 10.5 对话路由、延迟与真实 smoke

runtime v7 job `29637886` 使用同一正式模型/adapter、5 条固定对话和一张 600x400 真实商品
图片。current→deterministic：首轮三键合规 `0.4→1.0`、纠错 `0.6→0`、失败 `0.6→0`、
状态召回/值准确率 `0.5/0.5→1/1`、状态精确率 `1→1`、整状态准确率 `0.4→1`；候选
5/5 均由代码组装契约，模型 fallback 为 `0`。

固定图片 512→384 token cap：mean/P50/P95
`5006.81/5006.33/5028.50→5000.55/5002.16/5009.02 ms`，输入/输出 tokens 均为
`3685/295`，Schema/failure=`1/0`，5/5 exact match。冷启动 `31.44 s` 是 MIG 结果；峰值
allocated/reserved `6.736/8.433 GB`。v6 图片 cap + cache 候选虽然 cache hit `5/6`，mean
却 `4871.60→4874.44 ms`，因此 v7 release 关闭 processor cache 且不设置视觉 cap。

release smoke job `29638236` `COMPLETED 0:0`，`00:01:07`，状态 `PASS`：商品/售后/行程
首轮 Schema-valid；对话为 `DETERMINISTIC_CONTRACT`、无 attempt/fallback，达到
`DIALOGUE_BETA`。证据 SHA-256 `086133ec...85030`，绑定 release config SHA-256
`9defb3e7...ef749` 和正式 adapter SHA-256 `c2fbb5c7...eaa2a`。单条冷启动商品 smoke
为 `36.13 s`，只作 smoke，不混入稳态 P50/P95。

### 10.6 真实 Milvus Lite 混合检索

v2 因 query 进程无法加载已建 collection 而保留真实失败；修复跨进程显式
`load_collection` 后，v3 development job `29636996` 使用
`pymilvus[milvus_lite]==2.6.16`、backend `milvus_lite_flat_cosine`、offline fallback=false，
从 CLIP、metadata rerank、hybrid RRF、hybrid weighted 中锁定 `hybrid_weighted`。

唯一 final job `29637070` 的 144 queries（relevance support 128）：NDCG@10
`0.125654→0.564459`、Recall@10 `0.018090→0.142734`、NDCG@5
`0.127023→0.656263`、Recall@5 `0.013827→0.117405`。过滤正确率 `1`、无结果率
`0.071813`、失败率 `0`、可追溯率 `1`。mean/P50/P95
`12.156/12.211/12.756→14.427/14.503/14.905 ms`；质量收益伴随约 `2.15 ms` P95 开销。

### 10.7 当前完成项与仍待优化项

已完成：全自动 fresh source/锁/错误切片、三 Prompt fresh development、两阶段负实验、
continuation SFT 10% checkpoint 与回载、单次商品 final、确定性首轮路由、真实图片延迟、
真实 Milvus Lite 混合检索 final、release v7 smoke、配置/测试/文档和分支推送。正式 release、
冻结历史产物、`dev/stg/main` 和标签均未修改。

仍待优化及准确原因：

- 商品 final 的业态准确率下降 `0.016667`、风格 micro-F1 下降 `0.002059`；当前 fresh
  合法酒店/景点总量仅 `1/7`，且在无人工前提下不能用新人工 gold 定向修复。
- known-price support 仍为 `0`；隔离后的合法图片没有可靠可见 tier 价格证据，不能把 Yelp
  元数据价格伪装为视觉事实。exact unknown set accuracy 也只有 `0.033333`。
- 两阶段 evidence Schema pass 低且 caption silver 正标签严重不足；checkpoint-5 已证明
  小规模 continuation 只带来有限改善，继续训练会强化 unknown/空标签塌缩。
- 稳态商品延迟的 token-cap 收益很小；processor cache/视觉 cap 未证明 mean 改善，故没有
  为更好看的延迟数字启用它们。单条 cold smoke 仍约 36 秒。
- 检索 relevance 仍是 programmatic silver 而非真实用户判断，并以约 2.15 ms P95 换取
  相关性；在明确不再有人工作后，无法新增人工相关性 gold。

### 10.8 终态验证与身份复核

- `python -m unittest discover -s tests -v`：`594/594 PASS`；`python -m compileall -q
  src scripts tests`：`PASS`。
- Spartan v7 lock validator：`PASS`，train/development/test=`400/60/60`、失败项 `0`；
  锁本身保留创建时的 `LOCKED_UNCONSUMED` 状态，独立不可覆盖的
  `test_consumption.json` 已为 `COMPLETED`，SHA-256
  `852fe2e4fda86fbb33a493a20bd1aeb63590dd556f2194991cfd03da070495b9`。
- final comparison/release/runtime/smoke SHA-256 复算分别为
  `5dc83953...f3829`、`9defb3e7...ef749`、`3cc8c0c2...a6b6f1`、
  `086133ec...85030`；正式 checkpoint-87 adapter 文件复算仍为
  `c2fbb5c7...eaa2a`。
- `git diff --check`、tracked secret signature scan、tracked file 大于 10 MiB 扫描均为
  `PASS`；没有把模型权重、凭据、数据集、输出或 Slurm 日志纳入 Git。
