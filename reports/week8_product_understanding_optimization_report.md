# Week 8 商品理解与并行优化报告

> 2026-08-30 交付更新：用户授权以“v13 运行配置 + v12 商品验收身份”作为正式 final v1。
> 本报告中“未替换正式 release”等文字保留当时实验时序，不再代表当前分支状态；自动 silver、
> human=0、价位 0 支持和 v18 development-only 等质量边界保持不变。

## 1. 结论

**2026-08-29 当前：v12已完整通过自动silver候选验收及双端交接，达到本轮可晋级候选目标；未替换正式发布。**
后续 v10 虽在 development 和新 final 都有小幅风格收益，但新 final 请求失败 1/100，
因此不晋级；v11 虽无请求失败，却因业态准确率低于同批正式基线而再次被拒绝。继续优化
v12 development综合0.754617→0.774020，final与同场v9各指标持平，
不得声称最终再提升。最新完整表见16.22。v9/v10/v11已生成产物均不覆盖。
v9历史通过证据见14.8—14.9，当前v12结果与交接见16.22—16.24；九项确定性缺陷及后续复审反例已修复。v7 的
猜测/行程问题、v8 的无效最终参考均保留为失败证据。此结论不是人工视觉准确率或无条件
生产可用性。以下第 1—13 节的旧选择和分数为历史记录，不能作为现行候选的晋级依据。

**2026-08-27 复审更正：以下 Week 8 数值是混合 metadata/caption 银标匹配分，不是图像事实
准确率。** 固定 development 60/60 条混有商家元数据，56/60 条的业态已知值与
`unknown_fields` 矛盾，60/60 条将混合设施/风格错误归因于 caption。不能据此声称商品视觉
理解问题已解决。复审修复与新增诊断见第 12 节；历史数据、输出和数值原样保留。

Week 8 在独立 `feature/week8-product-understanding` 分支完成了商品、对话、延迟与检索的
全自动扩展执行。自 2026-08-27 起没有任何人工标注、复核或验收；新增数据与 target 均为
`programmatic_silver`，human annotation/review/acceptance=`0/0/0`。

最终继续选择正式 checkpoint-87 adapter 和 `week8_product_field_check_v1`。fresh v7 的
唯一 final test 将商品综合从同口径正式 Prompt 的 `0.819003` 提高到 `0.857729`
（`+0.038725`）；银标口径下设施 micro-F1、price unknown 和完整性提高，业态与风格的轻微回退
如实保留。JSON/Schema 仍为 `100%/100%`，失败率为 `0`，支持数未改变。可观察证据
两阶段方案和 checkpoint-5 continuation SFT 均经 development 实测后被拒绝，没有用
final test 选择 Prompt、adapter 或 checkpoint。

对话首轮改为代码组装三键契约，固定 5 条样本的格式合规、状态召回/准确与整状态准确均为
`1.0`，纠错、失败和 fallback 均为 `0`。这五条仅测协议和状态，不验证实际回答/推荐质量。
真实图片延迟只选择已通过商品银标 test 的
384 token 上限；图片 cap + processor cache 因 mean 变慢而未进入最终 release。检索最终
选择真实 Milvus Lite `hybrid_weighted`，NDCG@10 `0.125654→0.564459`，代价是 P95
增加约 `2.15 ms`。release v7 四场景真实 smoke 为 `PASS`。

后续剩余优化没有改变商品 release：两个额外 Prompt 在同一 development 上均回退，历史
哈希隔离和 OCR 审计也确认没有新的视觉价位正例，故不再启动缺少正支持的 SFT。性能侧的
prepared-input cache 因变慢被拒绝；检索侧有界 LRU 在质量完全一致时将稳态 P95 降低
`12.55%`，作为 development 已锁定候选保留。

## 2. 初始 v4 执行、历史基线与资产审计

本节至第 9 节保留 2026-08-26 初始 v4 的不可变历史证据；2026-08-27 的全自动扩展结果
见第 10 节；后续工作见第 11 节，当前复审后的解释以第 12 节为准。

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

## 11. 2026-08-27 剩余优化续行

### 11.1 商品 Prompt 的 development-only 复测

`product_prompt_refinement_20260828_v8` 通过
`development_lock_config=configs/week8/product_understanding_v7.json` 绑定既有 v7
development lock SHA-256 `321bea49...b0301`。该 overlay 明确设置
`development_only=true`，selection 的 `test_policy=DISABLED_DEVELOPMENT_ONLY`；final
执行函数也在读取 test 前 fail-closed。Spartan job `29643869` 在 A100 20 GB MIG 上
`COMPLETED 0:0`，耗时 `00:15:11`，60 条 development 的支持数完全一致。

| 指标 | v7 当前选择 | 字段检查 v2 | 保守证据约束 |
|---|---:|---:|---:|
| 商品综合 | **0.836536** | 0.701144 | 0.703235 |
| 业态准确率（n=60） | 0.916667 | 0.900000 | **0.950000** |
| 风格 P/R/F1 micro | **0.758065/0.712121/0.734375** | 0.657143/0.696970/0.676471 | 0.754098/0.696970/0.724409 |
| 设施 P/R/F1 micro | **0.887500/0.763441/0.820809** | 0.361111/0.419355/0.388060 | 0.358696/0.354839/0.356757 |
| label completeness | **0.803333** | 0.645000 | 0.636111 |
| known price support | 0 | 0 | 0 |
| price unknown accuracy/support | 1.000000/60 | 1.000000/60 | 1.000000/60 |
| exact unknown accuracy/support | 0.066667/60 | 0.066667/60 | 0.066667/60 |
| JSON / Schema / failure | 1/1/0 | 1/1/0 | 1/1/0 |
| mean / P50 / P95（ms） | **4630.48/4622.80/4892.21** | 4987.80/4826.38/5273.76 | 4766.09/4633.75/5077.63 |
| input/output tokens | **40,728/3,459** | 49,374/3,640 | 48,398/**3,445** |

两个候选都因 `composite_not_strictly_above_current_release` 被拒绝；selection 状态为
`SFT_ALLOWED_NO_PROMPT_WINNER`，SHA-256 `110d3630...aef8a`。这里的 `SFT_ALLOWED` 只表示
Prompt 选择器允许进入下一项可行性检查，并不代表已有合格训练数据。

### 11.2 未消费 silver 与视觉价位证据审计

独立配置 `product_silver_source_audit_20260827_v8` 不读取 v7 final 行或输出。Spartan CPU
job `29643962` `COMPLETED 0:0`，耗时 `00:01:12`；审计结果 SHA-256
`b425ab81...9e29`，candidate manifest SHA-256 `6ca17cc5...a32d`。新增 target/review/
acceptance 的 human 计数仍为 `0/0/0`，标签身份为 `programmatic_silver`。

- 完整未消费候选的 post-hash 前上限为餐饮/景点/酒店 `15934/6/27`；加入历史及 v7
  图片哈希排除后，45 个候选中 37 个被拒，只剩 8 个安全候选。
- v7 未使用的 480 张图全部为 restaurant；native caption 只有 48，style/facility caption
  支持为 `0/12`。商家 metadata 价位虽有 319 条，但明确标记为非视觉证据。
- caption 金额/tier 的 pre-hash 上限为 `11/1`；Tesseract 与 caption 精确 token 一致确认
  后，可见金额、可见 tier 和正 `price_range` 支持均为 `0/0/0`。数字金额也不会自动映射
  价位档。

因此未启动新的 continuation SFT：8 个安全候选无法形成酒店、景点、风格、设施和视觉价位
的完整正支持，继续训练只会重复已观察到的 unknown/空标签塌缩。该决定来自数据可行性审计，
不是把失败结果隐藏成“无需训练”。

### 11.3 商品输入缓存与检索有界 LRU

商品 prepared-input cache development job `29643870` 在固定模型、adapter、Prompt、图片
和 A100 MIG 上 `COMPLETED 0:0`。10 次 current→cache 的 mean/P50/P95 为
`4845.46/4839.64/4877.90→4868.88/4858.77/4920.32 ms`；cache hit/miss=`11/1`，
但 mean/P95 反而增加 `23.43/42.42 ms`。10/10 输出完全一致，tokens 均为
`7370/590`，Schema/failure=`1/0`。证据 SHA-256 `83e8b2ce...1161`；候选被拒，release
继续关闭该默认-off 开关。

检索 v5 使用新 development-only 身份并确定性排除 v3 development/final query group，
不读取 v3 final artifact。lock 的 index/development/final=`582/127/0`，五维隔离 PASS，
SHA-256 `a5fdf0a1...fbc9`。真实 Milvus Lite job `29644063` `COMPLETED 0:0`，耗时
`00:00:47`，无 offline fallback。

此前 v4 job `29643904` 首次证明 pool100 cache 可保持质量并降低延迟，而 pool50/25 会使
质量回退；但 v4 未显式锁定容量，也未记录预计算和内存成本，因此不作最终选择。v5 使用
新身份补齐这些约束，没有覆盖 v4 失败/中间证据。

| 指标 | pool100 uncached | pool100 LRU512 | 变化 |
|---|---:|---:|---:|
| NDCG@10 | 0.584776 | 0.584776 | 0 |
| Recall@10 | 0.172498 | 0.172498 | 0 |
| relevance support | 102 | 102 | 0 |
| filter / trace / source / failure | 1/1/1/0 | 1/1/1/0 | 0/0/0/0 |
| mean（ms） | 9.6001 | **8.3079** | -13.46% |
| P50（ms） | 9.2823 | **8.1101** | -12.63% |
| P95（ms） | 9.6339 | **8.4247** | -1.2092 / -12.55% |

两侧各有 508 个交错测量。LRU 容量 512、最终 entries 393、evictions 0；预计算
`1602.23 ms`，tracemalloc peak `22,991,100 B`，进程 `ru_maxrss` 高水位从
`168648→217032 KB`。预计算 hit/miss=`228/393`，稳态 measurement hit/miss=
`2484/0`。selection SHA-256 `61b6d8e4...8150`，锁定
`hybrid_weighted_pool100_lru512`；这是有完整成本记录的稳态 development 优化，未宣称
已替换正式 API release。

### 11.4 最终选择、完成项与仍待优化项

商品最终选择保持 checkpoint-87 adapter 加 `week8_product_field_check_v1`；release v7、
已消费商品 final、历史 adapter 和 smoke 均未改变，因此没有重跑 final 或制造新的 smoke
指标。对话确定性首轮路由也保持既有 5/5 结果。新增完成项是两个 Prompt 负实验、剩余
silver/OCR 可行性审计、商品 prepared-input cache 负实验及检索有界 LRU 正实验。

仍待优化及准确原因：商品合法新数据的酒店/景点、style 和视觉价位正支持不足，无法在
human=`0` 的前提下继续可靠训练；v7 final 的业态与风格轻微回退仍保留；商品稳态生成
延迟没有找到更快且可重复的缓存方案；检索 LRU 尚未接入正式 API/release，接入时还需将
本地 metadata 生命周期、预热成本和失效策略纳入生产配置。没有生成后续周计划。

### 11.5 续行验证

- 新增及相关定向 unittest：`76/76 PASS`；完整
  `python -m unittest discover -s tests -v`：`609/609 PASS`。
- `python -m compileall -q src scripts tests`、三份新增 Slurm 脚本 `bash -n`、
  `git diff --check`、tracked secret signature scan、tracked file 大于 10 MiB 扫描均为
  `PASS`；README 中两个新增 CLI 的 `--help` 实测通过。
- release config SHA-256 复算仍为 `9defb3e7...ef749`；Spartan 正式 checkpoint-87
  adapter SHA-256 复算仍为 `c2fbb5c7...eaa2a`。没有把运行输出、图片、模型权重、凭据或
  Slurm 日志加入 Git。

## 12. 全项目复审与商品证据修复（2026-08-27）

### 12.1 复审结论与修复边界

本次检查 API、推理、训练、评测、检索和行程入口，以商品链路为重点；不是宣称整个项目已
不存在缺陷。主工作树的用户文档改动未暂存、覆盖或提交。代码在独立 feature worktree
提交；Spartan 仍使用既有项目目录。没有新增人工标签、人工复核或人工验收。

| 优先级 | 实际问题 | 修复与验证方式 |
|---|---|---|
| P1 | metadata 停车场/吧台被写进 `visible_facilities`；provenance 与 unknown 矛盾 | 增加逐样本引用审计和视觉 SFT 输入拒绝；冻结 target 不改写，历史分数重新说明口径 |
| P1 | 两阶段 Schema 只传给解码器，模型看不到字段/枚举；纠错未带实际错误 | 新诊断 Prompt 显式展示 Schema、短观察事实与互斥规则；重试包含原输出和错误；映射抑制明确缺证据的肯定标签 |
| P1 | 两种 continuation SFT 的内存 backend 漏初始化缓存 | 统一 `from_loaded`，补齐两级缓存、执行锁与 readiness；两条训练回调的生成回归通过 |
| P1 | 并发首次缓存 miss 可重复构造模型；同一模型并发生成存在共享状态风险 | API 单例工厂加外层锁，VLM/CLIP backend 加执行锁，模型与 processor 全部加载成功后才发布实例 |
| P2 | HTTP 图片内容可能变化但 CPU 预处理缓存仍按 URL 命中 | HTTP 图片同时绕过 CPU/device 缓存；不改变默认关闭策略 |
| P1 | 失败后的占位 JSON 仍获格式/部分语义及 unknown 分；价位支持写死；缺行重复行未拒绝 | 失败预测零分但保留分母；价位支持从固定参考集计算，必须一条样本恰好一个结果，NaN/越界选优指标拒绝；原始输出保留后离线重计分 |
| P2 | development 校验会打开 test 标签；final 可接受不完整选择证据 | development 只查 train/dev 与五维 identity manifest；final 验证完整指标哈希与重新计算的选择结果，先写一次性标记再读取 test |
| P2 | 对话先匹配预算被后续修改覆盖、局部否定误伤、部分解析吞掉剩余修改 | 同字段最后一次明确修改优先、否定局限于分句、剩余变化使用受约束 fallback；明确更新不能被 fallback 覆盖 |
| P2 | 非法 days/budget 可进入状态；缺失字段与期望 null 被误计为一致 | 对非空数值检查类型和范围，保留合法取消时的 null；状态分数要求键实际存在 |
| P2 | 生产旧 planner 返回示例目录；搜索数量无边界 | 生产关闭示例 planner，保留真实任务入口；搜索参数在 API 层校验；旧模型不可用返回 503 |
| P1 | 生产环境误设 fallback 开关仍可伪造固定模型成功结果 | `APP_ENV=production` 无条件禁用旧客户端示例 fallback，开发模式行为保留 |

复审后的对话仍是 beta 状态/契约功能；普通推荐问题和新图请求的固定确认语不构成实质
任务完成。没有把“减少模型调用”包装为回答质量提升。检索 LRU 仍为 development 候选，
此次没有再次改变检索排序或宣称已接入正式 API。

### 12.2 商品弱标签与图像的直接矛盾

Spartan preflight 使用完整 60 条 v7 development，human=`0`。数据锁 SHA-256：
`321bea495df6e53813d79caa93fcd3478391ecf0b613f972500f7463224b0301`，五维隔离 PASS，
本次不读取 final 标签。60/60 条混有 metadata；56/60 条业态 known/unknown 自相矛盾，
60/60 条风格/设施 provenance 不准确。误差切片保留全部样本，没有删除困难项或降低支持。

在查看模型输出前固定索引 `0/15/30/45` 做自动视觉定性检查。它不是新增 gold，不参与
总体准确率计算，也不改变原 target：

| development sample 后缀 | 图像可直接观察的短事实 | 当前 v7 输出的冲突示例 |
|---|---|---|
| 0000 | 室内大型不锈钢酿造设备 | 输出 `parking`，画面没有停车场 |
| 0015 | 纸上的卷饼近景、木桌局部 | 输出 `parking`，无法由食品特写确认 |
| 0030 | 室内桌椅、窗、桌游盒和电脑 | 输出 `parking`，不能由此确认停车设施 |
| 0045 | 寿司、啤酒瓶与餐厅内景 | 输出 `bar, parking`，其中停车场未出现在图中 |

这四条也都只返回 `photo type: ...` 作为证据，说明 Schema 通过不能证明观察事实正确。
对这些样例的判断是自动定性证据，不能外推为 60 条或总体视觉准确率。

### 12.3 可复现实验入口

- 主要代码修复提交：`a099f3f`（商品/评测）、`217f6da`（运行时/API/状态）、`fdc49b0`
  （CLIP/训练回归）、`dd7f4e5`（生产 fallback）、`b4d6193`（失败占位零分/离线重计分）。
- `configs/week8/product_review_v1.json` 绑定完整固定 development、正式 adapter 与
  v7 release；比较现有商品 Prompt、旧证据链、修复证据契约、临时禁用 adapter 的基座
  消融。两阶段输出预算为 256；商品单阶段为 384。禁用 adapter 不写权重、不合并模型。
- `scripts/review_week8_product.py --audit-only` 只做身份/引用审计；去掉该参数运行 GPU
  对照。必须传 `--output-dir` 指向新的目录，已有输出拒绝覆盖。
- `--rescore-dir` 对已有四组 raw output 校验原哈希，再以修复后的失败零分协议写到新目录；
  原指标/输出不改写，模型新增请求数为零，不打开 final 标签。
- `configs/week8/runtime_review_v2.json` 固定 10 条对话，包括新增的重复预算、局部否定、
  部分更新、取消预算和非法天数；最终 `runtime_review_v3.json` 保留这 10 条，仅将图片
  绑定到已锁定真实 development 照片，并记录 SHA/尺寸。四条有图、六条纯文本。不是用
  额外样本替换原有失败样本。
- 本次无训练、无新 adapter、无新 final test。已消费 v7 final 不被再次调参使用。

### 12.4 完整 development 对照与失败重计分

本节表中 `product_release` 是已选 **Week 8 v7 RC** 的
`week8_product_field_check_v1`，不是历史正式 `system_repair_product_compact_v3`。
历史 fresh test 综合分 `0.780639` 保留在前文，不能与本节 development 跨集相减。
所有组使用同一正式 checkpoint-87；`base` 消融只暂时禁用 adapter，不生成新权重。

| GPU 作业 | 代码 | 完成状态/实际耗时 | 工作内容 |
|---|---|---|---|
| 29664584 | a099f3f | COMPLETED 0:0 / 47:37 | 四角色各 60 条商品 development |
| 29666004 | f129ea8 | COMPLETED 0:0 / 13:40 | RC 控制组及基座自由解码各 60 条 |
| 29666837 | f58707c | COMPLETED 0:0 / 03:00 | 真实照片 smoke、10 条对话、重复延迟 |

三作业均为 NVIDIA A100 80GB PCIe、torch `2.8.0+cu128`，顺序运行，没有争用同一 GPU。
首轮商品冷加载 `38423.918 ms`；完整首轮峰值 allocated/reserved 为
`8,143,745,536/8,432,648,192 B`。这是整轮高水位，不是每种 Prompt 的独立内存比较。

每组完整保留 60 条，业态 reference 为 restaurant/hotel/attraction=`57/1/2`。
style/facility 指标的行分母均 60，其中正标签行 `53/60`、正标签数量 `66/93`；
style 多标签/单标签/空值分别 `11/42/7`。视觉价位 known support=`0`，unknown support=`60`。
多主体/主体模糊没有独立可信标注，不能声称已测得该切片准确率。五维 manifest 检查无
跨集碰撞；模板身份在本组均为空（适用数 0），不伪装成有模板覆盖。

以下为修正后 `week8_product_failure_zero_credit_v2` 分数：失败留在分母但零分；综合分
沿用原 macro-F1 权重并排除无支持的已知价位项。为便于比较，表内 P/R/F1 单独列 micro。
所有语义数值只表示有缺陷的 metadata/caption silver 匹配，不表示图像正确率。

| 角色 | 综合 | 业态 acc | style P/R/F1 | facility P/R/F1 | completeness |
|---|---:|---:|---|---|---:|
| v7 RC 控制 | 0.836046 | 0.916667 | 0.770492 / 0.712121 / 0.740157 | 0.886076 / 0.752688 / 0.813953 | 0.800556 |
| 旧证据链 | 0.587549 | 0.633333 | 0.767442 / 0.500000 / 0.605505 | 0.959184 / 0.505376 / 0.661972 | 0.551667 |
| 修复证据契约 | 0.694199 | 0.916667 | 0.589041 / 0.651515 / 0.618705 | 0.512821 / 0.430108 / 0.467836 | 0.648611 |
| 基座＋受约束解码 | 0.269641 | 0.383333 | 0.387755 / 0.287879 / 0.330435 | 0.419355 / 0.139785 / 0.209677 | 0.245000 |
| 基座＋自由解码 | 0.510065 | 0.650000 | 0.426667 / 0.484848 / 0.453901 | 0.436364 / 0.258065 / 0.324324 | 0.437778 |

已知价位 accuracy 五组均 N/A（support=0），不能报 100%。unknown 集合匹配五组均
`4/60=0.066667`，受到参考自身 56 条 known/unknown 矛盾影响，不能解释为真实 unknown
使用正确率。单独 price unknown 的有效正确数依表中顺序为 `60/43/60/27/58`，分母均 60。
RC 的完整语义严格匹配仅 `19/60`，多风格切片 `0/11`、hotel `0/1`、attraction `0/2`；
这些仍是银标一致性切片，不是经人工确认的错图数量。

| 角色 | 原始 JSON syntax | 模型 Schema | 有效 JSON/Schema 计分 | 请求失败 | 发生重试的样本 | 首次成功 |
|---|---:|---:|---:|---:|---:|---:|
| v7 RC 控制 | 100% | 100% | 100% / 100% | 0/60 | 0 | 60/60 |
| 旧证据链 | 100% | 71.6667% | 71.6667% / 71.6667% | 17/60 | 53 | 7/60 |
| 修复证据契约 | 100% | 100% | 100% / 100% | 0/60 | 0 | 60/60 |
| 基座＋受约束解码 | 45% | 45% | 45% / 45% | 33/60 | 33 | 27/60 |
| 基座＋自由解码 | 100% | 96.6667% | 96.6667% / 96.6667% | 2/60 | 2 | 58/60 |

原始 JSON syntax 指最终模型输出能否解析；有效格式计分另将请求失败置零，不能把有效
计分率下降误称为 JSON 语法错误。旧证据链的 Schema 失败是重复 `uncertainty_reasons`；
基座受约束解码失败表现为中文字符串中途结束。自由解码保持相同 256-token 上限、完整
Schema 后校验和最多一次重试，剩余两条为重复 `observable_facts`，未被自动抹掉。
旧证据链角色也使用本次共享的重试错误上下文修复，不是未改动历史 v4 的重新运行。

| 角色 | mean / P50 / P95（ms） | 输入 / 输出 token 总量 |
|---|---|---|
| v7 RC 控制（首轮） | 3966.894 / 3925.841 / 4077.809 | 40728 / 3455 |
| 旧证据链 | 17630.520 / 18585.745 / 19452.544 | 35420 / 8262 |
| 修复证据契约 | 7999.279 / 7919.372 / 8481.089 | 40608 / 3238 |
| 基座＋受约束解码 | 14467.119 / 14414.522 / 16171.269 | 66691 / 7446 |
| v7 RC 控制（第二轮） | 3914.785 / 3869.962 / 4018.004 | 40728 / 3455 |
| 基座＋自由解码 | 6232.580 / 6051.702 / 7271.232 | 42317 / 7834 |

结论：Schema 可见契约确实修复了旧证据链的运行失败，并缩短该链延迟；但仍猜测不可见
设施，facility 银标分也回退。基座自由解码减少字符串错误，但定性检查仍有物体/文字
错读。没有候选同时证明商品视觉正确性、格式和延迟优于当前选择，因此 **不更换 RC
Prompt 或 adapter**。不能仅凭 composite 的格式项回升就称语义优化成功。

### 12.5 真实图片与对话再次回归

复审发现 `data/samples/images/cafe_001.jpg` 是 64×64 的底色加圆形占位图，SHA 为
`fa3858fd0d08b1788606095cc4c18d470e927644d61b9696840749a1ae3644f7`。
前两作业随附 smoke 和重复延迟使用该图，**只能作为真实模型连通性证据**，不能称为
真实商品照片基准。两轮各 60 条商品 development 本身使用真实图片，不受此问题影响。

最终作业绑定原 development `week8-product-v2-development-0030` 的 533×400 室内照片，
SHA `4522e1aa84ef6f0800b2b138068f56db88e8096a622ef1f842e652b9024cf6d8`。
`f69797e` 在加载模型前检查图片 SHA，并写入尺寸和 provenance；CLI 替换图也必须通过
身份校验。`runtime_review_v3.json` SHA 为
`cd2aad55361c1c88d7308a00620b9b6dee2784dcda34f3769669b09e5c61ee24`。

runtime v2 曾暴露两个真实错误：取消预算仍保留 2000、负一天被改写为正一天。
`f58707c` 新增明确取消置 null、非法字段保护和部分失败回复；语义 fallback 仍可更新
其他字段，但不得覆盖被拒字段。最终原 10 条全部保留并再次运行：

| 对话指标 | 模型生成 current | 修复后确定性路由＋语义 fallback |
|---|---:|---:|
| 首轮格式合规 | 90% | 100% |
| 二次格式纠错触发 | 10% | 0 |
| 请求失败率 | 10% | 0 |
| 上下文字段召回 | 0.88 | 1.00 |
| 状态值正确率（25 字段） | 0.80 | 1.00 |
| 完整状态 exact（10 条） | 0.40 | 1.00 |
| 非预期状态键数量 | 15 | 0 |

修复组 9 条不调用模型，儿童数量部分更新 1 条使用真实模型 fallback，成功保留预算和
其他状态。进程内 wall mean/median 为 `567.325/0.055652 ms`，P95（nearest-rank）
`5672.375 ms`；这不是 HTTP 延迟。汇总中的模型耗时 P50=0 只代表不调用模型，不能声称
服务响应耗时为零。普通推荐/新图分支仍只是确认语；该测试不证明推荐或图片问答完成。

同一模型/adapter/Prompt/真实图片、复用模型和 processor、缓存关闭，每组 1 次预热再
5 次测量。加载耗时 `21434.521 ms`，生成上限 512 与已有 RC 的 384 比较：

| 商品真实照片基准 | 512 | 384 |
|---|---:|---:|
| mean（ms） | 3894.280 | 3901.957 |
| P50（ms） | 3902.640 | 3908.000 |
| P95（ms） | 3943.654 | 3926.002 |
| 输入 / 输出 token 总量 | 3565 / 285 | 3565 / 285 |
| 生成速度（token/s） | 14.636852 | 14.608055 |
| Schema / 请求失败率 | 100% / 0 | 100% / 0 |
| 配对输出完全相同 | 5/5 | 5/5 |
| 峰值 allocated / reserved（B） | 6731133440 / 8432648192 | 6731133440 / 8432648192 |

两组每次都只生成 57 token，降低上限没有降低实际生成量；P95 小幅波动但 mean 略慢，
样本少且未交错，**不认定稳定提速**。输出一致仅证明没有改变输出，不代表输出正确：
该图仍输出不可见的 `parking`。不改变发布参数，不以此代替商品质量提升。

真实照片 smoke 的商品/售后/行程三场景 JSON/Schema 通过，对话确定性三键契约通过。
商品第一次调用含冷加载为 `40470.734 ms`，不与稳态 3.9 s 混用。对话 smoke 本身不调用
模型，真实 fallback 由上述 runtime 覆盖。行程输出仍有“简短摘要”“简短活动”等模板
复述，因此 `status=PASS` 只代表技术契约，不能描述为业务语义全部正确。

### 12.6 证据身份、验证与最终状态

原始证据目录位于 `outputs/week8/review/`，全部忽略、不入 Git：

| 相对目录/文件 | SHA-256 |
|---|---|
| week8_product_review_20260827_v1/summary.json | f2234d5958e64b8d6f35129b5853e23eaf4a8c2a30b1bbff1c4b07d450dcbff9 |
| week8_product_decode_review_20260827_v2/summary.json | c14956942e4d3072d26681cf9b280e6dd19c5df8632c18b5bee172ffbbc23100 |
| week8_product_review_20260827_v1_rescored/summary.json | add2379dc9f2d23b88390882bc42d5d931d831e491799a8614cf870140df8dd3 |
| week8_product_decode_review_20260827_v2_rescored/summary.json | 259b9ce9b97193095cb7803e1cb0ecb10b4da8b92d0fc2ad4da2d9c699ff23a6 |
| week8_real_image_runtime_20260827_v3/model_smoke.json | b6532cf4f2cbc15db604537909e2da95f31222fc68713e472677a4bc6f8d0734 |
| week8_real_image_runtime_20260827_v3/runtime.json | 0701c1e7299c8c3e0c90b241273d4602f555bb409af76c285363ee393e6742a4 |

离线重计分代码 `f69797e`，两次均 `new_model_requests=0`，原 raw 输出 SHA 逐组校验；
本节以新 summary 为准，不覆盖首轮含失败占位得分的历史文件。

- 新增定向：`python -m unittest tests.test_week8_review_repairs tests.test_api_review_repairs tests.test_clip_review_repairs -v`，`45/45 PASS`。
- 全量：`python -m unittest discover -s tests -v`，最终复验 `654/654 PASS`（33.380 s）。
- `python -m compileall -q src scripts tests`、两个新增 Slurm 脚本 `bash -n`、
  `git diff --check`、tracked secret signature scan、tracked 文件大于 10 MiB 扫描均 PASS。
- README 新增路径存在；商品 review、runtime benchmark、model smoke 三个 CLI 的
  `--help` 实测通过。
- development 五维身份检查 PASS；最终 test 本次未运行、未读取标签，不产生新的最终
  评测结论。冻结 lock 内的状态文字不代替已消费 final 的外部一次性 marker。
- 本次不训练，无新 adapter 回载结果；已有内存训练 backend 的单元回归及正式 adapter
  的真实模型加载已验证，不能将其写成已完成新训练。
- formal manifest 的 Git blob 与 Spartan LF 文件 SHA 为
  `3c71e0d58ea834a70d2d65a780bf1f790f38c78f5d4e16a3f9ac9d0c91ef3f6b`；Windows 当前
  工作副本因换行转换，字节 SHA 为 `88984350e083a75ac13e944ffb4a2cf4eb1ebbdb54a2515203b8a9df3ad619fa`。
  两端 `git hash-object` 均为 `348f92b338f5cda36d06e58bde455ba11917f4e7`，内容未改；
  v7 RC manifest SHA 为 `9defb3e7e346bef32d3e290b65f8aaf48f50793959d32e1753684590186ef749`；
  adapter SHA 为 `c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a`。

已完成：可复现商品数据口径审计、证据契约和失败计分修复、基座解码消融、真实照片身份
及重复基准、对话状态再次修复、运行时/API 并发与生产失败保护、全量回归和文档纠偏。

仍未解决：不可见设施猜测与弱证据复述、酒店/景点及多主体/风格的可靠视觉评测、已知
价位正支持、通用推荐/行程的实质任务完成度、商品稳定提速、检索 LRU 正式接入。原因是
现有银标混合非视觉 metadata、有效独立视觉参考与正样本不足，现有 adapter 仍复述弱
训练模式；本次诊断没有证明可安全替代的语义方案。不会伪造 gold、安排人工标注或把
失败实验写成完成。保留 checkpoint-87 与 `week8_product_field_check_v1` 的既有 RC
选择，未升级正式 release、未合并到 dev/stg/main、未打标签。

## 13. c01b732 九项审查修复（2026-08-27 执行，2026-08-28 复验）

### 13.1 修复范围与版本身份

实现提交 `327f764`，配置 `configs/week8/audit_repair_v1.json`，运行身份
`week8_audit_repair_20260827_v1`。沿用同一 Spartan 项目目录和
`feature/week8-product-understanding` 分支；不修改主工作树、冻结数据、adapter、正式
release 或已消费 final。本轮人工标注/复核/验收均为 0，没有新增 SFT。

| 审查问题 | 修复与验证 |
|---|---|
| P1 标签错误 | 新 `caption_evidence_v2`：结构化解析 bool/字典、完整词匹配、否定处理；merchant_metadata 不并入 visible_facilities。False parking、mushroom、spacious、no parking 反例通过。历史生成器仅在明确 legacy 协议中保留。 |
| P1 选优口径 | `reference_semantics` 成为选优前提；缺少视觉依据、metadata 代理或标签矛盾均返回 `DIAGNOSTIC_ONLY_INVALID_REFERENCES`。可靠参考下价位支持为 0 不阻断其他字段，价位单列 N/A。 |
| P1 对话未执行 | 保留确定性状态解析，分派至商品、行程或真实检索；记录 tool_calls、attempts、task_result 和 task_status。仅确认、工具未执行、约束未应用时返回 NOT_COMPLETED。 |
| P1 行程假通过 | Schema 后检查天数、日序、明确约束覆盖与占位文本；沿用一次纠错。技术 smoke 与业务 smoke 独立，失败不再被 PASS 掩盖。 |
| P2 金额截断 | `2,000` 完整解析为 2000；`1e3`、错误千分位、范围等不支持格式保留旧值并提示，不能截取数字前缀。 |
| P2 图片轮次 | 支持 user 轮次级 image_urls；兼容顶层图片绑定最新 user 轮，保留历史图，不再挂到第一轮。总引用数有界。 |
| P2 检索闭环 | 排序只使用用户条件/模型预测，参考 metadata 留在评分侧；生产路由接通 keyword/embedding/hybrid 和配置化 weighted fusion，显示未应用文字条件。 |
| P2 输入 500 | 商品/售后一图约束、行程非空文字约束在请求模型中校验；三项真实 HTTP TestClient 反例返回 422，未加载模型。 |
| P2 配置错验 | CLI/服务共用配置解析；显式参数优先于环境变量。Compose 经 tripctl 传入相同绝对文件，验证返回路径/SHA。不存在配置退出 1；实际候选及 Compose 静态验证通过。 |

复测还纠正 CLI 对 `quality.dialogue` 的硬编码假设：该字段是历史说明，不应将可运行的
确定性候选误判为无效。未放宽运行参数校验，也未改动 manifest。

### 13.2 真实数据重建与 Prompt 诊断

只读取原 development 的 60 条和原生 caption parquet，按 photo_id/business_id 关联，
验证每张图片 SHA；train/test 仅读取身份清单。新增目录
`outputs/week8/review/week8_audit_repair_20260827_v1/labels/caption_evidence_v2/`。
五维隔离 PASS：sample_id/source_id/image_sha256/group_id/constraint_template_id；实际图片
无模板，模板值保持 null，不发明模板身份。train/development/test 身份数为 400/60/60，
跨 split 碰撞 0，60 条 development 全部保留。新标签全部为 programmatic_silver、权重不超过 0.5。

| 引用正支持（样本数） | 旧混合标签 | 新 caption-only silver |
|---|---:|---:|
| 总样本 | 60 | 60 |
| 已知业态 | 60 | 3 |
| 非空风格 | 53 | 0 |
| 非空设施 | 60 | 3 |
| 已知价位 | 0 | 0（N/A） |
| parking | 58 | 0 |

这是错误标签/来源纠正，**不是删除难例、降低支持以提高指标，也不是完整视觉真值**。
caption 未提及某物不等于图片没有该物；所以新参考仍不能证明视觉准确率或召回率。
风格正支持为 0，不能把空参考下实现约定的 recall=1 当作真实召回。

原 3 组各 60 条 development 模型输出只读重计分，新增模型请求 0；raw SHA 写入 summary。

| Prompt | 旧 metadata/caption composite | 新 caption-only diagnostic composite | 新选优 |
|---|---:|---:|---|
| system_repair_product_compact_v3 | 0.782941 | 0.541176 | 不锁定 |
| week8_product_field_check_v1 | 0.836536 | 0.539869 | 不锁定 |
| week8_product_evidence_guard_v1 | 0.740866 | 0.584314 | 不锁定 |

同一 raw 的 JSON/Schema 仍为 100%/100%、请求失败 0，延迟和 token 未变；这些不是新的推理成绩。
仅作为污染敏感性例子，正式 Prompt 的设施 micro P/R/F1 从旧参考的
0.887097/0.591398/0.709677 变为新参考的 0.048387/1.000000/0.092308；不能解读为视觉性能变化。
业态新值 1.0 仅支持 3 条，风格无正支持、价位 N/A。两套参考的选择器均明确拒绝视觉锁定。

历史正式 fresh test composite=0.780639、风格/设施/价位支持 25/30/5 仅作历史，不与上述
诊断混算。本轮没有 Week 8 新最终 test 结果；v7 final 不重跑、不读取标签或输出调参。
冻结 lock 的 `test_status=LOCKED_UNCONSUMED` 是历史字段，不代表已消费 v7 final 可再次使用。

### 13.3 实际检索与真实模型验证

生产检索路由在新建隔离 Milvus Lite FLAT 集合中运行，复用原发布的 1,000 条 CLIP 向量。
固定查询输入来自配置，不使用查询参考 metadata。5 项结果数为 5/5/0/5/5，字段过滤全部正确；
Indianapolis→New Orleans 改变结果，hotel 无匹配时返回空集，不返回餐厅冒充酒店。
keyword 两次不调用图像 encoder，hybrid/embedding 使用身份绑定的既有 CLIP 缓存向量。
这是实际生产路由/真实 Milvus SDK 的闭环证据，**不是新 CLIP 编码、在线 HNSW 部署或图片
相关性提升证据**。旧 NDCG 的 oracle 查询 metadata 增益不能继承为新策略的业务准确率；LRU 尚未接入生产。

GPU job `29667548`，代码 `327f764`，A100 MIG 1g.20gb，`COMPLETED 0:0`，2 分 46 秒。
初始整卡排队后，在验证 MIG 20GB 的既有运行兼容性及可用资源后调整同一待排作业到 MIG；
未新建竞争作业，walltime 为 15 分钟。图片仍为 533×400 的固定 development 实图，
SHA `4522e1aa84ef6f0800b2b138068f56db88e8096a622ef1f842e652b9024cf6d8`。

- 技术 smoke PASS，业务 smoke FAIL。上海两日行程首轮仅一天、含占位文本；纠错后仍一天，
  请求明确失败。对话行程实际调用模型两次，纠错后生成四天，返回 NOT_COMPLETED、FAILED tool call。
- 商品对话实际调用模型 1 次，约 4678.876 ms，返回完整商品 task_result，不再只有确认语。
  该执行 COMPLETED 不代表视觉标签正确；`business_valid=null`，仍含不可见 parking。
- 相同模型/adapter/Prompt/图片预热后商品重复 5 次：mean/P50/P95(nearest-rank)
  =4686.114/4684.040/4702.216 ms；每次 input/output=713/57 token，总计 3565/285。
  商品 JSON/Schema 100%/100%、请求失败 0/5、输出完全一致 5/5，均保留 parking 猜测。
- 冷加载 36987.748 ms；峰值 allocated=8143745536 B。P95 在 n=5 时是最大观测值，不是稳定尾延迟。
  本次为 MIG，不能与此前整卡约 3.9 s 比较宣称提速或回退；未调整 Prompt、adapter 或推理上限。

### 13.4 证据、验证与未完成项

以下路径均相对 `outputs/week8/review/week8_audit_repair_20260827_v1/`，原始文件不入 Git：

| 文件 | SHA-256 |
|---|---|
| labels/summary.json | 960e0d8e05d83ca356edc56526e44c2e17c72a1e1b8ce9a18d54ae9746688d3c |
| labels/caption_evidence_v2/diagnostic_silver.jsonl | bf3bd50cb9df3ac619fbdf75a51fcaeb6b2f1b67401bc5a978ce8605b7568382 |
| retrieval/summary.json | 14ff41ef31e870caa6faa9033e2d9010763a73f7ee39dbb5dc03fd0dc0cba468 |
| runtime/model_smoke.json | 8df93d5c566ca876be8d5015b87f335e9602bd1bab9a6794b4676c8e95f4eca3 |
| runtime/product_dialogue.json | 423f823d1609e61fbf33e32c9399d94f91154cdd290fd36d96c07282cb237f24 |
| runtime/summary.json | c81af248d47b457604b3ea15b58c2ae248590cf765198e8466eda1027a9f2cb1 |

复现命令见 README“c01b732 审查后的修复入口”。25 条新增定向 unittest、完整 679 条 unittest、
compileall、Slurm `bash -n`、分支完整 `git diff --check dev...HEAD`、密钥签名/大文件扫描、
显式 release CLI 和 Compose 静态配置检查已执行；真实数据重建与五维隔离通过。
旧本地四层交接包另行复验 PASS；这是历史完整性验证，不代表新业务 smoke 通过。
正式/RC manifest SHA 仍分别为 `3c71e0d5...ef3f6b`（Git/Spartan LF）和
`9defb3e7...ef749`；实际加载的 adapter SHA 仍为 `c2fbb5c7...eaa2a`。

已完成的是九项确定性缺陷修复、回归保护、真实失败不误报、无泄漏诊断和检索接口接通。
仍未完成的是商品视觉猜测治理、酒店/景点/多主体与风格的可靠视觉评测、已知价位正支持、
实质行程质量、未建模检索条件和独立相关性提升、商品稳定提速。现有 caption silver 正支持
不足，checkpoint-87 模板复述/错误标签倾向仍在；不通过扩大训练、伪造金标或人工工作掩盖。
当前正式模型不变，v7 Prompt/adapter 仅保留为历史候选，不晋级、不合并、不打标签。

## 14. 持续复审：独立图像 silver 与真实服务候选（2026-08-28）

### 14.1 口径和修复

不再用 merchant metadata 或 caption 是否提到某物作为图像视觉真值。独立 qwen3.7-plus
只接收原 development 的 60 张图和版本化观察协议，不接收旧 target、metadata 或候选答案。
60 条全部生成成功，11 条使用一次格式纠错；标签仍是 model_generated_silver、权重 0.5，
人工标注/复核/验收数均 0。这是跨模型图像 silver 一致性，不能宣称人工准确率或统计独立。

商品采用“可观察事实→确定性字段映射”：每个风格/设施标签绑定短事实，多标签分别检查；
工厂设备不自动视作餐厅，食品特写不据此猜场所设施；商家属性不混入 visible_facilities。
价位没有可验证比较口径时为 unknown，原始价格文本仍留在 observation/raw。unknown_fields
由实际字段确定，避免已知字段和 unknown 声明矛盾。所有原始失败、空标签和难例保留。

不新增训练。development 对比正式 adapter、关闭 adapter 的观察 v1/v2；商品候选为
`product_visual_observation_v2` + Qwen3-VL-8B 底座，售后继续使用正式 checkpoint-87。
生产服务按请求持锁切换 adapter，退出或异常时恢复；没有覆盖旧 adapter 或正式 manifest。

### 14.2 固定 development 实测

GPU job `29684981`，执行 `7047093`，A100 MIG 1g.20gb；三组各 60/60 请求通过结构校验。
参考已知业态 39（restaurant 30 / hotel 1 / attraction 4 / other 4）、未知业态 21；
风格正支持 34 样本/42 标签，设施 37 样本/78 标签，价位 0（N/A），unknown 判断 240。
没有正标签的图片仍计入误报，不能免费猜测。三组使用完全相同样本和参考支持。

| 指标 | 正式 Prompt + adapter | 观察 v1 + base | 观察 v2 + base |
|---|---:|---:|---:|
| business_category_accuracy（已知 39） | 0.820513 | 0.820513 | 0.820513 |
| 含 unknown 业态准确率（60） | 0.533333 | 0.850000 | 0.850000 |
| style P/R/F1 | 0.266667/0.380952/0.313725 | 0.777778/0.166667/0.274510 | 0.696970/0.547619/0.613333 |
| facility P/R/F1 | 0.290323/0.230769/0.257143 | 0.943396/0.641026/0.763359 | 0.824324/0.782051/0.802632 |
| price_range_accuracy | N/A | N/A | N/A |
| unknown_accuracy | 0.470833 | 0.845833 | 0.920833 |
| label_completeness | 0.428571 | 0.545293 | 0.685825 |
| composite（3 个有支持字段等权） | 0.463794 | 0.619460 | 0.745493 |
| JSON/Schema | 1/1 | 1/1 | 1/1 |
| 请求失败率 | 0 | 0 | 0 |
| 字段内部一致率 | 0.016667 | 1.000000 | 1.000000 |
| mean/P50/P95（ms） | 4723.828/4717.584/5015.134 | 5344.243/5069.402/9064.609 | 6574.683/5714.439/11249.417 |
| input/output token 均值 | 682.8/59.167 | 817.3/79.3 | 923.8/98.717 |

观察 v1 因风格召回和 F1 回退被拒绝；v2 全指定字段不回退且综合分提升，仅取得
DEVELOPMENT_CANDIDATE。新增事实输出使 token 和延迟上升，不隐瞒或用旧 smoke 数字替代。
历史正式 fresh test 0.780639、支持 25/30/5 的参考口径不同，不能与此表横向比较。
development 正式组沿用诊断运行器的 384 输出上限（原正式 manifest 为 512）；60 次均单次
完成，最长输出仅 76 token，未触及上限。新最终正式组将直接读取原 manifest 的 512 上限。

### 14.3 错误切片（同一 60 条图像 silver）

| 切片 | 支持数 | 正式错误样本 | v2 错误样本 |
|---|---:|---:|---:|
| 已知业态 | 39 | 7 | 7 |
| 多主体/主体不明 | 4 | 4 | 3 |
| 食品特写 | 17 | 17 | 1 |
| 多风格标签 | 7 | 7 | 6 |
| 风格漏标/扩展（含空参考） | 60 | 48 | 20 |
| 设施漏识/误识（含空参考） | 60 | 53 | 23 |
| 无价位依据却给价位 | 60 | 57 | 0 |
| 应 unknown 却猜测或失败 | 60 | 59 | 3 |
| Schema 通过但至少一项语义不一致 | 60 | 60 | 36 |

错误切片不是人工 adjudication。残余多风格、多主体和类别错误明确保留，不能声称标签
全部正确。明细与原始输入哈希在 `outputs/week8/review/week8_development_error_slices_20260828_v1.json`。

### 14.4 证据身份和待完成验证

| development 产物 | SHA-256 |
|---|---|
| 教师 raw | 19a5eeb588158deb991724868b8c14fd2386af7dccae3d895520b8baef9ad194 |
| 正式 raw | bd7a705046231c3c3da1c4ad377227dc9ee0e751b60bae518b403058c3e02321 |
| 观察 v1 raw | 394a0944158f0b1a9797c1ff63da52aa9b677996a62e377da20e91a97350547b |
| 观察 v2 raw | 59fceb8ae9a1f7fa42c6cde82cf49f428960f5e47ec9b1612d23bb80f11df01a |

生产探针 v1 的 PASS 经复审发现“某文化空间”占位地点，不作为完整业务通过；v2 又发现
漏城市检查、未经核实的路线/来源引用。均新增业务校验和真实复验，不复制模型 satisfied
作为验收。商品三模式各八次输出一致且失败 0，CPU 缓存只改善约 0.29%，不是显著加速。
原生图片的模板身份必须为 null/N/A；第一次无标签封存的实验模板名已判定不应使用，
保留旧产物但不消费，以同 seed/source 顺序重新版本化身份，不按结果挑图。

新增 runtime 包隔离导入测试，补齐此前缺失的模块和观察/检索配置。当前全量 729 条
unittest 通过；新最终模型/教师尚未运行。本节不提前宣布可晋级，后续结论只基于真实最终
对比和完整业务复验。旧正式发布包、模型、数据和已消费 final 不变。

探针 v3（job `29693345`，执行 `818a25a`）已完成 PASS：商品/售后/行程/对话 smoke、
两/三/五日固定请求及商品/比较对话均完成，新增日期、具体地点、交通/引用检查通过。
三缓存模式各 8 次结果完全一致、失败 0；CPU 缓存 mean/P50/P95 为
9087.169/9085.690/9096.546 ms，对照 9095.954/9095.036/9100.914 ms，约 0.10% mean
差异，不证明稳定显著提速。新候选保留容量 8 的 CPU 缓存，不占用额外 GPU 输入缓存。
进程模型冷加载 21960.131 ms，峰值 allocated/reserved 为 8143745536/8432648192 B。
每次 input/output=958/140，端到端输出吞吐约 15.406 token/s。

最终数据 v2（CPU job `29693346`）100 张、模板均 null/N/A，五维历史碰撞 0；同此前未
消费 v1 的 source_id 顺序完全一致。数据锁 SHA 为
`6df52dfabceb59ebb3242b29d2475e0aeac0deb7ec2d4ed91250e0018d5f0410`。

### 14.5 锁定候选 v8 的单次最终尝试：未通过

最终推理 job `29693616` 完成，正式组及候选组各 100 条、请求失败均 0；执行的是原正式
manifest 的实际参数和锁定 v8。独立教师 100 条中 99 条有效，1 条在预定两次尝试后仍违反
“食品特写不能证明场所风格/设施”的观察契约。因此锁定评分命令退出 1，错误为
`invalid final reference; never drop or replace it`。没有计算或查看字段选优分数。

本轮结论为 `FAIL / INVALID_REFERENCE`，不是商品质量 PASS，也不是人工准确率。保留全部
100 条、所有教师尝试和两组模型输出；不删除异常样本、不修订最终参考、不重跑该最终集。
失败记录为最终目录下 `acceptance_failure.json`，不存在通过的 `promotion_acceptance.json`。

| 单次最终证据 | SHA-256 |
|---|---|
| candidate lock | 18b830c1d66ba8f6fd018bfd55a8a37beb8fae8485d2f4444c3bca8126738256 |
| 教师 raw | 1bd4bc67ea4b4edc4b5c33cbddea90c5e7c498936d5405e02c33673f4c4453ab |
| 正式 raw | b6387a82337f7110e155480361bb98a1b69794425b76152e20f4e8776f111168 |
| 候选 raw | ccc976e3d1c1acb5fb04c584e08d02be406695d023f521d367a93ed999f32844 |

后续工作仅使用既定 development：观察 v3 检验短事实能否在不损失标签的前提下降低延迟；
教师纠错 v4 将教师自身的无效输出作为上下文，最多四次、仍需通过原验证器。新的教师
development 产物只检验参考生成可靠性，不替换既定 development 参考做候选选优。只有
development 证明了实质改进并产生新候选，才允许使用另一个完全隔离的新最终身份；不能
反复换测试集来获取 PASS。当前仍未晋级，正式 release 不变。

新增教师纠错成功/耗尽/预算边界回归后，全量 unittest 为 733/733（22.600 秒）。实际检索
路由 job `29693617` 通过 5 个固定条件查询（返回 5/5/0/5/5），城市改变会改变结果，条件
过滤通过；推荐对话实际调用一次检索。未建模的“安静”条件明确返回 NOT_COMPLETED 并
保留部分结果。该测试复用身份锁定 CLIP 向量和真实 MilvusLite，不宣称新的视觉相关性提升。

教师纠错 v4 的 development 实测为 60/60 有效：48 条一次、11 条两次、1 条三次，额外
尝试 13 次，raw SHA `a42e2addf54a6c9103f7c241b0cd9d075731f8775996aa91eb8318b6ffd10108`。
旧固定选优参考没有替换。复审额外修复观察事实校验漏掉 `no parking`、`parking unavailable`
等明确否定的问题；两组既有教师 raw 分别 60/60 在更严格校验下映射不变，未修改任何标签。

新锁定流程还必须核对 development generation identity、实测观察配置、模型与 adapter 路由；
配置自身哈希正确但未实测也不能锁定。新增候选验收重放器独立复算最终原始输出、核对
准确 release 的业务 smoke/真实检索及全量测试日志；没有真实证据不会生成通过记录。

### 14.6 固定 development 修订：观察 v3

job `29693799`，执行 `1eddfd2`，同一 60 张图、原教师 v3 的固定参考与支持数；没有读取
旧最终语义分数。正式及观察 v3 各 60/60 成功，新严格否定校验下 raw 重放也通过。

| 指标 | 本次正式对照 | 观察 v3 |
|---|---:|---:|
| 已知业态准确率（39） | 0.820513 | 0.820513 |
| 含 unknown 业态准确率（60） | 0.533333 | 0.850000 |
| style P/R/F1（34 样本/42 标签） | 0.266667/0.380952/0.313725 | 0.604167/0.690476/0.644444 |
| facility P/R/F1（37 样本/78 标签） | 0.290323/0.230769/0.257143 | 0.818182/0.807692/0.812903 |
| 价位（0） | N/A | N/A |
| unknown accuracy（240） | 0.470833 | 0.920833 |
| label completeness | 0.428571 | 0.734441 |
| composite | 0.463794 | 0.759287 |
| JSON/Schema；请求失败率 | 1/1；0 | 1/1；0 |
| mean/P50/P95（ms） | 4857.312/4859.474/5160.187 | 6427.037/6388.496/10393.109 |
| input/output token 均值 | 682.8/59.167 | 995.8/94.9 |

相对旧观察 v2，综合分 +0.013794、风格 F1 +0.031111、设施 F1 +0.010272；但风格
precision -0.092803、设施 precision -0.006143，P50 也变慢。预先实现的修订比较器按
“原正式对照全部门槛通过且综合分进一步提高”接纳其为新 development 候选，不认定为
无质量代价的性能优化。观察 v3 加更严格否定验证进入候选 release v9，随后仍需新业务
探针与完全隔离的一次最终验收。没有给失败的 v8 换测试取得通过。

本次正式 raw SHA `df70f5f8aefa2d53e26267cbd1545fd16bf40216076a10da926f7b70ee82535e`；
候选 raw SHA `e4ae93ae9160d1fe21fcadaffef2891c3c707552dd5bc84f1861b215e3368494`；
comparison SHA `9a7b2de68f913ec98f4e49bdc993b760a42a65e2107a6ffb19a3f4b89459c044`。
v9 配置 SHA `8af73e995d9c94b65a60cac5ec92e3f49cfb1d9624aead7231937fda1755de55`，
观察配置 canonical SHA `ba7d15a4f86bfe344f4bb174044d0d1d611209f31e57c09af5c0dbc2864da0ab`。
包含准确 v9 runtime 隔离导入的全量 unittest 744/744 通过（23.139 秒）。

### 14.7 v9 准确配置的实际服务与最终身份

执行 `6fb9133`，GPU job `29694606`（申请 12 分钟，实用 8:55）完成 PASS：三场景及
行程对话 smoke、两/三/五日明确约束、商品对话和两轮图片比较均完成；业务规则独立重放
通过，不能把结构合规或模型自己的 satisfied 字段当作唯一证据。

| 固定图、各 8 次 | mean / P50 / P95（ms） | 失败 | 输出一致 |
|---|---|---:|---|
| 不缓存 | 9232.380 / 9231.263 / 9238.613 | 0 | 是 |
| CPU 预处理缓存 | 9221.647 / 9219.868 / 9233.382 | 0 | 是 |
| GPU prepared 输入缓存 | 9220.828 / 9219.173 / 9234.928 | 0 | 是 |

CPU 缓存保留容量 8；固定重复图为 7 hit / 1 miss，mean 仅改善约 0.12%，不宣称稳定显著
提速。每请求 input/output=1030/141，端到端 15.290 token/s；模型冷加载 22704.610 ms，
峰值 allocated/reserved 8143745536/8432648192 B。硬件为 A100 MIG 1g.20gb。
运行包独立导入 v9 与旧 adapter 哈希复验通过；Compose 仅执行静态配置验证，没有启动服务。

CPU job `29694607` 完成新身份及准确 v9 检索闭环：此前失败 final 永久排除，剩余 328
张未消费原图中按原 seed/source 排序封存 100 张，五维碰撞 0、模板 null/N/A。另在本地
独立比较新 final 与固定 development、已消费 v2 final，五维交集均 0，并验证全部图像哈希。

数据锁 `f828c9de74a38a0ef767db727cd071f4e7fd09c1905d7c62848486a111d3ce1c`；候选锁
`9e81df3ed069805eaed3f7d9f71383bd57e1940c32fe415ac34bc15331150d32`。锁定后启动 GPU
job `29694824` 和一次教师收集。教师 100/100 有效、失败 0：85 条一次、12 条两次、3 条
三次，全部尝试保留且始终 silver；raw SHA 为
`95639edbab5a9d0c7bda2edb7a36706c20ca39579fcfee4e23ac460de2f6a105`。
此时尚未完成最终配对评分，不提前填写候选质量通过。

### 14.8 v9 独立单次最终评测：PASS

job `29694824` 完成，实用 18:39，正式和锁定候选分别 100/100，请求失败均 0。教师、
正式组、候选组使用同一固定 100 张图，五维身份逐条一致；没有排除困难样本或更换参考。
正式组 100 条均一次生成，候选 99 条一次、1 条使用预定的一次纠错。先按锁定协议评分，
再从原始输出独立重放核验；没有用 final 排序、换 Prompt、改 adapter 或调整阈值。

全部参考为 qwen3.7-plus 图像 silver；human annotation/review/acceptance 均为 0。
业态正支持 62（restaurant 50、other 9、attraction 2、hotel 1），unknown 38；风格 57
样本/77 标签，设施 51 样本/106 标签，价位 0（N/A），unknown 判断 400。两组支持完全一致。

| 指标 | 当前正式模型（同口径） | 锁定 v9 |
|---|---:|---:|
| business_category_accuracy（62） | 0.806452 | 0.822581 |
| 含 unknown 业态准确率（100） | 0.500000 | 0.890000 |
| style precision / recall / F1 | 0.282828 / 0.363636 / 0.318182 | 0.636364 / 0.636364 / 0.636364 |
| facility precision / recall / F1 | 0.166667 / 0.160377 / 0.163462 | 0.777778 / 0.726415 / 0.751220 |
| price_range_accuracy（0） | N/A | N/A |
| unknown_accuracy（400） | 0.432500 | 0.937500 |
| label_completeness（63 个有已知标签的样本） | 0.403099 | 0.693764 |
| composite（3 个有支持字段） | 0.429365 | 0.736721 |
| JSON / Schema 合规率 | 1 / 1 | 1 / 1 |
| 请求失败率 | 0 | 0 |
| 字段内部一致率 | 0 | 1 |
| mean / P50 / P95（ms） | 4925.687 / 4902.253 / 5664.163 | 6016.703 / 5965.712 / 10297.217 |
| input / output token 均值 | 681.73 / 60.04 | 1005.38 / 87.20 |

综合分 +0.307356，所有有支持的预定质量指标不回退；mean 延迟增加 22.15%，input/output
增加 47.47%/45.24%，不能把本次质量提升写成加速。进程冷加载 22153.315 ms，峰值
PyTorch allocated 8143745536 B。与历史 fresh test 的 0.780639 是不同参考与样本，不横比。

最终错误切片（正式→v9）：已知业态 12→11/62；食品特写 37→0/37；多主体/不明 1→1/1；
多风格 18→10/18；风格漏标/扩展 78→40/100；设施漏识/误识 96→37/100；无证据价位
99→0/100；应 unknown 却猜测 99→4/100；至少一项语义不一致 99→55/100。保留剩余
55 个与 silver 不完全一致的样本，不声称全部标签已正确；该结果不用于后续调参。

| 最终证据 | SHA-256 |
|---|---|
| formal raw | 0e1a756393970818229cdf4414ae6786ae0aef921c091014b5df70e13962dfd6 |
| candidate raw | 2e09395c0c7f428d0f10f552198147217320c94511dadfa929fc7d1f1df0e2b3 |
| final comparison | a267e4b43e00af8b8cf198b7d626347571af59067a144a1f4718d1b88d27719d |
| promotion acceptance | 760dc92c44a9aca1e96535499a7060e92ed9946f150e4a976532127d2f787953 |

### 14.9 完整交接、已完成与仍有限制

候选最终选择：`product_visual_observation_v3` + Qwen3-VL-8B 底座；商品、行程、对话关闭
原 adapter，售后保留正式 checkpoint-87。本轮没有新增 SFT 或新 adapter，原权重 SHA
仍为 `c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a`。

三个固定真实对话的首次响应核心契约 3/3、实际任务完成 3/3、失败 0/3；可核验的输入
状态字段保留 2/2（单个行程对话，非通用上下文准确率）。其中行程需要一次下游模型纠错，
不能说完全消除了二次生成；图片比较直接调用模型，tool_calls 为空不表示未执行。
准确 v9 的真实检索服务及推荐分派完成；未建模的“安静”条件明确报告未完成/部分结果。

交接目录：`outputs/releases/trip-qwen3-vl-8b-week8-visual-silver-v9-rc1`。四层包中保留新
final 的全部 100 张图、教师/两组原始输出、数据及候选锁、development/真实业务/检索证据、
746 条测试日志和 v8 失败记录。包内哈希、模型身份、原图完整性、候选验收和隔离运行层
导入均 PASS，`eligible_for_automatic_silver_candidate=true`；没有执行正式发布或合并。

| 交接层 | SHA-256 |
|---|---|
| runtime | 66206880321c66df3c98a12b26d01abc1865d21b47e987911f579e232b2130f6 |
| adapter | f74c078738fa0229574114986c58040bbc280e11ba4ec06558c9a488c2de619d |
| retrieval | 3cdb98f4d50bc72ae53c4e7e96d823ea5b08af93f41df5d14ff1118d12d1a15b |
| evidence | 0f75687d63cb487228f350729106e609560214adc5d9c91d6f6f59206827d035 |
| release manifest | 593d9e058535d31e2358659a0a86c07d393f98f481fb5a45fbb1104b81ef0162 |
| handoff verification | e53eab1e5622f7062535c1f7b612fa1933829b6a6b6c0a64433cd355801591f9 |

验证：新增定向回归、全量 `python -m unittest discover -s tests -v` 746/746（22.729 秒）、
五维隔离与图片哈希、development 比较、单次最终推理及 raw 重放、真实业务 smoke、固定
重复延迟、准确 release/adapter 和完整包校验均已执行。显式 CLI 与环境变量配置错误都会
失败；Compose 仅静态配置验证，未作生产部署。新 adapter 回载不适用（本轮未训练）。
模型运行代码 `6fb9133`，交接校验工具 `31f688e`，开发仅在 feature 分支；最终文档提交及
推送状态见 Git 交付记录，不合并 dev/stg/main、不打标签。

仍待优化：缺少价位正支持，无法评价价位等级准确率；hotel/attraction 和多主体支持很小；
风格/设施误报漏报仍存在；较长观察输出使延迟高于正式模型，缓存收益不显著；未建模的
检索条件与图片相关性提升尚无独立业务证明；未做实时 POI 开放时间、交通路线或预订核验。
这些限制明确保留，自动 silver 候选通过不能被写成人工准确率、零错误或无条件生产通过。

交接补充复审：同一包在 Spartan Python 3.11/FastAPI 0.141.1 复验也 PASS。其内部路由
对象数量为 6，本地为 14，但实际 OpenAPI 的 10 个路径相同；不能靠对象数量判定端点
完整性。已新增七个必需业务/健康端点的明确检查，以及“app 可导入但无业务路由”的反例。
该修复仅作用于交接验证器，34 个锁定模型/评分文件和全部归档字节未变，没有重新运行模型
或消费 final。新版交接复验记录独立保存，不覆盖原记录。
补充全量 unittest 为 747/747（24.750 秒），日志 SHA
`f3906cdba59b023e9a76547f50890da0588a0d9a1055d888517e25d7564c56bc`；本地新版交接验证
SHA `76d1fc166b9449db9fcaeb06cb39f08ba439e3329881ec5a9fd2bfd92f569664`。包内原 746 条
  初验日志保持不变，新增日志单独留存，不把旧记录改写成新结果。

## 15. v9 后继续优化：仅 development 的新实验

本轮由用户“继续优化”授权，保留 v9 已通过候选及原四层交接包，不重新消费已用 final。
商品仍优先于其他方向；所有新增检查与参考均为自动结果，人工参与为 0。

### 15.1 变更与执行身份

- `contract_ablation_v5.json` 固定原 60 条 development 和教师 v3 raw；四组依次为
  formal、v9 观察 v3、观察 v4 紧凑 label→fact、观察 v5 紧凑结构加语义边界。后两配置
  使用新的 `product_visual_observation_v4` wire protocol，公开商品 Schema 不变。
- 逐标签事实没有删除；保留空对象 unknown、十个事实及 80 字符上限，拒绝重复 JSON 键
  和“菜单暗示座位”等推断证据。原协议不改写；补充校验幂等性防止紧凑对象二次校验失败。
- 新 incumbent 比较器要求所有有支持字段不低于同场 v9。纯性能选优还要求 mean 至少
  降低 5%、输出 token 至少降低 10%、P50/P95 不回退；不把一般抖动或精度损失当改进。
- 运行代码提交 `1764324`。首次 job `29697329` 在模型加载前因定向 fetch 未更新远端
  跟踪引用而缺少配置；零模型请求、零数据消费，保留错误日志。改为核验 `FETCH_HEAD`
  快进与实际 HEAD，并预检配置、数据及图片后，串行恢复为 job `29697351`。配置 SHA
  `2bcd4a0ec73c705f3facb0030cb81fae20d03358f058ffd0299a75b09353463b`，30 分钟 A100 MIG。

检索补充复审：显式 `price_range=budget` 与“推荐奢华餐厅”冲突时，旧实现把未应用的
“奢华”删掉并返回 COMPLETED。新实现仅移除实际应用条件的别名，保留冲突或歧义文字；
ASCII 城市也按完整词匹配，不从其他词中删字母。新真实探针包含 8 个查询和 4 个对话，
不声称这项条件覆盖修复提高了图像相关性。

### 15.2 第一组完整 development 结果：保留 v9

job `29697351` 完成，`COMPLETED 0:0`，实际 26:10；四组各保留完整 60 条。
v9 的 60 个公开结果与上轮逐条一致，当前实现从 raw 重放同样全部一致。固定支持为
业态 39、风格 34 样本/42 标签、设施 37 样本/78 标签、价位 0、unknown 判断 240。

| 指标 | 同场 v9（观察 v3） | 紧凑 v4 | 紧凑加语义 v5 |
|---|---:|---:|---:|
| business_category_accuracy | 0.820513 | 0.820513 | 0.615385 |
| style P/R/F1 | 0.604167/0.690476/0.644444 | 0.451220/0.880952/0.596774 | 0.404762/0.809524/0.539683 |
| facility P/R/F1 | 0.818182/0.807692/0.812903 | 0.744681/0.897436/0.813953 | 0.723404/0.871795/0.790698 |
| unknown_accuracy | 0.920833 | 0.912500 | 0.829167 |
| label_completeness | 0.734441 | 0.839867 | 0.768771 |
| composite | 0.759287 | 0.743747 | 0.648588 |
| JSON / Schema | 1/1 | 1/0.983333 | 1/0.966667 |
| 请求失败 | 0/60 | 1/60 | 2/60 |
| mean / P50 / P95（ms） | 6417.182/6378.806/10377.189 | 6850.315/7205.041/9450.711 | 7578.558/7381.099/12016.908 |
| input / output token 均值 | 995.80/94.90 | 1612.52/94.97 | 1662.25/106.55 |

同场正式模型 composite 为 0.463794，不作为替换 v9 的唯一标准。紧凑表示提高部分召回
但增加误报、失败和 token；两个新方案均被 incumbent 比较器拒绝，不删除失败样本。
一次请求的两次输出均含 `seating: No visible seating`，校验器正确拒绝，不能删掉该标签
后伪造原模型通过。该 development 反例触发独立 v6 Schema 表达实验，不改变已运行配置。

comparison SHA `dfcba0f0bcfdf66d208b80b0f72426a040cf7a36016591011cfee527cf128024`；
incumbent decision SHA `8da3ddf72c6a8a7384041955f5c3dee90e8550cef9ab04aeaee3453929d82ec9`。
原始文件与全部字段指标位于 `outputs/week8/review/week8_contract_development_20260828_v5`
和 `outputs/week8/review/week8_contract_comparison_20260828_v5`。价位始终 N/A，不把
unknown 的正确使用写成价位等级识别改善。

### 15.3 真实检索条件覆盖复验：PASS

CPU job `29697507` 在项目内独立校验 worktree、提交 `8afd53c` 完成，耗时 7 秒。
v3 预检失败（job `29697455`，4 秒，未创建集合/执行查询）保留；v4 补齐既有 Yelp
图片引用，先验证固定图片 SHA，再使用原 1,000 条向量和 metadata 创建隔离 Milvus Lite。
没有修改正在运行商品作业的目录。

- 8/8 查询的过滤和状态检查通过；结果数 5/5/0/5/5/5/5/5，换城市改变结果。
- “奢华餐厅”与显式 budget 冲突、两种价位、两种业态均保留未应用文字，返回
  `PARTIAL_UNSUPPORTED_CONSTRAINTS`，不把部分候选说成满足全部条件。
- 4/4 对话状态检查通过，全部实际调用一次检索：普通餐厅推荐完成；安静、双价位、
  双业态请求明确未完成。后面三项不是业务条件已被支持。
- 使用身份绑定的已有 CLIP 向量，不是新图片相关性/NDCG 实验；不宣称相关性提升。

summary SHA `93291db9bab1ad3dee57f582fe3c5c4a04c721c29fd930d0c6fb1e67cab356d5`；
向量 SHA `021f09d764038a3ce53d28d348b4c1b6f5b50ba82f51d69ccd2b1acfeee059ee`，metadata
SHA `7a79894fb027e2f0e6e6aa943a5af21c10c72b084f0dd866c931405debfcd42d`。

### 15.4 等价简洁 Schema 追加实验：仍保留 v9

`contract_ablation_v6.json` 绑定原 development/reference，执行代码 `fb49de6`。采用
`propertyNames` 提示词表达可选标签，内部仍用完整展开 Schema 校验。job `29697591`
仅在 `29697351` 完成后串行执行，同场复跑 formal/v9/观察 v6，申请 22 分钟，实际
18:39，`COMPLETED 0:0`。三组各 60 条，原数据、教师、字段支持不变。

| 指标 | 同场 v9（观察 v3） | 简洁 Schema v6 |
|---|---:|---:|
| business_category_accuracy（39） | 0.820513 | 0.666667 |
| style P/R/F1（34 样本/42 标签） | 0.604167/0.690476/0.644444 | 0.415584/0.761905/0.537815 |
| facility P/R/F1（37 样本/78 标签） | 0.818182/0.807692/0.812903 | 0.750000/0.846154/0.795181 |
| price_range_accuracy（0） | N/A | N/A |
| unknown_accuracy（240） | 0.920833 | 0.854167 |
| label_completeness | 0.734441 | 0.757918 |
| composite | 0.759287 | 0.666554 |
| JSON / Schema | 1/1 | 1/0.966667 |
| 请求失败 | 0/60 | 2/60 |
| mean / P50 / P95（ms） | 6424.501/6385.584/10389.095 | 6794.864/6416.848/12866.893 |
| input / output token 均值 | 995.80/94.90 | 1176.65/98.78 |

新 Schema 减少了相对紧凑 v5 的输入长度，但相对同场 v9 仍更慢、token 更多，且业态、
风格 precision、设施 precision、unknown 与失败率回退。`KEEP_V9_CANDIDATE`，不新增
final、不更换模型。两次完整实验累计 420 个 development 请求（另保留纠错尝试），不是
420 个独立样本。A100 MIG 1g.20gb，第二组冷加载 34727.378 ms，峰值 allocated
8143745536 B；第一组冷加载 35504.525 ms、峰值相同。这是批次观测，不是新的固定图
重复性能基准，也不声称 P95 具有独立稳定性结论。

| 追加实验身份/证据 | SHA-256 |
|---|---|
| contract_ablation_v6.json | d81e84e8b9d988ec88f5845dee3872518219b88dd882c7f0ab999caa995a0fbb |
| formal raw | 419b3d1fcc63af131465d50772ebe5e7fcdb00d1034545062fdf231912709761 |
| v9 raw | 497fdce9dcea8766ef67edf5611a73c75b5645a66d7f273ea953fdb033eedca4 |
| 简洁 Schema v6 raw | ed6f9ec82fecf6c57760a35aafdc231d5b16a3d67acc9f7057041c29ea660188 |
| comparison | 933bb3d96d006cb07cc649da2a75b862721faafed3796549da6eaafee44de406 |
| incumbent decision | 100f3dc48a7de273c8e3f5b861791a4c73346c451a853c3e71411d35569cb270 |

### 15.5 development 错误切片与检索补充

全部原始输出经过当前 mapper 重放；以下是与固定图像 silver 不一致的样本数，不是人工
判错数。v9 取第一组，第二组同一 v9 的质量指标一致。失败仍在分母内，没有删难样本。

| 错误切片（支持） | v9 | 紧凑 v4 | 语义 v5 | Schema v6 |
|---|---:|---:|---:|---:|
| 已知业态（39） | 7 | 7 | 15 | 13 |
| 食品特写（17） | 1 | 1 | 6 | 5 |
| 多主体/主体不明（4） | 3 | 3 | 4 | 3 |
| 多风格（7） | 5 | 5 | 6 | 6 |
| 风格漏标或扩展（60） | 24 | 30 | 36 | 32 |
| 设施漏标或扩展（60） | 22 | 24 | 28 | 27 |
| 应为 unknown 时猜测或请求失败（60） | 4 | 11 | 19 | 14 |
| Schema 合格但语义不一致（60） | 36 | 37 | 43 | 40 |
| 至少一项语义错误或请求失败（60） | 36 | 38 | 45 | 42 |

所有成功输出价位均为 unknown；价位切片中的 0/1/2/2 错误计数来自请求失败，不应描述为
生成了错误价位。价位等级正支持仍为 0。业态支持细分 restaurant/hotel/attraction/other
为 30/1/4/4，另有 unknown 21，不能据此宣称稀疏业态和多主体问题全面解决。
切片产物 `outputs/week8/review/week8_continuation_development_slices_20260828_v1.json`，
SHA `e215f8452ab02788831b1fadb1945387fb2fdc48fa39da396908bee35240b842`。

英语复数业态修复后，CPU job `29698776` 在提交 `06f1b48` 完成 v5 真实检索复验，
`COMPLETED 0:0`、9 秒，10/10 查询和 4/4 对话状态检查 PASS。新增
`find cheap restaurants` 应用 restaurant+budget，返回 5 条；`find hotels` 应用 hotel，
返回空集，明确只表示查询执行完成，不代表找到合适推荐。原有冲突、歧义条件的三条对话
仍返回 NOT_COMPLETED。所有对话各实际调用一次检索，不使用确认语假装业务完成。
使用既有 1,000 条向量的 Milvus Lite FLAT，非新 CLIP 或相关性评测。
summary SHA `1e9e915d140b0ed53d635d465222f53fe52aeb0649841587ad7da8f3498bee98`。

### 15.6 本轮验证、复现与交付边界

- 最新定向 44/44（0.027 秒）、完整 unittest 769/769（23.892 秒）通过；完整日志
  `outputs/week8/review/week8_full_unittest_20260828_v28.log`，SHA
  `7315eb0538efc302dba72e99263ce71c3f7cdd34f8eacfc4ec7388d63036c949`。
- 固定 development manifest SHA
  `39670e793fdbb9d26e255465c66361e3b49cbabeb41d2fc7ab5ccf3840d8cce5`，原教师 raw SHA
  `19a5eeb588158deb991724868b8c14fd2386af7dccae3d895520b8baef9ad194`。两组评分均核验
  generation/config/reference/raw 身份和逐条五维配对，human=0。没有生成新标签、读取
  final 进行选优、改变历史隔离或降低字段支持。
- v9 显式 release 验证、原运行包隔离导入与四层归档哈希复验通过；正式与 v9 配置、
  observation v3、原 adapter 保持不变。本轮不重复 final，不新增 SFT 或 adapter 回载。
- `compileall`、工作区及完整 `git diff --check dev...HEAD` 通过；627 个跟踪文件扫描
  无大于 10 MiB 的文件、无所检私钥/AWS/OpenAI 密钥特征命中。Spartan 原 adapter SHA
  复验仍为 `c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a`。
- 商品新增方案均被拒绝，故未开展新候选的业务 smoke 或重复图加速验收；不把历史
  v9 的此类结果写成本轮重新执行。检索的真实 API/分派复验是本轮实际执行。
- 只提交/推送 `feature/week8-product-understanding`，主工作树 34 个既有改动不处理。
  检索修复在 feature 源码中，未重新打包或部署；第 14 节的候选通过只绑定原 v9 包。

从已同步既有产物的仓库根目录复核（下列 `recheck` 目录/文件必须尚不存在；不调用模型）：

```bash
python scripts/score_week8_visual_silver.py --config configs/week8/contract_ablation_v5.json --output outputs/week8/review/compact_v5_recheck
python scripts/compare_week8_incumbent.py --comparison outputs/week8/review/compact_v5_recheck/comparison.json --output outputs/week8/review/compact_v5_incumbent_recheck.json
python scripts/score_week8_visual_silver.py --config configs/week8/contract_ablation_v6.json --output outputs/week8/review/compact_v6_recheck
python scripts/compare_week8_incumbent.py --comparison outputs/week8/review/compact_v6_recheck/comparison.json --output outputs/week8/review/compact_v6_incumbent_recheck.json
python -m unittest tests.test_product_observation_compact tests.test_query_constraint_coverage -v
python -m unittest discover -s tests -v
```

最终选择不变：`product_visual_observation_v3` + base；不把未通过的 v4/v5/v6 接入 release。
本轮完成紧凑协议实验、严格选优及检索条件修复，未取得新的商品语义或稳定提速收益。
风格误报、设施漏误识、稀疏业态/多主体与价位支持不足继续如实保留；所有新增参考依旧
自动 silver，不安排人工工作，不宣称人工准确率或正式生产晋级。

## 16. 持续自主优化：字段复查与实际视觉输入控制（进行中）

用户进一步要求持续寻找路径直到达成优化目标；第 15 节的失败实验保留，但不作为
本次目标完成。当前仍以 v9 为已通过候选，不覆盖其配置、adapter、Prompt 或交接包。

### 16.1 风格专项复查

`contract_ablation_v7.json` 保留完整原 60 条 development 与固定教师 v3。四组为
formal、v9、非食品场景独立风格替换、已有风格场景仅补充遗漏标签。新配置为
`product_observation_v7.json` / `product_observation_v8.json`，二阶段只接收原图和
必要的候选自产风格，绝不接收教师 target、商家 metadata 或样本编号。首阶段 Prompt
与 v9 完全相同，业态/设施/价位保留；不是先按参考挑出错行再只修这些行。

逐阶段原始输出、两次以内纠错和 token/时延完整留存；复查失败计为请求失败。评分从
完整生成序列重建合并结果，不信任派生字段或最终 style JSON 的自述。事实与 Schema
上限不降低，不截断正标签。新增定向 54 条通过。

Spartan 首次 CPU 预检发现缺少本地原教师副本，未启动 GPU；补传原始 raw/identity 并
核验 SHA 后，60 条与五维隔离预检 PASS。GPU job `29704676` 执行 `151a8f2`、申请
38 分钟 A100 MIG，目前排队，尚无商品对比结果；不提前声明质量提升或进入 final。
配置 SHA `27069dc8aa52cfd731dc5ddda58e2c9a0698d7f09327b6738757fa8310e3515c`。

### 16.2 像素参数真实生效修复：CPU 验证通过

固定 Transformers 4.57.1 的 Qwen2VLImageProcessorFast 实际由 `size` 或成对 min/max
派生缩放尺寸；旧 runtime 单独修改 max_pixels，可能被既有 size 覆盖。新兼容函数同时
更新有效 size 和 min/max，保留 legacy 支持、最小视觉块检查及缓存身份变化。
v9 的 visual_max_pixels=None 完全不变。

CPU job `29704717` 使用独立校验 worktree 的 `6f3e31f`，22 秒完成；真实固定图与
原处理器配置核验如下，不加载 VLM 权重、不使用 GPU：

| 模式 | 实际像素 | visual tokens | input tokens | 上限生效 |
|---|---:|---:|---:|---|
| 原处理器 | 208896 | 204 | 1030 | N/A |
| 旧方式仅 max_pixels=131072 | 208896 | 204 | 1030 | 否 |
| 修复后 131072 | 119808 | 117 | 943 | 是 |
| 修复后 65536 | 55296 | 54 | 880 | 是 |

图像 SHA `cc5034c59eb75c3777457be2272604f635cb86a0929185725a4b17f07510f2e5`；summary
`outputs/week8/review/week8_processor_limit_probe_20260828_v1/summary.json`，SHA
`ba73c6effe87e7368e19d909855b2e75a32d6680658dfcd9048c68616dd812fb`。这证明配置修复，
不证明降低分辨率后语义不回退，也不是重复 GPU 延迟基准。未切换任何 release 参数。
历史采用单属性设置的像素实验不能被当作本实现的有效缩放证据。

最新像素定向 31/31、完整 unittest 792/792（40.222 秒）通过，日志 v30 SHA
`4f98caa3781ff59db5ab1e5985deba6e3183cac855cbf0e5c0d8a81407c149c8`。本阶段没有新
人工工作、teacher、SFT 或最终 test。风格质量、低像素质量和端到端性能结果仍待实际验证，
自主优化目标保持进行中，不以工程测试通过替代模型收益。

### 16.3 场所风格证据范围：过滤实验未通过，转入条件式视觉复查

固定 v9 development 原始输出中，3 条风格依据仅描述衣物或饮品。独立版本
`product_observation_scope_v1.json` 按完整词、否定及场所上下文识别范围，原始事实和
标签保留在 raw 与排除审计中，不更改教师或删除任何样本。直接过滤实验的结果为：

| 指标 | v9 | 范围过滤 replay |
|---|---:|---:|
| style precision | 0.604167 | 0.622222 |
| style recall（42 标签） | 0.690476 | 0.666667 |
| style F1 | 0.644444 | 0.643678 |
| label_completeness | 0.734441 | 0.729790 |
| composite | 0.759287 | 0.759031 |

其余字段、JSON/Schema 和失败率不变，但 recall/完整性回退，结论 KEEP_V9_CANDIDATE。
3 条中有 1 个标签虽然匹配 silver，原模型却用外套支持 casual；直接删除没有验证该场所
是否本来就有 casual 风格。这说明需要重新看图，而不是仅从文字依据判定最终标签。

`outputs/week8/review/week8_style_scope_development_20260828_v1` 是基于 `6f3e31f` 加未
提交 scope 实现的确定性诊断，不是新 GPU 执行；记录继承的时延/token 只用于保留原始
生成来源，不作新性能测量。所有失败证据保留，所用新增实现随后纳入版本控制。

新增 `product_observation_scope_review_v1.json` 将范围错误用作真实风格复查触发条件。
无范围错误不增加调用、也不改标签；有范围错误时重新从图像判断场所风格，新事实仍
越界则进入最多一次纠错，持续失败明确返回失败。定向 69 条通过，尚待实际模型结果。

### 16.4 独立银标依据修订与完整风格复查实测

对全部 60 条原教师参考作范围审计，4 条把衣物或食品当作场所风格依据。v2 范围词表
明确区分 seating/sofa 等真实场所上下文，避免只见 clothing 就删除含场所证据的条目。
独立 qwen3.7-plus 只重看这 4 张图的风格，不接收旧标签、候选、商家属性或样本 ID。
4 次请求全部成功，其余 56 条原样继承；类别/设施/价位及五维身份不变，模板仍为不适用。
新增观察找到的是场所物件与装修，原正标签未减少，style 支持从 34 图/42 标签变为
34 图/44 标签。所有参考均是 model_generated_silver、权重 0.5、human=0。

参考目录 `outputs/week8/review/week8_visual_teacher_style_revision_20260828_v1`：
raw SHA `29a34f8aff360286c1e4053c0e53e24fe143ab877538d7bf848314cf1f9a51aa`；
identity SHA `a4e5f3ad652178f8046f30e35207ef728de06057163c80e35a92ff7f50445df9`；
scope audit SHA `b304253ab7e00862c0b9d0fc6ccf8450874900afb47f4fad039379bc214532fb`。
旧参考不改写；以下各列全部按新参考统一复算，不与 0.759287 的旧参考分数作跨口径比较。

GPU `29704676` 在 `151a8f2` 完成四组完整 60 图，用时 30:19，退出 0。这里的正式
adapter 列是本次 development 推理，不是历史 fresh test 或 37.63 秒单条 smoke。

| 指标 | 正式 adapter | v9 | 全面风格替换 | 仅补充风格 |
|---|---:|---:|---:|---:|
| category accuracy（39） | 0.820513 | 0.820513 | 0.820513 | 0.820513 |
| style precision | 0.266667 | 0.604167 | 0.342342 | 0.378947 |
| style recall（44 标签） | 0.363636 | 0.659091 | 0.863636 | 0.818182 |
| style F1 | 0.307692 | 0.630435 | 0.490323 | 0.517986 |
| facility precision | 0.290323 | 0.818182 | 0.826667 | 0.818182 |
| facility recall（78 标签/37 图） | 0.230769 | 0.807692 | 0.794872 | 0.807692 |
| facility F1 | 0.257143 | 0.812903 | 0.810458 | 0.812903 |
| price accuracy（0） | N/A | N/A | N/A | N/A |
| unknown accuracy（240） | 0.470833 | 0.920833 | 0.900000 | 0.920833 |
| label completeness | 0.423034 | 0.727243 | 0.765282 | 0.763344 |
| composite | 0.461783 | 0.754617 | 0.707098 | 0.717134 |
| JSON / Schema | 100% / 100% | 100% / 100% | 100% / 98.33% | 100% / 100% |
| 请求失败数 | 0 | 0 | 1 | 0 |
| 平均延迟 ms | 4706.566 | 6425.277 | 9985.256 | 8541.111 |
| P50 / P95 ms | 4711.528 / 4978.496 | 6400.012 / 10406.655 | 10027.268 / 17151.679 | 8222.790 / 15583.181 |
| 平均 input / output tokens | 682.8 / 59.167 | 995.8 / 94.9 | 1414.95 / 148.95 | 1329.083 / 125.817 |

结论为 KEEP_V9_CANDIDATE。全面复查虽增加 recall，却同时扩大误报；替换组还在一张
图上连续产生重复 casual 标签而失败，不能用原阶段成功掩盖。全部原始输出与失败保留。
复算目录 `week8_contract_comparison_20260828_v7_style_revision_replay_v2`，comparison SHA
`320c7ef0ffc0d91ae1baf977b3aa687a8fbfc9c452cfb09d75eeb0ded833c47a`。

### 16.5 有范围的后续验证与交付完整性修复（执行中）

`contract_ablation_v8` 在 `fffb6b1`/GPU `29705244` 比较 v9、仅复查越界假设和实际
131072 像素上限，各 60 图。定点复查不扩展其他风格，正确范围的标签及其他字段保留；
像素参数在执行锁内临时应用并恢复，缓存键随真实尺寸变化。申请 26 分钟来自前组实测
吞吐加启动/纠错余量。此处尚无新质量或延迟结论，不生成或消费新 final。

新增独立教师范围校验：不允许推理过滤器悄悄减少参考正标签，越界必须重新看图或失败。
新候选锁要求按修订参考重算全部 development raw、真正优于 incumbent，并绑定实际
像素和新复查依赖；配对推理不会把候选像素参数泄漏给正式基线。发布打包自动带上选中
配置，拒绝非配置/越界路径，已做新配置的隔离运行层导入验证。原 v9 包与配置 SHA 未变。

最新定向锁/参考 37 条、打包 17 条、完整 unittest 827/827（27.813 秒）通过；完整日志
`week8_full_unittest_20260828_v35.log` SHA
`ec52e19f259c108c249b14ecaedac196727290c521bd06e5cc532c15cd7d87ac`。跟踪文件密钥模式
及大于 5 MiB 文件扫描均无命中，主工作树仍为 dev 上 34 项既有改动。自主优化目标未
完成；没有新增人工、训练、最终 test、正式发布、长期分支合并或标签。

### 16.6 定点改写与像素实测：不接受无收益或质量下降

`29705244` 在 `fffb6b1` 完成 180 次请求，用时 19:58、退出 0，三组均无请求失败。
同一修订参考、同一 60 图、相同硬件与模型，结果如下：

| 指标 | v9 | 仅重写越界风格依据 | 有效 131072 像素 |
|---|---:|---:|---:|
| composite | 0.754617 | 0.754617 | 0.715351 |
| style precision / recall / F1 | 0.604167 / 0.659091 / 0.630435 | 同 v9 | 0.521739 / 0.545455 / 0.533333 |
| facility precision / recall / F1 | 0.818182 / 0.807692 / 0.812903 | 同 v9 | 0.802632 / 0.782051 / 0.792208 |
| unknown accuracy | 0.920833 | 0.920833 | 0.916667 |
| label completeness | 0.727243 | 0.727243 | 0.690587 |
| 平均延迟比（对 v9） | 1 | 1.013055 | 0.978721 |
| 平均输出 token 比 | 1 | 1.014226 | 0.989463 |

category accuracy 均为 0.820513（39），price 为 N/A（0）；style/设施支持仍为
34 图/44 标签、37 图/78 标签，JSON/Schema 均 100%，失败率均 0。两个方案都不替换 v9。
像素设置虽然确实生效，但质量下降；定点改写仍把“扭纹玻璃杯”作为 classy 场所依据，
因此只是改写事实而没有修正标签。原始目录 `week8_contract_development_20260828_v8`，
配对重放目录 `week8_contract_comparison_20260828_v8_style_revision_v1`。

新增 `product_observation_scope_repair_v2` 对餐具、衣物等非场所事实使用明确的 unknown
策略，保留全部 raw 和 `style_evidence_abstentions`；这只处理结构合法却没有场所证据
的假设，重复键/标签、长度、推断句和未请求标签继续失败。玻璃幕墙/吊灯/门窗的装修
上下文不被当作餐具，教师参考仍必须纠错或失败，不能用推理弃权过滤标签。原参考按
扩展范围审查无新增问题，没有再次改变数据或支持。新对照 `29705434` 在 `a878a0c`
执行 `contract_ablation_v9`，当前尚无完整结果，不提前锁定候选。

独立教师显式餐具边界在 60 图可靠性检查通过（64 请求），但不拿该新输出替换选优参考。
协议 raw SHA `6a62f136ec921464008cae3c5ca00e9d85ce080913790624ee1eeed6763e198b`。
更早的范围协议虽 60/60 结构通过，却有 1 条餐具事实；新范围重放失败的日志也保留，
不把形式正确等同正确参考。最新完整测试 833/833（27.936 秒）通过。

后续若 development 真正胜出，新单次 final 将同时保留正式模型与 v9：原严格正式
基线提升要求不变，再核对相对 v9 不回退；汇总和交接不会只展示较容易通过的那一项。
目前仅检查未用身份尚有 228 张，未建立新 holdout、未读取旧最终结果调参或运行新 final。

### 16.7 非场所证据弃权取得小幅 development 收益（v10 待验）

GPU `29705434` 在 `a878a0c` 上完成两组各 60 条，13:35、退出 0。固定修订 silver
参考（human=0，权重 0.5），没有改变任何图片、标签或支持。商品首阶段 Prompt 与 v9
相同，仅对越界的自有风格假设复查；仍没有场所依据时明确 unknown。鸡尾酒玻璃杯不再
支持 classy 场所，消除一条误报；其他字段及正确范围的风格保持不变。

| 指标 | 同场 v9 | 锁定待验 v10 | 支持 |
|---|---:|---:|---|
| category accuracy | 0.820513 | 0.820513 | 39 |
| style precision / recall / F1 | 0.604167 / 0.659091 / 0.630435 | 0.617021 / 0.659091 / 0.637363 | 34 图 / 44 标签 |
| facility precision / recall / F1 | 0.818182 / 0.807692 / 0.812903 | 同 v9 | 37 图 / 78 标签 |
| price accuracy | N/A | N/A | 0 |
| unknown accuracy | 0.920833 | 0.925000 | 240 字段决策 |
| label completeness | 0.727243 | 0.727243 | 43 条非空参考（60 图全部保留） |
| composite | 0.754617 | 0.756926 | 同参考 |
| JSON / Schema / 请求失败率 | 100% / 100% / 0 | 同 v9 | 60 请求 |
| 平均 / P50 / P95 ms | 6453.674 / 6385.795 / 10414.480 | 6544.186 / 6410.724 / 10721.846 | 同场 |
| 平均输入 / 输出 token | 995.8 / 94.9 | 1020.1 / 96.25 | 同场 |

development 状态为 `IMPROVED_DEVELOPMENT_CANDIDATE`，不是最终晋级。平均约慢 1.4%，
不属于提速方案。冷启动 31453.344 ms、峰值分配显存 8143745536 B。原始目录
`week8_contract_development_20260828_v9`；同参考对比目录
`week8_contract_comparison_20260828_v9_style_revision_v1`，comparison SHA
`cd71580aa8db3b5f39b822b634850d398d5e08bb7325a2981c4db8d186c65d0e`。

固定 `configs/releases/qwen3_vl_system_week8_v10.json` 与
`configs/week8/product_observation_scope_repair_v2.json`，保留 v9 冻结包；没有 SFT。
真实 runtime/retrieval 和新单次 final 配置已准备，新增实际生产弃权分支检查，确保不是
只在离线评测生效。新 final 必须同时保留正式模型与 v9，且锁定后不调参。当前尚未运行
这些新验收或创建新 holdout；最新完整 unittest 834/834（27.759 秒）通过，日志 v39。

后续切片复算（`week8_v10_development_slices_20260828_v1.json`）：类别错误 7/39、
多主体/模糊 3/4、多风格 7/8、设施错误 22/60 均不变；风格错误样本 25/60→24/60，
存在其他字段错误的重叠使总语义错误仍为 36/60。price 无证据猜测 0/60。支持未减少。
切片工具新增显式旧候选 observation 身份，不把 v9 原始观察错当正式直接 JSON。
完整测试 835/835（28.904 秒），日志 v40；完整分支空白检查和密钥/大文件扫描通过。

新无标签 holdout 已创建（`29705564`，5 秒），从 228 未使用图片固定抽取 100 张，
五维隔离及原始身份重推导通过；锁为
`8f3044e1362d90232d0631c7795bde62f3bc76aa4c16a778bc9c4d7fe9dfeb10`。
真实检索 `29705571` 已通过 10 查询/4 对话状态；首次错误 venv 导入失败保留。
runtime `29705563` 的商品/售后/行程 smoke、三条约束行程、商品对话、图片比较和
实际风格弃权分支通过，重复延迟仍执行中。此时尚无新 final 教师标签或候选输出。

### 16.8 v10 真实业务与性能验收；独立 final 执行中

runtime `29705563`（3072446）9:18 完成，PASS：商品/售后/行程 smoke、三条多日约束
行程、商品对话、图片比较和实际范围弃权分支均通过。三条独立约束行程各一次生成；
三个真实对话核心响应/任务完成 3/3，失败 0/3，其中行程对话仍用一次下游纠错。city/days
保留 2/2 仅来自固定 smoke，不是通用上下文准确率。图片比较实际调用模型，不能因
tool_calls 为空误判未执行。检索 `29705571` 通过 10 查询/4 对话状态检查，三条未支持
条件仍为 NOT_COMPLETED，不宣称完整条件覆盖或图像相关性提升。

| 同图固定基准（每组 8 次） | uncached | processor_cached（保留） | prepared_cached |
|---|---:|---:|---:|
| mean ms | 9235.946 | 9226.931 | 9226.790 |
| P50 ms | 9235.783 | 9225.362 | 9224.543 |
| P95 ms | 9240.167 | 9236.647 | 9242.446 |
| 生成 token/s | 15.266 | 15.281 | 15.282 |
| 输入 / 输出 token（每请求） | 1030 / 141 | 1030 / 141 | 1030 / 141 |
| 请求失败 / 标签变化 | 0 / 0 | 0 / 0 | 0 / 0 |

冷启动 36127.780 ms；A100 MIG 1g.20gb，峰值分配/保留显存 8143745536/8432648192 B。
逐 raw 重放生成与重新计算全部性能汇总一致。约 0.1% 的缓存差异不作为显著提速结论；
低像素方案此前因质量下降明确拒绝。Compose 仅静态验证及隔离运行层七个必需端点通过，
没有生产部署；原 v9 四层包和 adapter SHA 再次通过。

候选在 `dedc859` 锁定，绑定 49 项源代码/配置及 33 个 runtime 文件：
`369bd627d01a6b01afca26e5bdc734df1c010a1b46249b35e347b5bac5a7f910`。
新单次 final `29705792` 对 100 图分别运行正式 adapter、v9、锁定候选，独立教师只看图；
依据 development 实测预计约 30.1 分钟并申请 37 分钟时限。当前执行中，尚无最终质量
或晋级结论；不根据最终结果调参、删除困难样本或修改教师协议。全部新增标签为 silver。

### 16.9 v10 最终验收失败：保留收益与失败，不晋级

GPU `29705792` 在 `dedc859` 上 28:20 完成三组各 100 图；教师 100/100 有效，113 次
请求，raw 重放确认五维一致、无重复键或范围错误。全部自动 silver，human=0。

| 新 final 指标 | 正式模型配置 | v9 配置 | v10 | 同口径支持 |
|---|---:|---:|---:|---|
| category accuracy | 0.791667 | 0.895833 | 0.895833 | 48 |
| style P / R / F1 | 0.222222 / 0.265060 / 0.241758 | 0.666667 / 0.481928 / 0.559441 | 0.677966 / 0.481928 / 0.563380 | 47 图 / 83 标签 |
| facility P / R / F1 | 0.115385 / 0.134831 / 0.124352 | 0.802198 / 0.820225 / 0.811111 | 同 v9 | 40 图 / 89 标签 |
| price accuracy | N/A | N/A | N/A | 0 |
| unknown accuracy | 0.352500 | 0.920000 | 0.920000 | 400 决策 |
| label completeness | 0.312103 | 0.673435 | 0.673435 | 全部 100 图保留 |
| composite | 0.385926 | 0.755462 | 0.756775 | 同参考 |
| JSON / Schema | 100% / 100% | 100% / 99% | 100% / 99% | 100 |
| 请求失败率 | 0 | 1% | 1% | 100 |
| mean / P50 / P95 ms | 4910.006 / 4921.349 / 5337.123 | 5874.139 / 4774.143 / 10532.118 | 5954.379 / 4775.024 / 10537.579 | 同场 |
| 平均输入 / 输出 token | 682.53 / 60.07 | 1027.72 / 84.76 | 1042.93 / 85.83 | 同场 |

v9/v10 同一条请求两次生成均违反“food_closeup 不建立场所风格/设施”契约；两组内部
一致性也仅 99%。验收 `29706091` 退出 1，正式基线与 incumbent 检查都 FAIL，未产生
promotion_acceptance 或新候选交接包。字段不退及 0.001313 的综合收益不能替代完整通过。
总语义错误 54/100、有效 Schema 但语义错误 53/100；多主体/模糊 7/8、多风格 25/29，
仍有明显缺陷。新 final 无酒店正支持、景点仅 1，价位 N/A，不能宣称这些类别提升。

final comparison SHA `0a15524685f6b002d61d4e8c24b1fd47efedd3a40c339c897c133e7b0dc0fc8a`；
全量 raw/数据/协议只读复算一致，失败集及 v10 配置冻结，不重跑、换图、改参考或放宽
验收。`acceptance_failure.json` 是 MODEL_CONTRACT_FAILURE，不是无效参考重试许可。
后续仅回到已有 development：此前 contract v5/v6 已有 21 次同类错误，可用来验证
通用纠错历史缺失问题；不把本次 final 的图像、模型答复或标签作为新方案输入。

### 16.10 development 纠错实验失败与后续约束提示验证

`observation_retry_probe_v1` 固定收集既有 development v5/v6 的全部 34 条首轮错误
（22 张图）：食品特写矛盾 21、推断设施 7、否定设施 4、缺字段 2。没有读入 final
样本或教师 target；原错误保留，紧凑键值证据只做无损数组转换。

真实作业 `29707041`（60cafe3，5:18）中，旧纠错 34/34，新有界历史纠错 33/34；后者
仍遗漏必需的 price_text。因此拒绝该方案，不执行其预备的完整 development 或新 final。
逐输入消息哈希、原始答复和失败计数独立重放通过，不将耗时下降当作合格加速。

下一项 `product_observation_guarded_v1` 保留旧纠错，只在首阶段 Schema 提示中加入
已有字段约束注释。实际 Schema、语义校验、词表、样本和支持数均不改变；模型仍输出
矛盾时请求失败，不自动删标签。完整双组 `29707190`（dce9aac）执行中，未锁定新候选。

复审另修复旧 development revision 比较函数的遗漏：综合分提升也必须保证各字段不退，
原有支持字段变成 N/A 不能视作持平。当前 incumbent 比较本已有该保护，历史结果不改。
新增 11 类字段回退与 N/A 反例，完整 unittest 852/852（30.222 秒，日志 v45）通过。
补充澄清 completeness 的分母：60 图全部保留，指标对其中 43 条非空参考取均值，
其余空参考仍参与误报、unknown、格式和失败率统计。不是减少评测样本。

字段提示对照已完成（29707190，13:00）：60/60 均有效，但新方案 style F1 降至
0.487179、category 0.794872、composite 0.698462，低于 v9 的 0.630435/0.820513/0.754617；
明确拒绝。平均耗时下降约 4.56% 伴随正标签漏识别，不能算无损性能收益。
下一步只在原首轮失败时验证完整主体关系的解码纠错，不改首轮判断或降低后置验收。

### 16.11 仅纠错解码约束：验证中，不提前晋级

现有 LMFE 0.11.3 的 CPU 反例证实 maxItems=0 仍允许首个非空元素；新推理纠错使用
显式字面空数组，不修改依赖。模型仍选择 food/nonfood，原始输出不裁剪，否定设施等
语义规则继续后置验证。首轮 Prompt、图片、词表及一次纠错上限保持不变。

第一版真实诊断 `29707533` 两组各 34/34、原始重放通过，但追加格式检查发现 food 与
nonfood 接受不同的空格/键顺序，可能影响主体选择，因此拒绝，不进入完整质量或 final。
第二版为所有主体共享紧凑序列化，18 条真实解码器契约反例及 32 条格式检查通过，
861 条 unittest 通过；随后启动真实诊断 `29707641`。上述合成 JSON 是结构测试，
不是人工标注或视觉准确率证据；仍需固定 60 图质量对照和独立最终验收。

共享格式诊断 `29707641` 因 Slurm 时限被回收（TIMEOUT，12:10），不是主动释放；
旧纠错已完成 34 条，共享格式完成 22 条，均无已完成请求的契约失败。保留全部文件与
中断记录，新 `observation_retry_probe_v4` 校验原源码/模型/数据及 raw 前缀，只补余下
12 条，在作业 `29707912` 中执行。中断时的部分结果不计完整通过；完整结果见下。

补跑已结束（29707912，ce138f5，7:45、0:0）：旧组与共享格式组各 34/34，通过完整
raw/提示/解码配置及原始前缀重放。一次 TIMEOUT 及可能丢失的在途请求仍单列；旧输出
未修改。纠错均值 4194.372→29244.857 ms，累计输入 34735→34735、输出 1927→1624，
没有速度收益。此诊断只覆盖历史 development 错误，不是端到端视觉质量验收。

新完整 development 作业 `29708114`（22dae4e）运行 `contract_ablation_v13`，固定
原 60 张图及修订后的同一 silver 参考，与 v9 同场比较。首轮完全不改，只在失败时
启用新约束；继续保留越界风格定点复查。尚无新 release、final 或晋级结论。

本轮全量 unittest 再验 866/866（29.750 秒，日志 v51）；冻结 v9 交接包及隔离 runtime
导入、v9/v10 配置哈希、分支完整 diff 空白检查通过。已跟踪代码密钥模式命中 0、敏感
路径 0、超过 5 MiB 文件 0；主 dev 的 34 项原有改动未触碰。这些不替代商品最终验收。

### 16.12 纠错语义复审：缩小约束范围，未晋级

`29708114`（22dae4e，13:24）完整双组 60 图均通过；v9/统一约束的 composite
0.754617/0.756926，style F1 0.630435/0.637363，其他支持字段不退。均值
6444.934/6534.785 ms，P95 10395.651/10705.899 ms，输入 995.8/1020.1、输出
94.9/96.25。两组首轮错误都是 0，新纠错路径未在此完整集合上触发。

补充用既有固定 silver 参考审计 34 个历史纠错案例（22 张重复图片），设施 TP/FP/FN
从 7/2/3 变为 6/6/4；风格从 3/3/8 变为 7/6/4。21 条食品矛盾的公共语义标签
完全不变，设施回退来自其他错误。这个切片不是独立视觉质量估计，但已足以否决统一
约束方案；不能因为全量 development 自动 eligible 就忽略已知回退。

因此新版本 `product_observation_food_retry_v1.json` 仅在原校验器明确报告食品特写与
场所字段矛盾时启用共享主体解码，其他错误沿用原纠错。首轮、完整词表、参考与支持数
不变，模型仍可选择其他主体，仍只有一次纠错。补齐实际路由、无上下文拒绝、原始输出
及分错误语义重放测试；完整 875/875（30.341 秒）通过。新 34 条诊断/60 图对照待运行，
没有新 release、final、人工标注或晋级结论。

新诊断 `29708314`（28db2d1）执行中，18 条真实解码器契约/32 条格式预检已通过。
纠错分切片语义检查已接入候选锁、final 消费前校验和交接证据哈希检查；不会仅凭
全量 development 汇总分数忽略已知回退。交接时还会逐原始行核对所有组的失败率，
基线失败单列，不再误算为候选失败；新候选仍要求零失败。全量 884/884（25.629 秒）
通过，冻结 v9 交接包按新检查复验 PASS。当前目标仍未完成。

针对性方案诊断已完成（29708314，11:17、0:0）：两组 34/34，无中断，所有逐错误
切片的类别/风格/设施统计完全一致；facility TP/FP/FN 均 7/2/3，style 均 3/3/8。
原始输入、解码参数和输出在 Spartan 与本地重放一致。旧/新均值 4188.667/14988.250 ms，
输入累计均 34735、输出 1927/1697；不能宣称速度提升。新全量 development 对比
`29708515`（54ad6cd，contract_ablation_v14）执行中，仍未创建新 final 或晋级候选。

### 16.13 针对性方案通过 development，选定 v11 待验

`29708515`（54ad6cd，13:24、0:0）完成，同一 60 图与修订 silver 参考的完整复算一致。

| development 指标 | v9 同场 | v11 待验方案 | 支持 |
|---|---:|---:|---|
| category accuracy | 0.820513 | 0.820513 | 39 |
| style P/R/F1 | 0.604167/0.659091/0.630435 | 0.617021/0.659091/0.637363 | 34图/44标签 |
| facility P/R/F1 | 0.818182/0.807692/0.812903 | 同 v9 | 37图/78标签 |
| price accuracy | N/A | N/A | 0 |
| unknown accuracy | 0.920833 | 0.925000 | 240决策 |
| completeness | 0.727243 | 0.727243 | 43条非空参考，保留60图 |
| composite | 0.754617 | 0.756926 | 同参考 |
| JSON/Schema；失败率 | 100%/100%；0 | 100%/100%；0 | 60 |
| mean/P50/P95 ms | 6441.776/6386.300/10390.829 | 6530.159/6384.178/10695.503 | 同场 |
| input/output token 均值 | 995.8/94.9 | 1020.1/96.25 | 同场 |

风格错误 25→24/60，但总语义错误仍 36/60，不能声称全部标签正确。模型/adapter 未变，
商品继续走底座、原观察 Prompt 与定点风格复查，只对食品矛盾使用共享解码纠错。
配置 `qwen3_vl_system_week8_v11.json` 状态为 candidate_evaluation；新 runtime v6 /
retrieval v7 / final v5 配置已准备。最终集保留 100 图，明确排除已消费 v2/v3/v4，
未创建数据集或执行 final。最终验收未完成，原 v9/v10 产物不变，人工工作为零。

### 16.14 v11 真实运行通过并锁定独立 final（2026-08-29）

版本标识沿用预注册的 `20260828`，本节交付日期为 2026-08-29。GPU `29708734`
在 `2eb33ce` 上 9:11、0:0 完成：商品/售后/行程及对话 smoke PASS，上海2日、北京3日、
杭州5日业务约束逐原始输出重放通过；商品分派与双图比较都实际调用模型，均 COMPLETED。
非场所风格弃权探针通过，未把 Schema 合规当作视觉标签正确。
商品识别、双图比较、三组显式行程和三场景 smoke 均首轮有效；对话中的行程分派仍因
首轮漏掉三个契约字段而使用一次纠错，最终完成。确定性路由不等于下游模型永不纠错。

| 固定同图基准（每组8次） | uncached | processor_cached（保留） | prepared_cached |
|---|---:|---:|---:|
| mean ms | 9237.738 | 9227.763 | 9228.019 |
| P50 ms | 9237.300 | 9226.616 | 9225.968 |
| P95 ms | 9243.110 | 9234.774 | 9243.103 |
| 生成 token/s | 15.263 | 15.280 | 15.280 |
| 输入/输出 token（每请求） | 1030/141 | 1030/141 | 1030/141 |
| 失败/标签变化 | 0/0 | 0/0 | 0/0 |

冷启动 30429.037 ms，A100 MIG 1g.20gb，峰值分配/保留显存 8143745536/8432648192 B。
本地完整复算 24 条 raw、三类 smoke、三组业务行程与两类对话，所有汇总一致；约0.1%
差异不作为实质提速。summary SHA
`19a68466eae5f5643fe96426965238c977e724a8fedfc1c7007a52fec8f7e620`。

CPU `29708735`（8秒、0:0）在相同 release 上通过10组真实检索查询和4组对话状态；
生产路由实际执行、查询变化影响结果，不向排序传入参考 metadata。未建模的安静、歧义
或冲突条件继续明确 NOT_COMPLETED。该探针不证明新的图像相关性/NDCG 提升。summary SHA
`7009f844cbb0ceb27c7346cc7254767058acbc1cb9145a6353a047918e42480f`。

在 Spartan 原项目目录对128张未消费候选按固定种子选100张，五维历史重叠全0，原生图片
模板为 null/N/A；未看标签选样。数据锁
`ddee2e4e31a55afbee3ce8f1f0bf5617a88aef9f1367f560c38e00c8fb7f5c03`，候选锁
`4a344fe1ad6e82d788001273e9cef3c1b04f193772c3c7e8b148267c0b948d7b`。
GPU `29708885` 正运行正式/v9/v11各100图，基于此前28:20实测申请37分钟；独立教师仅看图，
无 metadata 或候选输出。无人工工作，不改锁定源码、参考、样本或阈值；单次结果待验。

教师已完成100条有效参考，113次请求中13次中间错误保留；无最终参考失败、重复键或
范围错误。逐raw映射及五维身份复算一致，raw SHA
`b2b71e874a45c3f7d78228359e8918884e8a7399cfd80aaf6932e318f3b17490`。
CPU验收作业 `29708959` 仅在推理结束后生成一次对照并独立重放，不在运行中调整方案。
本轮定向51/51、全量884/884（44.975秒，日志v56）通过；准确release校验、Compose静态
检查、冻结v9四层复验、分支完整空白检查、已跟踪密钥模式/敏感路径/大文件扫描均通过。
全量日志 SHA `6a05c23782c9f1055571c7655edd4574bc5495d82d9d89103504580b33b1e8ad`。
主dev仍保留原34项改动，本轮只改独立feature的交付文档；没有部署服务或改长期分支。

### 16.15 v11 最终：风格收益保留，业态回退使验收失败

GPU `29708885` 在2eb33ce上28:15、0:0完成三组各100图，均无请求失败；CPU验收
`29708959` 在5秒后退出1。正式模型业态47/56，v9/v11均45/56，因此正式基线非回退
检查FAIL，incumbent检查PASS。不得把综合分提高解释为完整通过，没有生成新交接包。

| 新final指标 | 同批正式模型 | 同批v9 | 锁定v11 | 支持 |
|---|---:|---:|---:|---|
| category accuracy | 0.839286 | 0.803571 | 0.803571 | 56 |
| category含unknown | 0.480000 | 0.860000 | 0.860000 | 100 |
| style P/R/F1 | 0.298969/0.337209/0.316940 | 0.734375/0.546512/0.626667 | 0.746032/0.546512/0.630872 | 49图/86标签 |
| facility P/R/F1 | 0.186275/0.174312/0.180095 | 0.833333/0.733945/0.780488 | 同v9 | 54图/109标签 |
| price accuracy | N/A | N/A | N/A | 0 |
| unknown accuracy | 0.402500 | 0.915000 | 0.915000 | 400决策 |
| completeness | 0.426457 | 0.649754 | 0.649754 | 保留100图 |
| composite | 0.445440 | 0.736909 | 0.738311 | 同一参考 |
| JSON/Schema；失败率 | 100%/100%；0 | 100%/100%；0 | 100%/100%；0 | 100 |
| 内部一致性 | 4% | 100% | 100% | 100 |
| mean/P50/P95 ms | 4862.335/4857.542/5513.863 | 5817.903/4618.575/10358.290 | 6020.544/4620.300/10863.967 | 同场 |
| input/output token均值 | 679.37/60.00 | 1003.26/84.13 | 1007.88/84.09 | 同场 |

全部自动silver，human=0；本地与Spartan逐原始输出、数据锁、参考、配置及完整指标复算
一致。comparison SHA `bb8a295c1ec8ceb4b9fe7c7728d6fabc0c3d06cf6397511883f8105c8be6730f`，
重放日志 `outputs/week8/review/week8_v11_failed_final_raw_replay_20260829_v1.log`。
v11配置、final v5全量图像/参考/答复及锁永久保留，不重跑、不改标签、不据其逐图错误
调参。后续只回到既有development：那里已存在餐饮门面误判零售、场景功能误判等类别
问题。未启动新增训练，没有以猜测价位或减少正支持来提高结果。

### 16.16 主体复查：实现与固定development验证准备

新 `product_observation_subject_review_v1` 保留v11首轮、风格复查和食品纠错，随后只对
当前输出为零售/工业/景点/未识别的场景独立重看图片。模型检查可见业务功能、入口招牌、
生产设备或陈列，不接收旧类别、教师、样本ID或商家metadata；最多输出两个主体字段与
80字符内短事实，最多一次纠错。只替换subject_kind/subject_fact，风格/设施/价位不改；
若新的食品判断与原有场所字段矛盾，直接失败，不裁剪标签来伪造成功。

新增14条主体复查回归覆盖范围、raw重放、原字段保留、失败/纠错、重复键及源码身份；
完整898/898（39.645秒，日志v57）通过。初次一个测试mock误用异常构造参数，修正fixture
后通过，没有修改运行行为来迁就测试。新阶段的六项生成源码按统一换行哈希绑定，不能
用变更后的实现重解释已生成证据。`contract_ablation_v15` 固定原60图及同一silver参考，
真实GPU对比尚待执行，不宣称新候选已经胜出，也未创建新的final。

### 16.17 主体复查v1失败与收窄v2

`contract_ablation_v15` 在原60图完成双组实测并逐raw跨主机复算一致：v9→广泛复查的
category为32/39→30/39，style F1为0.630435→0.637363，facility F1保持0.812903，
composite为0.754617→0.739832。JSON/Schema均100%、失败0；price仍N/A，支持不变。
平均/P50/P95由6430.467/6374.826/10369.609变为7046.271/7139.636/11387.205ms，
输入/输出token均值995.8/94.9→1168.133/102.783。15次主体复查中，修正了两处餐饮
门面，却误改了原本正确的商店/景点；不锁定、不晋级、不消费新final。
comparison SHA `36282524a6acf87edd1cc7beba562c28c6b6bdb4347e25600957545f8eca8075`。

作业29709265首次因远端旧HEAD缺少新配置而启动失败，没有模型请求或输出。保留原
allocation，核实无输出后快进到d463c56，在同一allocation的step .0恢复，13:54/0:0
完成120条；batch最终仍记录FAILED/1:0、15:30，不能写成整体PASS。初始失败和恢复日志
均保留。后续远端fetch后显式快进FETCH_HEAD并验证完整提交身份，再提交作业。

v2只在当前主体为retail_space且短事实同时含正向、完整词的餐饮功能与场所上下文时
触发同一个独立看图Prompt；关键词不直接赋类别，不含样本ID或品牌。保留原首轮、风格
复查、食品纠错与其余字段；排除否定、食物特写及仅商品陈列的触发。新配置
`product_observation_subject_review_v2`/`contract_ablation_v16` 仍使用原60图同一参考。
旧v15证据冻结；新版本额外绑定所复用的否定解析源码，不重解释旧输出。
定向16/16及完整900/900（26.405秒，日志v58）通过；全量日志SHA
`916d3d7f746bcd84fae5fd21d9a2d5ecee3e5e4b76dc7a6984a3263b4aaee110`。

只读身份审计29709285还确认：原6000张已解压合法图片排除全部历史（含final v5）后，
只有77图/72独立商家组。没有按标签筛图、没有生成新测试集。下一候选若通过development，
须从已保存的合法原始压缩包补充未消费身份，不能降低100图要求或复用已消费final。

### 16.18 新无标签身份池与交付来源检查

`unlabeled_source_pool_v1` 固定原7.447GB照片ZIP及business/photos表哈希，仅读取
business_id/categories确定OTA范围、photo_id/business_id确定身份；不读取caption、
图片标签、设施或价格参与选择。按固定组种子和图片种子选择最多4000张，流式提取，
排除所有已消费身份（含失败final v5）。缺失、不可读、重复与历史哈希逐项保留，无人工
或模型标签。此处为source pool，不是新final；最终仍须独立固定选择100张并复核五维。

新增池锁、身份文件、请求清单、拒绝清单和摘要随最终数据携带；发布包复验检查这些
原始摘要和最终样本归属，不依赖原ZIP或省略拒绝记录。旧final/交接包不改，冻结v9再次
复验PASS。新池及交接定向18项、完整908/908（26.544秒，日志v60）通过，日志SHA
`6fd7dcc790275cee6408a4e08c74c92067daa4adf2c934f429bec4dcf0e5fab1`。
尚未实际提取，因此没有声称新的可用图片数量或最终质量收益。

`observation_retry_probe_v6` 准备同一34条/22张历史development纠错，在准确subject v2
配置上重测已有纠错阶段；它不运行独立主体复查，不能替代全60图v16质量评测。身份
audit-only通过，保留21食品矛盾/7推断设施/4否定设施/2缺价位字段，不输入参考答案。

### 16.19 收窄主体复查通过development，登记v12待验

GPU29709486在1dd16aa运行13:29/0:0，两组各60条完整；本地/Spartan逐raw重放的完整
comparison哈希一致：`3a7b74b28371aafbb999f32765b9b9fb0cd6cd81f4ab796e12e3fc28c2dd03cc`。

| 固定development | 同场v9 | subject v2（v12待验） | 支持 |
|---|---:|---:|---|
| business_category_accuracy | 0.820513 | 0.871795 | 32/39→34/39 |
| category含unknown | 0.850000 | 0.883333 | 60 |
| style P/R/F1 | 0.604167/0.659091/0.630435 | 0.617021/0.659091/0.637363 | 34图/44标签 |
| facility P/R/F1 | 0.818182/0.807692/0.812903 | 同v9 | 37图/78标签 |
| price | N/A | N/A | 0 |
| unknown accuracy | 0.920833 | 0.925000 | 240决策 |
| label_completeness | 0.727243 | 0.746622 | 保留60图 |
| composite | 0.754617 | 0.774020 | 同一silver参考 |
| JSON/Schema；失败 | 100%/100%；0 | 100%/100%；0 | 60 |
| mean/P50/P95 ms | 6442.539/6391.060/10399.058 | 6605.533/6636.552/10699.544 | 同场 |
| input/output token均值 | 995.8/94.9 | 1038.467/97.183 | 同场 |

类别只修正餐饮门面两例，保留全部风格/设施/价位；两次复查的其他主体类别没有扩展修改。
总语义错误36→35/60，已知类别7→5/39，风格25→24/60，设施22/60不变；多主体3/4、
多风格7/8、食品1/17、unknown猜测4/60仍待改善，价位无证据猜测0/60。切片SHA
`bc442f5f7e8821c96cfe6a05290d2c991a5e1ce584e255ebaf10a667478ef58d`。
相较失败v11的原development，综合再增0.017094且字段不退；独立revision复算通过，
未读取旧final逐图结果。平均延迟比v9增2.53%，不是速度优化。

登记 `qwen3_vl_system_week8_v12`，release SHA
`d8a8874c46135466e8d6979377e4b47533186c350a75d08a64e31c31796b3fd0`，observation canonical
`28b87c3bbb1c558576f30832350711490e8c0d9b5840002774da960b1a7757fd`。准确CLI配置验证通过，
没有新训练或覆盖adapter。runtime v7保留原业务/缓存复测并加入两张已修复门面的真实
服务探针，要求正确类别且确实执行主体复查。修复探针把最后一次主体输出误当风格输出
的问题；完整raw先重放，再识别阶段。定向41/41、完整910/910（26.404秒，v61）通过，
全量日志SHA `25f9eccdef9daaf0ddcce2d39d1616c71a67f7a736bd9f50ab4d2d66004e67e0`。
原九项审查缺陷相关64条回归再次通过。真实runtime/retrieval及新final尚未执行。

CPU29709694在7733494完成归档身份池（1:08/0:0）：200100条照片元数据，排除历史
source/group184771、范围外1074；5666个可选组中按种子取4000，接受1039、拒绝2961。
池锁 `a547219957ab76b5ff7e835caaa8af3f00dee1543f9ca89802c21882e6075fce`。所有数据仍无
参考标签；`visual_final_v6` 配置固定100张新测试，未消费。纠错GPU29709806正在原34
切片上双组验证，不与其他GPU任务争用。全部新标签继续只允许model_generated_silver。

### 16.20 v12纠错、数据与发布预检

29709806在7733494完成（11:16/0:0），旧/新纠错各34/34，逐错误切片语义完全一致：
category错误3，style TP/FP/FN=3/3/8，facility=7/2/3。均时4190.704/14950.089ms，
输入34735/34735、输出1927/1697；新纠错仍慢，不宣称提速。原始重放及非回退检查通过，
receipt SHA `acb13afa05dcda30d4a887ea6c3cbe768565e173c5ee9931a3ae6b7b3e930fe1`，
semantics SHA `ec318bd3e1ccc50d0d4a20e8a833168f8245dda55a4e232332fe98d878419167`。

29710019在938addf完成（12秒/0:0）：同池1039身份中固定选100，无标签；历史五维重叠
全0，数据锁 `6222e7e7e388b6ed911a63bc6ac9a37aaf28f1b921eed4f030f78dd5d1e7b6fd`。
池拒绝2961=历史哈希2952+重复1+不可读8。59份来源/历史依赖回传后哈希一致，本地完整
重放纠错、来源池和final身份通过。一次SCP连接中断后仅补齐缺失文件，已有文件未覆盖。
该身份锁不是最终模型已通过，也未生成或读取新final标签。

相同v12的真实生产检索10查询/4对话通过，summary SHA
`5dd3c9ab172a2c870342a97eccfc2df9d630a44a89d7afc765258f17ab2864d2`；不宣称视觉NDCG
提升。v12独立运行层导入、14路由注册、准确CLI/Compose静态检查及旧v9四层复验通过。
29710020正在原项目目录运行真实业务与24次性能基准，额外检查两张餐饮门面的主体复查。

带v12环境变量跑全量时，v62有1条历史正式Prompt测试失败：fixture误把环境选择的候选
当作正式模型。修正为显式指定正式配置，未改服务解析；同样v12环境复跑v63为910/910
（30.995秒）。保留v62失败日志，不把它计作通过；此前v61默认环境910/910仍为历史结果。
v63日志SHA `e4a4a900f44f4c3e46a7d405aa4b0ce694d55a2473551b89353a4be1f6d06af1`。

### 16.21 v12真实业务通过，锁定单次final v6

29710020在938addf完成（9:20/0:0）。准确v12的商品/售后/行程smoke首轮通过；上海2天、
北京3天、杭州5天三条显式业务约束通过。商品对话和两图比较实际调用模型并完成，不是
只返回确认语；smoke行程对话仍需2次生成后完成，不能宣称消除了全部纠错。三个商品
探针均通过：一条风格范围弃权、两条餐饮门面主体复查，完整raw重放确认实际阶段和结果。
35份runtime产物及24条缓存raw已独立复算，summary SHA
`ce879ced879f691e549bbbd6dda5674de6549befc12be1faa6af70f9aa44335e`。

| 相同固定输入，各8次 | uncached | processor_cached | prepared_cached |
|---|---:|---:|---:|
| mean ms | 9233.776 | 9225.335 | 9224.645 |
| P50 ms | 9233.139 | 9224.719 | 9222.583 |
| P95 ms | 9239.038 | 9230.762 | 9240.296 |
| 生成token/s | 15.2700 | 15.2840 | 15.2851 |
| 输入/输出token总量 | 8240/1128 | 8240/1128 | 8240/1128 |
| 失败；标签是否相同 | 0；是 | 0；是 | 0；是 |

硬件A100 80GB MIG 1g.20gb；冷启动22133.187ms，峰值allocated/reserved为
8143745536/8434745344 bytes。约0.1%的耗时差异不足以宣称实质加速，保留已锁定的
processor缓存8、prepared缓存0，不以改变质量换速度。runtime原始复算receipt SHA
`d747797c353e55c5aa435bfe505171f0bf5e3e91af754e330f33b4b09a9aa27f`。

在53dd1db锁定final v6：candidate lock
`df70790e5a16c56549fc439756f2c08cb81af51fed5dc81c23723e9cb1bc6903`，data lock仍为
`6222e7e7e388b6ed911a63bc6ac9a37aaf28f1b921eed4f030f78dd5d1e7b6fd`。
GPU29710151按历史三组28:15及余量申请37分钟，同场各100图正式/v9/v12仅运行一次。
运行中不改源码、Prompt、参考或验收条件；CPU29710265依赖GPU成功后一次评分和验收。

独立qwen3.7-plus教师已完成100/100、最终错误0，114次请求保留14次中间错误：食品/场所
矛盾8、风格范围3、transport3。原有4次/120秒有界重试内完成，未重启或补换样本。
教师只收图片，没有metadata或候选输出；raw重放、重复键/范围检查和五维身份均通过。
raw SHA `ec05cb91605ae14a70c9efec3272aa813fd65aa8801baf7eb82c7dbd00e11ee2`。
这100条仍是model_generated_silver、权重0.5、human=0；教师完成不等于候选质量通过。

### 16.22 v12单次final v6质量验收通过

GPU29710151在53dd1db完成（27:12/0:0），正式/v9/v12各100条，仅运行一次；
CPU29710265（7秒/0:0）完成一次评分和独立验收。正式基线严格改善与v9非回退均PASS。
完整原始输出、来源隔离、配置/源码锁和验收已在本地再次重放，所有结果一致，没有新请求。

| final v6，同100图silver | 同场正式ck87 | 同场v9 | 锁定v12 | 相同支持 |
|---|---:|---:|---:|---|
| business_category_accuracy | 0.703704 | 0.740741 | 0.740741 | 38/54→40/54→40/54 |
| category含unknown | 0.380000 | 0.830000 | 0.830000 | 100 |
| style P/R/F1 | 0.247525/0.287356/0.265957 | 0.824561/0.540230/0.652778 | 同v9 | 49图/87标签 |
| facility P/R/F1 | 0.156863/0.156863/0.156863 | 0.823529/0.686275/0.748663 | 同v9 | 46图/102标签 |
| price_range_accuracy | N/A | N/A | N/A | 0 |
| unknown accuracy | 0.380000 | 0.897500 | 0.897500 | 400决策 |
| label_completeness | 0.319356 | 0.629598 | 0.629598 | 同一有支持参考，保留全部100图 |
| composite | 0.375508 | 0.714061 | 0.714061 | 同一参考 |
| JSON/Schema；失败 | 100%/100%；0 | 100%/100%；0 | 100%/100%；0 | 各100请求 |
| 内部语义一致性 | 0.020000 | 1.000000 | 1.000000 | 同一自动规则 |
| mean/P50/P95 ms | 4950.239/4904.136/5993.955 | 5518.823/3644.320/10472.603 | 5583.754/3645.222/10469.610 | 同模型/硬件/图片 |
| input/output token均值 | 676.22/60.67 | 989.22/79.31 | 1004.52/80.18 | 含所有纠错/复查 |
| input/output token总量 | 67622/6067 | 98922/7931 | 100452/8018 | 各100条 |

最终v12相对v9**没有综合或单字段指标增益**，平均耗时增加约1.18%；不宣称进一步泛化
提升或加速。最终有2次主体复查，97/100完整商品JSON逐项相同，其余差异没有改变上述
汇总指标。新方案资格来自原development真实改善与此次独立非回退，不是重新排列final
方案，也不是因为多次换测试集挑中高分；失败v10/v11与广泛主体复查均完整保留。
这些小规模顺序实验没有统计显著性或人工视觉准确率结论。

最终选择保持已锁定的`product_observation_subject_review_v2`：短视觉事实、定点风格弃权、
仅食品矛盾的纠错约束、仅可见功能冲突触发的主体复查。商品使用Qwen3-VL-8B底座，
不启用ck87；售后仍用原ck87，未进行新SFT、未改LoRA或覆盖adapter。price正支持0，
只能报告不无依据猜价，不能宣称价位识别已解决。冷启动21895.956ms，峰值allocated
8143745536 bytes；这些为本次测试实测，不把历史37.63秒smoke当稳定性能基线。

关键身份：

- final comparison SHA `634d4ed9f3bccc40fe8772e5b88b4a0b672bc04afdb37cc30d53e72782c732af`；
  acceptance SHA `eb58892b9b5ba468bec77b0a0ceff75c0d628c0585ae2c306a113bd8c9eb860c`。
- 正式/v9/v12 raw SHA分别为
  `19f00c6f6e78ddde8948321a545bf6c580d0df81931c6a1baed672235c93ba7c`、
  `88d797966af3a3305dc419fca0ff935a162ded3262a37abc03ccb2db879e280f`、
  `09e9eb591a7e5fcce83608c59d524323de906152da10fbee44d3f8af7342a1e4`。
- 本地完整重放receipt SHA `1010bd601bb18e0bed662433632e8d5d1f8e9351eeb7c98d23b245d01b864ca5`。

### 16.23 跨平台测试环境复审

额外CPU全量29710278首次失败（29秒/1:0）：886条中5个subtest失败、1个模块导入错误。
数据环境缺少TestClient依赖，另因`data/eval`为冻结历史数据的符号链接，Git拒绝沿链接
执行check-ignore。保留原日志SHA
`a6d9159785fe3d0374eff4b0ce43e06ace9af74f2eaec1ae887dbac2e7f4c46e`；没有移动或改写挂载。

按本地已验证版本补充httpx0.28.1/httpcore1.0.9，不改GPU环境或FastAPI版本；新增
`requirements-test.txt`明确测试依赖，不加入生产运行层。测试改在临时Git目录校验相同的
版本化`.gitignore`，并反查源码不被忽略；实际工作树未跟踪数据的另一检查仍保留。
此修正发生在final及验收完成后，没有改动其锁定的模型、数据、评分或源码。
Spartan原缺陷定向29710335为64/64（0.226秒），本地完整v64为910/910（35.084秒），
v64日志SHA `36de771c358627733a767a0e52f7c3aa3178d9ee735400f84cc23f77577e845e`。
Spartan在36a8348完整复跑29710448通过（作业33秒/0:0，910/910测试17.540秒），随后
完整final原始/身份/业务验收再次PASS。日志SHA
`fdfac8b5d55025b21e13db6e7aa435a120c39949a5ff4faab54fc0bbfa4f2060`；新增验收receipt位于
`outputs/week8/review/week8_v12_spartan_full_suite_acceptance_20260829_v1.json`，没有覆盖
原final验收记录。两端原始数据/adapter哈希一致，旧v9完整交接包再次复验PASS。

### 16.24 v12交接完成与当前边界

交接目录为`outputs/releases/trip-qwen3-vl-8b-week8-visual-silver-v12-rc1`，本地验证及
Spartan CPU29710473（36a8348，4秒/0:0）均PASS：四层哈希、锁、100张图片、原始
参考/推理、完整样本/失败披露、纠错切片及新无标签来源池快照均通过。模型协议源码锁
仍是53dd1db，后续只改测试fixture/依赖和交付说明，没有改模型、参考或最终评分实现。
manifest SHA `1a9868e6d4a28e1dfcfa619839f3526a0a5277146550aa49202bf04418325ab0`。

| 层 | bytes / 文件数 | SHA-256 |
|---|---:|---|
| runtime | 125880 / 143 | `5c2f842472f2180946503d0a330fa4ccd3f514e1f70adc40e8231b4ca90c957b` |
| adapter | 57850259 / 4 | `f74c078738fa0229574114986c58040bbc280e11ba4ec06558c9a488c2de619d` |
| retrieval | 1951172 / 3 | `3cdb98f4d50bc72ae53c4e7e96d823ea5b08af93f41df5d14ff1118d12d1a15b` |
| evidence | 7147110 / 213 | `82c54c56bd8dc8aafe788c87706222116983e84f76fa9bb94ab5d2091d90df6e` |

adapter/retrieval与冻结v9完全相同。运行层和实际所选配置在脱离工作树的临时环境导入，
两端10条OpenAPI路径相同且必需业务路径完整。本地FastAPI0.136.1/Starlette1.0.1与
Spartan0.141.1/1.6.0的顶层路由对象计数为14/6；远端有两个`_IncludedRouter`，这是
框架对象表示差异，不等于业务接口缺失。没有声称两份验证JSON逐字相同；四层内容和
实际路径集合一致。跨主机receipt SHA
`e009474a5da090022cdc0159f7e6584545ceea5818cdc6380723bae8edd9fb22`。

`configs/week8/candidate_handoff_v12.json`登记37个证据输入及来源版本。按该清单在本地新的
临时目录实际重建，manifest及四层逐字节一致，重建包独立交接检查也PASS；原包不改。
recipe SHA `502cdfc73006dc120229454ea90fbb35d1da5e8d4402cafd7c6c9b9b4abdaf25`，重建
receipt SHA `a9a3d1af4cc1445cf65d1837a899b7bd1a90e3141d46b8c0c641d0edd529a54c`。
该字节重现结论针对相同字节源；跨主机验证的是已分发的同一包，不声称任意换行/权限
环境重新打包也天然逐字节相同。

现有包可直接只读复验，不启动GPU或消费final：

```bash
python -c "from pathlib import Path; from scripts.verify_week8_candidate_handoff import verify; print(verify(Path('outputs/releases/trip-qwen3-vl-8b-week8-visual-silver-v12-rc1'), 'evidence/week8_visual_holdout_20260829_v6/promotion_acceptance.json'))"
python scripts/tripctl.py --release-config configs/releases/qwen3_vl_system_week8_v12.json validate
```

重建时从recipe读取`adapter_dir`、`retrieval_dir`、`release_config`及`evidence_paths`，
交给已有`build_release_bundle.build_bundle`，输出必须是新目录。版本化清单不是重新
运行teacher/inference的授权；final v6已永久消费，不能复跑或用于调参。

当前完成：商品标签/证据口径与原九项缺陷修复、原development实测改进、新100图单次
最终验收、真实商品/对话/约束行程与检索、重复性能基准、完整原始重放和可复现交接。
本地/Spartan全量910/910、原缺陷定向64/64分别通过；CLI/Compose静态检查通过，未
启动Compose服务。715个tracked/新增文件的密钥特征/敏感路径/>5MB扫描均0，工作树
及`dev...HEAD`的diff检查通过。主`dev`的34项既有改动保留，未reset/stash/暂存/提交。
交付仅在`feature/week8-product-understanding`，没有合并、打标签或改正式默认配置。

仍待优化的准确边界：

- 新final指标与v9持平；development的两例餐饮门面修复不能推出整体泛化能力再提高。
- 业态/多主体和风格、设施仍有漏识别/误识别；完整切片见16.19，不能以结构完整等同语义正确。
- 价位视觉正支持0，N/A；没有依据时保持unknown，不增加人工或用商家价格替代视觉证据。
- 缓存没有实质加速，v12 final均时比v9约慢1.18%；保留失败的降像素/压缩/纠错尝试，不能写成提速完成。
- 行程对话仍有一次模型纠错，检索未支持条件仍明确NOT_COMPLETED；检索闭环通过不等于证明新的视觉相关性提升。
- 所有新增参考仍为model_generated_silver，权重0.5，人工参与0；不声称人工准确率、统计显著性或无条件生产可用。

### 16.25 证据约束Prompt复审：设施改善但整体拒绝

`product_observation_evidence_guard_v1`只改商品主观察的`task_prompt`，在相同60图、
相同自动silver参考、相同底座且两组均显式关闭adapter的条件下完成双组实测。首次恢复
配置使用了不存在的Prompt资产，另一次新role沿用了隐式adapter路由；这些运行在模型调用
前失败或只形成未评分的部分输出，均冻结保留。修复后的`contract_ablation_v17_recovery_v4`
作业29725648完成两组各60/60，失败0；原始输出在本地重新评分，未读取final。

| 固定development | v12 incumbent | 证据约束Prompt | 支持 |
|---|---:|---:|---|
| business_category_accuracy | 0.871795 | 0.846154 | 39 |
| category含unknown | 0.883333 | 0.866667 | 60 |
| style P/R/F1 | 0.617021/0.659091/0.637363 | 0.629630/0.386364/0.478873 | 34图/44标签 |
| facility P/R/F1 | 0.818182/0.807692/0.812903 | 0.870130/0.858974/0.864516 | 37图/78标签 |
| price_range_accuracy | N/A | N/A | 0 |
| unknown accuracy | 0.925000 | 0.883333 | 240决策 |
| label_completeness | 0.746622 | 0.691971 | 60图 |
| composite | 0.774020 | 0.729848 | 3个有支持字段 |
| JSON/Schema；失败 | 100%/100%；0 | 100%/100%；0 | 各60请求 |
| mean/P50/P95 ms | 6653.730/6669.674/10754.921 | 6634.371/5183.954/11258.384 | 同场 |
| input/output token均值 | 1038.467/97.183 | 1225.650/89.750 | 含全部阶段 |

设施TP/FP/FN从63/14/15改善为67/10/11，但风格从29/18/15变为17/10/27；类别、
unknown、完整度和综合分也下降。目标无关的可观察证据诊断从20个矛盾标签/17图降至
14个/11图，但该诊断不产生正向晋级资格，只用于阻止明显证据回退。最终决策为
`KEEP_INCUMBENT_CANDIDATE`，comparison SHA
`9e570244633d73d8aaa47cc38a42d789c7f76ed49fe4b3c1675b6c045542b03e`；不锁定Prompt、
不改release、不运行final。此结果证明设施规则有可隔离价值，也证明不能用局部改善掩盖
风格召回和业态回退。

### 16.26 检索析取与行程首轮修复的真实闭环

生产检索现在把同字段析取保留为列表并传到Milvus `IN`过滤、metadata评分和缓存键：
“酒店或餐厅”“便宜或高档餐厅”可完整执行；跨字段“酒店或便宜餐厅”无法无歧义表达时
明确返回部分完成，不伪装为成功。恢复v1真实执行后暴露验收器错误地把标量与整个列表
比较，原FAIL保留；修复后的29725754/recovery v2在隔离Milvus Lite上11/11查询和5/5
对话路由通过，查询变化会改变结果，encoder实际调用3次，参考metadata未参与排序。
summary SHA `751bb3f0eecb157f63f049feb76d8fa8ddaf6d0d9afadba0b1b773b7af6fbd6e`。
这证明生产路由与条件过滤闭环，不证明新的图片视觉相关性提升。

行程Prompt v5只调整九个顶层键的输出顺序和逐键完整性要求；v13与v12 release除
`release_id`和行程Prompt外逐字段相同。GPU29725755在固定三条直接行程及一条对话行程
上完成真实复测：直接请求仍3/3首轮通过，累计input/output token为4733/2445，生成延迟
154629.553ms；v12对应4649/2444和153725.420ms，约增加0.59%延迟。对话行程从2次生成、
首轮0/1通过改为1次生成、首轮1/1通过；input/output token从3418/1064降至1579/515，
生成延迟67870.154→32767.911ms（-51.72%）。比较协议状态PASS，SHA
`235543be288159395f4644c13820a2fb94eff5480f62cf04782a7ad82c782d6c`。
因此v13是通过的行程运行时派生候选；商品完整交接仍由v12包承载，不把release-only
派生误写成重新完成了商品final验收。

### 16.27 设施独立复查取得development增益，但不重跑final

根据16.25的可复现切片，新增`visual_facility_review_v1`：先完整运行v12商品管线，
再由同一图片进行一次独立设施复查，只允许整体替换`facility_evidence`。该阶段看不到旧设施、
参考标签、样本身份或商家metadata；类别、风格和价位由代码深拷贝保留，重复键、否定事实、
推断句、食品特写与十事实上限继续失败关闭。原始重放从attempt序列重建主观察、风格与主体
阶段边界，不信任自报的阶段计数或最终公共字段。实现、组合配置与生成脚本均绑定LF规范化哈希。

最初按旧模板提交的29725877在57秒时主动取消：同场v17已实测120请求约需14分钟，
本次还增加设施阶段，15分钟上限会强制中断健康运行。部分目录改名保留且从未评分；随后以
相同`run_id`、配置、60图和代码提交申请22分钟，唯一恢复作业29725887在15:36/0:0完成，
两组各60/60。该恢复不是失败样本重跑或换数据，前一作业尚未形成任何计分结果。

| 固定development | v12 incumbent | 独立设施复查 | 支持 |
|---|---:|---:|---|
| business_category_accuracy | 0.871795 | 0.871795 | 34/39 |
| category含unknown | 0.883333 | 0.883333 | 60 |
| style P/R/F1 | 0.617021/0.659091/0.637363 | 相同 | 34图/44标签 |
| facility P/R/F1 | 0.818182/0.807692/0.812903 | 0.900000/0.807692/0.851351 | 37图/78标签 |
| price_range_accuracy | N/A | N/A | 0 |
| unknown accuracy | 0.925000 | 0.933333 | 240决策 |
| label_completeness | 0.746622 | 0.751827 | 60图 |
| composite | 0.774020 | 0.786836 | 3个有支持字段 |
| JSON/Schema；失败 | 100%/100%；0 | 100%/100%；0 | 各60请求 |
| mean/P50/P95 ms | 6651.645/6667.466/10752.924 | 8532.572/9258.038/14382.060 | 同场 |
| input/output token均值 | 1038.467/97.183 | 1444.683/122.667 | 含全部阶段 |

类别、风格、价位及其支持逐项相同；设施TP/FN保持63/15，FP从14降到7，故precision与F1
提高且没有靠删除困难样本或减少参考支持。总错误图35→34，目标无关证据矛盾20标签/17图
降到11标签/9图。选优器同时要求这些诊断不得增加，因此不是只凭silver综合分锁定。
代价是mean/P50/P95增加28.28%/38.85%/33.75%，input/output token均值增加
39.12%/26.22%；这是一项质量优先的development改进，不是延迟优化。

决策为`IMPROVED_DEVELOPMENT_CANDIDATE`，但`promotion_allowed=false`、release未改变、
human视觉准确率声明为false。comparison SHA
`d04b3772e0a27037e5ce55e231409bb5b90bd43783f853d417e3915ce5041a31`，decision SHA
`36eb033f5906f2ef301fde842dbbfff3a3aec02cc141b8717c59396a742c3ab2`。Week 8 final v6已按
锁定协议消费，不能为了该development结果再次读取或调参，因此不把设施阶段写入v12正式
候选或重新打包；当前完整可交接商品选择仍是v12的
`product_observation_subject_review_v2`与商品禁用adapter。没有执行continuation SFT。

本轮收口后的未完成边界：设施误报已下降，但仍有15个参考设施漏识别；业态、多主体和
风格错误仍存在。价位支持仍为0/N/A，只能保持unknown。商品缓存没有实质提速，设施复查
又明确增加延迟；若要求低延迟，不能在没有同口径质量复验的情况下启用它。新增参考与验收
仍全部为自动silver、human=0，不声称人工视觉准确率或统计显著性。

### 16.28 最终回归、跨主机重放与交付状态

本地最终完整回归为941/941（25.250秒），日志SHA
`4607e7186b29589fee993c8f5752f2cfe067589a01392ae62c7b63f5a7cc17d6`；原审查缺陷、
对话/检索/行程定向102/102通过。Spartan第一次命令未加载Python模块，解释器在测试前
失败；第二次误用缺FastAPI/pyarrow的GPU环境，仅加载755项并失败，两份日志均保留且
不计为通过。改用已有独立数据/测试环境后完整941/941（16.636秒），日志SHA
`a635367fdb23d5bafd7c0d04376993df6fc55cd8356b036ba3acb824ad728f7b`。

Spartan对v18原始输出重新评分的comparison/decision SHA与本地逐字节相同：
`d04b3772e0a27037e5ce55e231409bb5b90bd43783f853d417e3915ce5041a31` /
`36eb033f5906f2ef301fde842dbbfff3a3aec02cc141b8717c59396a742c3ab2`。
v12/v13 release SHA为`d8a8874c…b3fd0`/`74403534…80b1`，ck87 adapter SHA仍为
`c2fbb5c7…aa2a`，跨主机一致。v12四层候选包及隔离导入PASS；当前工作树直接调用final
raw replayer被“implementation unlocked”按设计拒绝，因为final锁定后源码已有版本化修改，
没有绕过该保护。五维身份、100图及原始证据由冻结v12包的独立交接验证完成，不重新生成。

`compileall`、v12/v13准确CLI、Compose静态配置、候选包验证与`git diff --check`通过。
737个tracked文件中大于5MB为0、私钥/常见令牌特征为0；5个`.env.example`均为版本化
占位配置，不是密钥。主`dev`仍为原34项未提交改动，未暂存、覆盖、格式化或提交。
最终只推送`feature/week8-product-understanding`；未合并dev/stg/main、未打标签、未替换
正式默认release。v12是完整自动silver商品候选，v13是行程派生候选，v18是尚未通过新final
的development设施改进，三者身份和完成层级不得混写。

### 16.29 质量、成本与证据等级的最终权衡

最优选择不是把development分数最高的阶段全部串联。默认商品链路首先要求已完成锁定
development、单次final、原始重放和四层交接；在满足这些条件的方案中保留v12的
`product_observation_subject_review_v2`，商品禁用adapter。v18虽有更高设施precision，
但只有development证据且mean延迟增加28.28%，不能替代该身份。

为判断能否低成本取得v18收益，新增目标无关反事实复算。脚本从候选attempt的最短严格
可重放前缀重建v12主观察，路由只读取该观察的subject、facility fact及确定性证据矛盾；
sample id、metadata和参考目标不进入路由，silver参考只在决策完成后评分。结果文件为
`outputs/week8/review/week8_facility_routing_tradeoff_20260829_v1.json`，SHA
`e33d487ef41febc47c66f3514a51faf9364b27b8eef0c7613a1ae0e0c6d653cc`。

| development反事实 | 触发 | facility TP/FP/FN | facility P/R/F1 | composite | mean比不复查 | input/output均值比 |
|---|---:|---:|---:|---:|---:|---:|
| 不复查 | 0/60 | 63/14/15 | 0.818182/0.807692/0.812903 | 0.774020 | 1.0000 | 1.0000/1.0000 |
| 仅证据冲突 | 15/60 | 60/10/18 | 0.857143/0.769231/0.810811 | 0.773323 | 1.0992 | 1.1415/1.0900 |
| 冲突或酒店/餐饮/零售/工业场景设施为空 | 17/60 | 62/10/16 | 0.861111/0.794872/0.826667 | 0.778608 | 1.1089 | 1.1616/1.0979 |
| 全部合格场景复查 | 41/60 | 63/7/15 | 0.900000/0.807692/0.851351 | 0.786836 | 1.2855 | 1.3912/1.2622 |

最自然的“冲突才复查”会损失3个TP并使F1和综合回退，明确拒绝。17条可观察不确定性
路由形成较温和的development帕累托点，但仍增加10.89% mean延迟，而且只是复用既有
输出的反事实，不是新真实路由运行或新final；不能把它写成已晋级优化。不存在可用参考
标签或样本身份路由的例外。

整体运行选择v13：它与v12除`release_id`和行程Prompt外逐字段相同，因此商品质量、
adapter与数据身份继续由v12验收包承担；真实固定复测中直接行程3/3首轮通过不退，对话
行程从2次生成降为1次并把延迟减少51.72%。因此后续联调以v13为默认候选配置，完整商品
交接仍明确指向v12；不重新打包成伪造的v13商品final，不替换正式release。

最终选择为：**v13运行配置 + v12商品验收身份 + 不启用v18设施复查**。这是当前证据下
质量、延迟、token、失败风险和可交付性的最优组合，不声称价位已解决、silver等于人工
视觉准确率或商品延迟已经优化完成。

新增路由定向4/4及完整`python -m unittest discover -s tests -v` 945/945通过（51.616秒）；
分析器compileall及跨平台源码哈希绑定PASS，v13准确CLI与运行时比较PASS，选择结果仍为v13；
v12四层包隔离导入和验收再次PASS。该复算未启动GPU、未读取或重跑final、未生成标签。
