# Week 8 商品理解与并行优化报告

## 1. 结论

Week 8 在临时分支 `feature/week8-product-understanding` 完成了导师既有四个方向的并行执行。
商品主任务在新的、未消费的 `silver` development/test 上锁定
`week8_product_field_check_v1`，没有执行 continuation SFT。单次最终 test 的商品综合分从
同口径正式模型的 `0.804239` 提高到 `0.861085`，差值 `+0.056846`；JSON/Schema
保持 `100%/100%`，请求失败率保持 `0`，字段支持没有因 Prompt 比较发生变化。

对话首轮路由减少了格式错误和纠错依赖，但固定 4 条真实模型样本仍有 1 条失败；纯延迟
优化只有约 `4.51 ms` 的平均收益，不构成实质性能改善；检索 metadata rerank 在一次性
独立 final test 上显著提高 NDCG/Recall，但该结果是离线精确向量基准，不是 Milvus 网络
服务延迟。上述未完全解决项均保留真实结果，没有用状态更新替代实际执行。

## 2. 范围、历史基线与资产审计

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
