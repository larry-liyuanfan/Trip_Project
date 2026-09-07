# v10 合成卡片渲染修复

2026-09-07：审计发现 v9 旧渲染器按 scale=3 每行最多显示 20 字符，超出后静默截断。
训练商品卡 384 张中 68 张被截断；开发商品卡 84 张中 20 张被截断，涉及 29 行。
旧字体还缺少下划线，MID_RANGE 的下划线会被替换为问号。部分 v9 development 预测
包含 mod、coa 等截断词，因此原 style 回退不能全部归因于训练遗忘；完整归因待配对诊断。

新渲染器按行选择 scale=2/3，保留完整文本，补齐下划线字形；超过最低可读尺寸、
不支持的字符或超过 7 行均抛错。像素级测试核验末尾字形与下划线，并覆盖越界和覆盖写入。
历史 v4/v7/v9 生成器及结果保持原字节；修复仅用于新版本。

配对诊断入口：`scripts/run_card_render_diagnostic_v10.py`。
固定 v7/v9 adapter、基座 revision、Prompt 和 generation，分别跑原始/修复渲染。
四个单元各 84 条商品样本，共 336 条；样本来自已查看过的 v9 development，
仅用于诊断渲染影响，不是独立测试、不进行训练，也不能支持模型发布。
在推理前锁定 manifest canonical SHA：

- original：`df69b729a674b86fa2aaa5a09781805a3a2622cfb262d6cec5884961865d569c`
- repaired：`b51764b57ee4a226719f76f2a3a42f396bdc1304792fd927f719159646288419`

预先定义诊断成功条件：v9 style F1 提升，其他商品字段 F1 回退不超过 .05。
该条件不替代原模型质量门；无论结果如何均保留原 v9 负实验。
Fresh Test 未读取或重跑。新训练需基于输入质量修复后的独立开发周期再验证。

正式 Git 外交付包已定位到主 checkout 的 outputs/releases 目录。
从独立 worktree 执行 verify_final_delivery.py 对该目录的只读复核为 PASS：四层 SHA、
adapter 身份和 runtime 隔离导入均通过；948 为包内历史单元测试记录。

本轮完整回归运行 983 项，OK（2 skipped）；tripctl validate、Compose 配置和 diff check
通过。实现提交 `cebb5745bfac8377f2b1399497809b288a737b7d`；代码归档 SHA
`e4e7be092f6d91a9a016ca15579919ce905c487d3409dbab05d8170f1561bd4a` 在 Spartan 解压前核对。
作业 `30211170` 已提交至 Iris yzhang3504，单 L40S、8 CPU、64 GiB、最多 30 分钟；
提交后曾为 PENDING/Priority；现已 COMPLETED 0:0，耗时 7 分 19 秒。
机器记录见 `experiments/card_render_repair_v10.json`。

## 配对诊断实测

作业位于 spartan-gpgpu007。独立复核四组各 84 条、共 336 条，输入图 SHA、manifest、
adapter、相同 Prompt/generation、配对 gold，以及原始结果重算全部通过。

| Adapter/渲染 | 业态 F1 | 风格 F1 | 设施 F1 | 价位 F1 |
| --- | ---: | ---: | ---: | ---: |
| v7 原图 | .9895 | 1.0 | .9074 | .8936 |
| v7 修复图 | .9895 | .9831 | 1.0 | .9231 |
| v9 原图 | 1.0 | .8571 | .9815 | 1.0 |
| v9 修复图 | 1.0 | 1.0 | 1.0 | .9412 |

字段参考分母依次为 48/36/36/24；风格为多标签，v9 修复后 TP=60、FP=FN=0。
v9 价位 TP=24、FP=3、FN=0，P=.8889、R=1.0。三个错误均来自冲突价位卡
v9_dev_conflict_000/004/008：参考应为 unknown，模型输出 mid_range。
unsupported hallucination 从 0/192 增至 3/192；四组首轮 JSON 均 84/84，纠错均 0/84。

因此渲染修复在本次配对诊断中消除了 v9 的风格错误，并改善设施识别；但价位 F1 回退
.05882 超过预定 .05，整体诊断门 FAIL。该结果不支持全部问题已解决。
它没有新增训练，使用已查看的开发集，不能称独立模型泛化提升；原 v9 负实验仍保留。
剩余技术事项为：在完整可读的新训练/开发数据中修复冲突价位弃权，并检查风格保留；
此外，现有服务性能仍缺更广的输入/负载覆盖。正式 release、Fresh Test 与主简历不变。

summary SHA 为 `cec2ab6d60d38410f75fca6869dfc58f1f6819a0b6ca639804dc3f02caa63741`；
预注册文件 SHA 为 `0ec0356045fd11d45214ca04d83730abc373c840f504d3b5f60ab30c103a2fd6`。
