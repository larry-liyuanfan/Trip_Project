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
