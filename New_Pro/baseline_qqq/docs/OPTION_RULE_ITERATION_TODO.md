# Option Rule Iteration TODO

目标: 在不依赖 TFT 的前提下, 先证明期权规则层本身可控。每个实验必须回答一个明确问题, 输出可复跑报告, 并给出保留/修改/废弃结论。

当前原则:
- 先验证规则和执行, 后接模型信号。
- 不把单日盈利当作规则有效; 必须覆盖趋势日、假突破日、震荡日、反转日。
- OTM runner 用来保留尾部收益; debit spread 用来降低错误方向亏损, 不能混用目标。
- 所有开盘交易必须使用真实秒级 quote gate, 不再用 1m high/low 近似点差。

## 0. 数据与回放基线

- [x] 接入 AAPL 9DTE 秒级 quote parquet 样本。
  - 证据: `/Users/fangshuai/Downloads/AAPL_2026-03-18.parquet`
  - 发现: 09:30:00-09:30:02 quote 很乱, 09:30:03 后部分合约点差明显收敛。

- [x] 建立 quote-only 规则验证脚本。
  - 脚本: `New_Pro/baseline_qqq/tools/quote_rule_validation.py`
  - 报告: `New_Pro/baseline_qqq/reports/aapl_2026-03-18_quote_rule_validation.json`
  - 限制: 当前没有正股价格, 只能验证期权 quote/结构/退出, 不能完整验证 V0 突破信号。

- [ ] 批量接入多天 AAPL 9DTE quote 文件。
  - 目的: 避免单日过拟合。
  - 输入: 至少 20 个交易日, 优先覆盖开盘冲高回落、单边趋势、震荡、尾盘反转。
  - 通过标准: 脚本可一次生成 per-day + aggregate 报告。
  - 下一步动作: 给 `quote_rule_validation.py` 增加 `--glob` / `--batch-out`。

- [ ] 补齐正股 1s/1m 数据对齐。
  - 目的: 把 quote-only 动量替换/补充为正股突破、VWAP、OR、回撤等规则。
  - 通过标准: 每个期权 timestamp 能 join 到同秒或前向填充的 AAPL stock quote/bar。
  - 下一步动作: 明确服务器上 AAPL stock 秒级数据路径或从现有缓存补齐。

## 1. 开盘执行规则

- [x] 验证 09:30 不必机械等到 09:45。
  - 发现: 2026-03-18 样本里, 9DTE call 在 09:30:03 后已有可交易窗口。
  - 当前建议: 09:30:00-09:30:02 只观察, 09:30:03 后允许 quote gate 通过的交易。

- [ ] 校准 quote gate。
  - 候选参数:
    - `min_seconds_after_open`: 3, 5, 10
    - `max_spread_pct`: 0.04, 0.06, 0.08
    - `min_bid_ask_size`: 2, 5, 10
    - `stable_quote_seconds`: 2, 3, 5
  - 通过标准: 假突破日少吃坏盘口, 趋势日不明显错过开盘主升段。
  - 输出: 各参数组合的入场延迟、成交成本、错失收益、坏成交率。

- [ ] 建立 fill sensitivity。
  - 候选参数: `entry_frac/exit_frac = 0.65/0.65, 0.775/0.775, 0.90/0.90`
  - 目的: 防止规则只在乐观成交假设下有效。
  - 通过标准: 保守 fill 下仍不过度依赖单笔大收益。

## 2. Playbook 分层验证

- [ ] Baseline: Directional Tight Single。
  - 目的: 作为 V0 单腿方向策略的对照组。
  - 验证重点: 假突破最大亏损、连续亏损、trailing 是否过早/过晚。
  - 暂定结构: `LONG_CALL_TIGHT` / `LONG_PUT_TIGHT`

- [ ] OpenDrive OTM Runner。
  - 目的: 捕捉 5x/10x/几十倍期权行情。
  - 核心约束: 小仓位预算, 不用 debit spread 封顶, 但必须有早期利润保护。
  - 待测参数:
    - 入场: OTM 5s/20s 动量、正股 OR/VWAP 确认
    - 风控: `risk_frac`, `hard_stop`, `trail_trigger`, `trail_keep`
    - 出场: 是否分批止盈, 是否留 1 张 lottery runner
  - 通过标准: 趋势日能保留凸性; 假突破日单笔亏损受预算约束。

- [ ] Debit Spread Defense。
  - 目的: 方向有优势但假突破风险高时降低亏损。
  - 核心约束: 不用于捕捉几十倍行情。
  - 待测问题:
    - 何时从 runner/single 降级为 debit spread?
    - spread 宽度用相邻 strike 还是 delta/权利金比例?
    - credit leg 的流动性会不会显著拖累退出?
  - 通过标准: 假突破日亏损明显低于单腿, 趋势日收益被封顶但仍可接受。

- [ ] Straddle / Strangle Volatility。
  - 目的: 方向不清楚但波动要爆发时使用。
  - 入场候选: call/put 两侧 premium 同时走强、正股开盘区间扩大、方向分歧。
  - 风险: theta 和双腿点差成本。
  - 通过标准: 只在方向不明样本提升收益/回撤, 不在普通趋势日稀释 runner。

## 3. 退出规则收敛

- [ ] 单腿紧退出网格。
  - 参数: hard stop, soft stop, time stop, trailing trigger/keep, max hold。
  - 输出: 每类市场状态下的退出原因分布。
  - 失败信号: 大多数盈利来自过早 trailing 或单笔偶然大涨。

- [ ] Runner 退出规则。
  - 目标: 普通趋势保住 20%-50% 收益, 极端趋势允许继续扩张。
  - 待测设计:
    - 30%-50% 浮盈后保护本金/部分利润
    - 达到 100% 后提高 trailing keep
    - 可选: 分批卖出, 留尾仓
  - 通过标准: 不再从 +50% 回吐到接近 0; 极端日仍有尾部暴露。

- [ ] 假突破专用杀跌规则。
  - 目的: 防止方向错误时单腿扩大亏损。
  - 输入: 正股跌回 OR/VWAP、期权 mid 连续走弱、spread 扩大。
  - 通过标准: 假突破日最大单笔亏损和日亏损显著下降。

## 4. Router 决策规则

- [ ] 定义市场状态标签。
  - 候选状态:
    - `clean_open_drive`
    - `weak_breakout`
    - `ambiguous_volatility`
    - `mean_reversion_chop`
    - `late_day_trend`
  - 目的: 让 router 选择 playbook, 而不是直接预测涨跌。

- [ ] Router 第一版规则。
  - `clean_open_drive` -> 小仓位 OTM runner
  - `weak_breakout` -> debit spread 或不交易
  - `ambiguous_volatility` -> strangle/straddle
  - `mean_reversion_chop` -> 不交易
  - `late_day_trend` -> 更严格 quote gate + 更短持有

- [ ] Router 失败归因表。
  - 每笔亏损必须归因:
    - bad signal
    - bad structure
    - bad entry quote
    - bad exit
    - unavoidable chop
  - 通过标准: 后续改动能对应具体失败原因, 而不是盲目调参数。

## 5. 批量报告与验收标准

- [ ] 生成日级报告。
  - 字段: trades, pnl, win rate, profit factor, max drawdown, worst trade, best trade, structures, exit reasons。

- [ ] 生成交易明细。
  - 字段: entry/exit ts, structure, contracts, debit, exit value, gross/net ROI, max ROI, reason。

- [ ] 生成状态切片报告。
  - 按开盘类型、趋势/震荡、价差质量、入场时间段切分。

- [ ] 设定阶段性通过标准。
  - 单日不作为通过标准。
  - 初版目标:
    - worst trade 明显受控
    - 假突破日亏损低于单腿 baseline
    - 趋势日 runner 不被 debit spread 错杀
    - fill sensitivity 下结论不反转

## 6. 接入 TFT 前置条件

- [ ] 规则层通过批量验证后, TFT 只作为 playbook selector 的一个输入。
  - 不是直接输出买 call/put。
  - 输出形式建议:
    - direction confidence
    - trend persistence confidence
    - volatility expansion confidence
    - no-trade confidence

- [ ] TFT 接入前必须保留 rule-only baseline。
  - 目的: 防止模型提升只是来自更多交易或更高风险。
  - 通过标准: TFT hybrid 在同等风险预算下优于 rule-only。

## 当前最近下一步

1. 给 `quote_rule_validation.py` 增加批量模式。
2. 从服务器/本地挑选至少 20 天 AAPL 9DTE quote。
3. 先跑四个 playbook 的独立结果, 不急着合并 router。
4. 对亏损交易做失败归因, 再决定下一轮参数。
