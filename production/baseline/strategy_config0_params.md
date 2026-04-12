# StrategyConfig0 参数说明

本文记录 `strategy_config0.py` 中 V0 策略参数的作用、当前取值含义，以及调参方向。

V0 对应 `strategy_core_v0.py`，偏向“短周期动量狙击 + 微利快速保护”。它通常更灵敏，目标不是持有大波段，而是在期权价格短促、有利的小波段里反复收割。

## 启用方式

策略选择由 `strategy_selector.py` 统一控制。

```bash
STRATEGY_CORE_VERSION=V0 python3.10 s4_run_historical_replay_s2_1s.py --date 20260302
```

如果不显式设置环境变量，则以 `config.py` 中的 `STRATEGY_CORE_VERSION` 为准。

## 资金与仓位

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `INITIAL_ACCOUNT` | `50000.0` | 回测初始资金。只影响回测权益口径，不代表实盘必须使用该规模。 |
| `MAX_POSITIONS` | `4` | 最大同时持仓数量。V0 当前最多 4 笔，避免信号过多时过度分散或过度暴露。 |
| `POSITION_RATIO` | `0.25` | 单笔目标仓位比例。配合 `MAX_POSITIONS=4`，理论满仓约 100%，但还会被全局敞口限制约束。 |
| `MAX_TRADE_CAP` | `150000.0` | 单笔交易名义资金上限。资金规模变大后，避免单笔订单过大冲击盘口。 |
| `GLOBAL_EXPOSURE_LIMIT` | `0.90` | 总敞口上限。防止所有持仓合计接近满仓。 |
| `COMMISSION_PER_CONTRACT` | `0.65` | 每张合约手续费。用于回测成本估算。 |

调参方向：提高 `MAX_POSITIONS` 或 `POSITION_RATIO` 会提高收益弹性，也会提高同向风险和滑点；实盘建议优先提高标的数量和流动性过滤，而不是直接放大单笔仓位。

## 交易时间

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `START_TIME` / `START_HOUR` / `START_MINUTE` | `09:45:00` | 最早允许开仓时间。V0 避开开盘前 15 分钟的高噪声。 |
| `NO_ENTRY_TIME` / `NO_ENTRY_HOUR` / `NO_ENTRY_MINUTE` | `15:30:00` | 禁止新开仓时间。 |
| `CLOSE_TIME` / `CLOSE_HOUR` / `CLOSE_MINUTE` | `15:40:00` | 强制清仓时间。 |

注意：如果 alpha 生成从 10:00 才开始，策略时间窗允许 09:45 并不等于实际能从 09:45 开始交易。需要先保证 FCS/replay warmup 能在 09:45 前准备好。

## 开仓阈值

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `VOL_MIN_Z` | `-1` | 波动率 z-score 下限。V0 比 V1 更宽松，允许低波环境下开仓。 |
| `VOL_MAX_Z` | `4.0` | 波动率 z-score 上限。过高波动会被过滤，避免追在极端波动尾部。 |
| `ALPHA_ENTRY_THRESHOLD` | `0.8` | 基础 alpha 入场阈值。越低交易越多，越高交易更少但质量更强。 |
| `ALPHA_ENTRY_STRICT` | `1.2` | 高置信 alpha 阈值。用于高置信逻辑和部分动态风控判断。 |
| `MIN_CS_ALPHA_Z` | `0.5` | 横截面 alpha 过滤阈值。当前 V0 core 中主要保留为兼容参数。 |

V0 的 `_calculate_dynamic_alpha_threshold` 会在 `vol_z > 2.0` 时提高 alpha 门槛：`dynamic_threshold = base + (vol_z - 2.0) * 0.5`，最高限制为 `3.0`。这能避免在高波动环境里用过低 alpha 追单。

## 动量与趋势过滤

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `STOCK_MOMENTUM_TOLERANCE` | `0.001` | 正股 5min 趋势容忍度。做多时不允许正股明显下跌，做空时不允许正股明显上涨。 |
| `MIN_LAST_SNAP_ROC` | `0.0001` | 1min 瞬时动量门槛。做多要求短线仍向上，做空要求短线仍向下。 |
| `MAX_SNAP_ROC_LIMIT` | `0.01` | 尖刺过滤。1min 瞬时涨跌过大时不追，避免冲高/杀跌后被反抽。 |
| `MIN_TREND_ROC` | `0.0001` | 趋势 ROC 兼容参数。 |
| `MAX_TREND_ROC` | `0.0030` | 趋势 ROC 上限兼容参数。 |

调参方向：如果交易太少，可以降低 `MIN_LAST_SNAP_ROC`；如果经常追在尖峰末端，可以降低 `MAX_SNAP_ROC_LIMIT`。

## Rolling Signal 与震荡过滤

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `ROLLING_WINDOW_MINS` | `30` | rolling 统计窗口。用于滚动状态或兼容逻辑。 |
| `CORR_THRESHOLD` | `-0.1` | 相关性阈值兼容参数。 |
| `REGIME_GUARD_ENABLED` | `True` | regime 总开关。 |
| `REGIME_ENTRY_GUARD_ENABLED` | `True` | 是否用优化后的 VIXY regime 拦截开仓。 |
| `REGIME_ADAPTIVE_STOCK_STOP_ENABLED` | `False` | 是否根据市场 regime 自动切换正股硬止损档位；当前关闭，保持统一 `0.003 / 0.005`。 |
| `REGIME_REVERSAL_THRESHOLD` | `6` | 指定窗口内 VIXY 反转次数超过该值时标记 VIXY 波动候选。 |
| `REGIME_WINDOW_MINS` | `30` | 反转统计窗口。 |
| `REGIME_REVERSAL_PERCENT` | `0.001` | 判定一次反转的涨跌幅阈值。 |
| `REGIME_VIXY_ROC_THRESHOLD` | `0.003` | VIXY 5 分钟正向跳升超过该阈值时标记 VIXY 波动候选。 |
| `REGIME_REQUIRE_NEUTRAL_INDEX_FOR_ENTRY_GUARD` | `True` | 只有大盘方向不清楚时才允许 VIXY 波动候选触发入场拦截。 |

V0 当前把 hard stop 简化为固定 `0.003 / 0.005`，regime 只负责禁开仓：VIXY 反转/正向跳升先标记波动候选，只有大盘方向不清楚时才真正拦截。

## 风控与事件参数

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `STOCK_HARD_STOP_TIGHT` | `0.003` | 平静 regime 下的普通置信度正股止损。 |
| `STOCK_HARD_STOP_LOOSE` | `0.005` | 平静 regime 下的高置信度正股止损。 |
| `STOCK_HARD_STOP_TIGHT_VOLATILE` | `0.003` | 兼容字段；当前关闭动态止损切档，与默认紧止损一致。 |
| `STOCK_HARD_STOP_LOOSE_VOLATILE` | `0.005` | 兼容字段；当前关闭动态止损切档，与默认宽止损一致。 |
| `STOCK_HARD_STOP_EVENT` | `0.008` | 事件行情下正股止损兼容参数。 |
| `EVENT_PROB_THRESHOLD` | `0.7` | 事件概率阈值。 |
| `EVENT_HODL_MINS` | `30` | 事件单持有分钟数兼容参数。 |
| `COOLDOWN_MINUTES` | `60` | 平仓后冷却时间，防止同一标的过度重复进出。 |
| `CIRCUIT_BREAKER_THRESHOLD` | `3` | 熔断触发次数阈值。 |
| `CIRCUIT_BREAKER_MINUTES` | `30` | 熔断后暂停时间。 |
| `MIN_OPTION_PRICE` | `1.0` | 低价期权过滤线。避免价差占比过高、跳动过粗的合约。 |

## 流动性

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `MAX_SPREAD_PCT_ENTRY` | `0.1` | 开仓最大点差比例。当前约 10%。 |
| `MAX_SPREAD_PCT_EXIT` | `0.2` | 平仓最大点差比例。当前约 20%，比开仓更宽，避免风险退出被过度阻塞。 |
| `MAX_SPREAD_DIVERGENCE` | `0.02` | 点差异常散度过滤。 |

V0 当前不再使用额外的执行侧 Quote Guard。原因是秒级期权 quote 本身不连续，直接用 quote gap 或 bid 消失做额外折价/拦截会让回测过度悲观，甚至掩盖策略本身的真实表现。当前保留原有点差过滤逻辑，由策略和撮合模块共同决定是否成交与平仓。

## 止损与信号反转

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `STOP_LOSS` | `-0.10` | 条件止损启动线。亏损小于 10% 时一般不触发条件止损。 |
| `ABSOLUTE_STOP_LOSS` | `-0.15` | 绝对硬止损。亏损超过该线直接平仓。高 IV 下 core 会进一步收紧到约 `-0.12`。 |
| `TIME_STOP_MINS` | `30` | 时间止损窗口。 |
| `TIME_STOP_ROI` | `0.05` | 持仓超过 `TIME_STOP_MINS` 后，如果收益仍低于 5%，触发时间止损。 |
| `ALPHA_FLIP_THRESHOLD` | `0.8` | alpha 反向阈值。持仓 2 分钟后，如果 alpha 明显反向，则触发 `FLIP`。 |
| `HIGH_CONFIDENCE_THRESHOLD` | `1.2` | 高置信 alpha 阈值。影响正股硬止损宽松程度。 |

V0 的止损偏紧，适合当前“微利多次”的风格；如果希望持有更大波段，需要整体放宽 `STOP_LOSS`、`TIME_STOP_ROI`、`MACD_FADE_MIN_ROI` 和阶梯止盈，而不是只改一个值。

## Plan A 兼容参数

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `EARLY_STOP_MINS` | `5` | 早期止损窗口兼容参数。 |
| `EARLY_STOP_ROI` | `-0.05` | 早期止损收益阈值兼容参数。 |
| `NO_MOMENTUM_MINS` | `5` | 无动量观察窗口兼容参数。 |
| `NO_MOMENTUM_MIN_MAX_ROI` | `0.02` | 无动量最小 max ROI 兼容参数。 |

这些参数主要为了与新配置结构保持一致，V0 core 当前并不原生依赖完整 Plan A grid 逻辑。

## 执行参数

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `SLIPPAGE_PCT` | `0.001` | 回测滑点比例。 |
| `LIMIT_BUFFER_ENTRY` | `1.03` | 开仓限价 buffer。当前偏向 mid 上浮，而不是直接追 ask。 |
| `LIMIT_BUFFER_EXIT` | `0.97` | 平仓限价 buffer。当前偏向 mid 下浮，模拟更容易成交的退出。 |
| `ORDER_TIMEOUT_SECONDS` | `30` | 订单超时时间。 |
| `ORDER_MAX_RETRIES` | `3` | 最大撤单重试次数。 |

在 IBKR 实盘里，底层会向成交侧靠近；当前 mid + 固定比例 buffer 的价值是控制滑点，避免一开仓就因为直接用 ask 触发策略止损。

## 利润保护与阶梯止盈

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `LADDER_TIGHT` | 多档 | 默认利润阶梯。V0 当前固定使用 tight 阶梯。 |
| `LADDER_WIDE` | 多档 | 宽阶梯。只有启用动态阶梯且初始 alpha 足够高时才使用。 |
| `FLASH_PROTECT_TRIGGER` | `0.05` | 闪电保护触发线。最大收益达到 5% 后进入快速保本监控。 |
| `FLASH_PROTECT_EXIT` | `0.02` | 闪电保护退出线。达到 5% 后若回落到 2% 附近，触发退出。 |
| `TRAILING_TRIGGER_ROI` | `5.50` | 暴利追踪触发线。当前约 550%，很少触发。 |
| `TRAILING_KEEP_RATIO` | `0.92` | 暴利追踪保留比例。 |
| `COUNTER_TREND_PROTECT_TRIGGER` | `0.25` | 逆势单利润保护触发线。 |
| `COUNTER_TREND_PROTECT_EXIT` | `0.10` | 逆势单利润回撤退出线。 |
| `MACD_FADE_MIN_ROI` | `0.03` | MACD 动能衰竭最小利润门槛。最大收益超过 3% 且持仓超过 1 分钟后，动能衰竭即可退出。 |

当前 `LADDER_TIGHT` 第一档从 `8% -> 5%` 开始保护，已经比旧版 `15%` 更适合秒级微利策略。若仍觉得太敏感，优先上调 `MACD_FADE_MIN_ROI` 或延长秒级确认，而不是单独拔高第一档 ladder。

## 动态阶梯

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `DYNAMIC_LADDER_ENABLED` | `False` | 是否根据初始 alpha 选择 tight/wide 阶梯。V0 默认固定 tight。 |
| `HIGH_ALPHA_WIDE_THRESHOLD` | `2.5` | 使用 wide 阶梯的高 alpha 阈值。 |

V0 当前更像短线现金流策略，因此固定 tight 更符合它的性格。如果要追 30m/60m 大波段，可以考虑启用动态阶梯，但需要重新回测收益分布和最大回撤。

## 僵尸单、小利保护与指数反转退出

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `ZOMBIE_EXIT_MINS` | `20` | 持仓 20 分钟且收益绝对值小于 2% 时，认为缺乏动能，触发僵尸单退出。 |
| `COUNTER_TREND_MAX_MINS` | `5` | 逆势单最长持有时间。 |
| `INDEX_REVERSAL_EXIT_ENABLED` | `True` | 是否启用大盘趋势反转退出。 |
| `SMALL_GAIN_THRESHOLD` | `0.08` | 小利保护触发线。最大收益曾达到 8% 后进入保护。 |
| `SMALL_GAIN_MINS` | `15` | 小利保护最短持仓时间。 |
| `SMALL_GAIN_LOCKED_ROI` | `0.04` | 小利保护退出线。达到 8% 后若回落到 4% 以下，触发退出。 |

这些参数让 V0 不愿意“等太久”。这也是它很难吃完整大波段，但回撤相对更可控的主要原因之一。

## MACD 与慢牛通道

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `MACD_HIST_CONFIRM_ENABLED` | `True` | 开仓时是否要求 MACD 柱线方向确认。 |
| `MACD_HIST_THRESHOLD` | `0.05` | MACD 柱线确认阈值。 |
| `SLOW_BULL_CHANNEL_ENABLED` | `False` | 是否启用慢牛通道 B。 |
| `SLOW_BULL_MAX_VOL_Z` | `0.5` | 慢牛通道最大波动率 z-score。 |
| `SLOW_BULL_ALPHA_THRESHOLD` | `0.75` | 慢牛通道 alpha 门槛。 |
| `SLOW_BULL_MACD_THRESHOLD` | `0.02` | 慢牛通道 MACD 阈值。 |
| `SLOW_BULL_MIN_INDEX_ROC` | `0.0005` | 慢牛通道大盘顺风阈值。 |

通道 A 是当前主路径；通道 B 是为“低波、慢牛、大盘顺风”预留的做多通道，当前关闭。

## 大盘 Index Guard

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `INDEX_GUARD_ENABLED` | `True` | 是否启用开仓方向的大盘过滤。关闭后 `_check_index_guard` 会直接放行。 |
| `INDEX_GUARD_SHORT_BLOCK_ENABLED` | `True` | 在大盘上涨或短线拉升时，是否强制禁止开 PUT。 |
| `INDEX_ROC_THRESHOLD` | `-0.01` | 做 CALL 时的大盘 ROC 下限。当前较宽，允许一定下跌中抄底。 |

`INDEX_GUARD_SHORT_BLOCK_ENABLED` 是更细的一层 PUT 拦截：即使允许 CALL 抄底，也仍可禁止“上涨趋势中顶风做空”。如果要观察逆势 PUT 的真实表现，需要同时确认当前运行的是 V0，并检查该开关是否符合测试目的。

## Guard Switches

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `ENTRY_MOMENTUM_GUARD_ENABLED` | `True` | 开仓动量保护开关，当前主要为配置兼容。 |
| `ENTRY_LIQUIDITY_GUARD_ENABLED` | `True` | 开仓流动性保护开关，当前主要为配置兼容。 |
| `EXIT_COUNTER_TREND_ENABLED` | `True` | 逆势退出保护开关。 |
| `EXIT_INDEX_REVERSAL_ENABLED` | `True` | 指数反转退出保护开关。 |
| `EXIT_STOCK_HARD_STOP_ENABLED` | `True` | 正股硬止损开关。 |
| `EXIT_ZOMBIE_STOP_ENABLED` | `True` | 僵尸单退出开关。 |
| `EXIT_MACD_FADE_ENABLED` | `True` | MACD 衰竭退出开关。 |
| `EXIT_SIGNAL_FLIP_ENABLED` | `True` | alpha 反转退出开关。 |
| `EXIT_LIQUIDITY_GUARD_ENABLED` | `True` | 平仓流动性保护开关。 |
| `EXIT_COND_STOP_ENABLED` | `True` | 条件止损开关。 |
| `EXIT_SMALL_GAIN_ENABLED` | `True` | 小利保护开关。 |

注意：部分开关是为后续统一策略框架保留的兼容字段。修改前最好先确认 `strategy_core_v0.py` 是否直接读取该开关。

## 秒级平仓确认

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `EXIT_CONFIRM_SECONDS_1S` | `8` | 秒级路径下，同一退出原因需要连续确认的秒数。用于降低 1s quote 噪声造成的误平仓。 |
| `EXIT_CONFIRM_REASON_PREFIXES` | 多项 | 需要做秒级连续确认的退出原因前缀。 |

当前确认列表包含硬止损、条件止损、trailing、阶梯保护、flash protect、时间止损、小利保护、MACD fade、正股止损、僵尸单、点差止损等。  

如果把 `EXIT_CONFIRM_SECONDS_1S` 从 1 提到 5 或 8 后收益变化不大，说明收益差异大概率不在“秒级过早止盈止损”这一层，而可能在执行价格、撮合、信号对齐、数据口径或策略本身的退出逻辑。

## 严格校验模式

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `PARITY_STRICT_MODE` | `True` | V0 对齐/回归测试时使用的严格模式标记。 |

## 常见调参方向

更保守：

- 提高 `ALPHA_ENTRY_THRESHOLD`。
- 降低 `MAX_SNAP_ROC_LIMIT`，减少追尖峰。
- 降低 `MAX_SPREAD_PCT_ENTRY`。
- 提高 `MACD_FADE_MIN_ROI`，减少过早微利出场。
- 缩小 `POSITION_RATIO` 或 `MAX_POSITIONS`。

更多交易：

- 降低 `ALPHA_ENTRY_THRESHOLD`。
- 放宽 `VOL_MIN_Z` / `VOL_MAX_Z`。
- 降低 `MIN_LAST_SNAP_ROC`。
- 放宽 `REGIME_REVERSAL_THRESHOLD`。

更愿意持有波段：

- 放宽 `STOP_LOSS` 和 `ABSOLUTE_STOP_LOSS`。
- 提高 `TIME_STOP_MINS` 或降低 `TIME_STOP_ROI`。
- 提高 `MACD_FADE_MIN_ROI`。
- 放宽 `SMALL_GAIN_THRESHOLD` / `SMALL_GAIN_LOCKED_ROI`。
- 考虑启用 `DYNAMIC_LADDER_ENABLED`，让高 alpha 单走 `LADDER_WIDE`。

更接近实盘执行安全：

- 保持较严的 `MAX_SPREAD_PCT_ENTRY`。
- 对低流动性标的降低单笔仓位上限。
- 对 bid 消失或 quote 断流先做日志和连续性统计，不建议直接把单次 quote gap 转换成额外平仓或大幅折价。
