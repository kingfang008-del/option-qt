# Parity 与执行时序说明

## 目的

这份文档用于记录最近一轮 `1s` 对 `1m` parity 收敛过程中达成的关键结论，重点包括：

- 为什么分钟级 Greeks 重算后更容易对齐
- 左对齐分钟语义与 OMS 延后一根 bar 交易之间的关系
- OMS 延迟执行开关的设计原则
- Mock IBKR 与 OMS 如何共用一套执行延迟配置
- 阈值化 parity 验收脚本的用途

## 1. 为什么分钟级 Greeks 重算后更容易对齐

分钟发球机开启 Greeks 重算后，parity 明显改善，这并不一定说明“离线 Greeks 错了”，更准确地说，是因为之前分钟链路和秒级链路没有走同一条计算路径。

实际含义是：

- 离线分钟快照中的 Greeks，可能是历史生产链路在当时根据另一套锚点算出来的
- 秒级链路在运行时，会使用当前时刻的 `spot`、`mid`、时间戳、无风险利率、到期日重新计算
- 当两边一个使用“离线预计算 Greeks”，另一个使用“运行时重算 Greeks”时，就会出现路径依赖的小偏差

运行时重算路径使用的核心输入包括：

- 当前正股价格 / `spot`
- 分钟锚点时间
- 运行时读取的无风险利率 `rfr`
- 运行时解析得到的期权到期日

相关代码：

- [realtime_feature_engine.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/realtime_feature_engine.py)
- [feature_compute_service_v8.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/feature_compute_service_v8.py)
- [s2_run_realtime_replay_sqlite_s2.py](/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/minute/s2_run_realtime_replay_sqlite_s2.py)

工程结论：

- 如果目标是“与当前实时交易链路一致”，应优先信任统一后的运行时 Greeks 重算路径
- 如果目标是“精确复现某一份历史离线库”，那份离线 Greeks 也可能是它当时那条链路下的正确结果
- 对当前系统最重要的目标，也就是“秒级模拟实盘 + 与线上执行一致”，统一运行时重算更值得作为主真值

## 2. 左对齐分钟语义 与 OMS 延后一根 bar 的关系

这两个问题有关联，但不应混为一谈。

### 左对齐分钟语义

对 `1s` 回放链路来说，当时钟跨入新一分钟时，系统结算的是上一分钟的数据。

例如：

- 在 `10:01:00` 触发结算时，实际产出的是 `10:00` 这根 bar 的特征
- 也就是说，该特征语义上属于 `10:00`，而不是 `10:01`

这是训练与实时推理中比较正确的因果写法，也正是当前 parity 对齐所依赖的语义。

### OMS 延后一根 bar 交易

OMS 延后一根 bar，不能通过“平移特征时间戳”来实现。

正确做法应该是：

- 特征时间仍然标记为它所描述的那根 bar
- 信号仍然基于该 bar 的结算特征生成
- OMS 只是在执行层把真实下单动作延后 `N` 根 bar

这样可以避免把以下三件事混在一起：

- 特征属于哪根 bar
- 信号是在什么时候生成的
- 订单是什么时候允许发出的

工程结论：

- 保持特征时间左对齐
- 把“延后交易”实现为 OMS 的执行策略，而不是修改特征时间戳

## 3. OMS 延迟执行开关

当前 OMS 已经支持“延迟执行”模式，而且这个延迟不会改变特征和标签时间语义。

相关文件：

- [config.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/config.py)
- [execution_engine_v8.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/execution_engine_v8.py)

当前行为：

- OMS 收到 `BUY/SELL` 信号后，可以先进入延迟队列
- 信号仍然在原始时刻产生
- 只有真实发单时机被推迟

默认策略：

- 默认不延迟
- 如果开启延迟，默认只延迟 `BUY`
- `SELL` 不会默认延迟，除非显式配置

为什么默认只延迟 `BUY`：

- 延迟开仓，是比较常见的“回测更接近真实执行”的做法
- 如果默认把平仓也一起拖后，容易影响止损、强平、EOD 清仓等风险控制动作

## 4. 执行延迟配置已经统一

为了避免 OMS 与 Mock IBKR 各改各的、后续遗忘导致配置漂移，现在执行延迟已经统一成同一套配置来源。

相关文件：

- [config.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/config.py)
- [mock_ibkr_historical.py](/Users/fangshuai/Documents/GitHub/option-qt/production/history_replay/mock_ibkr_historical.py)
- [mock_ibkr_historical_1s.py](/Users/fangshuai/Documents/GitHub/option-qt/production/history_replay/mock_ibkr_historical_1s.py)

当前统一配置项：

- `EXECUTION_DELAY_BARS`
- `EXECUTION_DELAY_SECONDS`
- `OMS_SIGNAL_DELAY_ACTIONS`

关系如下：

- `OMS_SIGNAL_DELAY_BARS` 现在直接继承 `EXECUTION_DELAY_BARS`
- `MockIBKRHistorical` 与 `MockIBKRHistorical_1s` 的默认延迟，也都从 `EXECUTION_DELAY_BARS / EXECUTION_DELAY_SECONDS` 读取

推荐使用方式：

### 1 分钟级延迟执行

```bash
export EXECUTION_DELAY_BARS=1
export EXECUTION_DELAY_SECONDS=0
```

### 1 秒级高仿真延迟执行

```bash
export EXECUTION_DELAY_BARS=0
export EXECUTION_DELAY_SECONDS=1
```

### 只延迟 BUY

```bash
export OMS_SIGNAL_DELAY_ACTIONS=BUY
```

### BUY 和 SELL 都延迟

```bash
export OMS_SIGNAL_DELAY_ACTIONS=BUY,SELL
```

重要说明：

- OMS 延迟控制的是“信号何时真正释放给执行层”
- Mock IBKR 延迟控制的是“回放时成交价如何重新锚定”

它们现在已经共用一套配置来源，因此不容易再出现“一个改了一个没改”的问题，但二者的业务含义仍然不是完全同一件事。

## 5. Parity 阈值验收脚本

当前已经有一个基于阈值的 parity 回归验收脚本：

- [verify_parity_thresholds.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/verify_parity_thresholds.py)

这个脚本支持：

- 对已知残差项设定默认阈值
- 通过命令行覆盖单个 key 的阈值
- 从 JSON 文件读取阈值配置
- 对回归失败项给出更明确的失败提示
- 在输出中解释“为什么运行时 Greeks 重算会提升 parity”

示例：

```bash
python3 production/baseline/verify_parity_thresholds.py \
  --left /path/to/left.npz \
  --right /path/to/right.npz
```

带单项阈值覆盖的示例：

```bash
python3 production/baseline/verify_parity_thresholds.py \
  --left /path/to/left.npz \
  --right /path/to/right.npz \
  --threshold hist_vwap_30=0.05 \
  --show-passing
```

## 6. 实际操作建议

如果目标是“秒级发球机尽量模拟实盘，并与分钟基准保持高一致性”，建议按下面这个优先级理解和维护系统：

1. 保持分钟特征左对齐且满足因果性
2. 用统一的运行时路径重算期权 Greeks
3. 把延后交易实现为 OMS 执行策略，不去修改特征时间
4. 让 Mock IBKR 与 OMS 共用同一套执行延迟配置
5. 使用阈值化 parity 脚本做回归验收，而不是每次手工盯差异表

## 7. 当前阶段的总体结论

当前系统应理解为：

- 特征使用左对齐、因果正确的分钟语义
- 期权 Greeks 通过共享运行时重算路径实现高一致性
- 交易延迟是 OMS 层的策略开关，不应反向污染特征时间语义
- Mock 成交重锚与 OMS 延迟已共享一套配置来源，降低了遗忘和错配风险

这种职责分离方式更容易长期维护，也更能避免未来引入 lookahead 或配置漂移问题。
