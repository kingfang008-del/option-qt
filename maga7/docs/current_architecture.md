# Mag7 当前统一架构与时钟契约

> 状态日期：2026-07-18  
> 生产候选：`m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1`  
> 研究基线 overlay（Watchdog / Hunter，默认 off）：见 [`watchdog_stack_architecture.md`](watchdog_stack_architecture.md)

## 1. 统一原则

Mag7 的股票事实源是 `/mnt/s990/data/raw_1s/stocks` 下的逐秒数据。系统不再用
“外部 1 分钟数据是左标签还是右标签”解释因果性；统一使用以下事件语义：

- `source_ts`：一秒行情真实所属的时间；
- `minute_key = floor(source_ts, 1min)`：该秒归属的分钟；
- 分钟 `M` 覆盖半开区间 `[M, M+1min)`；
- 收到下一分钟的第一帧后，分钟 `M` 才成为 completed bar；
- `feature_ts`：分钟 `M` 的 `minute_key`，用于特征、regime 和审计归属；
- `decision_ts`：completed bar 实际可见并允许下单的时间，必须满足
  `decision_ts >= M+1min`。

因此，因果正确性取决于“秒数据归桶、分钟完成、决策可用”三件事一致，而不是文件
中的 1 分钟时间戳被称为什么标签。

`bar_availability_delay_seconds=60` 是给预聚合 1 分钟适配路径使用的可用性契约：

- 从原始 1 秒流实时聚合时，`MinuteBarAggregator` 在分钟滚动时自然产生真实
  `available_ts`，不能再叠加一次 60 秒；
- 直接读取预聚合 1 分钟研究表时，用 `feature_ts + 60s` 构造同等
  `decision_ts`；
- 两条路径最终必须产生相同的 `(feature_ts, decision_ts)`，禁止双重延迟。

## 2. 数据事实源与派生层

| 数据 | 事实源/入口 | 用途 |
|---|---|---|
| Mag7 股票 1 秒 | `/mnt/s990/data/raw_1s/stocks/{SYM}/{SYM}_{date}.parquet` | Scanner、Redis S5、离线 1s 审计 |
| IBKR 股票实时 | `Mag7IbkrConnector` 的逐秒 completed frame | G4–G6，语义等价于股票 1 秒事实源 |
| 期权 1 秒 bid/ask | profile 的 `paths.quote_1s_root` | 离线成交、S5 Redis quote |
| IBKR 期权实时 | 已锁合约的 ticker bid/ask | Shadow/Paper/Live 成交与退出 |
| 预聚合股票 1 分钟 | `paths.stock_root` | 研究加速缓存，不是独立时钟事实源 |
| 开盘锁图 | `paths.open_locked_map` | G0–G3 历史锁约/选约 |
| 实时锁约清单 | `live_sessions/.../locks.json` | G4–G6 当日锁约恢复与审计 |

预聚合 1 分钟表可以继续用于快速 Offline/Parity，但必须视为由秒级事实源派生的缓存，
并通过统一 `decision_ts` 适配。最终上线证据还必须经过原始 1 秒 S5 和真实行情
session，不能只凭 1 分钟缓存放行。

## 3. 分层架构

```text
配置与数据层
  Profile + 股票1s + 期权1s + open-lock 数据
             │
             ▼
合约层
  09:30 spot → 0/1/2 trading-DTE → ATM+OTM1..5 ladder
             │
             ▼
行情层
  Offline 1s Pitcher 或 IBKR Connector
  → run-scoped fused_market_stream:maga7:<run_id/session_id>
             │
             ▼
时钟与策略层
  frame 校验 → options 入 book → 股票1s归桶 → completed 1m
  → Rule-A / mf10 / streak / QQQ-VIXY regime / TopK
  → （可选研究）Watchdog Degrade/Halt + Hunter 短窗
             │
             ▼
OMS 层
  only_win + concurrent sizing + Redis/IBKR quote
  → Shadow fill 或 IBKR Paper/Live order lifecycle
  → mf_flip / TP / SL / T+30 / EOD flatten
             │
             ▼
恢复与审计层
  scanner_state + oms_state + locks + manifest
  + signals/trades/order/fill/commission/reconcile
             │
             ▼
只读监控层
  dash/：G0 → G1 → G2 → G3 → G4 → G5 → G6
```

QQQ 的 TFT/FCS、`ExecutionEngineV8`、4-bucket delta 锁约和共享
`fused_market_stream` 不属于 Mag7 策略链。Mag7 只复用其 IBKR 生命周期、状态恢复和
风控设计经验。

## 4. 三条执行路径

### 4.1 G1/G2：快速规则验证

```text
预聚合1m缓存
  → feature_ts
  → decision_ts = feature_ts + bar_availability_delay_seconds
  → Offline Replay / StreamEngine
  → 共用 ContractBooks、RegimeGate、simulate_trade
  → 逐笔 parity
```

用途是快速验证规则、`only_win` 顺序、选约和收益计算。它不验证 Redis、墙钟、订阅或
券商订单生命周期。

### 4.2 G3：隔离 Redis S5

```text
股票1s + 期权1s
  → Mag7FusedPitcher
  → run-scoped Redis stream
  → Mag7RedisScannerLoop
  → MinuteBarAggregator
  → Mag7Scanner
  → Mag7OmsStub（只使用已经到达 Redis 的期权报价）
```

每个 run 拥有独立 stream、consumer group、ACK、回放时钟和结果目录。消费者要求：

1. 一个 frame 只有一个 `run_id/frame_id/ts`；
2. 同帧 symbol 不重复；
3. frame 严格递增，foreign/duplicate/rejected 可审计；
4. 同秒先把期权报价放入 quote book，再处理股票、退出和新入场；
5. 禁止整日 warm book 和磁盘报价回退。

### 4.3 G4–G6：真实会话

```text
IBKR 股票/期权 ticker
  → Mag7IbkrConnector 逐秒封帧
  → run-scoped Redis
  → Mag7LiveFrameEngine
  → Mag7Scanner
  → Mag7BrokerOms
       shadow: 不发券商订单
       paper : IBKR Paper placeOrder
       live  : 多重硬锁后 IBKR Live placeOrder
```

会话由 `maga7.tools.run_live_session` 编排，覆盖 NYSE 日历、09:30 锁约、行情订阅、
Scanner/OMS、订单回报、partial fill、commission、reconcile、断线恢复和 EOD flatten。

## 5. 开盘锁约与订阅

生产候选采用 strike-ladder，不使用 QQQ 的 delta bucket：

1. 09:30 ET 获取当日真实 spot；
2. 对 0/1/2 trading-DTE 分别查询并 qualify 合约；
3. Call/Put 各锁 ATM+OTM1..5；
4. `locks.json` 原子持久化，全天不重新定义 ladder；
5. 首先订阅首选可用 DTE；fallback DTE 已 qualify，在实际选中后按需订阅；
6. 信号时按方向和 signal spot 从已锁 ladder 中选约，不使用未来链数据。

历史回放的 `open_locked_map` 与实时 `locks.json` 必须保持相同 bucket、DTE 和
localSymbol 语义。

## 6. Scanner 与 OMS 顺序

单个 completed fused frame 的固定顺序是：

1. 校验 frame；
2. 导入该秒所有 option quote；
3. 先更新 QQQ/VIXY reference minute state；
4. 再更新 Mag7 股票 minute state；
5. 生成本分钟候选信号；
6. 先评估已有仓位退出；
7. 再处理新入场；
8. 原子 ACK，并持久化 Scanner 进度。

OMS 负责：

- 活动持仓互斥、TopK 和最多两腿并发；
- `only_win` 只在真实/模拟平仓结果后回写 Scanner；
- quote 新鲜度、spread fill、仓位 sizing 和日亏熔断；
- pending signal、pending order、partial fill、commission 和 broker position 对账；
- DISARM 只阻止新入场，不阻止已有仓位退出。

## 7. 状态权威与恢复

| 状态 | 权威载体 |
|---|---|
| 当日锁约 | `locks.json` |
| Scanner 窗口、TopK、only_win | Redis scanner snapshot + `scanner_state.json` |
| 持仓、订单意图、待报价信号 | `oms_state.json` |
| 已消费 frame | run-scoped ACK + engine health |
| 券商真实订单/持仓 | IBKR，恢复时必须 reconcile |
| 会话结论 | `manifest.json` |

同一交易日 `--resume` 才允许恢复。恢复时 profile fingerprint、mode、account、锁约和
broker position 任一不一致都应 fail closed。

## 8. G0–G6 门禁

| Gate | 验证对象 | 通过依据 |
|---|---|---|
| G0 | 配置、股票/期权数据、锁图 | 数据覆盖与 profile 可加载 |
| G1 | Offline 基准 | 当前 fingerprint 的完整日期范围结果 |
| G2 | Offline ↔ Stream | 交易集合、收益、size、exit reason 全一致 |
| G3 | Redis S5 | 帧完整、纯 Redis quote、与 Offline 逐笔一致 |
| G4 | 真实行情 Shadow | LIVE data、真实锁约、完整 session artifact、无遗留仓位 |
| G5 | IBKR Paper | order/status/fill/commission/reconcile 真实证据 |
| G6 | 小仓 Live | G4/G5 先通过，多重武装，完整平仓和盘后对账 |

代码实现完成不等于 Gate 通过。Gate 只接受当前 strategy/live fingerprint 对应的产物。

截至 2026-07-16，新的时钟与指纹口径已完成 2026-05-01 单日 S5 冒烟：

- `pitcher_ticks = consumer_batches = unique_frames = 23400`；
- duplicate/foreign/rejected 均为 0；
- 期权成交全部来自 Redis，`n_path_disk=0`；
- S5 与 Offline 3/3 匹配，`max_abs_ret_diff=0`。

该结果证明单日链路正确，不替代完整日期范围 G1–G3 冻结，也不替代 G4–G6 真实交易日
证据。

## 9. 关键实现

| 组件 | 路径 |
|---|---|
| 秒级加载与分钟聚合 | `maga7/common/bar_agg.py` |
| Offline 成交与 mf_flip | `maga7/common/replay.py` |
| Stream 规则对拍 | `maga7/common/stream_engine.py` |
| 开盘 ladder 选约 | `maga7/common/open_lock.py`、`entry_contract.py` |
| Redis key/序列化 | `maga7/live/redis_fused.py` |
| S5 Pitcher/Consumer | `maga7/live/redis_pitcher.py`、`redis_consumer.py` |
| 实时 Scanner | `maga7/live/scanner.py` |
| IBKR 行情与订阅 | `maga7/live/ibkr_connector.py` |
| 实时锁约 | `maga7/live/live_contract_lock.py` |
| Live frame 编排 | `maga7/live/live_engine.py` |
| Shadow/Paper/Live OMS | `maga7/live/broker_oms.py` |
| Scanner 恢复 | `maga7/live/scanner_state.py` |
| 会话入口 | `maga7/tools/run_live_session.py` |
| 全流程只读监控 | `dash/` |

运维命令和硬锁见
[`live_session_operations.md`](live_session_operations.md)，规则与 Redis 对拍细节见
[`scanner_oms_integration.md`](scanner_oms_integration.md)。
