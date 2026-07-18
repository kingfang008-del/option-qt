# Mag7 实时交易会话运维

`maga7.tools.run_live_session` 把同一套 Mag7 profile 依次接到：

1. IBKR 股票/期权实时行情；
2. 真实开盘 spot 的 0/1/2 trading-DTE、ATM+OTM ladder 锁约；
3. run-scoped `fused_market_stream:maga7:<session_id>`；
4. 因果 1 秒→1 分钟 Scanner、QQQ/VIXY regime、实时 mf_flip；
5. Shadow / IBKR Paper / IBKR Live OMS；
6. orderStatus、partial fill、commission、持仓对账和可恢复审计。

实时股票帧与 `/mnt/s990/data/raw_1s/stocks` 的历史事实源使用同一秒级契约。
Scanner 只在分钟滚动后消费 completed minute；不依赖“1 分钟左/右标签”判断可见性。
完整分层与时钟定义见 [`current_architecture.md`](current_architecture.md)。

`--lock-time/--end-time` 默认 `auto`，从 NYSE calendar 读取开盘和“收盘前
5 分钟”，因此半日市不会按 15:55 错跑。

代码能力不代表阶段已经通过。G4/G5/G6 只由 `dash/` 对真实 session
artifact 判定，不能用单元测试或 Redis replay 代替。

## 事件日禁入（开盘前）

可预知宏观日（FOMC / Mag7 财报 / 巨型 IPO）在开盘前写入黑名单，当日不入场：

```bash
# 1) 编辑日期文件（JSON dates 列表）
#    maga7/CONFIG/event_calendar_live.json
export MAG7_EVENT_CALENDAR_PATH=maga7/CONFIG/event_calendar_live.json

# 2) 或 Redis（开盘前写入）
# redis-cli SET maga7:event_blackout '["2026-06-17"]'

# 3) 或临时强制今天停手
# export MAG7_EVENT_BLACKOUT_TODAY=1
```

启动后若今日命中，日志会出现 `EVENT_BLACKOUT active today=...`，OMS `day_halted`。  
研究消融见 [`event_calendar_block_research.md`](event_calendar_block_research.md)。

## 盘前一天流式对拍（开盘前 / 改代码后）

对齐 production：打一天数据 → `trade_log` → 对 offline 开平仓：

```bash
cd maga7/SHELL
./run_day_stream_check.sh                 # 默认 2026-05-28；有 Redis 走 S5
./run_day_stream_check.sh 2026-06-02 --force-local
```

详见 [`premarket_hardening.md`](premarket_hardening.md)。通过仍不能替代下方 G4/G5/G6 真实 session 证据。

## 前置条件

- IB Gateway/TWS 已开启 API，Paper 默认端口 `4002`，Live 默认 `4001`。

- Paper/Live 必须显式传入 `--account`；账户不在 IBKR managed accounts 时 fail closed。
- 行情账户具备美股和 OPRA 实时权限；`market_data_type=3/4` 只能观察，不能通过 G4/G6。
- Redis 可用；实时默认 DB 0，Replay 默认 DB 1，避免相互污染。
- 使用生产 open-ladder profile：

```bash
PROFILE=maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json
```

期权订阅先覆盖每个标的首选可用 DTE 的 ATM+OTM ladder；1/2DTE fallback
已 qualify，在信号实际选择后按需订阅。默认最多 90 条期权行情，需与账户
market-data line 配额核对。

## G4：真实行情 Shadow

09:25 ET 前启动，09:30 ET 用真实 spot 锁约。

一键脚本（推荐，含 Redis/IB 端口检查与日志；在 `maga7/SHELL/`，**不要**用 `production/SHELL`）：

```bash
cd maga7/SHELL
./start_maga7_live_session.sh start shadow
# 状态 / 停止 / 盘前同步日历
./start_maga7_live_session.sh status
./start_maga7_live_session.sh stop
./start_maga7_live_session.sh sync-calendar
```

等价手工命令：

```bash
python -m maga7.tools.run_live_session \
  --profile "$PROFILE" \
  --mode shadow \
  --scheme m5_circuit \
  --lock-time 09:30 \
  --end-time 15:55 \
  --redis-db 0
```


Shadow 不发券商订单，但使用同一实时行情、锁约、Scanner、限价模型和退出状态机。
一个 session 只有同时满足以下条件才通过 G4：

- `state=DONE`，无遗留持仓；
- IBKR `data_mode=LIVE`，锁约 `LOCKED`；
- 消费帧大于 0，foreign/rejected frame 为 0；
- session 有 `manifest.json`、`locks.json`、`signals.*`、`order_events.jsonl`。

## G5：IBKR Paper

只有 G4 证据通过后运行：

```bash
python -m maga7.tools.run_live_session \
  --profile "$PROFILE" \
  --mode paper \
  --account <IBKR_PAPER_ACCOUNT> \
  --ib-port 4002 \
  --max-qty 1 \
  --redis-db 0
```

Paper 启动前必须完成 broker reconcile。G5 还要求真实产生并留存
`ORDER_SUBMITTED`、`ORDER_STATUS`、`FILL`、`COMMISSION` 和 `RECONCILE`
事件；无交易日不能用“进程正常”冒充 G5 通过。

## G6：Live 硬锁

Live 同时需要四层许可：

1. CLI 明确传入 `--mode live --live-orders`；
2. IBKR 端口必须是 `4001`，行情必须是 `LIVE`；
3. 环境变量 `MAG7_LIVE_TRADING=1`；
4. `MAG7_LIVE_CONFIRM=<纽约交易日>:<profile_hash前12位>` 且 Redis
   `meta:runtime_trading_controls:maga7.trading_enabled=1`。

先从 Paper session 的 manifest 读取 `profile_hash`，再显式武装：

```bash
export MAG7_LIVE_TRADING=1
export MAG7_LIVE_CONFIRM=2026-07-16:0123456789ab
redis-cli -n 0 HSET meta:runtime_trading_controls:maga7 trading_enabled 1

python -m maga7.tools.run_live_session \
  --profile "$PROFILE" \
  --mode live --live-orders \
  --account <IBKR_LIVE_ACCOUNT> \
  --ib-port 4001 \
  --max-qty 1 \
  --redis-db 0
```

任一条件缺失都会在启动或入场前 fail closed。运行时 DISARM：

```bash
redis-cli -n 0 HSET meta:runtime_trading_controls:maga7 trading_enabled 0
```

DISARM 阻止新入场，不阻止已有真实持仓的风险退出。

## 跳空 / 过期报价硬闸（OMS）

`trade.risk`（见 peer3 因果基线 profile）在 Shadow/Paper/Live 统一生效：

| 闸门 | 行为 |
|------|------|
| 股票 staleness | `connector.last_stock_tick` 超过 `max_stock_staleness_sec` → 入场 `ENTRY_WAIT` |
| 期权 staleness | 报价年龄超过 `max_option_staleness_sec` → 不可见，等同缺报价 |
| 价差 / 入场 mid 跳空 | 超 `max_spread_pct` 或 `max_entry_mid_jump_pct` → 拒单或等待 |
| 持仓 mid 跳空 | 超 `max_exit_mid_jump_pct` 时挂起 SL/TP/T+；连续 `max_gap_hold_ticks` 后 `GAP_FLATTEN` |
| 不利成交 | fill 穿 ask/bid 或 spread frac 超限 → 记 `ADVERSE_FILL`，入场后立即 `ADVERSE_FILL_FLATTEN` |
| 出场追单 | 出场撤单/拒单累计 ≥ `max_exit_chase` → `EXIT_CHASE_CAP`（按 bid 强平） |
| 日熔断 | `day_circuit` 触发后除停开仓外，`day_circuit_force_flatten` 对剩余 OPEN 仓 `DAY_CIRCUIT` 强平 |

纯函数实现：`maga7/live/risk_guards.py`；接线：`maga7/live/broker_oms.py`。

## 动态询价（cancel-replace）

Paper/Live 在限价超时后不再只撤单放弃，而是借鉴 `New_Pro/baseline_qqq/oms` 的精神做 **LMT cancel-settle → requote**：

- 入场：在 `entry_frac` 基础上按 `entry_frac_step` 向 ask 推进，**不穿 ask**；总滑点帽 `entry_max_slippage_pct`（相对首笔 mid），单步帽 `entry_step_cap_pct`。
- 出场：普通理由逐步贴 bid；`SL` / 强平类理由允许小幅穿 bid（urgent）。
- 每轮超时默认入场 3s / 出场 2s；超过 `max_*_requotes` 后回退到原有 `EXIT_CHASE_CAP` / 入场放弃。
- Shadow 仍瞬时按模型价成交，不走 requote（研究口径不变）。

定价：`maga7/live/requote.py`；配置：`trade.risk.requote`。

## 断线、重启与恢复

连接心跳失败后会重连并重订股票和已锁期权。同交易日进程重启使用原 session：

```bash
python -m maga7.tools.run_live_session \
  --profile "$PROFILE" \
  --mode paper \
  --account <IBKR_PAPER_ACCOUNT> \
  --session-id <原session_id> \
  --resume \
  --ib-port 4002 \
  --redis-db 0
```

恢复会：

- 保留原 Redis stream/group，不清空；
- 校验并重新 qualify `locks.json`；
- 恢复 `oms_state.json`；
- claim pending Redis frame；
- 重新绑定 IBKR open trade callback；
- 查询 completed orders，补入停机期间的 fill；
- 对齐 broker option positions；不一致时停止新入场。

恢复只允许同一交易日、同一 mode、同一 profile hash。不要跨日复用 session。

## 盘后与回滚

15:55 ET 默认执行 EOD flatten，等待成交并再次 reconcile。若仍有持仓，manifest
写为 `DONE_WITH_OPEN_POSITIONS` 且进程返回非零；必须在 IBKR 人工确认，不得把它
标记为成功。

回滚顺序：

1. Redis runtime DISARM；
2. 保持进程运行，让已有仓位继续退出和对账；
3. 必要时在 IBKR 人工平仓；
4. 确认 broker 与 `oms_state.json` 均为空；
5. 再停止会话进程。

全流程面板（分层同 `qqq_btc/dashboard`）：

```bash
python dash/run.py
```

侧栏：Download → Offline Replay → Stream Parity → Live。  
对拍与实盘共用同一 profile；Dashboard 只读，不提供发单、武装或停进程按钮。
