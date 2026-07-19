# L2 还剩什么（白话版）

**日期：** 2026-07-19  
**基线：** peer3 **L2+TT1 05**（`max_otm_50bps` + 仅二四 confirm2；见 [`tt1_uplift_05.md`](tt1_uplift_05.md) / [`research_full_day_peer3_baseline.md`](research_full_day_peer3_baseline.md)）  
**说明：** 05 已写入 research profile；下列验收项仍按「先别再拧 α 旋钮」执行。

## 两类事

| 类型 | 是什么 | 例子 |
|------|--------|------|
| **本地就能验** | 用历史回放对照 | 熔断、去票、滑点 |
| **上线过渡** | 要对着 Shadow/实盘会话 | Scanner 是否打出 Hunt、日终对账 |

---

## 本地三件事（已跑）

工具：`python -m maga7.tools.run_l2_local_checks`  
结果：[`../results/watchdog/l2_local_checks/README.md`](../results/watchdog/l2_local_checks/README.md)

| # | 问题 | 结果 | 进基线？ |
|---|------|------|----------|
| 1 | Hunt 大亏后当天还做不做？ | 停手会伤强窗（砍掉救命的基线单） | **关** |
| 2 | 成交贵一点会怎样？ | 0.9 档弱窗很痛 | 不改默认；上线盯滑点 |
| 3 | 是不是靠某一两只票？ | 去任一只仍赚；去 TSLA 掉最多 | 不改池子 |

---

## 工程对齐（本地已跑，2026-07-19）

工具：`python -m maga7.tools.run_hunt_scanner_align`  
结果：`results/watchdog/hunt_scanner_align/`

| 检查 | 结果 |
|------|------|
| 12 个历史 Hunt 日：begin_day 候选 ⊇ offline 成交 | **通过** |
| stream vs offline（票/向/时刻） | **通过** |
| 直播 `scanner.py` 会不会把 Hunt 变成下单 | **已补**（2026-07-19）：`_schedule_hunts` + `drain_hunts`，`event_source=hunt` |
| live `stock_by` 晨间累积 + 延迟 arm Hunt | **已接**（2026-07-19）：`on_reference_second` / `_append_stock_bar` / `_maybe_refresh_watchdog`；`run_live_session` + `live_engine` + `redis_consumer` 喂 QQQ/标的 |

Smoke：07-02 空 `stock_by` 按序喂 1m → ~09:51 arm → 发出 `AMD UP` Hunt。Redis / live_engine 帧时钟也会 `drain_hunts`。

## 上线过渡（还没做，本地替不了）

1. **Shadow 连跑几天**（确认 live 路径也能打出 Hunt）  
2. **日志里能分清** Hunt / 基线单  
3. **看板** 能看见今天 Halt / Hunt  
4. **日终** 实盘日志 vs 当天回放对一下  

本地工程侧：`stock_by` 已从空缓存按分钟累积，不必再预灌全日数据。下一关就是 Paper/Shadow。

---

## 明确先不做

股票袖仓、gamma、`mf_backfill` 进基线、放宽 Hunt 洗盘阈值、为故事日关 QQQ。

## 已收窄（2026-07-19）

`ladder_otm_rungs` / `lock.otm_rungs`：**5 → 3**（peer3 基线及同族 profile）。依据 `results/ladder_rung_ablation_peer3_may_jul/`。

## Entry iceberg（最小可用，2026-07-19）

`trade.iceberg`：按 `ask_size×frac` / `fallback_notional` 拆开仓限价；Shadow 同步走 clip；Paper/Live 成交后续挂。见 `maga7/live/iceberg.py`。
