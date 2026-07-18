# Mag7 操作手册（补数据 → Offline → 对拍 → 实盘）

面向日常研究 / Shadow / Paper。分层对齐 `qqq_btc/dashboard`：**对拍与实盘共用同一套逻辑，只换数据源与成交方式。**

| 层 | 数据源 | 成交 | 用途 |
|---|---|---|---|
| Offline | 磁盘 1s / quote | 模型 fill | 金标收益、改 profile |
| Stream Parity / Day check | 历史 1s 流式打入（或 Redis S5） | 模型 fill | 开平仓一致性 |
| Live Shadow | IBKR 实时 | 模型 fill | 真实行情、不发单 |
| Live Paper / Live | IBKR 实时 | 券商限价 | 真成交 / 真钱 |

**默认 profile（research freeze）**  
`maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json`  
要点：open_ladder OTM5 · peer3 · QQQ align · T30→T45（MTM≥0）· full_day 事件 · TCN off · `topk_backfill` off · `rank_by=earliest`  
基线数字见 [`research_full_day_peer3_baseline.md`](research_full_day_peer3_baseline.md)。

---

## 0. 一分钟导航

```bash
# 仓库根
export PYTHONPATH=$PWD
cd maga7/SHELL

# 监控（侧栏四板）
python ../../dash/run.py

# 盘前主检查：打一天 → trade_log → 对 offline
./run_day_stream_check.sh 2026-05-28

# 开盘前 Shadow
./start_maga7_live_session.sh sync-calendar
./start_maga7_live_session.sh start shadow
```

| 文档 | 内容 |
|---|---|
| 本文 | 日常操作顺序 |
| [`live_session_operations.md`](live_session_operations.md) | G4–G6 硬锁、恢复、风险闸 |
| [`premarket_hardening.md`](premarket_hardening.md) | day stream / 故障注入 |
| [`replay_stream_parity.md`](replay_stream_parity.md) | G2 对拍细节 |
| [`scanner_oms_integration.md`](scanner_oms_integration.md) | S0–S5 拓扑 |
| [`SHELL/menu.md`](../SHELL/menu.md) | 命令速查 |

---

## 1. Dashboard（看）

```bash
export PYTHONPATH=$PWD
python dash/run.py
# http://127.0.0.1:8501
```

侧栏 Board：

1. **Download** — 可配起止日期 / 标的 → 扫缺数 → 一键日历与股票 1s → 页面看日志（不改策略）  
2. **Offline Replay** — 离线金标结果  
3. **Stream Parity** — G2 / S5 / `trade_log` 对拍结果  
4. **Live** — 同时持仓、滑动窗 `mf10/mf_fast/streak`、session 证据、启停命令  

Download 可后台启停补数；其余不写 Redis、不代启停 Live、不发单。Live Redis 默认 **DB 0**（S5 研究常用 DB 1）。

---

## 2. 补数据（Download）

目标：股票 1s、期权 quote、锁约 map、事件日历就绪。

**推荐：在 Dashboard Download 页完成**（无需进终端）：

1. 填 Start / End date、Symbols（默认带 profile universe + QQQ）  
2. 「检查缺数」看哪些 `(symbol, day)` 缺 1s  
3. 「① 同步事件日历」/ 「② 下载股票 1s」  
4. 下方 Live log 看进度；可「⏹ 停止任务」  

任务状态：`{stock_1s_root}/logs/dash_mag7_backfill_job.json`  
需 `MASSIVE_API_KEY` 或 `POLYGON_API_KEY`（页面可填，或环境变量）。

### 2.1 事件日历（等价 CLI）

```bash
cd maga7/SHELL
./start_maga7_live_session.sh sync-calendar
# 或指定区间
./start_maga7_live_session.sh sync-calendar 2026-07-01 2026-09-30
```

文件：`maga7/CONFIG/event_calendar_live.json`  
强制今日停手：`export MAG7_EVENT_BLACKOUT_TODAY=1`

### 2.2 股票 1s（等价 CLI）

事实源：`/mnt/s990/data/raw_1s/stocks`（profile `stock_1s_root`）。

```bash
python -m preprocess.download.download_stock_1s \
  --symbols NVDA,TSLA,AMD \
  --start-date 2026-05-01 --end-date 2026-05-10 \
  --stock-output-dir /mnt/s990/data/raw_1s/stocks
```

缺日会导致对拍/实盘该票无法成交（见 [`amd_1s_gap_backfill.md`](amd_1s_gap_backfill.md)）。

### 2.3 锁约 / 期权 quote

- open_ladder 锁约 map：profile `open_locked_map`  
- 期权 1s：`quote_1s_root`  
在 Dashboard **Download** 页看路径是否 `exists`（本页暂不启期权全链路下载）。

**边界**：本阶段不改 TopK / 出场 / regime；那些只动 freeze profile。

---

## 3. Offline Replay（金标）

```bash
export PYTHONPATH=$PWD
python -m maga7.tools.run_replay_offline \
  --profile maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json \
  --scheme single \
  --start-date 2026-05-01 --end-date 2026-07-16
```

- `scheme` 默认用 profile `recommended_scheme`（freeze = `single`）  
- 结果：`maga7/results/...`（Dashboard Offline 板可看）  
- 改规则：**只改 profile JSON**，勿在脚本里硬编码分叉  

研究窗口参考：May–Jul ~+875% / MaxDD −13.2%（至 07-16）。

---

## 4. 对拍（Parity）— 盘前主路径

原则：**同一 profile + Scanner + 退出**；数据=历史流，成交=模拟。

### 4.1 一天流式（推荐，对齐 production trade_log）

```bash
cd maga7/SHELL
./run_day_stream_check.sh                 # 默认黄金日 2026-05-28
./run_day_stream_check.sh 2026-06-02
./run_day_stream_check.sh 2026-05-28 --force-local   # 无 Redis
```

| 有 Redis | 无 Redis |
|---|---|
| S5：1s → fused → scanner → OMS | 进程内 1s → scanner → OMS dry |

产物目录内：

- `trade_log.csv` — 流式 OPEN/CLOSE  
- `trade_log_offline.csv` — offline 对照  
- `day_stream_check.json` — `ok=true` 才过  

### 4.2 G2 规则对拍（offline ↔ stream）

```bash
python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json \
  --scheme single \
  --start-date 2026-05-28 --end-date 2026-05-28 \
  --stock-source stock_1s \
  --tag parity_freeze_smoke
```

### 4.3 G3 Redis S5（最接近实盘拓扑）

```bash
# Redis 需可用；研究库常用 DB 1
python -m maga7.tools.run_maga7_redis_sim \
  --profile maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json \
  --scheme single \
  --start-date 2026-05-28 --end-date 2026-05-28 \
  --options --compare-offline --sync
```

要求：`frame_integrity_ok`、与 offline 逐笔一致、`n_path_disk=0`。

### 4.4 可选：故障注入单测

```bash
./run_premarket_hardening.sh faults-only
```

改 `risk_guards` / OMS 纯函数时跑；日常优先 §4.1。

---

## 5. 实盘（Live）

代码在 ≠ Gate 过。顺序：**对拍过 → G4 Shadow →（有成交日）G5 Paper → G6 Live**。

### 5.1 前置

- IB Gateway/TWS API 开着；Shadow/Paper 端口 **4002**，Live **4001**  
- 行情 `market_data_type=LIVE`（3/4 延迟不能当 G4/G6 证据）  
- Redis 可用（Live DB **0**）  
- 当日日历已 sync  

### 5.2 G4 Shadow（真实行情，不发单）

```bash
cd maga7/SHELL
./start_maga7_live_session.sh sync-calendar
./start_maga7_live_session.sh start shadow
./start_maga7_live_session.sh status
tail -f ../../logs/maga7/live_session.log
```

Dashboard → **Live** → 看同时持仓、滑动窗、session。

G4 通过（由 dash / manifest 判，不是「进程没挂」）：

- `state=DONE`，无残留仓  
- `data_mode=LIVE`，锁约 `LOCKED`  
- frames>0，foreign/rejected=0  
- 有 `manifest.json` / `locks.json` / `signals.*` / `order_events.jsonl`  

产物：`maga7/results/live_sessions/<date>/<session_id>/`

### 5.3 G5 Paper

```bash
MAG7_ACCOUNT=DUxxxxxx \
  ./start_maga7_live_session.sh start paper --account DUxxxxxx
```

需真实留下 `ORDER_SUBMITTED` / `ORDER_STATUS` / `FILL` / `COMMISSION` / `RECONCILE`。无成交日不能冒充通过。

### 5.4 G6 Live（硬锁）

见 [`live_session_operations.md`](live_session_operations.md)：`--mode live --live-orders` + 端口 4001 + `MAG7_LIVE_TRADING` + `MAG7_LIVE_CONFIRM` + Redis `trading_enabled`。建议 `max_qty=1`。

### 5.5 停机 / DISARM / 恢复

```bash
# 停进程
./start_maga7_live_session.sh stop

# Live 运行中停新开仓（不阻止已有仓退出）
redis-cli -n 0 HSET meta:runtime_trading_controls:maga7 trading_enabled 0

# 同日恢复
./start_maga7_live_session.sh resume <session_id> shadow   # 或 paper/live
```

勿跨日复用 session。

---

## 6. 推荐日程

### 改代码后（非交易日也可）

1. `./run_premarket_hardening.sh faults-only`（若动了 risk/OMS）  
2. `./run_day_stream_check.sh 2026-05-28`（或 `--force-local`）  
3. 必要时 G2 parity 短窗  

### 交易日前夜 / 盘前

1. `sync-calendar`  
2. Download 板确认 1s / lock / quote  
3. `./run_day_stream_check.sh <最近完整日>`  
4. IB Gateway LIVE → `start shadow`  
5. Dashboard Live 板盯持仓与滑动窗  

### 盘后

1. 看 Live session manifest / order_events  
2. 若异常：对照当日 `trade_log` 对拍复现（同一 profile）  
3. 需要时再开 Paper 验证成交链路  

---

## 7. 环境变量速查

| 变量 | 含义 | 默认 |
|---|---|---|
| `MAG7_PROFILE` | 策略 profile | freeze full_day peer3 |
| `MAG7_MODE` | shadow / paper / live | shadow |
| `MAG7_ACCOUNT` | IB 账户 | 空 |
| `MAG7_IB_PORT` | API 端口 | paper/shadow 4002，live 4001 |
| `MAG7_REDIS_DB` | Redis DB | 0（Live） |
| `MAG7_EVENT_CALENDAR_PATH` | 事件禁入 JSON | `maga7/CONFIG/event_calendar_live.json` |
| `MAG7_EVENT_BLACKOUT_TODAY` | 强制今日停手 | 未设 |
| `PYTHONPATH` | 仓库根 | 必设 `$PWD` |

---

## 8. 常见坑

| 现象 | 处理 |
|---|---|
| day stream FAIL，entry_ts 对不齐 | 换黄金日 `2026-05-28`，或先修 scanner 时序；研究可用 `--allow-dry-mismatch`（hardening） |
| 某票永远不成交 | 查 1s 是否缺日；期权短 DTE 是否在 lock map |
| Shadow 无帧 | IB 行情是否 LIVE；Redis stream 是否有 fused |
| 对拍过但 Live 行为不同 | 确认 Live 用的是同一 `MAG7_PROFILE`，未开 research-only 开关（TCN / backfill / dyn_trail） |
| `topk_backfill` / 成交额 TopK | **研究开关，freeze 默认关**；见 research 文档，勿当生产默认 |

---

## 9. Freeze 明确不做的事

- 不默认开 `mf_flip` / 短持仓 / `dyn_trail` / `mtm_floor` 软砍（已测伤收益）  
- 不默认开 `topk_backfill_on_block`、`rank_by=dollar_vol`  
- `tcn_gate.enabled=false`  
- Dashboard 不提供发单 / 武装 Live  

细节与消融：[`research_full_day_peer3_baseline.md`](research_full_day_peer3_baseline.md) Non-goals。
