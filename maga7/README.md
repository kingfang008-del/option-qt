# maga7 — Mag7 Rule-A money-flow Top2 short-DTE scalp

规则策略路径（非 TFT）。成交模型复用 `qqq_btc.common.fills_model` /
`qqq_btc.live.oms_adapter`。

**临时生产 profile**：
`CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json`

- `open_ladder` OTM5 + `only_win` + concurrent p20 + `exit_mode=mf_flip`
- 股票事实源：`/mnt/s990/data/raw_1s/stocks`
- 预聚合 1 分钟研究缓存：`~/train_data/spnq_train`（不是独立时钟事实源）

当前统一架构、秒级时钟契约、恢复和 G0–G6 门禁见
[`docs/current_architecture.md`](docs/current_architecture.md)。

研究基线 + Watchdog/Hunter 三层栈（默认 off、过拟合边界）见
[`docs/watchdog_stack_architecture.md`](docs/watchdog_stack_architecture.md)；
完善顺序见 [`docs/watchdog_optimization_roadmap.md`](docs/watchdog_optimization_roadmap.md)。

研究结论见 [`docs/open_ladder_live_package_results.md`](docs/open_ladder_live_package_results.md)；
版本对比见 [`docs/jan_jul_replay_versions.md`](docs/jan_jul_replay_versions.md)。

## 策略摘要

- 信号：规则 A（mf10 streak≥8，|from_prev|≥2%，vol_z≥1，10:30–14:00）
- 选股：当日 Top2（最早触发）
- 选约：生产候选 `open_ladder`（开盘阶梯 OTM5）；对照可用 day_lock ATM
- 出场：TP 1.6x / SL 0.4x / 超时 30m；生产候选 `mf_flip`
- 仓位：concurrent（独处 20%，并发第二腿 10%，最多 2 腿）
- 成交：点差 frac=0.8（`bid + 0.8*(ask-bid)` 买入）

## 目录

```text
maga7/
  CONFIG/strategy_profiles/   命名 profile 与 catalog
  common/   signals, replay, stream_engine, bar_agg, open_lock, provenance
  live/     scanner, redis_*, ibkr_connector, broker_oms, live_engine
  tools/    prepare_*, run_replay_offline, run_stream_parity,
            run_maga7_redis_sim, run_live_session, OMS dry/stub
  results/  回测、对拍、Redis S5、live_sessions
  docs/     架构、对拍、锁约流水线、实时运维
```

## 数据与时钟

- 股票权威源是秒级 parquet：`/mnt/s990/data/raw_1s/stocks/{SYM}/{SYM}_{date}.parquet`
- 秒数据按 `[M, M+1min)` 聚合；分钟完成后再决策
- 不再用“外部 1 分钟左/右标签”解释因果性
- `bar_availability_delay_seconds=60` 只用于预聚合 1 分钟缓存适配；
  原始 1 秒流使用实际 `available_ts`，禁止双重延迟

```bash
export MASSIVE_API_KEY=...   # 或 POLYGON_API_KEY
cd /path/to/option-qt
export PYTHONPATH=$PWD

# 日锁对照流水线
python -m maga7.tools.prepare_jan_jul_data --step all --max-workers 12

# 开盘阶梯锁约 + 1s quote
python -m maga7.tools.prepare_open_lock_quotes --step all
```

## Offline / Stream / Redis

```bash
# G1 Offline
python -m maga7.tools.run_replay_offline \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json \
  --scheme m5_circuit

# G2 Stream parity
python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json \
  --scheme m5_circuit \
  --tag parity_open_ladder_otm5_mf_flip_p20_jan_jul

# G3 Redis S5（股票/期权秒级进总线）
python -m maga7.tools.run_maga7_redis_sim \
  --start-date 2026-05-01 --end-date 2026-05-01 \
  --options --compare-offline
```

## Scanner / OMS 仿真

```bash
# S2：正股 1s → 因果聚合 1m
python -m maga7.tools.run_scanner_from_1s --start-date 2026-05-06 --end-date 2026-05-10

# S3 / S4
python -m maga7.tools.run_oms_dry_run --start-date 2026-05-06 --end-date 2026-05-10 --compare-offline
MAG7_MAX_QTY=1 python -m maga7.tools.run_oms_live_stub \
  --start-date 2026-05-06 --end-date 2026-05-10 --compare-offline
```

## 操作手册

日常顺序（补数据 → Offline → 对拍 → 实盘）见：

[`docs/maga7_operations_guide.md`](docs/maga7_operations_guide.md)

```bash
cd maga7/SHELL
./run_day_stream_check.sh 2026-05-28      # 盘前：打一天 → trade_log → 对 offline
python ../../dash/run.py                  # Download / Offline / Parity / Live
```

## Live session（G4–G6）

一键脚本在 `maga7/SHELL/`（不要用 `production/SHELL` 旧栈）：

```bash
cd maga7/SHELL
./start_maga7_live_session.sh start shadow
./start_maga7_live_session.sh status
```


或直接：

```bash
python -m maga7.tools.run_live_session --help
```

运维、硬锁、恢复和盘后步骤见
[`docs/live_session_operations.md`](docs/live_session_operations.md)、[`SHELL/menu.md`](SHELL/menu.md)。
代码存在不等于 Gate 通过；G4/G5/G6 需要真实 session 证据。

## Dashboard

权威全流程监控：

```bash
python dash/run.py
```

`qqq_btc/dashboard` 内的 Mag7 board 仅保留为旧 Offline/Parity 面板，不再作为
G0–G6 / Live session 的权威入口。
