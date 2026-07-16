# Mag7 Scanner → OMS 接入草图

## 目标

把 **多标的 Rule-A TopK** 接到现有 OMS / fill audit，**不经过** QQQ TFT / FCS 主信号链路。

## 拓扑

```text
┌──────────────────────────────────────────────────┐
│ 实时 / 历史 正股 1s (tick or 1s OHLCV)            │
│        ↓ 因果聚合 MinuteBarAggregator            │
│ Mag7 Scanner (maga7.live.scanner)                │
│  决策时钟 = RTH **1m** bars                       │
│  Rule-A → TopK earliest                          │
│  contract_mode: day_lock | open_lock (+ clear_otm)│
│  regime gate (QQQ align)                         │
└──────────────────┬───────────────────────────────┘
                   │ signal_audit.jsonl
                   │ (optional) Redis ORCH_SIGNAL-like payload
                   ▼
┌──────────────────────────────────────────────────┐
│ OMS Adapter                                      │
│  qqq_btc.live.oms_adapter                        │
│  fill_frac=0.8 (profile)                         │
│  订阅合约 **1s quote** → spread fill             │
│  TP/SL/hold from signal.meta                     │
└──────────────────┬───────────────────────────────┘
                   ▼
              IBKR / mock / shadow
```

**两层时钟（不要混）**

| 层 | 粒度 | 用途 |
|---|---|---|
| 信号 | 1m（由 1s 聚合） | mf10 / streak≥8 / vol_z / TopK |
| 成交 | 1s 期权 quote | `fill_frac=0.8` 限价与出场 |

QQQ FCS：**可选** regime 闸门（日后），不是 Mag7 信号源。

已实现（`maga7.common.regime`）：QQQ `from_prev` 对齐、VIXY `vix_reversal_count_30m`、DN 的 `vixy_z` Put 闸。  
Profile `regime.enabled=true` 时 **offline + stream + scanner** 共用闸门。  
选约：`maga7.common.entry_contract`（`day_lock` / `open_lock` / `signal_atm` + `clear_otm_ban_0dte_pct`）。

### 实盘因果选约（推荐）

```bash
# 开盘锁约 + 1s 补数（说明见 maga7/docs/open_lock_quote_pipeline.md）
python -m maga7.tools.prepare_open_lock_quotes --step all
python -m maga7.tools.prepare_open_lock_quotes --step all --add-symbols GOOGL

# stream ↔ offline 对拍（open_lock + clear_otm）
python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_lock_research_v1.json \
  --scheme m5_circuit --tag parity_open_lock_clear_otm_jan_jul
```

规则：开盘锁 0/1/2 DTE；信号时若 0DTE 相对现价明显 OTM（默认 ≥1%）→ 改用 1DTE+。

## 阶段

| 阶段 | 内容 | 状态 |
|---|---|---|
| S0 | Offline replay + stream parity（规则流） | ✅ `maga7.tools` |
| S1 | Shadow scanner（历史 1m parquet → audit） | ✅ `run_scanner_shadow` |
| S2 | **1s → 1m 聚合** → 同 scanner → audit | ✅ `run_scanner_from_1s` + `bar_agg` |
| S3 | OMS dry-run：限价 = fill_model(bid,ask,0.8) + 1s 闭合 | ✅ `run_oms_dry_run` |
| S4 | 独立 OMS stub（小仓 + fill_audit + 可选 Redis）对拍 offline | ✅ `run_oms_live_stub` |

S2 说明：不是「用秒级重写 Rule-A」，而是用秒级作为 ingest，**整分收盘后**再跑与回测相同的 1m 规则，便于与 live 同源、略早于第三方 1m 推送。

## Shadow 用法

```bash
export PYTHONPATH=$PWD

# S1：历史 1m（spnq_train）
python -m maga7.tools.run_scanner_shadow --start-date 2026-05-06 --end-date 2026-05-08

# S2：正股 1s → 聚合 1m（与实盘 ingest 对齐）
python -m maga7.tools.run_scanner_from_1s --start-date 2026-05-06 --end-date 2026-05-08
# → maga7/results/scanner_shadow_1s/signals_*.jsonl|.csv

# S3：Scanner → OMS dry-run（限价=fill_model 0.8，1s quote 闭合，无 IBKR）
python -m maga7.tools.run_oms_dry_run --start-date 2026-05-06 --end-date 2026-05-08 --compare-offline
# → maga7/results/oms_dry_1s_*/{orders_dry,fill_audit,trades,summary}.csv|json

# S4：独立 OMS stub（小仓 MAG7_MAX_QTY=1，fill_audit 流式追加，默认同源 1s 闭合）
MAG7_MAX_QTY=1 python -m maga7.tools.run_oms_live_stub \
  --start-date 2026-05-06 --end-date 2026-05-08 --compare-offline
# 可选：--redis  → xadd 映射后的 BUY/SELL 到 orch_trade_signals（不启动 QQQ OMS）
```

1s 与 1m parquet 来源不同时，信号时间可能有小偏差；对拍以 S0 stream parity（同 1m 源）为准，S2 验证的是 **ingest 管线**。S3/S4 与 offline `scheme=single` 在同信号上应 `compare ok`（逐笔 ret 一致）。

### S4 范围（重要）

- **做**：Mag7 独立 stub（`submit_buy/sell` + fill_audit + 可选 Redis payload）
- **不做**：挂整机 `run_live_exec_qqq` / `ExecutionEngineV8`（QQQ 单标的 + TFT 门控不匹配）
- Redis payload 用 `ScannerSignal.to_oms_exec_payload()`（顶层 `action=BUY|SELL`，`sig.meta.contract_id`），与审计用的 `to_orch_payload()`（CALL/PUT）不同

| Env | 默认 | 含义 |
|---|---|---|
| `MAG7_MAX_QTY` | 1 | 合约张数上限 |
| `MAG7_FILL_AUDIT_PATH` | results 下 `fill_audit_live.csv` | 流式 audit |
| `MAG7_REDIS_PUBLISH` | 0 | 1 时 xadd `orch_trade_signals` |
| `MAG7_OMS_MODE` | `MAG7_SHADOW` | audit `mode` 列 |

## OMS 对接约定

信号 payload（`ScannerSignal.to_orch_payload`）关键字段：

- `symbol`, `side` (CALL/PUT), `contract`, `rank`
- `meta.fill_frac`, `meta.tp_mult`, `meta.sl_mult`, `meta.hold_minutes`
- `source=maga7_mf10_top2`

OMS 侧：

1. 用 `contract` 订阅 1s quote；
2. 入场限价：`spread_interpolate(bid, ask, fill_frac, BUY)`；
3. 出场：TP/SL 按权利金倍数，或复用 `exit_rails` 的简化版；
4. **禁止**走 `select_entry_candidates_qqq_btc` 的单标的 TFT 排序。

## Dashboard

QQQ_BTC Dash → Board **Mag7** ：Run offline/parity、看 results、prepare data、看本文档。
