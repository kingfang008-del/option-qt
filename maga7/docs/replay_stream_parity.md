# maga7 mf10 Top2 — replay & stream parity

## Offline

`maga7.common.replay.run_offline_replay`：

1. **因果对拍默认**：从 `/mnt/s990/data/raw_1s/stocks` 聚合 1s→1m → mf 特征（`stock_1s`）  
   研究加速才允许读 `spnq_train` 1m 缓存（`cache_1m`，不作放行证据）
2. TopK Rule-A（+ 可选 m5 复入 / circuit）  
3. 选约（`entry_contract`）：`day_lock` | `open_lock` | `signal_atm`；可选 `clear_otm_ban_0dte_pct`  
4. Regime 闸门（对拍时 QQQ 亦从股票 1s 聚合）  
5. 期权 **仅** 1s quote + `FillSpec(frac=0.8)` 模拟 TP/SL/超时  

共享加载：`maga7.common.stock_1s.build_stock_by_from_1s`。  
预聚合 1m 路径用 `bar_availability_delay_seconds=60` 映射 `decision_ts`；
1s 实时聚合路径用真实 `available_ts`，禁止双重延迟（见 `current_architecture.md`）。

## Stream

`maga7.common.stream_engine.StreamEngine`：

- `StreamSignalState` 因果维护 streak/mf  
- 当日最早 K 个标的入选 TopK  
- **同一套** `ContractBooks` / `resolve_entry_contract` / `Mag7RegimeGate`  
- 成交与 offline 共用 `simulate_trade`  

## Live scanner

`maga7.live.scanner.Mag7Scanner` 同样走 `entry_contract`（含 open_lock + clear_otm），OMS 只消费 `sig.contract`。

## Parity

`run_stream_parity.py` 对比 `(date,symbol,dir,n_in_day)` 与 `ret` / `size_frac` / `reason`；不一致 exit 2。

**当前因果基线对拍（股票只用 1s，默认 `--stock-source stock_1s`）**：

```bash
python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json \
  --scheme single \
  --start-date 2026-07-01 --end-date 2026-07-13 \
  --stock-source stock_1s \
  --tag parity_peer3_jul_stock1s
```

Scanner / OMS 秒级路径（同 profile）：

```bash
python -m maga7.tools.run_replay_stock_1s \
  --profile maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json \
  --scheme single \
  --start-date 2026-07-01 --end-date 2026-07-13 \
  --tag replay_stock1s_peer3_jul
```

**已通过（2026-07-17）**：`results/parity_peer3_jul_stock1s/`  
三路一致（offline / stream / scanner）：**11 笔**，`only_*=0`，`ret` 差 = 0，账户收益 **+21.32%**。  
股票源：`/mnt/s990/data/raw_1s/stocks`；期权源：`quote_1s_root`。未使用 `spnq_train` 1m。  
数据缺口（不阻断对拍）：全员缺 `2026-07-03`；AMD 缺 `07-06/07`；VIXY 1s 几乎缺失（peer3 仅 QQQ align）。

### L1 Watchdog 流式对拍（2026-07-18）

Profile：`…_peer3_watchdog_v1`（Hunter off），`--stock-source stock_1s`。

| tag | period | n | ok |
|-----|--------|---|----|
| `parity_l1_watchdog_20260501_15` | 05-01..05-15 | 13 | true |
| `parity_l1_watchdog_20260528` | 05-28 | 1 | true |
| `parity_l1_watchdog_20260714_17` | 07-14..07-17 | 4 | true |

`only_*=0`，`ret` / `size_frac` / `reason` 差 = 0。  
修复：`Mag7RegimeGate`/`regime_gate_from_1s` 对 `regime` cfg deepcopy（防 Watchdog overlay 污染 profile）；stream 补齐 `mf_idio` 仓位缩放与 `loss_streak`。  
### L2 Hunt 流式对拍（2026-07-18）

`stream_engine` 已注入 Hunt。Profile：`…_watchdog_hunter_washout_reclaim_v1`。

| tag | period | n / hunt | ok |
|-----|--------|----------|----|
| `parity_l2_hunter_smoke_20260507_15` | 05-07..15 | 10 / 2 | true |
| `parity_l2_hunter_washout_reclaim_20260501_0717` | 05-01..07-17 | **60 / 12** | **true** |

升线总闸见 [`l2_hunter_validation_gates.md`](l2_hunter_validation_gates.md)（对拍过 ≠ 进基线）。  

### S1 research_baseline 对拍（2026-07-23）

Profile：`…_extend_mtm_full_day_peer3_v1`（含 soft `stock_path_confirm`）。  
指纹：`057410a6…c997df` · revision `2026-07-23_s1_path_soft`。

| tag | period | n | ok | note |
|-----|--------|--:|:--:|------|
| `parity_s1_research_baseline_jan_mar_stock1s` | 01-02..03-31 | 54 | true | offline↔stream |
| `parity_s1_research_baseline_apr_jul_stock1s` | 04-01..07-21 | 79 | true | offline↔stream |
| `parity_s1_fix_tox_hunt_20260701_21` | 07-01..07-21 | 15 | true | offline↔stream；+ scanner 三路 |

工程修复（同日）：
- `stream_engine`：接线 `trade_toxic` + `trade_path`（否则 7/01 TSLA 退出对不齐）
- `scanner`：Hunt 排程按**当日** `hunt_budget_remaining`（勿用累计 `n_hunt_emitted`）
- `stock_path_confirm_ok(..., asof_ts=)` + scanner `pending_path`（实盘因果等待）

双窗收益验收（cache offline PRE vs S1）：`KEEP_S1_RESEARCH_BASELINE` — 见 [`research_full_day_peer3_baseline.md`](research_full_day_peer3_baseline.md) · 汇总包 `results/s1_research_baseline_offline_pack_20260723/`。

Scanner 对拍时挂 `scanner.stock_by`，`peer_align` 与 offline 同用 `feature_ts` asof；纯 IBKR live 无 stock_by 时退回各标的 live mf（秒级完成时刻可能差 1–2s）。

**注意**：G2 是进程内 offline ↔ stream / scanner 规则对拍（股票 1s 聚合 + 期权 1s quote），
**不是** Redis 墙钟回放。真·Redis 见 `scanner_oms_integration.md` S5。

```bash
# 旧：开盘锁对拍（cache_1m 研究口径，勿当 stock-1s 放行）
python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_lock_research_v1.json \
  --scheme m5_circuit --stock-source cache_1m --tag parity_open_lock_clear_otm_jan_jul
```

已通过：`results/parity_open_lock_clear_otm_jan_jul`（145 笔，ret 差 = 0）。

```bash
# 临时生产：开盘阶梯 OTM5 + only_win + concurrent p20 + mf_flip
python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json \
  --scheme m5_circuit \
  --tag parity_open_ladder_otm5_mf_flip_p20_jan_jul
```

**历史口径已通过（2026-07-16）**：
`results/parity_open_ladder_otm5_mf_flip_p20_jan_jul`，247 笔，
`only_*=0`，`ret` / `size_frac` / `reason` 差 = 0。该产物早于当前时钟/指纹，
只保留作研究记录，不作为当前 G2 放行证据。

冒烟：`parity_open_ladder_otm5_mf_flip_p20_may_smoke`（5/1–5/15，37 笔，同样全一致）。

### Scanner / OMS（S3/S4）

```bash
# 默认挂 p20 production；m5 交错 fill（only_win 回写）
python -m maga7.tools.run_oms_dry_run \
  --start-date 2026-05-01 --end-date 2026-05-15 --ingest 1m --compare-offline

python -m maga7.tools.run_oms_live_stub \
  --start-date 2026-05-01 --end-date 2026-05-15 --ingest 1m --compare-offline
```

`ingest=1m` 用于快速对齐；`ingest=1s` 直接从统一股票秒级事实源验证 live 聚合。
两者必须对齐 `feature_ts` 和 `decision_ts`；如有差异，应视为聚合/缺秒/派生缓存
问题，而不是用标签命名解释。

完整架构和时钟契约见 [`current_architecture.md`](current_architecture.md)。
