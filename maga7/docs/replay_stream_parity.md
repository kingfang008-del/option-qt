# maga7 mf10 Top2 — replay & stream parity

## Offline

`maga7.common.replay.run_offline_replay`：

1. 加载 `spnq_train` 1m → mf 特征  
2. TopK Rule-A（+ 可选 m5 复入 / circuit）  
3. 选约（`entry_contract`）：`day_lock` | `open_lock` | `signal_atm`；可选 `clear_otm_ban_0dte_pct`  
4. Regime 闸门  
5. 1s / day_iv quote + `FillSpec(frac=0.8)` 模拟 TP/SL/超时  

## Stream

`maga7.common.stream_engine.StreamEngine`：

- `StreamSignalState` 因果维护 streak/mf  
- 当日最早 K 个标的入选 TopK  
- **同一套** `ContractBooks` / `resolve_entry_contract` / `Mag7RegimeGate`  
- 成交与 offline 共用 `simulate_trade`  

## Live scanner

`maga7.live.scanner.Mag7Scanner` 同样走 `entry_contract`（含 open_lock + clear_otm），OMS 只消费 `sig.contract`。

## Parity

`run_stream_parity.py` 对比 `(date,symbol,dir,n_in_day)` 与 `ret`；不一致 exit 2。

```bash
# 生产日锁
python -m maga7.tools.run_stream_parity --scheme m5_circuit --tag parity_day_lock

# 实盘因果：开盘锁 + 明显 OTM 禁 0DTE
python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_lock_research_v1.json \
  --scheme m5_circuit --tag parity_open_lock_clear_otm_jan_jul
```

已通过：`results/parity_open_lock_clear_otm_jan_jul`（145 笔，ret 差 = 0）。
