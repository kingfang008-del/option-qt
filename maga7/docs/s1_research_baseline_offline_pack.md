# S1 soft path — research_baseline offline pack (2026-07-23)

**Profile:** `CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json`  
**Revision:** `2026-07-23_s1_path_soft`  
**Fingerprint:** `057410a6f1e58e47d4f38a7fea528f76a79fe6639f42d752d633b46b9fc997df`  
**Decision:** **`KEEP_S1_RESEARCH_BASELINE`**（生产 freeze / `default_profile` 仍不动）

## What landed

| Item | Setting |
|------|---------|
| Gate | `trade.stock_path_confirm` soft：thr_pos=+15bp / thr_neg=−30bp / max_wait=300s / **on_timeout=allow** / delay_on_pos=false / tod 10:30–14:00 |
| Live | `stock_path_confirm_ok(..., asof_ts=)` + scanner `pending_path` |
| Overlay **not** default | `WAVE_ABORT` UP-only（`…_peer3_s1_wa_up_v1`） |

## Dual-window accept (cache offline, PRE = S1 off)

Script: `tools/run_s1_research_baseline_accept.py`  
Results: `results/s1_research_baseline_accept_apr_jul_jan_mar_v1/`

| Window | PRE total_ret | S1 total_ret | keep | MaxDD PRE→S1 | path_ok / block |
|--------|-------------:|-------------:|-----:|--------------|----------------:|
| Strong Apr–Jul →07-21 | +4205% | **+4504%** | **1.07** | −16.1%→**−14.1%** | 73 / 9 |
| Weak Jan–Mar | +90.6% | **+97.7%** | 1.08 | −15.5%→**−15.3%** | 70 / 8 |
| July slice | +95.7% | +95.7% | **1.00** | same | 16 / 0 |

Pass rule: strong keep≥0.85 **and** (weak ret↑ or MaxDD↑) **and** july keep≥0.95 → all green for **S1**.

`S1_WA_UP` vs PRE: strong keep≈0.82 → **`RECONSIDER`** as global default；七月更好，仅 overlay。

## Stream / scanner parity (stock_1s)

See also [`replay_stream_parity.md`](replay_stream_parity.md) § S1.

| Tag | Period | n | ok |
|-----|--------|--:|:--:|
| `parity_s1_research_baseline_jan_mar_stock1s` | 01-02..03-31 | 54 | true |
| `parity_s1_research_baseline_apr_jul_stock1s` | 04-01..07-21 | 79 | true |
| `parity_s1_fix_tox_hunt_20260701_21` | 07-01..07-21 | 15 | true（+ scanner 三路） |

Fixes required for July three-way:
1. `stream_engine` wires `trade_toxic` + option `trade_path`
2. scanner Hunt reschedule uses **daily** `hunt_budget_remaining` (not cumulative `n_hunt_emitted`)

## Reproduce

```bash
# Dual-window accept (PRE / S1 / S1_WA_UP)
python -m maga7.tools.run_s1_research_baseline_accept

# Stream parity (example: strong window)
python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json \
  --scheme single --stock-source stock_1s \
  --start-date 2026-04-01 --end-date 2026-07-21 \
  --tag parity_s1_research_baseline_apr_jul_stock1s
```

## Not done here

- Production freeze / swap `catalog.default_profile`
- Live shadow trading day
- Full dual-window Mag7Scanner stock_1s replay（七月三路已覆盖）

汇总包：`/mnt/s990/data/maga7/results/s1_research_baseline_offline_pack_20260723/`
