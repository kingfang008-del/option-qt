# Second-level active ladder (research path)

**Status:** research only — does **not** mutate peer3 freeze / `extend_mtm` baseline.

## Why

0DTE/1DTE can print +20% in seconds. A 10m signal window + 30–45m hold is statistically OK on calm windows and **fails hard** on toxic-UP / mixed-wash days (e.g. 2026-07-20). Passive unmanaged mid-hold is not acceptable for this product.

## Path

Profile: `CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_sec_ladder_active_v1.json`

| Layer | Peer3 baseline | Sec-ladder v1 |
| --- | --- | --- |
| Signal | mf10 / streak8 | mf5 / streak5 (phase-1) |
| Hold | T30→T45 extend | hard `SEC_MAX` ≤ 300s |
| Exit | rails + toxic | stepped TP/SL + trail + stall + mf flip + toxic + Δ/ROI rails |
| Hunter | on | off (clean A/B) |

## Exit reasons

- `SL_LADDER{pct}` / `TP_LADDER{pct}` / `TRAIL_LADDER` / `PROFIT_STALL` / `SEC_MAX` / `MF_FLIP`
- Still honors outer `tp_mult`/`sl_mult` when `keep_outer_rails=true`
- Offline: `simulate_trade(..., exit_mode="ladder_active")`
- Live OMS: same rails on every quote tick

## Validation plan

1. Unit: `maga7/tests/test_ladder_active.py`
2. Ablation: same entries as peer3 with **only** exit_mode swapped (hold exit fairness), then full profile replay May–Jul + Feb–Apr
3. Stress day: fused replay / session `2026-07-20` — expect shorter losses, no 30m bleed
4. Promote only if dual-window scoreboard beats peer3 on maxDD and toxic-UP days without gutting strong days

## Grid findings (2026-07-20 night)

Tool: `python -m maga7.tools.run_sec_ladder_grid`

| Variant | May–Jul total_ret | maxDD | notes |
| --- | --- | --- | --- |
| peer3_extend | +1856% | −5.4% | control |
| ladder_tight always | −8% | −18% | kills winners |
| ladder_loose_15m always | +21% | −16% | still too much MF_FLIP |
| **ladder_cond_loose_wash** | **+515%** | **−6.2%** | 23 toxic days ladder / 30 extend |
| **ladder_cond_mid_wash** | **+484%** | **−5.5%** | same split |

**Promotion candidate:** `when=mixed_wash_up` conditional ladder (keep peer3 extend on clean days).

Config knobs: `trade.ladder_active.when` (`always` \| `mixed_wash_up`), `trade.ladder_fallback_exit_mode` (default `hold_extend`).

## Non-goals (yet)

- True 1s Rule-A scanner (phase-2)
- Partial scale-out ladders (full flatten only in v1)
- Touching frozen peer3 JSON
