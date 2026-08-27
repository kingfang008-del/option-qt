# Smooth + Impulse Stock Sleeve (research)

Own-path trend start (not 10:30 MF window, not cross-section #1) executed as **stocks**.

## Pipeline

```
smooth launch (10m grind)  ─┐
                            ├─ first per symbol/dir → ≤2 names/day → stock
impulse launch (5m ≥40bp)  ─┘
                                ↓
              exit: trail giveback / smooth break / time / EOD
```

## Code

- `maga7/common/smooth_trend.py`
- `python -m maga7.tools.run_smooth_impulse_stock_replay`
- Results: `/mnt/s990/data/maga7/results/research_smooth_impulse_stock_may_jul/`

## May–Jul snapshot

| Variant | Ret | MaxDD | Note |
|---------|----:|------:|------|
| trail 40bp UP+DN | −1.6% | −3.1% | over-chopped |
| **UP trail 120bp / 180m** | **+1.7%** | **−3.2%** | practical candidate |

Detection coverage was high; PnL is modest and **exit-sensitive**. Prefer stocks/weeklies over 0DTE for grind legs.

## QQQ × 0DTE/2DTE follow-up

Tool: `python -m maga7.tools.run_smooth_qqq_weekly_ablation`  
Results: `/mnt/s990/data/maga7/results/research_smooth_qqq_weekly_may_jul/`

- QQQ hard filter hurts stock grind PnL.
- 0DTE amplifies with ~−85% MaxDD; 2DTE better but still large DD.
- Prefer stock core; 2DTE only as small satellite without QQQ hard gate.
