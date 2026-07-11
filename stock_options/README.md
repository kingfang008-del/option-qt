# stock_options — single-stock option modeling path

`stock_options` is the independent path for NVDA/TSLA/MAG7-style equity option
models. It reuses stable primitives from `qqq_btc.common` where appropriate,
but keeps data profiles, anchors, thresholds, checkpoints, and replay reports
separate from the QQQ 0DTE path.

## Initial Scope

- Pilot symbols: `NVDA`, `TSLA`
- **Short-DTE path (current mainline)**: trading `dte∈{0,1,2}` after MAG7
  Mon/Wed expiries (~**2026-02** onward); Fri weekly existed earlier
- Legacy weekly-DTE path: Mon–Thu vs nearest Friday (`1..4DTE`); kept as
  secondary profile under `CONFIG/anchor_stock_weekly_dte.json`
- Do **not** copy QQQ 0DTE curated rules; framework shared, params independent

## Directory Layout

```text
stock_options/
  CONFIG/
    anchor_stock_weekly_dte.json
    symbol_map_stock.json
  common/
    weekly_dte_config.py
  nvda/
    config_weekly_dte.py
  tsla/
    config_weekly_dte.py
  tools/
    rebuild_weekly_dte_pipeline.py
  results/
```

## Design Rules

1. Do not import `qqq_btc.qqq.config` or QQQ 0DTE thresholds.
2. Keep fill/replay primitives shared through `qqq_btc.common` until a dedicated
   shared package is needed.
3. Validate every result by `symbol`, `weekday`, and `dte`; aggregate PnL is not
   enough.
4. Daily-expiry single-stock data can be added later as a separate profile after
   coverage and liquidity diagnostics pass.

