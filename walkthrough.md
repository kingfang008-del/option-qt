# Backtest Restoration Walkthrough - V0 Strategy

We have successfully restored the transaction flow for the V0 strategy. The "Zero Trades" issue was traced back to data starvation and state management locks in the orchestrator and replay script.

## Changes Implemented

### 1. Data Pipeline Repair (Universal Parser)
- **Component**: `system_orchestrator_v8.py`
- **Fix**: Implemented a robust JSON parser for `opt_snapshot` that supports both legacy list formats (indices 0, 2) and newer flat dictionary formats. This ensures Bid/Ask prices are extracted regardless of the data generation.

### 2. Market Vision Restoration (Index Passthrough)
- **Component**: `s4_run_historical_replay_stable.py`
- **Fix**: Modified the replay script to explicitly fetch `SPY` and `QQQ` prices from the market bars table. Previously, the inner-join logic filtered out indices, blinding the strategy's `Index Guard` and causing it to block all entries.

### 3. State Management (Spread Divergence Fix)
- **Component**: `system_orchestrator_v8.py`
- **Fix**: Ensured `last_spread_pct` updates every minute even without a position. This prevents the first trade of the day from being blocked by an artificially large spread divergence.

## Results Summary (2026-01-08)

| Metric | Value |
| :--- | :--- |
| **Total Trades** | 76 |
| **Win Rate** | 43.42% |
| **Net PnL** | $-6,387.00 (-12.77%) |
| **Commission** | $1,716.00 |

> [!NOTE]
> While the trade count is restored, the PnL is negative. This is consistent with the "Original V0" behavior which is highly aggressive and susceptible to noisy Alpha signals (IC was -0.035 for this day).

## Next Steps
- Consider adjusting the `VOL_MIN_Z` or `ALPHA_ENTRY_THRESHOLD` to filter out the high-frequency losers observed in the 01-08 counter-trend stats.
