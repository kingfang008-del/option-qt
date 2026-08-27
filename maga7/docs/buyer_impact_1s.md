# Buyer Impact Windows — 1s Stock Validation

**Tag:** `research_buyer_impact_1s`  
**Tool:** `maga7/tools/scan_buyer_impact_1s.py`  
**Artifacts:** `/mnt/s990/data/maga7/results/research_buyer_impact_1s/`

## Thesis

Option sellers win most of the clock. As a buyer you only need the rare **impact** windows (~10% intuition). Before buying L2/OBI, ask: can **causal stock 1s** features mark those windows?

## Protocol

- Window: AM 09:30–11:30, 60 session days, stride 15s → **122,061** probes
- Features at `t` (causal): `|ret_15/30|`, `vol_z`, `volume_ratio_60`, `mf100`, composite `impact_score`
- Label: ATM open-lock option oracle in **stock-move direction**, horizons 30/60/120s, slip 1%
- Buyer-good: `oracle_ret >= thr` (10% / 15% / 25% / 40%)
- Gates: rare impact percentiles vs FO≥0.8%; report hit_rate / lift / recall / frac_time

## Verdict

**Partial PASS on concentration, FAIL as a trade engine.**

1. **Rarity matches the framing.** Mild buyer-good (~+15% in 60s) is ~8% of clock; +25%/60s is ~2.4%; extreme +40%/30s is ~0.1%. The “~10%” story is about mild goods, not jackpots.
2. **1s impact does concentrate extremes.** For +40%/30s goods, median impact percentile ≈ **0.97**, **46.5%** sit in top-2% impact time. Lift of `impact_p98` / `volr≥2` reaches **~10–23×** on those rare labels.
3. **FO@0.8% does not mark buyer-good.** At H=60 / thr=25%, FO covers **33%** of clock with lift **0.96** (worse than base). Good events’ `frac_FO08` stays ~32% even as impact tops concentrate — FO is a time sponge, not an impact filter.
4. **Precision still too low to trade.** Best rare gate at H=60 / thr=25%: `abs_ret30≥20bp+volr15` → hit_rate **10.9%**, lift **4.6×**, only **1.2%** of time. Ranking signal exists; absolute hit rate is not a sleeve.

## Focus table (H=60, oracle≥25%, base≈2.37%)

| gate | frac_time | hit_rate | lift | recall |
|------|-----------|----------|------|--------|
| abs_ret30≥20bp+volr15 | 1.2% | 10.9% | 4.61 | 5.5% |
| impact_p98 | 2.0% | 10.4% | 4.39 | 8.8% |
| volr≥2 | 2.1% | 10.2% | 4.31 | 8.9% |
| impact_p95 | 5.0% | 6.6% | 2.77 | 13.9% |
| FO≥0.8% | 33.4% | 2.3% | **0.96** | 32.0% |

## Good-event location (where were the wins?)

| H | thr | frac_good | p50 impact pct | in top-2% | under FO08 |
|---|-----|-----------|----------------|-----------|------------|
| 30 | 25% | 0.68% | 0.77 | 22.7% | 28.7% |
| 30 | 40% | 0.12% | 0.97 | 46.5% | 20.8% |
| 60 | 15% | 8.3% | 0.56 | 5.0% | 35.3% |
| 60 | 25% | 2.4% | 0.60 | 8.8% | 32.0% |

Mild goods are **not** mostly in the top few % of impact — only the extreme tail is. That explains why FO/pocket grids keep failing: they chase mid-frequency regimes, not the rare bursts.

## Implications

- Stop expanding FO@0.8 / MF grids as “buyer edge finders.” They sample the wrong mass of the day.
- Stock 1s **rare** gates (`impact_p98`, `volr≥2`, short-ret ∩ vol) are valid **hard filters / research priors**, not yet entry engines (hit≈10% on +25%/60s).
- Next precision step is **L2/OBI** (or print imbalance closer to the book), not another 1s MF grid — 1s already shows concentration ceiling for stock-only features on mild goods.

## Reproduce

```bash
PYTHONPATH=. python -m maga7.tools.scan_buyer_impact_1s \
  --tag research_buyer_impact_1s --stride-sec 15
```
