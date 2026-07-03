# Phase 1 Bidirectional Audit Summary

- dates: `synthetic`
- mode: `synthetic`
- minutes: **390**
- tradable_ratio: **100.0%**
- oracle call/put share: **51.3%** / **47.9%**
- model call/put share: **27.9%** / **27.7%**
- side agreement (oracle vs model): **55.6%**
- missed put edge (mean ret when oracle=put, model≠put): **0.0154**
- missed call edge: **0.0139**

## By day_type

- `chop`: n=159 | oracle_put=44.7% | model_put=2.5% | agree=5.7%
- `dislocation`: n=3 | oracle_put=66.7% | model_put=0.0% | agree=0.0%
- `trend_down`: n=114 | oracle_put=100.0% | model_put=91.2% | agree=91.2%
- `trend_up`: n=114 | oracle_put=0.0% | model_put=0.0% | agree=91.2%
