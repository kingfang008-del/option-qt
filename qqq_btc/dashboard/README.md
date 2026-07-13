# qqq_btc dashboard

`qqq_btc_dash.py` is a focused, read-only Streamlit dashboard for the QQQ_BTC
path. It replaces the old scattered dashboard workflow with one page centered
on the current model contract:

- QQQ 0DTE single-symbol live path
- `qqq_btc.qqq.config.FILL_MODEL` as the displayed fill assumption
- Redis probes for FCS, Signal, OMS, and trade logs
- OMS live position projection
- `fill_audit.csv` parity summary and exit-reason distribution
- runbook commands for the qqq_btc live tools

Run:

```bash
python qqq_btc/tools/run_dashboard_qqq.py
```

or:

```bash
streamlit run qqq_btc/dashboard/qqq_btc_dash.py --server.port 8502
```

Useful environment variables:

- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`
- `QQQ_BTC_DASH_PORT`, `QQQ_BTC_DASH_HOST`
- `QQQ_BTC_FILL_AUDIT_PATH`
- `QQQ_BTC_LIVE`

The dashboard does not write trading controls or Redis state.
