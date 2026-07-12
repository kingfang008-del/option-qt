#!/usr/bin/env bash
# Step-2: July W1 开平仓对拍（特征已对齐后的 rolling stream）
# 每日 truncate fill_audit，避免多轮 append 污染 OPEN→first CLOSE 配对。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMS_MOCK_IBKR=1 REDIS_STREAM_SIM=1 QQQ_BTC_LIVE=1 QQQ_BTC_FILL_AUDIT=1
export FCS_NORMALIZER_STATS_UPDATE_INTERVAL="${FCS_NORMALIZER_STATS_UPDATE_INTERVAL:-1}"
export FCS_TA_MONTH_ISOLATED="${FCS_TA_MONTH_ISOLATED:-1}"
# put_gate 对齐 +37.7%：用 5min quote_features_test.vix_level asof（非 1min VIXY z）
export QQQ_BTC_PUT_GATE_MODE="${QQQ_BTC_PUT_GATE_MODE:-feature5m}"
export QQQ_BTC_PUT_GATE_5M_FEATURE="${QQQ_BTC_PUT_GATE_5M_FEATURE:-$HOME/train_data/july_w1_v4_databento/quote_features_test/QQQ/regular/09:30-16:00/5min}"
export QQQ_BTC_REGIME_GOLD_1M="${QQQ_BTC_REGIME_GOLD_1M:-$HOME/train_data/july_w1_v4_databento/quote_features_test/QQQ/regular/09:30-16:00/1min}"
# FCS alpha label 滞后已等价 offline entry_delay=1；勿再叠加 OMS 延迟（会双重 +1min）
# 强制写 0，避免父 shell 残留 DELAY=1 被 ${VAR:-0} 继承
export EXECUTION_DELAY_BARS=0
export OMS_SIGNAL_DELAY_BARS=0
# REALTIME_DRY 成交用 0.775（oms_integration patch），非 mid
export BACKTEST_OPT_FILL_SPREAD_FRAC="${BACKTEST_OPT_FILL_SPREAD_FRAC:-0.775}"
unset FCS_FROZEN_NORM_PATH || true
export FCS_FROZEN_NORM_PATH=""

CKPT="${CKPT:-$(realpath checkpoint/checkpoints_qqq_ft56_julw1/best.pth)}"
OPT_ROOT="${OPT_ROOT:-/mnt/s990/data/v4_original_jul5/databento_july_w1/raw_1s}"
STOCK_ROOT="${STOCK_ROOT:-$HOME/train_data/spnq_train}"
# 对齐 +37.7%：seed=已归一化 bak June（与 quote_features_test 组装一致），非 raw
ROLLING_NORM_SEED="${ROLLING_NORM_SEED:-$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00/1min/2026-06.parquet}"
GREEK_ROOT="${GREEK_ROOT:-$HOME/train_data/july_w1_v4_databento/quote_options_day_iv}"
MONTHLY_IV_ROOT="${MONTHLY_IV_ROOT:-$HOME/train_data/july_w1_v4_databento/quote_options_monthly_iv}"
OUT_DIR="${OUT_DIR:-$REPO/qqq_btc/results/july_w1_ft56_4c_stream_rolling}"
FCS_WAIT="${FCS_WAIT:-60}"
SPEED="${SPEED:-inf}"
POS_FRAC="${POS_FRAC:-0.25}"
DAYS=(2026-07-01 2026-07-02 2026-07-06 2026-07-07 2026-07-08 2026-07-09)

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/step2_run.log"
: > "$LOG"

echo "=== [0] seed PG market_bars (June) ===" | tee -a "$LOG"
"$PY" -u "$REPO/qqq_btc/tools/seed_pg_warmup_bars.py" \
  --root "$STOCK_ROOT" --symbols QQQ,VIXY --start 2026-06-01 --end 2026-06-30 \
  2>&1 | tee -a "$LOG"

prev=""
for d in "${DAYS[@]}"; do
  ymd="${d//-/}"
  echo "" | tee -a "$LOG"
  echo "======== STREAM $d ========" | tee -a "$LOG"

  if [[ -n "$prev" ]]; then
    "$PY" -u "$REPO/qqq_btc/tools/seed_pg_warmup_bars.py" \
      --root "$STOCK_ROOT" --symbols QQQ,VIXY --start "$prev" --end "$prev" \
      2>&1 | tee -a "$LOG"
  fi

  audit="$HOME/quant_project/shadow/fill_audit_${ymd}.csv"
  rm -f "$audit" \
    "$HOME/quant_project/shadow/signals_${d}.csv" \
    "$HOME/quant_project/shadow/se_alpha_${d}.csv"
  # 确保空文件头，避免旧行残留
  : > "$audit"

  export QQQ_BTC_FILL_AUDIT_PATH="$audit"
  export QQQ_BTC_VIXY_SEED_BEFORE="$ymd"
  export FCS_MINUTE_OPTION_MONTHLY_IV_ROOT="$MONTHLY_IV_ROOT"

  "$PY" -u "$REPO/qqq_btc/tools/run_qqq_btc_redis_sim.py" \
    --date "$ymd" --source raw \
    --option-root "$OPT_ROOT" \
    --checkpoint "$CKPT" \
    --deep-warmup --no-frozen-norm \
    --rolling-norm-seed "$ROLLING_NORM_SEED" \
    --greek-parity --greek-root "$GREEK_ROOT" \
    --speed "$SPEED" --fcs-wait "$FCS_WAIT" \
    2>&1 | tee -a "$LOG" | tee "$OUT_DIR/stream_${d}_step2.log"

  cp -f "$audit" "$OUT_DIR/fill_audit_${ymd}.csv"
  [[ -f "$HOME/quant_project/shadow/signals_${d}.csv" ]] \
    && cp -f "$HOME/quant_project/shadow/signals_${d}.csv" "$OUT_DIR/" || true
  prev="$d"
done

echo "" | tee -a "$LOG"
echo "======== AGGREGATE @${POS_FRAC} + vs offline ========" | tee -a "$LOG"
export OUT_DIR CKPT OPT_ROOT POS_FRAC
"$PY" - <<'PY' | tee -a "$LOG" | tee "$OUT_DIR/step2_summary.txt"
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

out = Path(os.environ["OUT_DIR"])
pos = float(os.environ.get("POS_FRAC", "0.25"))
days = ["2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"]
day_set = set(pd.Timestamp(d).date() for d in days)

off = pd.read_parquet("qqq_btc/results/ft56_julw1_with_vix/replay_trades.parquet")
off["entry_ny"] = pd.to_datetime(off["entry_ts"], utc=True).dt.tz_convert("America/New_York")
off["exit_ny"] = pd.to_datetime(off["exit_ts"], utc=True).dt.tz_convert("America/New_York")
off["entry_sb"] = (off["entry_ny"].dt.hour * 60 + off["entry_ny"].dt.minute) - (9 * 60 + 30)
off["exit_sb"] = (off["exit_ny"].dt.hour * 60 + off["exit_ny"].dt.minute) - (9 * 60 + 30)
off["date"] = off["entry_ny"].dt.date.astype(str)

frames = []
for d in days:
    fp = out / f"fill_audit_{d.replace('-', '')}.csv"
    if fp.exists() and fp.stat().st_size > 0:
        try:
            frames.append(pd.read_csv(fp))
        except Exception:
            pass
df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
trades = []
if not df.empty:
    df["action"] = df["action"].astype(str).str.upper()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts", kind="mergesort")
    df["ny_date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.date
    open_row = None
    for _, r in df.iterrows():
        if r["action"] == "OPEN":
            open_row = r
            continue
        if r["action"] == "CLOSE" and open_row is not None:
            d = open_row["ny_date"]
            if d in day_set:
                trades.append({
                    "date": str(d),
                    "leg": str(open_row.get("leg", "")),
                    "entry_sb": float(pd.to_numeric(open_row.get("session_bar"), errors="coerce") or np.nan),
                    "exit_sb": float(pd.to_numeric(r.get("session_bar"), errors="coerce") or np.nan),
                    "entry_px": float(pd.to_numeric(open_row.get("fill_px"), errors="coerce") or np.nan),
                    "exit_px": float(pd.to_numeric(r.get("fill_px"), errors="coerce") or np.nan),
                    "reason": str(r.get("exit_reason", "")),
                    "net_return": float(pd.to_numeric(r.get("net_return"), errors="coerce") or 0.0),
                })
            open_row = None

eq = 1.0
for t in trades:
    eq *= 1.0 + pos * float(t["net_return"])
rets = [t["net_return"] for t in trades]
by_leg = {}
for t in trades:
    by_leg[t["leg"]] = by_leg.get(t["leg"], 0) + 1

# day-level open/close align vs offline
day_cmp = []
for d in days:
    orows = off[off["date"] == d]
    srows = [t for t in trades if t["date"] == d]
    item = {"date": d, "offline_n": len(orows), "stream_n": len(srows), "pairs": []}
    n = min(len(orows), len(srows))
    for i in range(n):
        o = orows.iloc[i]
        s = srows[i]
        item["pairs"].append({
            "offline": {
                "leg": o["leg"], "entry_sb": int(o["entry_sb"]), "exit_sb": int(o["exit_sb"]),
                "net_return": float(o["net_return"]), "exit_reason": o["exit_reason"],
            },
            "stream": {
                "leg": s["leg"], "entry_sb": int(s["entry_sb"]), "exit_sb": int(s["exit_sb"]),
                "net_return": float(s["net_return"]), "exit_reason": s["reason"],
            },
            "entry_match": (o["leg"] == s["leg"] and int(o["entry_sb"]) == int(s["entry_sb"])),
            "exit_sb_delta": int(s["exit_sb"]) - int(o["exit_sb"]),
            "net_delta": float(s["net_return"]) - float(o["net_return"]),
        })
    day_cmp.append(item)

offline = 0.3767942218288456
summary = {
    "mode": "step2_open_close_rolling_stream",
    "checkpoint": os.environ["CKPT"],
    "position_frac": pos,
    "trades": len(trades),
    "acct25": float(eq - 1.0),
    "hit_rate": float(np.mean([x > 0 for x in rets])) if rets else None,
    "trades_by_leg": by_leg,
    "offline_baseline_acct25": offline,
    "delta_pp_vs_offline": float(eq - 1.0 - offline),
    "trades_detail": trades,
    "day_compare": day_cmp,
}
(out / "step2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print()
print(f"STREAM @25%: {summary['acct25']*100:+.2f}%  trades={len(trades)} legs={by_leg}")
print(f"OFFLINE:     +37.68%")
print(f"delta:       {summary['delta_pp_vs_offline']*100:+.2f} pp")
PY

echo "done → $OUT_DIR/step2_summary.json"
