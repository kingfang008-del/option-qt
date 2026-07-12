#!/usr/bin/env bash
# July W1 · ft56 · 4约基线 · Redis 实时流（OMS MOCK）→ 汇总账户 25% 仓利润
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMS_MOCK_IBKR=1
export REDIS_STREAM_SIM=1
export QQQ_BTC_LIVE=1
export QQQ_BTC_FILL_AUDIT=1

CKPT="${CKPT:-$(realpath checkpoint/checkpoints_qqq_ft56_julw1/best.pth)}"
OPT_ROOT="${OPT_ROOT:-/mnt/s990/data/v4_original_jul5/databento_july_w1/raw_1s}"
PARQUET="${PARQUET:-$REPO/qqq_btc/results/ft56_julw1_frozen_daily/test_infer.parquet}"
OUT_DIR="${OUT_DIR:-$REPO/qqq_btc/results/july_w1_ft56_4c_stream_frozen}"
FROZEN_NORM="${FROZEN_NORM:-$REPO/qqq_btc/CONFIG/frozen_norm_qqq_daily.npz}"
FCS_WAIT="${FCS_WAIT:-45}"
SPEED="${SPEED:-inf}"
POS_FRAC="${POS_FRAC:-0.25}"

DAYS=(2026-07-01 2026-07-02 2026-07-06 2026-07-07 2026-07-08 2026-07-09)
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"
: > "$LOG"

echo "ckpt=$CKPT" | tee -a "$LOG"
echo "opt=$OPT_ROOT" | tee -a "$LOG"
echo "parquet=$PARQUET" | tee -a "$LOG"
echo "frozen=$FROZEN_NORM" | tee -a "$LOG"

prev=""
for d in "${DAYS[@]}"; do
  ymd="${d//-/}"
  echo "" | tee -a "$LOG"
  echo "======== STREAM $d ========" | tee -a "$LOG"
  # 杀掉残留栈，避免串流
  pkill -9 -f 'feature_compute_service_v8|run_live_signal_qqq|run_live_exec_qqq|redis_fused_pitcher' 2>/dev/null || true
  sleep 1

  audit="$HOME/quant_project/shadow/fill_audit_${ymd}.csv"
  rm -f "$audit" \
    "$HOME/quant_project/shadow/signals_${d}.csv" \
    "$HOME/quant_project/shadow/se_alpha_${d}.csv"
  export QQQ_BTC_FILL_AUDIT_PATH="$audit"
  export QQQ_BTC_VIXY_SEED_BEFORE="$ymd"

  warm_args=()
  if [[ -n "$prev" ]]; then
    warm_args=(--warmup-from-date "${prev//-/}")
  else
    warm_args=(--warmup-from-date "$ymd")
  fi

  "$PY" -u "$REPO/qqq_btc/tools/run_qqq_btc_redis_sim.py" \
    --date "$ymd" \
    --source raw \
    --option-root "$OPT_ROOT" \
    --checkpoint "$CKPT" \
    --frozen-norm "$FROZEN_NORM" \
    --speed "$SPEED" \
    --fcs-wait "$FCS_WAIT" \
    "${warm_args[@]}" \
    2>&1 | tee -a "$LOG" | tee "$OUT_DIR/stream_${d}.log"

  if [[ -f "$audit" ]]; then
    cp -f "$audit" "$OUT_DIR/fill_audit_${ymd}.csv"
  else
    echo "[warn] missing fill_audit $audit" | tee -a "$LOG"
  fi
  if [[ -f "$HOME/quant_project/shadow/signals_${d}.csv" ]]; then
    cp -f "$HOME/quant_project/shadow/signals_${d}.csv" "$OUT_DIR/"
  fi
  prev="$d"
done

echo "" | tee -a "$LOG"
echo "======== AGGREGATE @${POS_FRAC} ========" | tee -a "$LOG"
export OUT_DIR CKPT OPT_ROOT POS_FRAC
"$PY" - <<'PY' | tee -a "$LOG" | tee "$OUT_DIR/summary.txt"
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

out = Path(os.environ["OUT_DIR"])
ckpt = os.environ["CKPT"]
opt_root = os.environ["OPT_ROOT"]
pos = float(os.environ.get("POS_FRAC", "0.25"))
days = ["2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"]
rows = []
for d in days:
    ymd = d.replace("-", "")
    fp = out / f"fill_audit_{ymd}.csv"
    if not fp.exists() or fp.stat().st_size == 0:
        rows.append({"date": d, "n_close": 0, "sum_net": 0.0, "acct": 0.0, "closes": []})
        continue
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        rows.append({"date": d, "n_close": 0, "error": str(e), "closes": []})
        continue
    if df.empty or "action" not in df.columns:
        rows.append({"date": d, "n_close": 0, "sum_net": 0.0, "acct": 0.0, "closes": []})
        continue
    closes = df[df["action"].astype(str).str.upper() == "CLOSE"].copy()
    if not closes.empty:
        closes["ts_dt"] = pd.to_datetime(closes["ts"], unit="s", utc=True, errors="coerce")
        closes = closes.dropna(subset=["ts_dt"])
        target = pd.Timestamp(d).date()
        closes = closes[closes["ts_dt"].dt.tz_convert("America/New_York").dt.date == target]
    closes["net_return"] = pd.to_numeric(closes.get("net_return"), errors="coerce")
    closes["session_bar"] = pd.to_numeric(closes.get("session_bar"), errors="coerce")
    closes["leg"] = closes.get("leg", pd.Series(dtype=str)).astype(str)
    closes["reason"] = closes.get("exit_reason", pd.Series(dtype=str)).astype(str)
    closes = closes.sort_values(["session_bar", "ts"], kind="mergesort")
    eq = 1.0
    for r in closes["net_return"].fillna(0.0):
        eq *= 1.0 + pos * float(r)
    recs = closes[["session_bar", "leg", "reason", "net_return"]].to_dict("records")
    rows.append({
        "date": d,
        "n_close": int(len(closes)),
        "sum_net": float(closes["net_return"].fillna(0).sum()) if len(closes) else 0.0,
        "acct": float(eq - 1.0),
        "closes": recs,
        "model_frac_median": float(pd.to_numeric(closes.get("model_frac"), errors="coerce").median()) if len(closes) else None,
        "fill_frac_median": float(pd.to_numeric(closes.get("fill_spread_frac"), errors="coerce").median()) if len(closes) else None,
    })

eq = 1.0
n = 0
all_rets = []
for r in rows:
    for c in r.get("closes") or []:
        nr = c.get("net_return")
        if nr is None or (isinstance(nr, float) and not np.isfinite(nr)):
            continue
        eq *= 1.0 + pos * float(nr)
        all_rets.append(float(nr))
        n += 1
hit = float(np.mean([x > 0 for x in all_rets])) if all_rets else None
summary = {
    "mode": "redis_stream_ft56_4c_baseline",
    "checkpoint": ckpt,
    "option_root": opt_root,
    "position_frac": pos,
    "days": days,
    "trades": n,
    "acct25": float(eq - 1.0),
    "hit_rate": hit,
    "sum_net": float(sum(all_rets)) if all_rets else 0.0,
    "offline_baseline_acct25": 0.3767942218288456,
    "by_day": rows,
}
(out / "stream_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print()
print(f"STREAM ft56 4c @25%: {summary['acct25']*100:+.2f}%  trades={n}  hit={hit}")
print(f"OFFLINE baseline:     +37.68%")
print(f"delta vs offline:     {(summary['acct25']-0.3767942218288456)*100:+.2f} pp")
PY

echo "done → $OUT_DIR/stream_summary.json"
