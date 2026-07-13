#!/usr/bin/env bash
# 诚实特征栈上的门控消融：只改 put_gate / regime_gold，不动 greek-parity。
# 基线（vixy_z + regime off）已有 july_w1_ft56_honest_3gate_week_g12pass → +10%。
#
# 用法:
#   bash qqq_btc/tools/ablate_honest_stream_gates.sh
#   ONLY=put5m_honest bash qqq_btc/tools/ablate_honest_stream_gates.sh
#   DAYS="2026-07-01 2026-07-02" bash ...
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

CKPT="${CKPT:-$(realpath checkpoint/checkpoints_qqq_ft56_julw1/best.pth)}"
OPT_ROOT="${OPT_ROOT:-/mnt/s990/data/v4_original_jul5/databento_july_w1_openwin/raw_1s}"
STOCK_ROOT="${STOCK_ROOT:-$HOME/train_data/spnq_train}"
HONEST_FEAT_ROOT="${HONEST_FEAT_ROOT:-$HOME/train_data/july_w1_v4_honest_openwin}"
GREEK_ROOT="${GREEK_ROOT:-$HONEST_FEAT_ROOT/quote_options_day_iv}"
FROZEN_NORM="${FROZEN_NORM:-$REPO/qqq_btc/CONFIG/frozen_norm_qqq_daily.npz}"
PUT5M_HONEST="${PUT5M_HONEST:-$HONEST_FEAT_ROOT/quote_features_test/QQQ/regular/09:30-16:00/5min}"
REGIME_HONEST="${REGIME_HONEST:-$HONEST_FEAT_ROOT/quote_features_test/QQQ/regular/09:30-16:00/1min}"
BASELINE_DIR="${BASELINE_DIR:-$REPO/qqq_btc/results/july_w1_ft56_honest_3gate_week_g12pass}"
OUT_ROOT="${OUT_ROOT:-$REPO/qqq_btc/results/july_w1_ft56_honest_gate_ablation}"
POS_FRAC="${POS_FRAC:-0.25}"
SPEED="${SPEED:-inf}"
FCS_WAIT="${FCS_WAIT:-45}"

if [[ -n "${DAYS:-}" ]]; then
  # shellcheck disable=SC2206
  DAY_ARR=($DAYS)
else
  DAY_ARR=(2026-07-01 2026-07-02 2026-07-06 2026-07-07 2026-07-08 2026-07-09 2026-07-10)
fi

# name|put_mode|put5m_path|regime_path
VARIANTS=(
  "put5m_honest|feature5m|$PUT5M_HONEST|0"
  "regime_honest|vixy_z||$REGIME_HONEST"
  "put5m_regime_honest|feature5m|$PUT5M_HONEST|$REGIME_HONEST"
)

_apply_honest_fcs_env() {
  export OMS_MOCK_IBKR=1
  export REDIS_STREAM_SIM=1
  export QQQ_BTC_LIVE=1
  export QQQ_BTC_FILL_AUDIT=1
  export FCS_NORMALIZER_STATS_UPDATE_INTERVAL=1
  export EXECUTION_DELAY_BARS=0
  export OMS_SIGNAL_DELAY_BARS=0
  export BACKTEST_OPT_FILL_SPREAD_FRAC="${BACKTEST_OPT_FILL_SPREAD_FRAC:-0.775}"
  export FCS_FROZEN_NORM_PATH="$FROZEN_NORM"
  export RECALC_GREEKS=1
  export FCS_FORCE_RECALC_GREEKS=1
  export FCS_DEBUG_RAW=1
  export FCS_TA_MONTH_ISOLATED=1
  export FCS_OPTION_T_LABEL=end
  export FCS_IV_PRICE_MODE=close
  export SLOW_FEATURE_CONFIG="${SLOW_FEATURE_CONFIG:-$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json}"
  unset FCS_MINUTE_PARITY_INJECT || true
  unset GREEK_PARITY_MODE || true
}

_run_one() {
  local name="$1" put_mode="$2" put5m="$3" regime="$4"
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"
  local log="$out/run.log"
  : > "$log"

  echo "======== ABLATE $name ========" | tee -a "$log"
  _apply_honest_fcs_env
  export QQQ_BTC_PUT_GATE_MODE="$put_mode"
  if [[ "$put_mode" == "feature5m" && -n "$put5m" ]]; then
    export QQQ_BTC_PUT_GATE_5M_FEATURE="$put5m"
  else
    unset QQQ_BTC_PUT_GATE_5M_FEATURE || true
  fi
  if [[ -n "$regime" && "$regime" != "0" ]]; then
    export QQQ_BTC_REGIME_GOLD_1M="$regime"
  else
    export QQQ_BTC_REGIME_GOLD_1M=0
  fi

  cat > "$out/manifest.json" <<EOF
{
  "mode": "honest_gate_ablation",
  "name": "$name",
  "put_gate": "$put_mode",
  "put5m": "${put5m:-}",
  "regime_gold": "$regime",
  "greek_parity": false,
  "baseline_ref": "$BASELINE_DIR",
  "checkpoint": "$CKPT",
  "days": [$(printf '"%s",' "${DAY_ARR[@]}" | sed 's/,$//')]
}
EOF

  pkill -9 -f 'feature_compute_service_v8|run_live_signal_qqq|run_live_exec_qqq|redis_fused_pitcher|run_qqq_btc_redis_sim' 2>/dev/null || true
  sleep 1

  echo "[0] seed June PG bars" | tee -a "$log"
  "$PY" -u "$REPO/qqq_btc/tools/seed_pg_warmup_bars.py" \
    --root "$STOCK_ROOT" --symbols QQQ,VIXY \
    --start 2026-06-01 --end 2026-06-30 \
    2>&1 | tee -a "$log"

  local prev=""
  for d in "${DAY_ARR[@]}"; do
    local ymd="${d//-/}"
    echo "" | tee -a "$log"
    echo "======== STREAM $d ($name) ========" | tee -a "$log"
    pkill -9 -f 'feature_compute_service_v8|run_live_signal_qqq|run_live_exec_qqq|redis_fused_pitcher' 2>/dev/null || true
    sleep 1
    if [[ -n "$prev" ]]; then
      "$PY" -u "$REPO/qqq_btc/tools/seed_pg_warmup_bars.py" \
        --root "$STOCK_ROOT" --symbols QQQ,VIXY --start "$prev" --end "$prev" \
        2>&1 | tee -a "$log"
    fi
    local audit="$HOME/quant_project/shadow/fill_audit_${ymd}.csv"
    rm -f "$audit" \
      "$HOME/quant_project/shadow/signals_${d}.csv" \
      "$HOME/quant_project/shadow/se_alpha_${d}.csv"
    export QQQ_BTC_FILL_AUDIT_PATH="$audit"
    export QQQ_BTC_VIXY_SEED_BEFORE="$ymd"

    "$PY" -u "$REPO/qqq_btc/tools/run_qqq_btc_redis_sim.py" \
      --date "$ymd" --source raw \
      --option-root "$OPT_ROOT" --greek-root "$GREEK_ROOT" \
      --checkpoint "$CKPT" --deep-warmup --frozen-norm "$FROZEN_NORM" \
      --speed "$SPEED" --fcs-wait "$FCS_WAIT" \
      2>&1 | tee -a "$log" | tee "$out/stream_${d}.log"

    [[ -f "$audit" ]] && cp -f "$audit" "$out/fill_audit_${ymd}.csv"
    [[ -f "$HOME/quant_project/shadow/signals_${d}.csv" ]] && cp -f "$HOME/quant_project/shadow/signals_${d}.csv" "$out/" || true
    prev="$d"
  done

  DAYS_CSV="$(IFS=,; echo "${DAY_ARR[*]}")"
  export OUT_DIR="$out" POS_FRAC DAYS_CSV NAME="$name"
  "$PY" - <<'PY' | tee -a "$log" | tee "$out/summary.txt"
import json, os
from pathlib import Path
import pandas as pd
import numpy as np

out = Path(os.environ["OUT_DIR"])
pos = float(os.environ.get("POS_FRAC", "0.25"))
days = [d.strip() for d in os.environ["DAYS_CSV"].split(",") if d.strip()]
day_set = set(pd.Timestamp(d).date() for d in days)
name = os.environ.get("NAME", out.name)

frames = []
for d in days:
    fp = out / f"fill_audit_{d.replace('-', '')}.csv"
    if fp.exists() and fp.stat().st_size > 0:
        try:
            frames.append(pd.read_csv(fp))
        except Exception:
            pass

trades = []
if frames:
    df = pd.concat(frames, ignore_index=True)
    df["action"] = df["action"].astype(str).str.upper()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts", kind="mergesort")
    df["ts_dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df["ny_date"] = df["ts_dt"].dt.tz_convert("America/New_York").dt.date
    open_row = None
    for _, r in df.iterrows():
        act = r["action"]
        if act == "OPEN":
            open_row = r
            continue
        if act == "CLOSE" and open_row is not None:
            d = open_row["ny_date"]
            if d in day_set:
                trades.append({
                    "date": str(d),
                    "leg": str(open_row.get("leg", "")),
                    "net_return": float(pd.to_numeric(r.get("net_return"), errors="coerce") or 0.0),
                    "reason": str(r.get("exit_reason", "")),
                })
            open_row = None

eq = 1.0
for t in trades:
    eq *= 1.0 + pos * float(t["net_return"])
rets = [t["net_return"] for t in trades]
by_leg = {}
for t in trades:
    by_leg[t["leg"]] = by_leg.get(t["leg"], 0) + 1
by_day = {}
for t in trades:
    by_day.setdefault(t["date"], []).append(float(t["net_return"]))
by_day_acct = {}
for d, rs in by_day.items():
    e = 1.0
    for r in rs:
        e *= 1.0 + pos * r
    by_day_acct[d] = {"n": len(rs), "acct25": float(e - 1.0), "sum_net": float(sum(rs))}

summary = {
    "name": name,
    "position_frac": pos,
    "trades": len(trades),
    "acct25": float(eq - 1.0),
    "hit_rate": float(np.mean([x > 0 for x in rets])) if rets else None,
    "trades_by_leg": by_leg,
    "by_day": by_day_acct,
    "trades_detail": trades,
}
(out / "stream_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print(f"ABLATION {name}: {summary['acct25']*100:+.2f}% trades={len(trades)} legs={by_leg}")
PY
}

mkdir -p "$OUT_ROOT"
ONLY="${ONLY:-}"

# copy baseline pointer
if [[ -f "$BASELINE_DIR/stream_summary_paired.json" ]]; then
  mkdir -p "$OUT_ROOT/baseline_honest_vixy_z"
  cp -f "$BASELINE_DIR/stream_summary_paired.json" "$OUT_ROOT/baseline_honest_vixy_z/stream_summary.json"
  cp -f "$BASELINE_DIR/manifest.json" "$OUT_ROOT/baseline_honest_vixy_z/manifest.json" 2>/dev/null || true
fi

for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r name put_mode put5m regime <<<"$spec"
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then
    continue
  fi
  _run_one "$name" "$put_mode" "$put5m" "$regime"
done

# compare table
export OUT_ROOT BASELINE_DIR POS_FRAC
"$PY" - <<'PY'
import json
from pathlib import Path
import os

root = Path(os.environ["OUT_ROOT"])
base = Path(os.environ["BASELINE_DIR"]) / "stream_summary_paired.json"
rows = []
if base.exists():
    b = json.loads(base.read_text())
    rows.append({
        "name": "baseline_honest_vixy_z",
        "acct25": b.get("acct25"),
        "trades": b.get("trades"),
        "legs": b.get("trades_by_leg"),
        "hit": b.get("hit_rate"),
    })
for p in sorted(root.glob("*/stream_summary.json")):
    d = json.loads(p.read_text())
    if p.parent.name == "baseline_honest_vixy_z":
        continue
    rows.append({
        "name": d.get("name") or p.parent.name,
        "acct25": d.get("acct25"),
        "trades": d.get("trades"),
        "legs": d.get("trades_by_leg"),
        "hit": d.get("hit_rate"),
    })
base_acct = rows[0]["acct25"] if rows else 0.0
print("\n=== GATE ABLATION @25% ===")
print(f"{'name':<28} {'acct25':>10} {'delta_pp':>10} {'trades':>7} legs")
for r in rows:
    acct = float(r["acct25"] or 0.0)
    delta = (acct - float(base_acct or 0.0)) * 100.0
    print(f"{r['name']:<28} {acct*100:+9.2f}% {delta:+9.2f} {r['trades'] or 0:7d} {r['legs']}")
out = {"position_frac": float(os.environ.get("POS_FRAC", "0.25")), "rows": rows, "offline_target_acct25": 0.5107873996354977}
(root / "ablation_compare.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
print(f"\nwrote {root / 'ablation_compare.json'}")
print(f"offline replay target: +51.08%")
PY

echo "done → $OUT_ROOT"
