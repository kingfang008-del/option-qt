#!/usr/bin/env bash
# 【开卷诊断】F56 July W1 实时流对拍 —— 读离线 day_iv / 5m·1m 金标，不可当实盘绿灯
#
# 实盘一致性请用:
#   bash qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh
#
# 本脚本保留用途：特征/门控开卷上界（greek-parity + feature5m + regime gold + rolling seed）
# 冻结口径（2026-07-12 开卷 PASS）：
#   - ckpt: checkpoints_qqq_ft56_julw1
#   - greek-parity + day_iv end-label（pitcher 注入 T+60）
#   - bak June 已归一化 seed + july_w1_v4_databento 金标特征
#   - REPLAY 默认含 put_early_vix + CALL TREND_SPENT
#   - Step-1: debug_slow vs quote_features_test_clean（ts_shift=+60）
#
# 用法:
#   bash qqq_btc/tools/restart_ft56_july_w1_stream_parity.sh
#   DAYS="2026-07-01" bash qqq_btc/tools/restart_ft56_july_w1_stream_parity.sh   # 单日
#   SKIP_STEP1=1 bash ...   # 只跑流，不对拍特征
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# --- 开卷诊断栈（显式金标路径；勿当 deploy 默认）---
export OMS_MOCK_IBKR=1
export REDIS_STREAM_SIM=1
export QQQ_BTC_LIVE=1
export QQQ_BTC_FILL_AUDIT=1
export FCS_NORMALIZER_STATS_UPDATE_INTERVAL="${FCS_NORMALIZER_STATS_UPDATE_INTERVAL:-1}"
export FCS_TA_MONTH_ISOLATED="${FCS_TA_MONTH_ISOLATED:-1}"
export QQQ_BTC_PUT_GATE_MODE=feature5m
export QQQ_BTC_PUT_GATE_5M_FEATURE="${QQQ_BTC_PUT_GATE_5M_FEATURE:-$HOME/train_data/july_w1_v4_databento/quote_features_test/QQQ/regular/09:30-16:00/5min}"
export QQQ_BTC_REGIME_GOLD_1M="${QQQ_BTC_REGIME_GOLD_1M:-$HOME/train_data/july_w1_v4_databento/quote_features_test/QQQ/regular/09:30-16:00/1min}"
export EXECUTION_DELAY_BARS=0
export OMS_SIGNAL_DELAY_BARS=0
export BACKTEST_OPT_FILL_SPREAD_FRAC="${BACKTEST_OPT_FILL_SPREAD_FRAC:-0.775}"
unset FCS_FROZEN_NORM_PATH || true
export FCS_FROZEN_NORM_PATH=""

CKPT="${CKPT:-$(realpath checkpoint/checkpoints_qqq_ft56_julw1/best.pth)}"
OPT_ROOT="${OPT_ROOT:-/mnt/s990/data/v4_original_jul5/databento_july_w1/raw_1s}"
STOCK_ROOT="${STOCK_ROOT:-$HOME/train_data/spnq_train}"
ROLLING_NORM_SEED="${ROLLING_NORM_SEED:-$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00/1min/2026-06.parquet}"
GREEK_ROOT="${GREEK_ROOT:-$HOME/train_data/july_w1_v4_databento/quote_options_day_iv}"
MONTHLY_IV_ROOT="${MONTHLY_IV_ROOT:-$HOME/train_data/july_w1_v4_databento/quote_options_monthly_iv}"
OFFLINE_CLEAN="${OFFLINE_CLEAN:-$HOME/train_data/july_w1_v4_databento/quote_features_test_clean/QQQ/regular/09:30-16:00/1min/2026-07.parquet}"
OUT_DIR="${OUT_DIR:-$REPO/qqq_btc/results/july_w1_ft56_4c_stream_rolling_spent_fix}"
FCS_WAIT="${FCS_WAIT:-60}"
SPEED="${SPEED:-inf}"
POS_FRAC="${POS_FRAC:-0.25}"
# spent 门控后 offline @25% 量级（非旧 +37.7%）
OFFLINE_BASELINE_ACCT25="${OFFLINE_BASELINE_ACCT25:-0.515}"

if [[ -n "${DAYS:-}" ]]; then
  # shellcheck disable=SC2206
  DAY_ARR=($DAYS)
else
  DAY_ARR=(2026-07-01 2026-07-02 2026-07-06 2026-07-07 2026-07-08 2026-07-09 2026-07-10)
fi

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"
: > "$LOG"

echo "=== stop prior parity stack ===" | tee -a "$LOG"
pkill -9 -f 'feature_compute_service_v8|run_live_signal_qqq|run_live_exec_qqq|redis_fused_pitcher|run_qqq_btc_redis_sim' 2>/dev/null || true
sleep 1

echo "=== [0] seed PG market_bars (June) for Deep Warmup ===" | tee -a "$LOG"
"$PY" -u "$REPO/qqq_btc/tools/seed_pg_warmup_bars.py" \
  --root "$STOCK_ROOT" \
  --symbols QQQ,VIXY \
  --start 2026-06-01 \
  --end 2026-06-30 \
  2>&1 | tee -a "$LOG"

echo "ckpt=$CKPT" | tee -a "$LOG"
echo "opt=$OPT_ROOT" | tee -a "$LOG"
echo "out=$OUT_DIR" | tee -a "$LOG"
echo "days=${DAY_ARR[*]}" | tee -a "$LOG"
echo "mode=PG_deep_warmup + greek-parity(end-label) + rolling ON + TREND_SPENT" | tee -a "$LOG"

prev=""
for d in "${DAY_ARR[@]}"; do
  ymd="${d//-/}"
  echo "" | tee -a "$LOG"
  echo "======== STREAM $d (deep-warmup) ========" | tee -a "$LOG"
  pkill -9 -f 'feature_compute_service_v8|run_live_signal_qqq|run_live_exec_qqq|redis_fused_pitcher' 2>/dev/null || true
  sleep 1

  if [[ -n "$prev" ]]; then
    echo "[seed] prior day $prev → PG" | tee -a "$LOG"
    "$PY" -u "$REPO/qqq_btc/tools/seed_pg_warmup_bars.py" \
      --root "$STOCK_ROOT" \
      --symbols QQQ,VIXY \
      --start "$prev" \
      --end "$prev" \
      2>&1 | tee -a "$LOG"
  fi

  audit="$HOME/quant_project/shadow/fill_audit_${ymd}.csv"
  rm -f "$audit" \
    "$HOME/quant_project/shadow/signals_${d}.csv" \
    "$HOME/quant_project/shadow/se_alpha_${d}.csv"
  export QQQ_BTC_FILL_AUDIT_PATH="$audit"
  export QQQ_BTC_VIXY_SEED_BEFORE="$ymd"
  export FCS_MINUTE_OPTION_MONTHLY_IV_ROOT="$MONTHLY_IV_ROOT"

  "$PY" -u "$REPO/qqq_btc/tools/run_qqq_btc_redis_sim.py" \
    --date "$ymd" \
    --source raw \
    --option-root "$OPT_ROOT" \
    --checkpoint "$CKPT" \
    --deep-warmup \
    --no-frozen-norm \
    --rolling-norm-seed "$ROLLING_NORM_SEED" \
    --greek-parity \
    --greek-root "$GREEK_ROOT" \
    --speed "$SPEED" \
    --fcs-wait "$FCS_WAIT" \
    2>&1 | tee -a "$LOG" | tee "$OUT_DIR/stream_${d}.log"

  if [[ -f "$audit" ]]; then
    cp -f "$audit" "$OUT_DIR/fill_audit_${ymd}.csv"
  else
    echo "[warn] missing fill_audit $audit" | tee -a "$LOG"
  fi
  [[ -f "$HOME/quant_project/shadow/signals_${d}.csv" ]] && cp -f "$HOME/quant_project/shadow/signals_${d}.csv" "$OUT_DIR/" || true
  prev="$d"
done

echo "" | tee -a "$LOG"
echo "======== AGGREGATE @${POS_FRAC} ========" | tee -a "$LOG"
DAYS_CSV="$(IFS=,; echo "${DAY_ARR[*]}")"
export OUT_DIR CKPT OPT_ROOT POS_FRAC OFFLINE_BASELINE_ACCT25 DAYS_CSV
"$PY" - <<'PY' | tee -a "$LOG" | tee "$OUT_DIR/summary.txt"
import json, os
from pathlib import Path
import pandas as pd
import numpy as np

out = Path(os.environ["OUT_DIR"])
ckpt = os.environ["CKPT"]
opt_root = os.environ["OPT_ROOT"]
pos = float(os.environ.get("POS_FRAC", "0.25"))
offline = float(os.environ.get("OFFLINE_BASELINE_ACCT25", "0.515"))
days = [d.strip() for d in os.environ["DAYS_CSV"].split(",") if d.strip()]
day_set = set(pd.Timestamp(d).date() for d in days)

frames = []
for d in days:
    fp = out / f"fill_audit_{d.replace('-', '')}.csv"
    if fp.exists() and fp.stat().st_size > 0:
        try:
            frames.append(pd.read_csv(fp))
        except Exception:
            pass
if not frames:
    summary = {"error": "no fill_audit", "acct25": None, "trades": 0}
    (out / "stream_summary_paired.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    raise SystemExit(0)

df = pd.concat(frames, ignore_index=True)
df["action"] = df["action"].astype(str).str.upper()
df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
df = df.dropna(subset=["ts"]).sort_values("ts", kind="mergesort")
df["ts_dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
df["ny_date"] = df["ts_dt"].dt.tz_convert("America/New_York").dt.date

trades = []
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
                "entry_sb": float(pd.to_numeric(open_row.get("session_bar"), errors="coerce") or np.nan),
                "exit_sb": float(pd.to_numeric(r.get("session_bar"), errors="coerce") or np.nan),
                "reason": str(r.get("exit_reason", "")),
                "net_return": float(pd.to_numeric(r.get("net_return"), errors="coerce") or 0.0),
            })
        open_row = None

eq = 1.0
for t in trades:
    eq *= 1.0 + pos * float(t["net_return"])
rets = [t["net_return"] for t in trades]
hit = float(np.mean([x > 0 for x in rets])) if rets else None
by_leg = {}
for t in trades:
    by_leg[t["leg"]] = by_leg.get(t["leg"], 0) + 1
summary = {
    "mode": "redis_stream_ft56_4c_pg_deep_warmup_rolling_spent_fix",
    "checkpoint": ckpt,
    "option_root": opt_root,
    "position_frac": pos,
    "frozen_norm": None,
    "warmup": "PG Deep Warmup (market_bars before REPLAY_START_TS)",
    "greek_parity": "day_iv end-label (minute_start+60)",
    "gates": ["put_early_vix", "call_trend_spent"],
    "days": days,
    "trades": len(trades),
    "acct25": float(eq - 1.0),
    "hit_rate": hit,
    "sum_net": float(sum(rets)) if rets else 0.0,
    "trades_by_leg": by_leg,
    "offline_baseline_acct25": offline,
    "delta_pp_vs_offline": float(eq - 1.0 - offline),
    "trades_detail": trades,
}
(out / "stream_summary_paired.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, default=str)
)
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print()
print(f"STREAM @25%: {summary['acct25']*100:+.2f}%  trades={len(trades)}  hit={hit}  legs={by_leg}")
print(f"OFFLINE spent baseline: {offline*100:+.2f}%")
print(f"delta vs offline:       {summary['delta_pp_vs_offline']*100:+.2f} pp")
PY

if [[ "${SKIP_STEP1:-0}" != "1" ]]; then
  echo "" | tee -a "$LOG"
  echo "======== STEP-1 feat parity (debug_slow vs clean) ========" | tee -a "$LOG"
  STEP1_DATES="$(IFS=,; echo "${DAY_ARR[*]}")"
  set +e
  "$PY" -u "$REPO/qqq_btc/tools/compare_debug_slow_offline.py" \
    --dates "$STEP1_DATES" \
    --offline "$OFFLINE_CLEAN" \
    --ts-shift-sec 60 \
    --out "$OUT_DIR/feat_parity_step1.json" \
    2>&1 | tee -a "$LOG" | tee "$OUT_DIR/feat_parity_step1.txt"
  step1_rc=${PIPESTATUS[0]}
  set -e
  echo "step1_exit=$step1_rc" | tee -a "$LOG"
  if [[ "$step1_rc" -ne 0 ]]; then
    echo "[FAIL] Step-1 overall_pass=false → $OUT_DIR/feat_parity_step1.json" | tee -a "$LOG"
    exit 2
  fi
  echo "[PASS] Step-1 → $OUT_DIR/feat_parity_step1.json" | tee -a "$LOG"
fi

echo "done → $OUT_DIR/stream_summary_paired.json"
