#!/usr/bin/env bash
# July W1 · ft56 · 4约基线 · PG Deep Warmup（对齐实盘开盘前预热）+ live rolling
#
# 实盘口径：
#   1) 历史 bars 先入 PG
#   2) 每日开盘启动 FCS → Deep Warmup(SKIP_DEEP_WARMUP=0, REPLAY_START_TS=当日开盘)
#   3) 只流当日盘中 1s；不做「流历史日」预热
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMS_MOCK_IBKR=1
export REDIS_STREAM_SIM=1
export QQQ_BTC_LIVE=1
export QQQ_BTC_FILL_AUDIT=1
export FCS_NORMALIZER_STATS_UPDATE_INTERVAL="${FCS_NORMALIZER_STATS_UPDATE_INTERVAL:-1}"
# 对齐离线「月 parquet 冷启动」ADX/BB/GK；生产连续 history 时请 unset
export FCS_TA_MONTH_ISOLATED="${FCS_TA_MONTH_ISOLATED:-1}"
# put_gate 对齐 +37.7%：5min quote_features_test.vix_level asof（非 1min VIXY z）
export QQQ_BTC_PUT_GATE_MODE="${QQQ_BTC_PUT_GATE_MODE:-feature5m}"
export QQQ_BTC_PUT_GATE_5M_FEATURE="${QQQ_BTC_PUT_GATE_5M_FEATURE:-$HOME/train_data/july_w1_v4_databento/quote_features_test/QQQ/regular/09:30-16:00/5min}"
# put_trend/open30 门控：默认读 offline 1min 金标（避免 SE 短历史早翻转）
export QQQ_BTC_REGIME_GOLD_1M="${QQQ_BTC_REGIME_GOLD_1M:-$HOME/train_data/july_w1_v4_databento/quote_features_test/QQQ/regular/09:30-16:00/1min}"
# FCS alpha label 滞后已等价 offline entry_delay=1；勿再叠加 OMS 延迟
export EXECUTION_DELAY_BARS=0
export OMS_SIGNAL_DELAY_BARS=0
export BACKTEST_OPT_FILL_SPREAD_FRAC="${BACKTEST_OPT_FILL_SPREAD_FRAC:-0.775}"
unset FCS_FROZEN_NORM_PATH || true
export FCS_FROZEN_NORM_PATH=""

CKPT="${CKPT:-$(realpath checkpoint/checkpoints_qqq_ft56_julw1/best.pth)}"
OPT_ROOT="${OPT_ROOT:-/mnt/s990/data/v4_original_jul5/databento_july_w1/raw_1s}"
STOCK_ROOT="${STOCK_ROOT:-$HOME/train_data/spnq_train}"
# 对齐 +37.7% 金标：offline 组装时把已归一化的 bak June 拷进 quote_features_test 再 rolling。
# buffer 语义=「上月输入」= bak_test（非 raw），否则会落到 clean 口径而非 test。
ROLLING_NORM_SEED="${ROLLING_NORM_SEED:-$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00/1min/2026-06.parquet}"
# 与 raw_1s（databento_july_w1）同源的分钟 IV/Greeks；禁止用异源 experiment day_iv 冒充盘口
GREEK_ROOT="${GREEK_ROOT:-$HOME/train_data/july_w1_v4_databento/quote_options_day_iv}"
MONTHLY_IV_ROOT="${MONTHLY_IV_ROOT:-$HOME/train_data/july_w1_v4_databento/quote_options_monthly_iv}"
OUT_DIR="${OUT_DIR:-$REPO/qqq_btc/results/july_w1_ft56_4c_stream_rolling}"
FCS_WAIT="${FCS_WAIT:-60}"
SPEED="${SPEED:-inf}"
POS_FRAC="${POS_FRAC:-0.25}"



DAYS=(2026-07-01 2026-07-02 2026-07-06 2026-07-07 2026-07-08 2026-07-09)
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"
: > "$LOG"

echo "=== [0] seed PG market_bars (June) for Deep Warmup ===" | tee -a "$LOG"
"$PY" -u "$REPO/qqq_btc/tools/seed_pg_warmup_bars.py" \
  --root "$STOCK_ROOT" \
  --symbols QQQ,VIXY \
  --start 2026-06-01 \
  --end 2026-06-30 \
  2>&1 | tee -a "$LOG"

echo "ckpt=$CKPT" | tee -a "$LOG"
echo "opt=$OPT_ROOT" | tee -a "$LOG"
echo "mode=PG_deep_warmup + July-only stream + rolling ON frozen=OFF" | tee -a "$LOG"

prev=""
for d in "${DAYS[@]}"; do
  ymd="${d//-/}"
  echo "" | tee -a "$LOG"
  echo "======== STREAM $d (deep-warmup) ========" | tee -a "$LOG"
  pkill -9 -f 'feature_compute_service_v8|run_live_signal_qqq|run_live_exec_qqq|redis_fused_pitcher' 2>/dev/null || true
  sleep 1

  # 把「昨日」bars 再灌进 PG，模拟盘后落库 → 次日开盘 Deep Warmup 可见
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
echo "======== AGGREGATE @${POS_FRAC} (OPEN→first CLOSE) ========" | tee -a "$LOG"
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
day_set = set(pd.Timestamp(d).date() for d in days)

frames = []
for d in days:
    fp = out / f"fill_audit_{d.replace('-','')}.csv"
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
offline = 0.3767942218288456
summary = {
    "mode": "redis_stream_ft56_4c_pg_deep_warmup_rolling",
    "checkpoint": ckpt,
    "option_root": opt_root,
    "position_frac": pos,
    "frozen_norm": None,
    "warmup": "PG Deep Warmup (market_bars before REPLAY_START_TS)",
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
(out / "stream_summary_paired.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print()
print(f"STREAM PG-warmup rolling @25%: {summary['acct25']*100:+.2f}%  trades={len(trades)}  hit={hit}  legs={by_leg}")
print(f"OFFLINE rolling:               +37.68%")
print(f"delta vs offline:              {summary['delta_pp_vs_offline']*100:+.2f} pp")
PY

echo "done → $OUT_DIR/stream_summary_paired.json"
