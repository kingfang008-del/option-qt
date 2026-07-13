#!/usr/bin/env bash
# F56 July W1 —— 诚实流式对拍（模拟实盘，禁止开卷）
#
# 三闸门（硬顺序，不可跳级）：
#   Gate-1  FCS debug_raw（秒级→指标，归一化前）vs 离线 quote_features_raw
#   Gate-2  仅 Gate-1 PASS 后：debug_slow（frozen/rolling norm）vs quote_features_test
#   Gate-3  仅 Gate-1+2 PASS 后：交易对拍汇总（否则只写 diagnostic，不宣称 parity）
#
# 与 restart_ft56_july_w1_stream_parity.sh（开卷诊断）的区别：
#   - 无 greek-parity / 无 day_iv 注入（FCS RECALC_GREEKS 自算）
#   - put_gate = vixy_5m（因果 5min raw z），禁止 feature5m 金标
#   - regime gold = off
#   - 归一化 = deploy 同款 frozen_norm_qqq_daily.npz + PG Deep Warmup
#
# 用法:
#   bash qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh
#   DAYS="2026-07-01" bash qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh
#   SKIP_GATES=1 bash ...          # 只跑流式+diagnostic 汇总，不跑门控
#   FORCE_GATE3=1 bash ...         # Gate 失败仍输出交易汇总（标为 ungated）
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# --- 诚实实盘模拟栈 ---
export OMS_MOCK_IBKR=1
export REDIS_STREAM_SIM=1
export QQQ_BTC_LIVE=1
export QQQ_BTC_FILL_AUDIT=1
export FCS_NORMALIZER_STATS_UPDATE_INTERVAL="${FCS_NORMALIZER_STATS_UPDATE_INTERVAL:-1}"
unset FCS_TA_MONTH_ISOLATED || true
export QQQ_BTC_PUT_GATE_MODE=vixy_5m
# asof=与向量化 raw5 put_gate 同序列；buffer=真因果（无桶内前视，目标不是 +64%）
export QQQ_BTC_VIXY_5M_SOURCE="${QQQ_BTC_VIXY_5M_SOURCE:-asof}"
export QQQ_BTC_PUT_GATE_RAW5="${QQQ_BTC_PUT_GATE_RAW5:-$HOME/train_data/july_w1_v4_honest_openwin/quote_features_raw/QQQ/regular/09:30-16:00/5min}"
export QQQ_BTC_REGIME_GOLD_1M=0
unset QQQ_BTC_PUT_GATE_5M_FEATURE || true
# 与离线 raw 5min put_gate / early_vix=0.6 同标尺；勿沿用 vixy_z 重标定阈值
unset QQQ_BTC_PUT_GATE_MIN || true
unset QQQ_BTC_PUT_EARLY_VIX_MIN || true
unset QQQ_BTC_EDGE_Q10_FLOOR || true
export QQQ_BTC_USE_LIVE_REPLAY="${QQQ_BTC_USE_LIVE_REPLAY:-1}"
export EXECUTION_DELAY_BARS=0
export OMS_SIGNAL_DELAY_BARS=0
export BACKTEST_OPT_FILL_SPREAD_FRAC="${BACKTEST_OPT_FILL_SPREAD_FRAC:-0.775}"
FROZEN_NORM="${FROZEN_NORM:-$REPO/qqq_btc/CONFIG/frozen_norm_qqq_daily.npz}"
export FCS_FROZEN_NORM_PATH="$FROZEN_NORM"
unset FCS_MINUTE_PARITY_INJECT || true
unset GREEK_PARITY_MODE || true
export RECALC_GREEKS=1
export FCS_FORCE_RECALC_GREEKS=1
# Gate-1 必需：落盘归一化前 raw
export FCS_DEBUG_RAW=1
# 对齐离线月冷启动 TA（ADX/BB/GK）
export FCS_TA_MONTH_ISOLATED="${FCS_TA_MONTH_ISOLATED:-1}"
# 期权 T / IV 定价对齐离线 option_cac（end-label + close/mid snap）
export FCS_OPTION_T_LABEL="${FCS_OPTION_T_LABEL:-end}"
export FCS_IV_PRICE_MODE="${FCS_IV_PRICE_MODE:-close}"

CKPT="${CKPT:-$(realpath checkpoint/checkpoints_qqq_ft56_julw1/best.pth)}"
OPT_ROOT="${OPT_ROOT:-/mnt/s990/data/v4_original_jul5/databento_july_w1_openwin/raw_1s}"
STOCK_ROOT="${STOCK_ROOT:-$HOME/train_data/spnq_train}"
HONEST_FEAT_ROOT="${HONEST_FEAT_ROOT:-$HOME/train_data/july_w1_v4_honest_openwin}"
# 仅注入分钟 volume（cbbo 无 trade volume）；不启用 --greek-parity
GREEK_ROOT="${GREEK_ROOT:-$HONEST_FEAT_ROOT/quote_options_day_iv}"
OFFLINE_RAW="${OFFLINE_RAW:-$HONEST_FEAT_ROOT/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet}"
OFFLINE_NORM="${OFFLINE_NORM:-$HONEST_FEAT_ROOT/quote_features_test/QQQ/regular/09:30-16:00/1min/2026-07.parquet}"
# 兼容旧变量名
if [[ -n "${OFFLINE_CLEAN:-}" && "${OFFLINE_NORM}" == "$HONEST_FEAT_ROOT/quote_features_test/QQQ/regular/09:30-16:00/1min/2026-07.parquet" ]]; then
  OFFLINE_NORM="$OFFLINE_CLEAN"
fi
OUT_DIR="${HONEST_OUT_DIR:-$REPO/qqq_btc/results/july_w1_ft56_honest_live_parity}"
export SLOW_FEATURE_CONFIG="${SLOW_FEATURE_CONFIG:-$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json}"
if [[ "$(basename "$SLOW_FEATURE_CONFIG")" == *v2* ]]; then
  export SLOW_FEATURE_CONFIG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json"
fi
FCS_WAIT="${FCS_WAIT:-60}"
SPEED="${SPEED:-inf}"
POS_FRAC="${POS_FRAC:-0.25}"

if [[ -n "${DAYS:-}" ]]; then
  # shellcheck disable=SC2206
  DAY_ARR=($DAYS)
else
  DAY_ARR=(2026-07-01 2026-07-02 2026-07-06 2026-07-07 2026-07-08 2026-07-09 2026-07-10)
fi

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"
: > "$LOG"

cat > "$OUT_DIR/manifest.json" <<EOF
{
  "mode": "honest_live_parity",
  "gates": ["1_raw", "2_norm", "3_trade"],
  "meaning": "simulate live: no greek-parity, no put_gate/regime gold files, frozen_norm=deploy",
  "crutches_disabled": [
    "greek-parity / day_iv inject",
    "QQQ_BTC_PUT_GATE_MODE=feature5m",
    "QQQ_BTC_REGIME_GOLD_1M",
    "bak June rolling-norm-seed",
    "FCS_TA_MONTH_ISOLATED"
  ],
  "norm": "$FROZEN_NORM",
  "put_gate": "${QQQ_BTC_PUT_GATE_MODE:-vixy_5m}",
  "regime_gold": "off",
  "checkpoint": "$CKPT",
  "option_root": "$OPT_ROOT",
  "greek_root_volume_only": "$GREEK_ROOT",
  "offline_raw_gate1": "$OFFLINE_RAW",
  "offline_norm_gate2": "$OFFLINE_NORM",
  "fcs_debug_raw": true,
  "fcs_ta_month_isolated": true,
  "fcs_option_t_label": "${FCS_OPTION_T_LABEL:-end}",
  "fcs_iv_price_mode": "${FCS_IV_PRICE_MODE:-close}"
}
EOF

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
echo "greek_root=$GREEK_ROOT (volume-only; no greek-parity)" | tee -a "$LOG"
echo "frozen=$FROZEN_NORM" | tee -a "$LOG"
echo "offline_raw=$OFFLINE_RAW" | tee -a "$LOG"
echo "offline_norm=$OFFLINE_NORM" | tee -a "$LOG"
echo "out=$OUT_DIR" | tee -a "$LOG"
echo "days=${DAY_ARR[*]}" | tee -a "$LOG"
echo "mode=HONEST 3-gate: raw → norm → trade | FCS_DEBUG_RAW=1" | tee -a "$LOG"

prev=""
for d in "${DAY_ARR[@]}"; do
  ymd="${d//-/}"
  echo "" | tee -a "$LOG"
  echo "======== HONEST STREAM $d ========" | tee -a "$LOG"
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

  "$PY" -u "$REPO/qqq_btc/tools/run_qqq_btc_redis_sim.py" \
    --date "$ymd" \
    --source raw \
    --option-root "$OPT_ROOT" \
    --greek-root "$GREEK_ROOT" \
    --checkpoint "$CKPT" \
    --deep-warmup \
    --frozen-norm "$FROZEN_NORM" \
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

GATE_DATES="$(IFS=,; echo "${DAY_ARR[*]}")"
gate1_rc=2
gate2_rc=2
gate1_pass=0
gate2_pass=0

if [[ "${SKIP_GATES:-0}" != "1" ]]; then
  echo "" | tee -a "$LOG"
  echo "======== GATE-1 RAW (debug_raw vs quote_features_raw) ========" | tee -a "$LOG"
  if [[ ! -f "$OFFLINE_RAW" ]]; then
    echo "ERROR: OFFLINE_RAW missing: $OFFLINE_RAW" | tee -a "$LOG"
    gate1_rc=2
  else
    set +e
    "$PY" -u "$REPO/qqq_btc/tools/compare_debug_raw_offline.py" \
      --dates "$GATE_DATES" \
      --offline "$OFFLINE_RAW" \
      --ts-shift-sec 60 \
      --corr-min "${GATE1_CORR_MIN:-0.90}" \
      --out "$OUT_DIR/feat_parity_gate1_raw.json" \
      2>&1 | tee -a "$LOG" | tee "$OUT_DIR/feat_parity_gate1_raw.txt"
    gate1_rc=${PIPESTATUS[0]}
    set -e
  fi
  echo "gate1_exit=$gate1_rc" | tee -a "$LOG"
  [[ "$gate1_rc" -eq 0 ]] && gate1_pass=1

  if [[ "$gate1_pass" -eq 1 ]]; then
    echo "" | tee -a "$LOG"
    echo "======== GATE-2 NORM (debug_slow vs quote_features_test) ========" | tee -a "$LOG"
    if [[ ! -f "$OFFLINE_NORM" ]]; then
      echo "ERROR: OFFLINE_NORM missing: $OFFLINE_NORM" | tee -a "$LOG"
      gate2_rc=2
    else
      set +e
      "$PY" -u "$REPO/qqq_btc/tools/compare_debug_slow_offline.py" \
        --dates "$GATE_DATES" \
        --offline "$OFFLINE_NORM" \
        --ts-shift-sec 60 \
        --corr-min "${GATE2_CORR_MIN:-0.90}" \
        --out "$OUT_DIR/feat_parity_gate2_norm.json" \
        2>&1 | tee -a "$LOG" | tee "$OUT_DIR/feat_parity_gate2_norm.txt"
      gate2_rc=${PIPESTATUS[0]}
      set -e
    fi
    echo "gate2_exit=$gate2_rc" | tee -a "$LOG"
    [[ "$gate2_rc" -eq 0 ]] && gate2_pass=1
  else
    echo "======== GATE-2 SKIPPED (Gate-1 FAIL) ========" | tee -a "$LOG"
  fi
else
  echo "[SKIP_GATES=1] feature gates skipped" | tee -a "$LOG"
fi

run_gate3=0
if [[ "$gate1_pass" -eq 1 && "$gate2_pass" -eq 1 ]]; then
  run_gate3=1
elif [[ "${FORCE_GATE3:-0}" == "1" ]]; then
  run_gate3=1
  echo "[FORCE_GATE3=1] emitting ungated trade summary" | tee -a "$LOG"
elif [[ "${SKIP_GATES:-0}" == "1" ]]; then
  run_gate3=1
fi

echo "" | tee -a "$LOG"
if [[ "$run_gate3" -eq 1 ]]; then
  echo "======== GATE-3 TRADE SUMMARY @${POS_FRAC} ========" | tee -a "$LOG"
else
  echo "======== GATE-3 BLOCKED (need Gate-1+2 PASS; diagnostic fills still archived) ========" | tee -a "$LOG"
fi

DAYS_CSV="$(IFS=,; echo "${DAY_ARR[*]}")"
export OUT_DIR CKPT OPT_ROOT POS_FRAC DAYS_CSV FROZEN_NORM
export GATE1_PASS="$gate1_pass" GATE2_PASS="$gate2_pass" RUN_GATE3="$run_gate3"
export GATE1_RC="$gate1_rc" GATE2_RC="$gate2_rc"
"$PY" - <<'PY' | tee -a "$LOG" | tee "$OUT_DIR/summary.txt"
import json, os
from pathlib import Path
import pandas as pd
import numpy as np

out = Path(os.environ["OUT_DIR"])
ckpt = os.environ["CKPT"]
opt_root = os.environ["OPT_ROOT"]
pos = float(os.environ.get("POS_FRAC", "0.25"))
frozen = os.environ.get("FROZEN_NORM", "")
days = [d.strip() for d in os.environ["DAYS_CSV"].split(",") if d.strip()]
day_set = set(pd.Timestamp(d).date() for d in days)
g1 = int(os.environ.get("GATE1_PASS", "0"))
g2 = int(os.environ.get("GATE2_PASS", "0"))
run_g3 = int(os.environ.get("RUN_GATE3", "0"))
gates = {
    "gate1_raw": {"pass": bool(g1), "exit": int(os.environ.get("GATE1_RC", "2"))},
    "gate2_norm": {"pass": bool(g2), "exit": int(os.environ.get("GATE2_RC", "2"))},
    "gate3_trade_allowed": bool(run_g3),
}
(out / "gates_status.json").write_text(json.dumps(gates, indent=2))

frames = []
for d in days:
    fp = out / f"fill_audit_{d.replace('-', '')}.csv"
    if fp.exists() and fp.stat().st_size > 0:
        try:
            frames.append(pd.read_csv(fp))
        except Exception:
            pass

def _empty_summary(extra=None):
    s = {
        "mode": "honest_live_parity",
        "gates": gates,
        "checkpoint": ckpt,
        "option_root": opt_root,
        "position_frac": pos,
        "frozen_norm": frozen,
        "acct25": None,
        "trades": 0,
        "note": "no fill_audit or gate3 blocked",
    }
    if extra:
        s.update(extra)
    return s

if not frames:
    summary = _empty_summary({"error": "no fill_audit"})
    (out / "stream_summary_paired.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
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
    "mode": "honest_live_parity",
    "gates": gates,
    "checkpoint": ckpt,
    "option_root": opt_root,
    "position_frac": pos,
    "frozen_norm": frozen,
    "warmup": "PG Deep Warmup",
    "greek_parity": False,
    "put_gate": os.environ.get("QQQ_BTC_PUT_GATE_MODE", "vixy_5m"),
    "regime_gold": False,
    "days": days,
    "trades": len(trades),
    "acct25": float(eq - 1.0) if run_g3 else None,
    "hit_rate": hit if run_g3 else None,
    "sum_net": float(sum(rets)) if (rets and run_g3) else None,
    "trades_by_leg": by_leg if run_g3 else None,
    "trades_detail": trades if run_g3 else None,
    "diagnostic_trade_count": len(trades),
}
if not run_g3:
    summary["note"] = (
        "Gate-3 blocked: Gate-1 raw and Gate-2 norm must both PASS before trade parity. "
        f"diagnostic fills archived ({len(trades)} closed trades); PnL not reported as parity."
    )
    summary["parity_status"] = "BLOCKED"
else:
    summary["parity_status"] = "PASS" if (g1 and g2) else "UNGATED"
    summary["note"] = (
        "Gate-3 trade summary (parity)" if (g1 and g2)
        else "ungated diagnostic PnL (FORCE_GATE3 or SKIP_GATES)"
    )

(out / "stream_summary_paired.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, default=str)
)
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print()
if run_g3 and summary.get("acct25") is not None:
    print(
        f"GATE3 STREAM @{pos}: {summary['acct25']*100:+.2f}%  "
        f"trades={len(trades)}  hit={hit}  legs={by_leg}  status={summary['parity_status']}"
    )
else:
    print(f"GATE3 BLOCKED | diagnostic_trades={len(trades)} | gates={gates}")
PY

echo "done → $OUT_DIR (manifest.json + gates_status.json + stream_summary_paired.json)"
echo "开卷诊断入口仍为: bash qqq_btc/tools/restart_ft56_july_w1_stream_parity.sh"
