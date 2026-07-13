#!/usr/bin/env bash
# 部署 F56 July W1 实盘栈（IBKR → FCS → SE → OMS）
#
# 开盘前自动化：
#   1) 检测 PG market_bars 预热覆盖（QQQ/VIXY）
#   2) 缺口：本地 spnq_train seed → 仍缺则 Massive/Polygon 下载
#   3) 再检测；不过关则拒绝启动
#   4) 过关后才拉起 Persistence / IBKR / FCS / SE / OMS
#
# 对齐诚实流式对拍口径（模拟实盘）：
#   - 门控/权重：F56 + qcfg.REPLAY（put_early / TREND_SPENT）
#   - 无 greek-parity；put_gate=vixy_z；regime gold off
#   - frozen_norm_qqq_daily + PG Deep Warmup
# 开卷诊断见 restart_ft56_july_w1_stream_parity.sh；诚实对拍见
# restart_ft56_july_w1_honest_live_parity.sh。
#
# 用法:
#   bash qqq_btc/tools/deploy_ft56_julw1_live.sh              # dry-run
#   LIVE_TRADE=1 bash qqq_btc/tools/deploy_ft56_julw1_live.sh # 真交易
#   bash qqq_btc/tools/deploy_ft56_julw1_live.sh stop
#   SKIP_WARMUP_GATE=1 bash ...   # 跳过预热门禁（不推荐）
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BASELINE="$REPO/New_Pro/baseline_qqq"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
LOG_DIR="${LOG_DIR:-$HOME/quant_project/logs}"
CKPT="${CKPT:-$REPO/checkpoint/checkpoints_qqq_ft56_julw1/best.pth}"
SLOW_CFG="${SLOW_FEATURE_CONFIG:-$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json}"
SYM_MAP="${FCS_SYMBOL_MAP:-$REPO/qqq_btc/CONFIG/symbol_map.json}"
FROZEN_NORM="${FCS_FROZEN_NORM_PATH:-$REPO/qqq_btc/CONFIG/frozen_norm_qqq_daily.npz}"
ENV_FILE="${ENV_FILE:-$BASELINE/config/minimal_stack.env}"
STOCK_ROOT="${STOCK_ROOT:-$HOME/train_data/spnq_train}"
WARMUP_LOOKBACK_TDAYS="${WARMUP_LOOKBACK_TDAYS:-12}"
WARMUP_MIN_DAY_BARS="${WARMUP_MIN_DAY_BARS:-300}"
WARMUP_MIN_TOTAL_1M="${WARMUP_MIN_TOTAL_1M:-1800}"
# Massive/Polygon：优先 MASSIVE_API_KEY，其次 POLYGON_API_KEY
export MASSIVE_API_BASE="${MASSIVE_API_BASE:-https://api.polygon.io}"

mkdir -p "$LOG_DIR"

stop_stack() {
  echo "=== stop F56 live stack ==="
  pkill -f 'run_live_signal_qqq.py' 2>/dev/null || true
  pkill -f 'run_live_exec_qqq.py' 2>/dev/null || true
  pkill -f 'feature_compute_service_v8.py' 2>/dev/null || true
  pkill -f 'ibkr_connector_v8.py' 2>/dev/null || true
  pkill -f 'data_persistence_service_v8' 2>/dev/null || true
  rm -f /tmp/replay_active.lock
  sleep 1
  echo "stopped"
}

if [[ "${1:-}" == "stop" ]]; then
  stop_stack
  exit 0
fi

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing checkpoint $CKPT" >&2
  exit 1
fi
if [[ ! -f "$PY" ]]; then
  echo "ERROR: python not found: $PY" >&2
  exit 1
fi
if [[ ! -d "$BASELINE" ]]; then
  echo "ERROR: baseline missing: $BASELINE" >&2
  exit 1
fi

stop_stack

# --- env ---
set -a
# shellcheck disable=SC1090
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
export QQQ_BTC_LIVE=1
export QQQ_BTC_TICK_EXITS="${QQQ_BTC_TICK_EXITS:-disaster_only}"
export SLOW_FEATURE_CONFIG="$SLOW_CFG"
export FCS_SYMBOL_MAP="$SYM_MAP"
export FCS_FROZEN_NORM_PATH="$FROZEN_NORM"
export QQQ_BTC_PUT_GATE_MODE="${QQQ_BTC_PUT_GATE_MODE:-vixy_z}"
export QQQ_BTC_REGIME_GOLD_1M="${QQQ_BTC_REGIME_GOLD_1M:-0}"
unset QQQ_BTC_PUT_GATE_5M_FEATURE || true
export EXECUTION_DELAY_BARS=0
export OMS_SIGNAL_DELAY_BARS=0
export BACKTEST_OPT_FILL_SPREAD_FRAC="${BACKTEST_OPT_FILL_SPREAD_FRAC:-0.775}"
export FCS_NORMALIZER_STATS_UPDATE_INTERVAL="${FCS_NORMALIZER_STATS_UPDATE_INTERVAL:-1}"
export RECALC_GREEKS=1
# 强制允许 Deep Warmup（从 PG 拉历史）
export SKIP_DEEP_WARMUP=0
unset FCS_TA_MONTH_ISOLATED || true
unset OMS_MOCK_IBKR || true
unset REDIS_STREAM_SIM || true
unset FCS_MINUTE_PARITY_INJECT || true
unset GREEK_PARITY_MODE || true
export OMS_MOCK_IBKR=0
export REDIS_STREAM_SIM=0

if [[ "${LIVE_TRADE:-0}" == "1" ]]; then
  export RUN_MODE="${RUN_MODE:-REALTIME}"
  export TRADING_ENABLED=1
  export IS_SIMULATED=0
  echo "!!! LIVE_TRADE=1 → RUN_MODE=$RUN_MODE TRADING_ENABLED=1 !!!"
else
  export RUN_MODE="${RUN_MODE:-REALTIME_DRY}"
  export TRADING_ENABLED="${TRADING_ENABLED:-0}"
  export IS_SIMULATED="${IS_SIMULATED:-0}"
  echo "mode=REALTIME_DRY (set LIVE_TRADE=1 for real orders)"
fi
set +a

if ! redis-cli ping >/dev/null 2>&1; then
  echo "Redis down → starting redis-server"
  redis-server --daemonize yes
  sleep 1
fi

# ========== 预热门禁（不过关不启动） ==========
if [[ "${SKIP_WARMUP_GATE:-0}" != "1" ]]; then
  echo "=== [warmup gate] detect → seed/download → recheck ==="
  WARMUP_OUT="$LOG_DIR/f56_warmup_ready.json"
  set +e
  "$PY" -u "$REPO/qqq_btc/tools/ensure_fcs_warmup_ready.py" \
    --symbols QQQ,VIXY \
    --lookback-tdays "$WARMUP_LOOKBACK_TDAYS" \
    --min-day-bars "$WARMUP_MIN_DAY_BARS" \
    --min-total-1m "$WARMUP_MIN_TOTAL_1M" \
    --stock-root "$STOCK_ROOT" \
    --api-base "$MASSIVE_API_BASE" \
    --out "$WARMUP_OUT" \
    2>&1 | tee "$LOG_DIR/f56_warmup_ready.log"
  warmup_rc=${PIPESTATUS[0]}
  set -e
  if [[ "$warmup_rc" -ne 0 ]]; then
    echo "ERROR: warmup gate FAILED (rc=$warmup_rc). Refuse to start." >&2
    echo "  log: $LOG_DIR/f56_warmup_ready.log" >&2
    echo "  report: $WARMUP_OUT" >&2
    exit 2
  fi
  echo "[PASS] warmup gate → $WARMUP_OUT"
else
  echo "[warn] SKIP_WARMUP_GATE=1 — starting without warmup check"
fi

cd "$BASELINE"
export PYTHONPATH="$REPO:$BASELINE${PYTHONPATH:+:$PYTHONPATH}"

start_one() {
  local name="$1"
  local script="$2"
  local log="$LOG_DIR/${name}.log"
  if [[ ! -f "$script" ]]; then
    echo "[warn] missing $script — skip $name"
    return 0
  fi
  echo "start $name → $log"
  nohup "$PY" -u "$script" >"$log" 2>&1 &
  local pid=$!
  echo "  pid=$pid"
  sleep 2
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "  ok"
  else
    echo "[warn] $name exited early; tail $log:"
    tail -n 15 "$log" || true
  fi
}

echo "=== deploy F56 live (honest) ==="
echo "ckpt=$CKPT"
echo "slow=$SLOW_FEATURE_CONFIG"
echo "frozen=$FCS_FROZEN_NORM_PATH"
echo "put_gate=$QQQ_BTC_PUT_GATE_MODE regime_gold=$QQQ_BTC_REGIME_GOLD_1M"
echo "run_mode=$RUN_MODE trading=$TRADING_ENABLED skip_deep_warmup=$SKIP_DEEP_WARMUP"

PERS=""
for cand in \
  "$BASELINE/DAO/data_persistence_service_v8_pg.py" \
  "$BASELINE/data_persistence_service_v8_pg.py" \
  "$REPO/production/baseline/DAO/data_persistence_service_v8_pg.py"
do
  [[ -f "$cand" ]] && PERS="$cand" && break
done
if [[ -n "$PERS" ]]; then
  start_one "f56_persistence" "$PERS"
else
  echo "[warn] persistence script not found; continue"
fi

start_one "f56_ibkr" "$BASELINE/DAO/ibkr_connector_v8.py"
start_one "f56_fcs" "$BASELINE/DAO/feature_compute_service_v8.py"
sleep "${FCS_WAIT:-8}"

# 确认 FCS 日志里出现 Deep Warmup Complete（最多等 60s）
echo "=== wait FCS Deep Warmup ==="
fcs_log="$LOG_DIR/f56_fcs.log"
ok_warm=0
for _i in $(seq 1 30); do
  if [[ -f "$fcs_log" ]] && rg -q "Deep Warmup Complete|Warmup.*Complete" "$fcs_log"; then
    ok_warm=1
    break
  fi
  # 若 FCS 已死，失败退出
  if ! pgrep -f 'feature_compute_service_v8.py' >/dev/null; then
    echo "ERROR: FCS process died before warmup complete" >&2
    tail -n 40 "$fcs_log" || true
    stop_stack
    exit 3
  fi
  sleep 2
done
if [[ "$ok_warm" -ne 1 ]]; then
  echo "ERROR: FCS Deep Warmup not confirmed in ${fcs_log}" >&2
  tail -n 40 "$fcs_log" || true
  stop_stack
  exit 3
fi
echo "[PASS] FCS Deep Warmup Complete"

nohup env QQQ_BTC_LIVE=1 SLOW_FEATURE_CONFIG="$SLOW_FEATURE_CONFIG" \
  "$PY" -u "$REPO/qqq_btc/tools/run_live_signal_qqq.py" \
  --checkpoint "$CKPT" \
  --feature-config "$SLOW_CFG" \
  >"$LOG_DIR/f56_signal.log" 2>&1 &
echo "start f56_signal pid=$! → $LOG_DIR/f56_signal.log"
sleep 2

nohup env QQQ_BTC_LIVE=1 \
  "$PY" -u "$REPO/qqq_btc/tools/run_live_exec_qqq.py" \
  >"$LOG_DIR/f56_oms.log" 2>&1 &
echo "start f56_oms pid=$! → $LOG_DIR/f56_oms.log"
sleep 2

echo ""
echo "=== process check ==="
ps -ef | grep -E 'ibkr_connector_v8|feature_compute_service_v8|run_live_signal_qqq|run_live_exec_qqq' | grep -v grep || true
echo ""
echo "logs: $LOG_DIR/f56_{warmup_ready,ibkr,fcs,signal,oms}.log"
echo "stop: bash $REPO/qqq_btc/tools/deploy_ft56_julw1_live.sh stop"
echo "done"
