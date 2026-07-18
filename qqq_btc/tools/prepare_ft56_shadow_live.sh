#!/usr/bin/env bash
# F56 Shadow 实盘准备 / 启动（不下真单）
#
# 对齐离线金标口径：
#   - LIVE_REPLAY 门控 + VX 日频 selector / CHOP
#   - PUT quarantine + 次日半仓
#   - 早盘 QQQ/VIXY 15m 反向确认
#   - put_gate=vixy_z；无 greek-parity / 无 regime gold
#
# 用法:
#   bash qqq_btc/tools/prepare_ft56_shadow_live.sh check   # 只检查依赖
#   bash qqq_btc/tools/prepare_ft56_shadow_live.sh start   # 启动 shadow 栈
#   bash qqq_btc/tools/prepare_ft56_shadow_live.sh stop
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
ACTION="${1:-check}"
SHADOW_ROOT="${SHADOW_ROOT:-$HOME/quant_project/shadow}"
LOG_DIR="${LOG_DIR:-$HOME/quant_project/logs}"
CKPT="${CKPT:-$REPO/checkpoint/checkpoints_qqq_ft56_julw1/best.pth}"
VX_TERM="${QQQ_BTC_VX_TERM_STRUCTURE:-/mnt/s990/data/raw_1m/vix_futures_databento/vx_term_structure_1d.parquet}"
SPOT_ROOT="${QQQ_BTC_SPOT_ROOT:-$HOME/train_data/spnq_train_resampled}"
FROZEN="${FCS_FROZEN_NORM_PATH:-$REPO/qqq_btc/CONFIG/frozen_norm_qqq_daily.npz}"

mkdir -p "$SHADOW_ROOT" "$LOG_DIR"

check_deps() {
  echo "=== shadow readiness ==="
  local rc=0
  [[ -f "$CKPT" ]] && echo "[ok] ckpt $CKPT" || { echo "[MISSING] ckpt $CKPT"; rc=1; }
  [[ -f "$FROZEN" ]] && echo "[ok] frozen_norm $FROZEN" || { echo "[MISSING] frozen $FROZEN"; rc=1; }
  [[ -f "$VX_TERM" ]] && echo "[ok] vx_term $VX_TERM" || { echo "[MISSING] vx_term $VX_TERM"; rc=1; }
  [[ -d "$SPOT_ROOT/QQQ/regular/09:30-16:00/1min" ]] \
    && echo "[ok] QQQ lookback $SPOT_ROOT" \
    || { echo "[MISSING] QQQ lookback under $SPOT_ROOT"; rc=1; }
  [[ -d "$SPOT_ROOT/VIXY/regular/09:30-16:00/1min" ]] \
    && echo "[ok] VIXY lookback $SPOT_ROOT" \
    || { echo "[MISSING] VIXY lookback under $SPOT_ROOT"; rc=1; }
  if redis-cli ping >/dev/null 2>&1; then
    echo "[ok] redis"
  else
    echo "[warn] redis down (start will try daemonize)"
  fi
  "$PY" - <<'PY'
from datetime import date
from qqq_btc.live.vx_term_live import prior_vx_curve_slope
from qqq_btc.live.rule_profile_live import select_live_profile, rule_profile_selector_enabled
today = date.today()
slope = prior_vx_curve_slope(today)
print(f"[ok] rule_profile_selector={rule_profile_selector_enabled()} today={today} vx_slope={slope}")
name, meta = select_live_profile(today, vx_curve_slope=slope)
print(f"[ok] profile_today={name} meta={ {k:meta.get(k) for k in ('vx_curve_slope','qqq_up_frac','qqq_range_mean')} }")
PY
  cat > "$SHADOW_ROOT/shadow_manifest.json" <<EOF
{
  "mode": "shadow_live",
  "trading_enabled": 0,
  "run_mode": "REALTIME_DRY",
  "checkpoint": "$CKPT",
  "vx_term": "$VX_TERM",
  "spot_root": "$SPOT_ROOT",
  "frozen_norm": "$FROZEN",
  "rule_profile_selector": "vx",
  "put_gate": "vixy_z",
  "audit_dir": "$SHADOW_ROOT",
  "governor_state": "$SHADOW_ROOT/governor_quantile.pkl"
}
EOF
  echo "wrote $SHADOW_ROOT/shadow_manifest.json"
  return "$rc"
}

stop_shadow() {
  bash "$REPO/qqq_btc/tools/deploy_ft56_julw1_live.sh" stop
}

start_shadow() {
  check_deps
  export LIVE_TRADE=0
  export TRADING_ENABLED=0
  export RUN_MODE=REALTIME_DRY
  export IS_SIMULATED=0
  export QQQ_BTC_FILL_AUDIT=1
  export QQQ_BTC_GOVERNOR_STATE="${QQQ_BTC_GOVERNOR_STATE:-$SHADOW_ROOT/governor_quantile.pkl}"
  export QQQ_BTC_RULE_PROFILE_SELECTOR="${QQQ_BTC_RULE_PROFILE_SELECTOR:-vx}"
  export QQQ_BTC_VX_TERM_STRUCTURE="$VX_TERM"
  export QQQ_BTC_SPOT_ROOT="$SPOT_ROOT"
  export QQQ_BTC_USE_LIVE_REPLAY="${QQQ_BTC_USE_LIVE_REPLAY:-1}"
  export QQQ_BTC_APPLY_PUT_ENTRY_QUANTILE="${QQQ_BTC_APPLY_PUT_ENTRY_QUANTILE:-0}"
  export LOG_DIR
  echo "=== starting shadow via deploy (LIVE_TRADE=0) ==="
  bash "$REPO/qqq_btc/tools/deploy_ft56_julw1_live.sh"
  echo "shadow audit dir: $SHADOW_ROOT"
  echo "logs: $LOG_DIR"
}

case "$ACTION" in
  check) check_deps ;;
  start) start_shadow ;;
  stop) stop_shadow ;;
  *)
    echo "usage: $0 {check|start|stop}" >&2
    exit 2
    ;;
esac
