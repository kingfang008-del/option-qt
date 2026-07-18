#!/usr/bin/env bash
# V0 V4 honest 无 VX：正式三闸门流式对拍入口。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

PROFILE="${V0_PROFILE:-$REPO/qqq_btc/CONFIG/strategy_profiles/v4_honest_v0_parity_v1.json}"
OUT_DIR="${V0_STREAM_OUT_DIR:-$REPO/qqq_btc/results/july_w1_v4_v0_gated_$(date +%Y%m%d_%H%M%S)}"
DAYS="${V0_DAYS:-2026-07-01 2026-07-02 2026-07-06 2026-07-07 2026-07-08 2026-07-09}"

[[ -f "$PROFILE" ]] || { echo "missing V0 profile: $PROFILE" >&2; exit 2; }

# 正式入口拒绝历史 SKIP/FORCE 路径，并清理可能让 profile 静默分叉的 QQQ_BTC_*。
unset SKIP_GATES FORCE_GATE3 CKPT SLOW_FEATURE_CONFIG FCS_FROZEN_NORM_PATH
unset BACKTEST_OPT_FILL_SPREAD_FRAC EXECUTION_DELAY_BARS OMS_SIGNAL_DELAY_BARS
while IFS= read -r name; do
  case "$name" in
    QQQ_BTC_*) unset "$name" ;;
  esac
done < <(compgen -v)

export QQQ_BTC_STRATEGY_PROFILE="$PROFILE"
export HONEST_OUT_DIR="$OUT_DIR"
export DAYS

echo "V0 profile: $PROFILE"
echo "V0 days:    $DAYS"
echo "V0 output:  $OUT_DIR"
echo "Gates:      required (SKIP_GATES/FORCE_GATE3 disabled)"

exec bash "$REPO/qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh"
