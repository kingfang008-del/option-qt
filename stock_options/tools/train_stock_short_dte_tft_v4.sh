#!/usr/bin/env bash
# MAG7 short-DTE → V4 dual-stream TFT (same train entry as QQQ V4).
#
# Usage:
#   bash stock_options/tools/train_stock_short_dte_tft_v4.sh NVDA
#   bash stock_options/tools/train_stock_short_dte_tft_v4.sh TSLA
#
# Assumes day_iv / monthly / bucketed already exist under the short_dte roots
# from stock_options.common.short_dte_config. If not, build those first.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

SYMBOL="${1:-NVDA}"
SYMBOL="$(echo "$SYMBOL" | tr '[:lower:]' '[:upper:]')"
EPOCHS="${EPOCHS:-20}"
DEVICE="${DEVICE:-auto}"
START_STEP="${START_STEP:-1}"
LOG="${LOG:-$REPO/stock_options/results/train_${SYMBOL,,}_short_dte_tft_v4.log}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"

echo "=== stock short-DTE TFT V4 symbol=$SYMBOL epochs=$EPOCHS device=$DEVICE ==="
echo "=== START_STEP=$START_STEP $(date -Is) ==="

run_step() {
  local n="$1"; shift
  if (( START_STEP > n )); then
    echo "=== skip step $n ==="
    return 0
  fi
  echo "=== [$n] $* ==="
  "$@"
}

PIPE="$PY stock_options/tools/rebuild_short_dte_pipeline.py --symbol $SYMBOL"

run_step 1 $PIPE --step show
run_step 2 $PIPE --step feature-config
run_step 3 $PIPE --step weekday-report
run_step 4 $PIPE --step feature-merge --force
run_step 5 $PIPE --step filter --force
run_step 6 $PIPE --step label
run_step 7 $PIPE --step lmdb
run_step 8 $PIPE --step train --epochs "$EPOCHS" --device "$DEVICE"

echo "=== done $(date -Is) ==="
echo "checkpoint: checkpoints_${SYMBOL,,}_stock_short_dte_tft_v4/best.pth"
