#!/usr/bin/env bash
# MAG7 non-0DTE (DTE∈{1,2}) → V4 TFT. Isolated build; does not touch QQQ 0DTE paths.
#
#   bash stock_options/tools/train_stock_non0dte_tft_v4.sh NVDA
#   bash stock_options/tools/train_stock_non0dte_tft_v4.sh TSLA
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"
SYMBOL="$(echo "${1:-NVDA}" | tr '[:lower:]' '[:upper:]')"
EPOCHS="${EPOCHS:-20}"
DEVICE="${DEVICE:-auto}"
LOG="$REPO/stock_options/results/train_${SYMBOL,,}_non0dte_tft_v4.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"
echo "=== non0dte TFT $SYMBOL epochs=$EPOCHS $(date -Is) ==="
$PY stock_options/tools/rebuild_stock_non0dte_tft.py --symbol "$SYMBOL" --step show
$PY stock_options/tools/rebuild_stock_non0dte_tft.py --symbol "$SYMBOL" --step all --epochs "$EPOCHS" --device "$DEVICE" --force
echo "=== done checkpoint=checkpoints_${SYMBOL,,}_stock_non0dte_tft_v4/best.pth $(date -Is) ==="
