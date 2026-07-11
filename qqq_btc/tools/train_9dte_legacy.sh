#!/usr/bin/env bash
# Legacy 9DTE: 完整流水线 + pretrain (数据源 /mnt/s990/data/raw_1s/options/QQQ)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"

echo "=== Legacy 9DTE pipeline + train ==="
"$PY" qqq_btc/tools/rebuild_9dte_legacy_pipeline.py --step all --epochs 20

echo "=== IC summary (last lines of train.log) ==="
tail -30 checkpoints_qqq_9dte_legacy/train.log

echo "done -> checkpoints_qqq_9dte_legacy/best.pth"
