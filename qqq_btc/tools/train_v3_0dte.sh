#!/usr/bin/env bash
# options_databento_v3 隔离 0DTE: 特征/LMDB/训练 (不触碰共享 quote_features_train)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"

echo "=== v3 0DTE isolated pipeline + train ==="
"$PY" qqq_btc/tools/rebuild_v3_features.py --step post --epochs 20
"$PY" qqq_btc/tools/rebuild_v3_features.py --step fit --epochs 20

echo "=== IC report ==="
cat qqq_btc/results/v3_0dte_ic_sanity.json
echo "done -> checkpoints_qqq_v3_0dte/best.pth"
