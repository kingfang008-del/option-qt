#!/usr/bin/env bash
# PUT-focused 1DTE experiment: reuse existing dte1 LMDB/features, train a separate checkpoint.
set -euo pipefail

REPO="/home/kingfang007/文档/GitHub/option-qt"
PY="/home/kingfang007/anaconda3/envs/ibkr/bin/python"
FEATURE_CFG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4_dte1_put_focus.json"
SYM_MAP="$REPO/qqq_btc/CONFIG/symbol_map.json"
CKPT_DIR="$REPO/checkpoints_qqq_v4_dte1_put_focus"
LMDB_ROOT="$HOME/train_data/lmdb"
TEST_FEAT="$HOME/train_data/quote_features_test_dte1"
OPTION_1M="/mnt/s990/data/raw_1m/dte1_options"
EVAL_OUT="/tmp/qqq_btc_test_eval_v4_dte1_put_focus"
LOG="$REPO/qqq_btc/results/train_v4_dte1_put_focus.log"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
mkdir -p "$CKPT_DIR" "$(dirname "$LOG")"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] === dte1 PUT-focused train start ==="
echo "config: $FEATURE_CFG"

rm -rf "$CKPT_DIR" "$EVAL_OUT"
mkdir -p "$CKPT_DIR"

"$PY" -m qqq_btc.model.train \
  --mode pretrain \
  --config "$FEATURE_CFG" \
  --data-root "$LMDB_ROOT" \
  --train-lmdb train_qqq_v4_dte1.lmdb \
  --val-lmdbs val_qqq_v4_dte1.lmdb \
  --checkpoint-dir "$CKPT_DIR" \
  --epochs 20 \
  --batch-size 1024 \
  --num-workers 8 \
  --device cuda \
  2>&1 | tee "$CKPT_DIR/train.log"

"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_DIR/best.pth" \
  --config "$FEATURE_CFG" \
  --feature-root "$TEST_FEAT" \
  --option-1m-root "$OPTION_1M" \
  --output-dir "$EVAL_OUT" \
  --device cuda

echo "[$(date '+%F %T')] === dte1 PUT-focused train done ==="
echo "checkpoint: $CKPT_DIR/best.pth"
echo "eval_out: $EVAL_OUT"
