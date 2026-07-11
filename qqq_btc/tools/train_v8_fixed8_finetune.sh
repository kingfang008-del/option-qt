#!/usr/bin/env bash
# V8 fixed-8: V4 checkpoint → fixed-8 LMDB finetune (28-dim V4 config 对齐)
# 对比 scratch pretrain: 验证 fixed-8 数据能否在 V4 权重上迁移
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
DATA_ROOT="$HOME/train_data/lmdb"
CONFIG="qqq_btc/CONFIG/slow_feature_qqq_v4.json"
SYM="qqq_btc/CONFIG/symbol_map.json"
CKPT_V4="checkpoints_qqq_v4/best.pth"
CKPT_OUT="checkpoints_qqq_v8_fixed8_finetune"
EVAL_OUT="/tmp/qqq_btc_test_eval_v8_fixed8_finetune"
FEAT_TEST="$HOME/train_data/quote_features_test_fixed8_v8"
LMDB_TRAIN="train_qqq_v8_fixed8.lmdb"
LMDB_VAL="val_qqq_v8_fixed8.lmdb"

if [[ ! -f "$CKPT_V4" ]]; then
  echo "missing V4 checkpoint: $CKPT_V4"
  exit 1
fi
if [[ ! -f "$DATA_ROOT/$LMDB_TRAIN/data.mdb" ]]; then
  echo "missing LMDB: $DATA_ROOT/$LMDB_TRAIN — 请先跑 train_v8_fixed8.sh Phase 7-8"
  exit 1
fi

echo "=== [1] V8 fixed-8 finetune: init=$CKPT_V4, config=$CONFIG ==="
mkdir -p "$CKPT_OUT"
"$PY" -m qqq_btc.model.train \
  --mode finetune \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --train-lmdb "$LMDB_TRAIN" \
  --val-lmdbs "$LMDB_VAL" \
  --checkpoint-dir "$CKPT_OUT" \
  --init-checkpoint "$CKPT_V4" \
  --epochs 20 \
  --device auto 2>&1 | tee "$CKPT_OUT/train.log"

echo "=== [2] test(4-6月) infer + strict replay ==="
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_OUT/best.pth" \
  --config "$CONFIG" \
  --feature-root "$FEAT_TEST" \
  --option-1m-root /mnt/s990/data/raw_1m/options_databento_fixed8_corrected \
  --output-dir "$EVAL_OUT" \
  --device auto

echo "done -> $CKPT_OUT/best.pth"
echo "eval -> $EVAL_OUT/replay_summary.json"
