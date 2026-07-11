#!/usr/bin/env bash
# v5 一轮训练: LMDB 重建 → finetune(v4 初始化) → test 推理回放
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
DATA_ROOT="$HOME/train_data/lmdb"
CONFIG="qqq_btc/CONFIG/slow_feature_qqq_v2.json"
SYM="qqq_btc/CONFIG/symbol_map.json"
CKPT_V4="checkpoints_qqq_v4/best.pth"
CKPT_V5="checkpoints_qqq_v5"
EVAL_OUT="/tmp/qqq_btc_test_eval_v5"
SEED="${SEED:-42}"
export QQQ_BTC_SEED="$SEED"

echo "=== [1/3] 重建 v5 LMDB (含 chop 特征) ==="
for stage in train val test; do
  "$PY" qqq_btc/tools/build_lmdb.py \
    --feature-root "$HOME/train_data/quote_features_${stage}" \
    --config "$CONFIG" \
    --symbol-map "$SYM" \
    --output "$DATA_ROOT/${stage}_qqq_v5.lmdb" \
    --symbols QQQ
done

mkdir -p "$CKPT_V5"
echo "=== [2/3] v5 finetune (init=$CKPT_V4, seed=$SEED, 仅训 stock塔+主头) ==="
mkdir -p "$CKPT_V5"
"$PY" -m qqq_btc.model.train \
  --mode finetune \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --train-lmdb train_qqq_v5.lmdb \
  --val-lmdbs val_qqq_v5.lmdb \
  --checkpoint-dir "$CKPT_V5" \
  --init-checkpoint "$CKPT_V4" \
  --epochs 20 \
  --seed "$SEED" \
  --device auto 2>&1 | tee "$CKPT_V5/train.log"

echo "=== [3/3] test 推理 + strict replay ==="
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_V5/best.pth" \
  --feature-root "$HOME/train_data/quote_features_test" \
  --option-1m-root /mnt/s990/data/raw_1m/options_databento \
  --output-dir "$EVAL_OUT" \
  --seed "$SEED" \
  --device auto

echo "done -> $CKPT_V5/best.pth"
echo "eval -> $EVAL_OUT/replay_summary.json"
