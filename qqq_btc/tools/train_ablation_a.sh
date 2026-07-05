#!/usr/bin/env bash
# 消融 A: V4 init + chop 特征 + rank_net=0 (复用 v5 LMDB)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
DATA_ROOT="$HOME/train_data/lmdb"
BASE_CONFIG="qqq_btc/CONFIG/slow_feature_qqq_v2.json"
ABLATION_CONFIG="/tmp/slow_feature_qqq_ablation_a.json"
SYM="qqq_btc/CONFIG/symbol_map.json"
CKPT_V4="checkpoints_qqq_v4/best.pth"
CKPT_OUT="checkpoints_qqq_v5a"
EVAL_OUT="/tmp/qqq_btc_test_eval_v5a"

"$PY" - <<'PY'
import json
from pathlib import Path
base = Path("qqq_btc/CONFIG/slow_feature_qqq_v2.json")
cfg = json.loads(base.read_text(encoding="utf-8"))
cfg["loss_weights"]["rank_net"] = 0.0
out = Path("/tmp/slow_feature_qqq_ablation_a.json")
out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"ablation config -> {out} (rank_net=0)")
PY

mkdir -p "$CKPT_OUT"
echo "=== [1/2] 消融 A finetune (init=$CKPT_V4, chop only, rank_net=0) ==="
"$PY" -m qqq_btc.model.train \
  --mode finetune \
  --config "$ABLATION_CONFIG" \
  --data-root "$DATA_ROOT" \
  --train-lmdb train_qqq_v5.lmdb \
  --val-lmdbs val_qqq_v5.lmdb \
  --checkpoint-dir "$CKPT_OUT" \
  --init-checkpoint "$CKPT_V4" \
  --epochs 20 \
  --device auto 2>&1 | tee "$CKPT_OUT/train.log"

echo "=== [2/2] test 推理 + strict replay ==="
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_OUT/best.pth" \
  --feature-root "$HOME/train_data/quote_features_test" \
  --option-1m-root /mnt/s990/data/raw_1m/options_databento \
  --output-dir "$EVAL_OUT" \
  --device auto

echo "done -> $CKPT_OUT/best.pth"
echo "eval -> $EVAL_OUT/replay_summary.json"
