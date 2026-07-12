#!/usr/bin/env bash
# Train V4 on prefer_primary rebuild features.
# Prereq: bash qqq_btc/tools/rebuild_0dte_prefer_primary.sh (full range done)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

FEAT_ROOT="${FEAT_ROOT:-$HOME/train_data/builds/0dte_prefer_primary}"
LMDB_ROOT="${LMDB_ROOT:-$HOME/train_data/lmdb}"
CKPT_DIR="${CKPT_DIR:-$REPO/checkpoint/checkpoints_qqq_v4_prefer_primary}"
FEATURE_CFG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json"
SYM_MAP="$REPO/qqq_btc/CONFIG/symbol_map.json"
OPT1M="$FEAT_ROOT/raw_1m_prefer_primary"
EVAL_H2="$REPO/qqq_btc/results/v4_prefer_primary_h2_after_train"
EVAL_TEST="$REPO/qqq_btc/results/v4_prefer_primary_test_after_train"
LOG="$REPO/qqq_btc/results/train_v4_prefer_primary.log"

mkdir -p "$CKPT_DIR" "$LMDB_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "=== build LMDB ==="
for stage in train val test; do
  out="$LMDB_ROOT/${stage}_qqq_v4_prefer_primary.lmdb"
  rm -rf "$out"
  "$PY" "$REPO/qqq_btc/tools/build_lmdb.py" \
    --feature-root "$FEAT_ROOT/quote_features_${stage}" \
    --config "$FEATURE_CFG" \
    --symbol-map "$SYM_MAP" \
    --output "$out" \
    --symbols QQQ \
    --window-step 1
done

echo "=== pretrain ==="
"$PY" -m qqq_btc.model.train \
  --mode pretrain \
  --config "$FEATURE_CFG" \
  --data-root "$LMDB_ROOT" \
  --train-lmdb train_qqq_v4_prefer_primary.lmdb \
  --val-lmdbs val_qqq_v4_prefer_primary.lmdb \
  --checkpoint-dir "$CKPT_DIR" \
  --epochs 20 \
  --batch-size 1024 \
  --num-workers 8 \
  --device cuda

echo "=== extract eval config ==="
"$PY" - <<PY
import json, torch
from pathlib import Path
st = torch.load("$CKPT_DIR/best.pth", map_location="cpu", weights_only=False)
out = Path("$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4_prefer_primary.json")
out.write_text(json.dumps(st["config"], indent=2))
print("wrote", out)
PY

echo "=== H2 replay (train months 2025-07..12) ==="
H2_FEAT="$FEAT_ROOT/h2_eval_features"
mkdir -p "$H2_FEAT/QQQ/regular/09:30-16:00/1min" "$H2_FEAT/QQQ/regular/09:30-16:00/5min"
for m in 2025-07 2025-08 2025-09 2025-10 2025-11 2025-12; do
  cp -f "$FEAT_ROOT/quote_features_train/QQQ/regular/09:30-16:00/1min/${m}.parquet" "$H2_FEAT/QQQ/regular/09:30-16:00/1min/"
  cp -f "$FEAT_ROOT/quote_features_train/QQQ/regular/09:30-16:00/5min/${m}.parquet" "$H2_FEAT/QQQ/regular/09:30-16:00/5min/"
done
"$PY" "$REPO/qqq_btc/tools/eval_test_set.py" \
  --checkpoint "$CKPT_DIR/best.pth" \
  --config "$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4_prefer_primary.json" \
  --feature-root "$H2_FEAT" \
  --option-1m-root "$OPT1M" \
  --output-dir "$EVAL_H2" \
  --device cuda

echo "=== test Apr-Jun replay ==="
"$PY" "$REPO/qqq_btc/tools/eval_test_set.py" \
  --checkpoint "$CKPT_DIR/best.pth" \
  --config "$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4_prefer_primary.json" \
  --feature-root "$FEAT_ROOT/quote_features_test" \
  --option-1m-root "$OPT1M" \
  --output-dir "$EVAL_TEST" \
  --device cuda

echo "DONE"
"$PY" - <<PY
import json
from pathlib import Path
for name in ["$EVAL_H2", "$EVAL_TEST"]:
    p=Path(name)/"replay_summary.json"
    s=json.load(open(p))
    print(Path(name).name, "ic=", s["label_metrics"]["ic"], "acct25=", s["total_net_return"], "trades=", s["trades"])
PY
