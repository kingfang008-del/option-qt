#!/usr/bin/env bash
# 用 cracked old_v2 血缘重训 V4（验证能否再次达到 ~0.2 IC）
# 切分: train 2023-04~2025-12 | val 2026-01~03 | test 2026-04~06
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export FEATURE_CONFIG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v2.json"

OUT_ROOT="${OUT_ROOT:-$HOME/train_data/builds/v4_old_v2_retrain}"
LMDB_ROOT="${LMDB_ROOT:-$HOME/train_data/lmdb}"
CKPT_DIR="${CKPT_DIR:-$REPO/checkpoint/checkpoints_qqq_v4_old_v2_retrain}"
FEATURE_CFG_TRAIN="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json"
SYM_MAP="$REPO/qqq_btc/CONFIG/symbol_map.json"
DAYIV_SRC="$HOME/train_data/_bak_pre4c/dayiv_old_dirs/standard_old_v2"
TOOL="$REPO/qqq_btc/tools/reproduce_bak_lineage.py"
OPT1M="${OPT1M:-/mnt/s990/data/raw_1m/options_databento}"
LOG="$REPO/qqq_btc/results/train_v4_old_v2_retrain.log"
EVAL_VAL="$REPO/qqq_btc/results/v4_old_v2_retrain_val"
EVAL_TEST="$REPO/qqq_btc/results/v4_old_v2_retrain_test"

MONTHS_TRAIN="2023-04,2023-05,2023-06,2023-07,2023-08,2023-09,2023-10,2023-11,2023-12,2024-01,2024-02,2024-03,2024-04,2024-05,2024-06,2024-07,2024-08,2024-09,2024-10,2024-11,2024-12,2025-01,2025-02,2025-03,2025-04,2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,2025-12"
MONTHS_VAL="2026-01,2026-02,2026-03"
MONTHS_TEST="2026-04,2026-05,2026-06"
MONTHS_ALL="${MONTHS_TRAIN},${MONTHS_VAL},${MONTHS_TEST}"

mkdir -p "$OUT_ROOT" "$CKPT_DIR" "$LMDB_ROOT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"

echo "=== OUT_ROOT=$OUT_ROOT ==="
echo "=== CKPT_DIR=$CKPT_DIR ==="

echo "=== [1/7] mount old_v2 day_iv ==="
mkdir -p "$OUT_ROOT/quote_options_day_iv/QQQ"
rm -rf "$OUT_ROOT/quote_options_day_iv/QQQ/standard"
cp -a "$DAYIV_SRC" "$OUT_ROOT/quote_options_day_iv/QQQ/standard"
echo "dayiv files=$(ls "$OUT_ROOT/quote_options_day_iv/QQQ/standard" | wc -l)"

echo "=== [2/7] monthly + bucketed ==="
"$PY" "$TOOL" --out-root "$OUT_ROOT" monthly-bucketed --months "$MONTHS_ALL"

echo "=== [3/7] feature-merge ==="
"$PY" "$TOOL" --out-root "$OUT_ROOT" feature-merge --months "$MONTHS_ALL" \
  --feature-config "$FEATURE_CONFIG"

echo "=== [4/7] split + rolling-norm ==="
"$PY" "$TOOL" --out-root "$OUT_ROOT" split-norm --feature-config "$FEATURE_CONFIG"

echo "=== [5/7] label stages ==="
"$PY" "$TOOL" --out-root "$OUT_ROOT" label-stages

echo "=== sanity vs bak_train options (2025-08) ==="
"$PY" - <<PY
import numpy as np, pandas as pd
from pathlib import Path
a=pd.read_parquet(Path("$OUT_ROOT")/"quote_features_train/QQQ/regular/09:30-16:00/1min/2025-08.parquet")
b=pd.read_parquet(Path.home()/"train_data/_bak_pre4c/quote_features_train_QQQ/regular/09:30-16:00/1min/2025-08.parquet")
opts=[c for c in a.columns if c.startswith("options_") and c in b.columns]
cors=[]; mx=[]
for c in opts:
    x,y=a[c].astype(float),b[c].astype(float)
    n=min(len(x),len(y)); x,y=x.iloc[:n],y.iloc[:n]
    m=x.notna()&y.notna()
    cors.append(float(x[m].corr(y[m])))
    mx.append(float((x[m]-y[m]).abs().max()))
print("options vs bak_train: med_corr", float(np.median(cors)), "max_abs", float(np.max(mx)), "n", len(cors))
PY

echo "=== [6/7] build LMDB ==="
for stage in train val test; do
  out="$LMDB_ROOT/${stage}_qqq_v4_old_v2_retrain.lmdb"
  rm -rf "$out"
  "$PY" "$REPO/qqq_btc/tools/build_lmdb.py" \
    --feature-root "$OUT_ROOT/quote_features_${stage}" \
    --config "$FEATURE_CFG_TRAIN" \
    --symbol-map "$SYM_MAP" \
    --output "$out" \
    --symbols QQQ \
    --window-step 1
done

echo "=== [7/7] pretrain ==="
"$PY" -m qqq_btc.model.train \
  --mode pretrain \
  --config "$FEATURE_CFG_TRAIN" \
  --data-root "$LMDB_ROOT" \
  --train-lmdb train_qqq_v4_old_v2_retrain.lmdb \
  --val-lmdbs val_qqq_v4_old_v2_retrain.lmdb \
  --checkpoint-dir "$CKPT_DIR" \
  --epochs 20 \
  --batch-size 1024 \
  --num-workers 8 \
  --device cuda

echo "=== extract config from ckpt ==="
"$PY" - <<PY
import json, torch
from pathlib import Path
st = torch.load("$CKPT_DIR/best.pth", map_location="cpu", weights_only=False)
out = Path("$OUT_ROOT/slow_feature_from_ckpt.json")
out.write_text(json.dumps(st["config"], indent=2))
print("wrote", out, "best_ic", st.get("best_ic"), "epoch", st.get("epoch"))
PY

echo "=== eval val (2026-01..03) ==="
"$PY" "$REPO/qqq_btc/tools/eval_test_set.py" \
  --checkpoint "$CKPT_DIR/best.pth" \
  --config "$OUT_ROOT/slow_feature_from_ckpt.json" \
  --feature-root "$OUT_ROOT/quote_features_val" \
  --option-1m-root "$OPT1M" \
  --output-dir "$EVAL_VAL" \
  --device cuda

echo "=== eval test (2026-04..06) ==="
"$PY" "$REPO/qqq_btc/tools/eval_test_set.py" \
  --checkpoint "$CKPT_DIR/best.pth" \
  --config "$OUT_ROOT/slow_feature_from_ckpt.json" \
  --feature-root "$OUT_ROOT/quote_features_test" \
  --option-1m-root "$OPT1M" \
  --output-dir "$EVAL_TEST" \
  --device cuda

echo "=== SUMMARY ==="
"$PY" - <<PY
import json
from pathlib import Path
ck = __import__("torch").load("$CKPT_DIR/best.pth", map_location="cpu", weights_only=False)
print("train best_ic", ck.get("best_ic"), "epoch", ck.get("epoch"))
for name in ["$EVAL_VAL", "$EVAL_TEST"]:
    p = Path(name) / "replay_summary.json"
    if not p.exists():
        print(name, "MISSING"); continue
    s = json.load(open(p))
    print(Path(name).name,
          "ic=", s.get("label_metrics", {}).get("ic"),
          "acct25=", s.get("total_net_return"),
          "trades=", s.get("trades"))
PY
echo "DONE"
