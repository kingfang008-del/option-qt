#!/usr/bin/env bash
# 按 7/5 qqq_pipeline_v4.sh 步骤 1-4 + eval，走规范路径（非 _v3 后缀）
set -euo pipefail

REPO="/home/kingfang007/文档/GitHub/option-qt"
PY="/home/kingfang007/anaconda3/envs/ibkr/bin/python"
LOG="/tmp/reproduce_july5_infer.log"
export FEATURE_CONFIG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v2.json"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

cd "$REPO"

log "=== [1/5] feature_merge (open30) → quote_features_raw ==="
"$PY" preprocess/ask_bid/feature_merge_option_raw.py

log "=== [2/5] split_raw_features → quote_features_{train,val,test} ==="
"$PY" preprocess/ask_bid/split_raw_features.py

log "=== [3/5] rolling_norm (分 stage) ==="
"$PY" preprocess/ask_bid/apply_rolling_norm_standalone.py

log "=== [4/5] label_pipeline (train/val/test, anchor_qqq_0dte) ==="
for stage in train val test; do
  log "label_pipeline $stage"
  "$PY" qqq_btc/tools/label_pipeline.py \
    --input "$HOME/train_data/quote_features_${stage}/QQQ/regular/09:30-16:00/1min" \
    --output "$HOME/train_data/quote_features_${stage}/QQQ/regular/09:30-16:00/1min" \
    --symbol QQQ \
    --anchor-config qqq_btc/CONFIG/anchor_qqq_0dte.json \
    --report "/tmp/label_report_${stage}_july5.json"
done

log "=== open30 验收 ==="
"$PY" - <<'PYEOF'
import pandas as pd
from pathlib import Path
p = sorted((Path.home() / "train_data/quote_features_train/QQQ/regular/09:30-16:00/1min").glob("*.parquet"))[-1]
df = pd.read_parquet(p, columns=["open30_max_ret", "open30_peak_dd", "open30_ret"])
print(p.name, df.describe())
assert (df["open30_max_ret"].abs().max() > 1e-6), "open30_max_ret 全零"
print("open30 OK")
PYEOF

log "=== [5/5] eval_test_set → /tmp/qqq_btc_test_eval_v4/test_infer.parquet ==="
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$REPO/checkpoints_qqq_v4/best.pth" \
  --config "$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json" \
  --feature-root "$HOME/train_data/quote_features_test" \
  --option-1m-root /mnt/s990/data/raw_1m/options_databento \
  --output-dir /tmp/qqq_btc_test_eval_v4 \
  --device cuda

log "=== DONE ==="
