#!/usr/bin/env bash
# V4.0 全链路重建：v3 1m → day_iv → monthly → bucket → merge → split → norm → label
#                    → LMDB → pretrain → test eval
#
# 数据切分（与 7/5 qqq_pipeline_v4 一致）:
#   train 2023-03 ~ 2025-12
#   val   2026-01 ~ 2026-03
#   test  2026-04 ~ 2026-06
#
# 代码 pin: a170dc6（7/5 v3→v4 pipeline 入库快照）
set -euo pipefail

REPO="/home/kingfang007/文档/GitHub/option-qt"
PY="/home/kingfang007/anaconda3/envs/ibkr/bin/python"
CODE_REF="a170dc6"
LOG="/tmp/rebuild_train_v4.0.log"
MANIFEST="$HOME/train_data/builds/v4.0/manifest.json"
CKPT_DIR="$REPO/checkpoints_qqq_v4.0"
LMDB_ROOT="$HOME/train_data/lmdb"
EVAL_OUT="/tmp/qqq_btc_test_eval_v4.0"
FEATURE_CFG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v2.json"
SYM_MAP="$REPO/qqq_btc/CONFIG/symbol_map.json"
ANCHOR_CFG="$REPO/qqq_btc/CONFIG/anchor_qqq_0dte.json"
OPTION_1M="/mnt/s990/data/raw_1m/options_databento"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export FEATURE_CONFIG="$FEATURE_CFG"

exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

cd "$REPO"
mkdir -p "$(dirname "$MANIFEST")" "$CKPT_DIR"

log "=== V4.0 full rebuild start ==="
log "git HEAD: $(git rev-parse --short HEAD) pin pipeline @$CODE_REF"

# --- pin 7/5 pipeline 代码 ---
git show "${CODE_REF}:preprocess/ask_bid/feature_merge_option_raw.py" > preprocess/ask_bid/feature_merge_option_raw.py
git show "${CODE_REF}:qqq_btc/CONFIG/slow_feature_qqq_v2.json" > qqq_btc/CONFIG/slow_feature_qqq_v2.json
git show "${CODE_REF}:qqq_btc/tools/label_pipeline.py" > qqq_btc/tools/label_pipeline.py
git show "${CODE_REF}:qqq_btc/qqq/anchor.py" > qqq_btc/qqq/anchor.py

log "=== purge QQQ pipeline outputs (force day_iv rebuild, no skip-if-exists) ==="
for sub in \
  quote_options_day_iv/QQQ \
  quote_options_monthly_iv/QQQ \
  quote_options_bucketed_v7/QQQ \
  quote_features_raw/QQQ \
  quote_features_train/QQQ \
  quote_features_val/QQQ \
  quote_features_test/QQQ; do
  rm -rf "$HOME/train_data/${sub}"
done
rm -rf "$CKPT_DIR" "$EVAL_OUT" \
  "$LMDB_ROOT/train_qqq_v4.0.lmdb" \
  "$LMDB_ROOT/val_qqq_v4.0.lmdb" \
  "$LMDB_ROOT/test_qqq_v4.0.lmdb"

log "=== [1/10] option_cac_day_vectorized_day (v3 1m → day_iv) ==="
"$PY" - <<'PY'
import multiprocessing
from preprocess.ask_bid.option_cac_day_vectorized_day import OptionIVCalculator

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

calc = OptionIVCalculator(
    db_path="/home/kingfang007/notebook/stocks.db",
    option_root="/mnt/s990/data/raw_1m/options_databento",
    data_root="/home/kingfang007/train_data/spnq_train_resampled",
    iv_option_root="/home/kingfang007/train_data/quote_options_day_iv",
)
calc.run(max_concurrent_stocks=12)
PY

log "=== [2/10] iv_day2month ==="
"$PY" preprocess/ask_bid/iv_day2month.py

log "=== [3/10] options_locked_feature (bucketed_v7) ==="
"$PY" preprocess/ask_bid/options_locked_feature.py

log "=== [4/10] feature_merge_option_raw ==="
"$PY" preprocess/ask_bid/feature_merge_option_raw.py

log "=== [5/10] split_raw_features ==="
"$PY" preprocess/ask_bid/split_raw_features.py

log "=== [6/10] apply_rolling_norm ==="
"$PY" preprocess/ask_bid/apply_rolling_norm_standalone.py

log "=== [7/10] label_pipeline (train/val/test) ==="
for stage in train val test; do
  log "label_pipeline $stage"
  "$PY" qqq_btc/tools/label_pipeline.py \
    --input "$HOME/train_data/quote_features_${stage}/QQQ/regular/09:30-16:00/1min" \
    --output "$HOME/train_data/quote_features_${stage}/QQQ/regular/09:30-16:00/1min" \
    --symbol QQQ \
    --anchor-config "$ANCHOR_CFG" \
    --report "/tmp/label_report_v4.0_${stage}.json"
done

log "=== label stats ==="
"$PY" - <<'PY'
import json
for stage in ("train", "val", "test"):
    rep = json.load(open(f"/tmp/label_report_v4.0_{stage}.json"))
    print(stage, "files", rep["files"], "avg_net_std", round(rep["avg_net_std"], 4))
PY

log "=== [8/10] build LMDB v4.0 (window-step=1) ==="
for stage in train val test; do
  "$PY" qqq_btc/tools/build_lmdb.py \
    --feature-root "$HOME/train_data/quote_features_${stage}" \
    --config "$FEATURE_CFG" \
    --symbol-map "$SYM_MAP" \
    --output "$LMDB_ROOT/${stage}_qqq_v4.0.lmdb" \
    --symbols QQQ \
    --window-step 1
done

log "=== [9/10] pretrain V4.0 ==="
"$PY" -m qqq_btc.model.train \
  --mode pretrain \
  --config "$FEATURE_CFG" \
  --data-root "$LMDB_ROOT" \
  --train-lmdb train_qqq_v4.0.lmdb \
  --val-lmdbs val_qqq_v4.0.lmdb \
  --checkpoint-dir "$CKPT_DIR" \
  --epochs 20 \
  --batch-size 1024 \
  --num-workers 8 \
  --device cuda \
  2>&1 | tee "$CKPT_DIR/train.log"

log "=== extract eval config from checkpoint ==="
"$PY" - <<PY
import json, torch
from pathlib import Path
st = torch.load("$CKPT_DIR/best.pth", map_location="cpu", weights_only=False)
out = Path("$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.0.json")
out.write_text(json.dumps(st["config"], indent=2))
print("wrote", out)
PY

log "=== [10/10] test eval + strict replay ==="
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_DIR/best.pth" \
  --config "$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.0.json" \
  --feature-root "$HOME/train_data/quote_features_test" \
  --option-1m-root "$OPTION_1M" \
  --output-dir "$EVAL_OUT" \
  --device cuda

log "=== write manifest ==="
"$PY" - <<PY
import json, subprocess, hashlib, glob, os
from pathlib import Path
from datetime import datetime, timezone

def count_parquet(root):
    return len(list(Path(root).glob("**/*.parquet"))) if Path(root).exists() else 0

manifest = {
    "build_id": "v4.0",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "code_ref": "$CODE_REF",
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "splits": {
        "train": "2023-03-01..2025-12-31",
        "val": "2026-01-01..2026-03-31",
        "test": "2026-04-01..2026-06-30",
    },
    "paths": {
        "option_1m": "$OPTION_1M",
        "day_iv": str(Path.home() / "train_data/quote_options_day_iv"),
        "monthly_iv": str(Path.home() / "train_data/quote_options_monthly_iv"),
        "bucketed": str(Path.home() / "train_data/quote_options_bucketed_v7"),
        "features_test": str(Path.home() / "train_data/quote_features_test"),
        "checkpoint": "$CKPT_DIR/best.pth",
        "eval_out": "$EVAL_OUT",
    },
    "counts": {
        "day_iv_qqq": count_parquet(Path.home() / "train_data/quote_options_day_iv/QQQ"),
        "monthly_qqq": count_parquet(Path.home() / "train_data/quote_options_monthly_iv/QQQ"),
        "test_1min": count_parquet(Path.home() / "train_data/quote_features_test/QQQ/regular/09:30-16:00/1min"),
    },
    "log": "$LOG",
}
if Path("$EVAL_OUT/replay_summary.json").exists():
    manifest["replay_summary"] = json.loads(Path("$EVAL_OUT/replay_summary.json").read_text())
if Path("$EVAL_OUT/test_infer.parquet").exists():
    manifest["test_infer"] = str(Path("$EVAL_OUT/test_infer.parquet"))

Path("$MANIFEST").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY

log "=== V4.0 DONE ==="
log "checkpoint: $CKPT_DIR/best.pth"
log "eval: $EVAL_OUT/test_infer.parquet"
log "manifest: $MANIFEST"
