#!/usr/bin/env bash
# 方案 B：新血缘隔离重建（不覆盖共享 quote_features_*）
#
# 原则:
#   1. 全部中间产物落在带 BUILD_ID 后缀的独立目录
#   2. 上游 1m 固定为 options_databento（= v3 聚合结果），禁止写回共享特征
#   3. 在本血缘上重新 label → LMDB → 训练（seed 固定）→ eval
#   4. 验收：2026-06 账户复利符号为正（允许与历史 97% 有差距）
#
# 用法:
#   bash qqq_btc/tools/rebuild_lineage_b.sh
#   SEED=42 BUILD_ID=v4_lineage_b bash qqq_btc/tools/rebuild_lineage_b.sh
#
# 跳过已有 day_iv 等重算（仅从已有隔离目录继续）:
#   SKIP_TO=label bash qqq_btc/tools/rebuild_lineage_b.sh
set -euo pipefail

REPO="/home/kingfang007/文档/GitHub/option-qt"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
BUILD_ID="${BUILD_ID:-v4_lineage_b}"
SEED="${SEED:-42}"
SKIP_TO="${SKIP_TO:-}"   # 空 | day_iv | monthly | bucket | merge | split | norm | label | lmdb | train | eval

STOCK_ROOT="${STOCK_ROOT:-/home/kingfang007/train_data/spnq_train_resampled}"
OPTION_1M="${OPTION_1M:-/mnt/s990/data/raw_1m/options_databento}"
FEATURE_CFG="${FEATURE_CFG:-$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json}"
SYM_MAP="$REPO/qqq_btc/CONFIG/symbol_map.json"
ANCHOR_CFG="$REPO/qqq_btc/CONFIG/anchor_qqq_0dte.json"

TRAIN_START="${TRAIN_START:-2023-03-01}"
TRAIN_END="${TRAIN_END:-2025-12-31}"
VAL_START="${VAL_START:-2026-01-01}"
VAL_END="${VAL_END:-2026-03-31}"
TEST_START="${TEST_START:-2026-04-01}"
TEST_END="${TEST_END:-2026-06-30}"

DAY_IV="$HOME/train_data/quote_options_day_iv_${BUILD_ID}"
MONTHLY_IV="$HOME/train_data/quote_options_monthly_iv_${BUILD_ID}"
BUCKETED="$HOME/train_data/quote_options_bucketed_v7_${BUILD_ID}"
RAW_FEAT="$HOME/train_data/quote_features_raw_${BUILD_ID}"
TRAIN_FEAT="$HOME/train_data/quote_features_train_${BUILD_ID}"
VAL_FEAT="$HOME/train_data/quote_features_val_${BUILD_ID}"
TEST_FEAT="$HOME/train_data/quote_features_test_${BUILD_ID}"
LMDB_ROOT="$HOME/train_data/lmdb"
CKPT_DIR="$REPO/checkpoint/checkpoints_qqq_${BUILD_ID}"
EVAL_OUT="$REPO/qqq_btc/results/${BUILD_ID}"
MANIFEST="$HOME/train_data/builds/${BUILD_ID}/manifest.json"
LOG="$REPO/qqq_btc/results/rebuild_${BUILD_ID}.log"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export FEATURE_CONFIG="$FEATURE_CFG"
export QQQ_BTC_SEED="$SEED"

cd "$REPO"
mkdir -p "$(dirname "$MANIFEST")" "$CKPT_DIR" "$(dirname "$LOG")" "$EVAL_OUT"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

should_run() {
  # SKIP_TO=X 表示从 X 步开始跑（含 X）；空则全跑
  local step="$1"
  local order=(day_iv monthly bucket merge split norm label lmdb train eval)
  if [[ -z "$SKIP_TO" ]]; then
    return 0
  fi
  local started=0
  for s in "${order[@]}"; do
    if [[ "$s" == "$SKIP_TO" ]]; then started=1; fi
    if [[ "$started" -eq 1 && "$s" == "$step" ]]; then return 0; fi
  done
  return 1
}

log "=== lineage-B rebuild start ==="
log "BUILD_ID=$BUILD_ID SEED=$SEED SKIP_TO=${SKIP_TO:-<none>}"
log "git HEAD: $(git rev-parse --short HEAD)"
log "option_1m(pinned)=$OPTION_1M"
log "splits: train ${TRAIN_START}..${TRAIN_END} | val ${VAL_START}..${VAL_END} | test ${TEST_START}..${TEST_END}"
log "isolated roots: day_iv=$DAY_IV raw=$RAW_FEAT test=$TEST_FEAT ckpt=$CKPT_DIR"

if should_run day_iv; then
  log "=== [1] day_iv (isolated) ==="
  rm -rf "$DAY_IV"
  "$PY" - <<PY
import multiprocessing
from preprocess.ask_bid.option_cac_day_vectorized_day import OptionIVCalculator
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass
calc = OptionIVCalculator(
    db_path="/home/kingfang007/notebook/stocks.db",
    option_root="$OPTION_1M",
    data_root="$STOCK_ROOT",
    iv_option_root="$DAY_IV",
)
calc.run(max_concurrent_stocks=12)
PY
fi

if should_run monthly; then
  log "=== [2] iv_day2month ==="
  rm -rf "$MONTHLY_IV"
  "$PY" - <<PY
import glob
from preprocess.ask_bid.iv_day2month import process_single_symbol
files = sorted(glob.glob("$DAY_IV/QQQ/standard/QQQ_*.parquet"))
print("day_iv files", len(files))
print(process_single_symbol(("QQQ", files, "$MONTHLY_IV")))
PY
fi

if should_run bucket; then
  log "=== [3] options_locked_feature ==="
  rm -rf "$BUCKETED"
  "$PY" - <<PY
import concurrent.futures, logging
from pathlib import Path
from tqdm import tqdm
from preprocess.ask_bid.options_locked_feature import process_single_file
RAW = Path("$MONTHLY_IV")
OUT = Path("$BUCKETED")
OUT.mkdir(parents=True, exist_ok=True)
tasks = [(p, OUT, "QQQ") for p in sorted((RAW / "QQQ" / "standard").glob("*.parquet"))]
with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
    futs = {ex.submit(process_single_file, t): t for t in tasks}
    for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
        r = f.result()
        if r:
            logging.warning(r)
print("bucketed months", len(tasks))
PY
fi

if should_run merge; then
  log "=== [4] feature_merge (isolated raw) ==="
  rm -rf "$RAW_FEAT"
  "$PY" - <<PY
from pathlib import Path
import preprocess.ask_bid.feature_merge_option_raw as fm
fm.OUTPUT_FEATURES_DIR = Path("$RAW_FEAT")
fm.OPTION_MONTHLY_DIR = Path("$MONTHLY_IV")
fm.AGG_OPTION_MONTHLY_DIR = Path("$BUCKETED")
fm.main()
PY
fi

if should_run split; then
  log "=== [5] split_raw_features (isolated) ==="
  rm -rf "$TRAIN_FEAT" "$VAL_FEAT" "$TEST_FEAT"
  "$PY" - <<PY
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from preprocess.ask_bid.split_raw_features import (
    process_and_copy_file, get_valid_symbols,
)
SOURCE, TRAIN, VAL, TEST = Path("$RAW_FEAT"), Path("$TRAIN_FEAT"), Path("$VAL_FEAT"), Path("$TEST_FEAT")
train_r = (pd.Timestamp("$TRAIN_START"), pd.Timestamp("$TRAIN_END"))
val_r = (pd.Timestamp("$VAL_START"), pd.Timestamp("$VAL_END"))
test_r = (pd.Timestamp("$TEST_START"), pd.Timestamp("$TEST_END"))
tasks = []
for sym in get_valid_symbols():
    sp = SOURCE / sym
    if sp.exists():
        tasks.extend(sp.glob("**/*.parquet"))
worker = partial(
    process_and_copy_file,
    source_dir=SOURCE, train_dir=TRAIN, val_dir=VAL, test_dir=TEST,
    train_range_ts=train_r, val_range_ts=val_r, test_range_ts=test_r,
)
with ProcessPoolExecutor(max_workers=32) as ex:
    list(tqdm(ex.map(worker, tasks), total=len(tasks), desc="split"))
print("split files", len(tasks))
PY
fi

if should_run norm; then
  log "=== [6] rolling_norm (isolated stages) ==="
  "$PY" - <<PY
import concurrent.futures, logging
from pathlib import Path
from tqdm import tqdm
import preprocess.ask_bid.apply_rolling_norm_standalone as arn
norm_cols = arn.load_target_features(arn.CONFIG_PATH)
for root in [Path("$TRAIN_FEAT"), Path("$VAL_FEAT"), Path("$TEST_FEAT")]:
    print("norm", root)
    if not root.exists():
        raise SystemExit(f"missing {root}")
    tasks = [(d, norm_cols) for d in arn.find_leaf_directories(root)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=arn.MAX_WORKERS) as ex:
        for res in tqdm(ex.map(arn.process_single_directory, tasks), total=len(tasks)):
            if res and str(res).startswith("ERROR"):
                logging.error(res)
    arn.verify_data_quality(root, norm_cols)
print("rolling norm done")
PY
fi

if should_run label; then
  log "=== [7] label_pipeline (fill labels on isolated features) ==="
  for stage_dir in "$TRAIN_FEAT" "$VAL_FEAT" "$TEST_FEAT"; do
    name=$(basename "$stage_dir")
    log "label $name"
    "$PY" qqq_btc/tools/label_pipeline.py \
      --input "$stage_dir/QQQ/regular/09:30-16:00/1min" \
      --output "$stage_dir/QQQ/regular/09:30-16:00/1min" \
      --symbol QQQ \
      --anchor-config "$ANCHOR_CFG" \
      --report "$EVAL_OUT/label_report_${name}.json"
  done
fi

if should_run lmdb; then
  log "=== [8] build LMDB (seed=$SEED, window-step=1) ==="
  rm -rf \
    "$LMDB_ROOT/train_qqq_${BUILD_ID}.lmdb" \
    "$LMDB_ROOT/val_qqq_${BUILD_ID}.lmdb" \
    "$LMDB_ROOT/test_qqq_${BUILD_ID}.lmdb"
  for pair in "train:$TRAIN_FEAT" "val:$VAL_FEAT" "test:$TEST_FEAT"; do
    stage="${pair%%:*}"
    root="${pair#*:}"
    "$PY" qqq_btc/tools/build_lmdb.py \
      --feature-root "$root" \
      --config "$FEATURE_CFG" \
      --symbol-map "$SYM_MAP" \
      --output "$LMDB_ROOT/${stage}_qqq_${BUILD_ID}.lmdb" \
      --symbols QQQ \
      --window-step 1
  done
fi

if should_run train; then
  log "=== [9] pretrain (seed=$SEED) ==="
  rm -rf "$CKPT_DIR"
  mkdir -p "$CKPT_DIR"
  "$PY" -m qqq_btc.model.train \
    --mode pretrain \
    --config "$FEATURE_CFG" \
    --data-root "$LMDB_ROOT" \
    --train-lmdb "train_qqq_${BUILD_ID}.lmdb" \
    --val-lmdbs "val_qqq_${BUILD_ID}.lmdb" \
    --checkpoint-dir "$CKPT_DIR" \
    --epochs 20 \
    --batch-size 1024 \
    --num-workers 8 \
    --device cuda \
    --seed "$SEED" \
    2>&1 | tee "$CKPT_DIR/train.log"

  "$PY" - <<PY
import json, torch
from pathlib import Path
st = torch.load("$CKPT_DIR/best.pth", map_location="cpu", weights_only=False)
out = Path("$EVAL_OUT/slow_feature_${BUILD_ID}.json")
out.write_text(json.dumps(st["config"], indent=2))
print("wrote", out, "ckpt_seed", st.get("seed"))
PY
fi

if should_run eval; then
  log "=== [10] test eval + replay (seed=$SEED) ==="
  CFG_EVAL="$EVAL_OUT/slow_feature_${BUILD_ID}.json"
  if [[ ! -f "$CFG_EVAL" ]]; then
    CFG_EVAL="$FEATURE_CFG"
  fi
  "$PY" qqq_btc/tools/eval_test_set.py \
    --checkpoint "$CKPT_DIR/best.pth" \
    --config "$CFG_EVAL" \
    --feature-root "$TEST_FEAT" \
    --option-1m-root "$OPTION_1M" \
    --output-dir "$EVAL_OUT" \
    --device cuda \
    --seed "$SEED"

  log "=== [11] acceptance: June acct >= 0 ==="
  "$PY" qqq_btc/tools/accept_lineage_replay.py \
    --trades "$EVAL_OUT/replay_trades.parquet" \
    --summary "$EVAL_OUT/replay_summary.json" \
    --require-month 2026-06 \
    --min-acct 0.0 \
    --out "$EVAL_OUT/acceptance.json" \
    || log "WARNING: acceptance failed — 新血缘 6 月仍为负，需检查特征/标签后再训"
fi

log "=== write manifest ==="
"$PY" - <<PY
import json, subprocess
from pathlib import Path
from datetime import datetime, timezone

def n_parquet(root):
    p = Path(root)
    return len(list(p.glob("**/*.parquet"))) if p.exists() else 0

manifest = {
    "build_id": "$BUILD_ID",
    "plan": "B_new_lineage",
    "seed": int("$SEED"),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "splits": {
        "train": "$TRAIN_START..$TRAIN_END",
        "val": "$VAL_START..$VAL_END",
        "test": "$TEST_START..$TEST_END",
    },
    "paths": {
        "option_1m": "$OPTION_1M",
        "day_iv": "$DAY_IV",
        "monthly_iv": "$MONTHLY_IV",
        "bucketed": "$BUCKETED",
        "features_raw": "$RAW_FEAT",
        "features_train": "$TRAIN_FEAT",
        "features_val": "$VAL_FEAT",
        "features_test": "$TEST_FEAT",
        "lmdb_train": "$LMDB_ROOT/train_qqq_${BUILD_ID}.lmdb",
        "checkpoint": "$CKPT_DIR/best.pth",
        "eval_out": "$EVAL_OUT",
    },
    "counts": {
        "option_1m_qqq": n_parquet("$OPTION_1M/QQQ"),
        "day_iv_qqq": n_parquet("$DAY_IV/QQQ"),
        "test_1min": n_parquet("$TEST_FEAT/QQQ/regular/09:30-16:00/1min"),
    },
    "log": "$LOG",
}
for name in ("replay_summary.json", "acceptance.json"):
    p = Path("$EVAL_OUT") / name
    if p.exists():
        manifest[name.replace(".json", "")] = json.loads(p.read_text())
Path("$MANIFEST").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print("wrote", "$MANIFEST")
PY

log "=== lineage-B DONE ==="
log "manifest: $MANIFEST"
log "checkpoint: $CKPT_DIR/best.pth"
log "eval: $EVAL_OUT"
log "NOTE: 以后推理必须用本 BUILD 的 features_test + checkpoint，禁止混用旧共享目录"
