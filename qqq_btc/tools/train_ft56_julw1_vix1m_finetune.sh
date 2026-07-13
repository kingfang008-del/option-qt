#!/usr/bin/env bash
# Step2：V4 → finetune，模型特征 vix_level 用 1min（对比 Step1：仅 put_gate=1min、模型仍 5min）
#
# 与 train_ft56_julw1.sh 相同数据切分（bak 5–6 月 finetune，July W1 评测），
# 唯一差异：CONFIG=slow_feature_qqq_v4_vix1m.json（vix_level resolution=1min）。
#
# 用法:
#   bash qqq_btc/tools/train_ft56_julw1_vix1m_finetune.sh
#   CKPT_V4=checkpoint/checkpoints_qqq_v4/best.pth bash ...
#
# 对比:
#   Step1 流式: put_gate=vixy_z + ckpt=ft56(5min vix 模型)
#   Step2 离线: put_gate 仍可用 1min raw；模型=本脚本 ckpt（1min vix 特征）
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
SEED="${SEED:-42}"
export QQQ_BTC_SEED="$SEED"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="$HOME/train_data/lmdb"
CONFIG="${CONFIG:-qqq_btc/CONFIG/slow_feature_qqq_v4_vix1m.json}"
SYM="qqq_btc/CONFIG/symbol_map.json"
CKPT_V4="${CKPT_V4:-checkpoint/checkpoints_qqq_v4/best.pth}"
CKPT_OUT="${CKPT_OUT:-checkpoint/checkpoints_qqq_ft56_julw1_vix1m}"
EVAL_FT="${EVAL_FT:-qqq_btc/results/ft56_julw1_vix1m}"
EVAL_BASE="${EVAL_BASE:-qqq_btc/results/ft56_julw1_vix5m_ref}"
RESULTS="${RESULTS:-qqq_btc/results/ft56_julw1_vix1m_compare}"

FEAT_BAK="${FEAT_BAK:-$HOME/train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00}"
# 评测默认用诚实 July 特征（与 Step1 流式同宇宙）；可改回 databento
FEAT_JUL="${FEAT_JUL:-$HOME/train_data/july_w1_v4_honest_openwin/quote_features_test}"
OPT1M_JUL="${OPT1M_JUL:-/mnt/s990/data/v4_original_jul5/databento_july_w1_openwin/options_1m}"
# fallback 旧路径
if [[ ! -d "$OPT1M_JUL/QQQ" ]]; then
  OPT1M_JUL="${OPT1M_JUL_FALLBACK:-$HOME/train_data/july_w1_v4_databento/options_1m_july_w1}"
fi

TRAIN_FEAT="/tmp/quote_features_ft56_vix1m_train"
VAL_FEAT="/tmp/quote_features_ft56_vix1m_val"
LMDB_TRAIN="train_qqq_ft56_vix1m.lmdb"
LMDB_VAL="val_qqq_ft56_vix1m.lmdb"

setup_months() {
  local dest="$1"; shift
  rm -rf "$dest"
  mkdir -p "$dest/QQQ/regular/09:30-16:00/1min" "$dest/QQQ/regular/09:30-16:00/5min"
  for ym in "$@"; do
    ln -sf "$FEAT_BAK/1min/${ym}.parquet" "$dest/QQQ/regular/09:30-16:00/1min/${ym}.parquet"
    ln -sf "$FEAT_BAK/5min/${ym}.parquet" "$dest/QQQ/regular/09:30-16:00/5min/${ym}.parquet"
  done
}

echo "=== [0] check inputs ==="
for ym in 2026-05 2026-06; do
  [[ -f "$FEAT_BAK/1min/${ym}.parquet" ]] || { echo "missing bak $ym"; exit 1; }
done
[[ -f "$FEAT_JUL/QQQ/regular/09:30-16:00/1min/2026-07.parquet" ]] || { echo "missing july feat: $FEAT_JUL"; exit 1; }
[[ -d "$OPT1M_JUL/QQQ" ]] || { echo "missing july option 1m: $OPT1M_JUL"; exit 1; }
[[ -f "$CKPT_V4" ]] || { echo "missing V4: $CKPT_V4"; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG"; exit 1; }

echo "=== [1] feature dirs: train=5+6, val=6 (config=$CONFIG) ==="
setup_months "$TRAIN_FEAT" 2026-05 2026-06
setup_months "$VAL_FEAT" 2026-06

echo "=== [2] build LMDB (vix_level from 1min tree) ==="
"$PY" qqq_btc/tools/build_lmdb.py \
  --feature-root "$TRAIN_FEAT" --config "$CONFIG" --symbol-map "$SYM" \
  --output "$DATA_ROOT/$LMDB_TRAIN" --symbols QQQ --window-step 1
"$PY" qqq_btc/tools/build_lmdb.py \
  --feature-root "$VAL_FEAT" --config "$CONFIG" --symbol-map "$SYM" \
  --output "$DATA_ROOT/$LMDB_VAL" --symbols QQQ --window-step 1

echo "=== [3] finetune V4 → vix1m (seed=$SEED) ==="
rm -rf "$CKPT_OUT"
mkdir -p "$CKPT_OUT"
"$PY" -m qqq_btc.model.train \
  --mode finetune \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --train-lmdb "$LMDB_TRAIN" \
  --val-lmdbs "$LMDB_VAL" \
  --checkpoint-dir "$CKPT_OUT" \
  --init-checkpoint "$CKPT_V4" \
  --epochs "${EPOCHS:-20}" \
  --batch-size 512 \
  --num-workers 4 \
  --seed "$SEED" \
  --device "${DEVICE:-cuda}" 2>&1 | tee "$CKPT_OUT/train.log"

echo "=== [4] July W1 offline replay: ft56(5min-vix ckpt) vs vix1m（同用 1min raw put_gate）==="
mkdir -p "$EVAL_FT" "$EVAL_BASE" "$RESULTS"
# 对照：现有 ft56 ckpt + 原 5min vix config（若存在）
CKPT_FT56="${CKPT_FT56:-checkpoint/checkpoints_qqq_ft56_julw1/best.pth}"
CONFIG_5M="${CONFIG_5M:-qqq_btc/CONFIG/slow_feature_qqq_v4.json}"
# Step1 门控：1min raw（不用 --causal-5m，避免误加 +5min）
PUT_GATE_1M="${PUT_GATE_1M:-$HOME/train_data/july_w1_v4_honest_openwin/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet}"
if [[ -f "$CKPT_FT56" ]]; then
  "$PY" qqq_btc/tools/eval_test_set.py \
    --checkpoint "$CKPT_FT56" --config "$CONFIG_5M" \
    --feature-root "$FEAT_JUL" --option-1m-root "$OPT1M_JUL" \
    --output-dir "$EVAL_BASE" --seed "$SEED" --device "${DEVICE:-cuda}" \
    --live-replay --put-gate-raw5 "$PUT_GATE_1M" || true
fi

"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_OUT/best.pth" --config "$CONFIG" \
  --feature-root "$FEAT_JUL" --option-1m-root "$OPT1M_JUL" \
  --output-dir "$EVAL_FT" --seed "$SEED" --device "${DEVICE:-cuda}" \
  --live-replay --put-gate-raw5 "$PUT_GATE_1M"

echo "=== [5] compare summary ==="
"$PY" - <<PY
import json
from pathlib import Path
import pandas as pd
from qqq_btc.common.event_replay import prepare_minute_frame
from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg

f = qcfg.LIVE_REPLAY.position_frac
kw = dict(
    edge_col="net_edge", edge_q10_col=qcfg.EDGE_Q10_COL,
    call_edge_col=qcfg.CALL_EDGE_COL, put_edge_col=qcfg.PUT_EDGE_COL,
    put_gate_col=qcfg.PUT_GATE_COL,
)
W1 = {"2026-07-01","2026-07-02","2026-07-06","2026-07-07","2026-07-08","2026-07-09","2026-07-10"}

def summarize(path, label):
    p = Path(path)
    if not p.exists():
        return {"label": label, "missing": True}
    df = prepare_minute_frame(pd.read_parquet(p))
    r = run_strict_replay(df, qcfg.FILL_MODEL, qcfg.LIVE_REPLAY, qcfg.EXIT_RAILS, **kw)
    s = r.summary(position_frac=f)
    trades = r.trades_frame()
    return {
        "label": label,
        "trades": int(s.get("trades") or 0),
        "acct25": float(s.get("total_net_return") or 0.0),
        "hit": float(s["hit_rate"]) if s.get("hit_rate") is not None else None,
        "legs": s.get("trades_by_leg") or {},
        "n_rows": int(len(trades)),
    }

rows = [
    summarize(Path("$EVAL_BASE") / "test_infer.parquet", "ft56_vix5m_model"),
    summarize(Path("$EVAL_FT") / "test_infer.parquet", "ft56_vix1m_model"),
]
out = {
    "seed": int("$SEED"),
    "step": "2_model_vix_1min_finetune",
    "init": "$CKPT_V4",
    "ckpt_vix1m": "$CKPT_OUT/best.pth",
    "config_vix1m": "$CONFIG",
    "feat_july": "$FEAT_JUL",
    "note": "Compare model-feature vix 5min vs 1min; put_gate still independent (prefer vixy_z in live).",
    "results": rows,
}
Path("$RESULTS").mkdir(parents=True, exist_ok=True)
(Path("$RESULTS") / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
PY

echo "done ckpt=$CKPT_OUT/best.pth"
echo "summary=$RESULTS/summary.json"
