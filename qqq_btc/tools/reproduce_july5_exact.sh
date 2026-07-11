#!/usr/bin/env bash
# 7/5 13:20 训练快照复现：
#   - 代码: commit a170dc6（含 open30，与 13:20 checkpoint 一致）
#   - 特征: _bak_pre4c/quote_features_test_QQQ（mtime 2026-07-05 13:19:59）
# 说明: 7/5 13:19 的上游 day_iv/monthly 中间态已不可从 v3 1m 精确重建；
#       仅 test 阶段特征快照完整保留，eval 可 1:1 复现 IC=0.111 / 697% replay。
set -euo pipefail

REPO="/home/kingfang007/文档/GitHub/option-qt"
PY="/home/kingfang007/anaconda3/envs/ibkr/bin/python"
CODE_REF="a170dc6"
BAK="$HOME/train_data/_bak_pre4c/quote_features_test_QQQ"
CANON="$HOME/train_data/quote_features_test/QQQ"
OUT="/tmp/qqq_btc_test_eval_v4"
LOG="/tmp/reproduce_july5_exact.log"

exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

log "=== [1/4] checkout pipeline @ ${CODE_REF} ==="
git show "${CODE_REF}:preprocess/ask_bid/feature_merge_option_raw.py" > preprocess/ask_bid/feature_merge_option_raw.py
git show "${CODE_REF}:qqq_btc/CONFIG/slow_feature_qqq_v2.json" > qqq_btc/CONFIG/slow_feature_qqq_v2.json
git show "${CODE_REF}:qqq_btc/tools/label_pipeline.py" > qqq_btc/tools/label_pipeline.py

log "=== [2/4] restore 7/5 13:19 test features → ${CANON} ==="
rm -rf "${CANON}.pre_restore."*
if [[ -d "$CANON" ]]; then
  mv "$CANON" "${CANON}.pre_restore.$(date +%s)"
fi
mkdir -p "$(dirname "$CANON")"
cp -a "$BAK" "$CANON"
log "restored files: $(find "$CANON" -name '*.parquet' | wc -l)"
log "sample mtime: $(stat -c '%y' "$CANON/regular/09:30-16:00/1min/2026-04.parquet")"

log "=== [3/4] eval v4 checkpoint ==="
mkdir -p "$OUT"
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$REPO/checkpoints_qqq_v4/best.pth" \
  --config "$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json" \
  --feature-root "$HOME/train_data/quote_features_test" \
  --option-1m-root /mnt/s990/data/raw_1m/options_databento \
  --output-dir "$OUT" \
  --device cuda

log "=== [4/4] verify vs 7/5 archived infer ==="
"$PY" - <<'PY'
import sys
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "/home/kingfang007/文档/GitHub/option-qt")
from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.common.event_replay import prepare_minute_frame
from qqq_btc.qqq import config as qcfg

arch = pd.read_parquet("/home/kingfang007/train_data/eval_v4/test_infer.parquet", columns=["timestamp", "net_edge"])
new = pd.read_parquet("/tmp/qqq_btc_test_eval_v4/test_infer.parquet", columns=["timestamp", "net_edge"])
m = arch.merge(new, on="timestamp", suffixes=("_arch", "_new"))
ic_pair = spearmanr(m["net_edge_arch"], m["net_edge_new"]).correlation
print(f"infer corr vs 7/5 archive: {ic_pair:.6f}")
print(f"max |diff|: {(m['net_edge_arch'] - m['net_edge_new']).abs().max():.6f}")

df = prepare_minute_frame(pd.read_parquet("/tmp/qqq_btc_test_eval_v4/test_infer.parquet"))
r = run_strict_replay(
    df, qcfg.FILL_MODEL, qcfg.REPLAY, qcfg.EXIT_RAILS,
    edge_col="net_edge", edge_q10_col=qcfg.EDGE_Q10_COL,
    call_edge_col=qcfg.CALL_EDGE_COL, put_edge_col=qcfg.PUT_EDGE_COL,
    put_gate_col=qcfg.PUT_GATE_COL,
)
s = r.summary(position_frac=0.25)
print(f"R1+R2 replay: trades={s['trades']} total={s['total_net_return']*100:+.2f}% legs={s.get('trades_by_leg')}")
PY

log "=== DONE: ${OUT}/test_infer.parquet ==="
