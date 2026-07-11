#!/usr/bin/env bash
# V4.1 — V4 同代码逻辑 + fixed-8 固定 8 合约数据，从头 pretrain
#
# 与 V4 对齐:
#   - config: slow_feature_qqq_v4.json (28-dim stock 塔)
#   - mode: pretrain (无 init-checkpoint)
#   - split: train 2023-03~2025-12 / val 2026-01~03 / test 2026-04~06
#
# 与 V4 唯一实质差异: 特征/标签/IV 来自 fixed-8 管线 (quote_features_*_fixed8_v8)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
DATA_ROOT="$HOME/train_data/lmdb"
CONFIG="qqq_btc/CONFIG/slow_feature_qqq_v4.json"
SYM="qqq_btc/CONFIG/symbol_map.json"
ANCHOR="qqq_btc/CONFIG/anchor_qqq_0dte_v8_fixed8.json"
CKPT_OUT="checkpoints_qqq_v4_1_fixed8"
EVAL_OUT="/tmp/qqq_btc_test_eval_v4_1_fixed8"
FEAT_TEST="$HOME/train_data/quote_features_test_fixed8_v8"
LMDB_TRAIN="train_qqq_v4_1_fixed8.lmdb"
LMDB_VAL="val_qqq_v4_1_fixed8.lmdb"

echo "=== [1] 确认 fixed-8 特征 + V4 config LMDB ==="
for stage in train val; do
  lmdb="$DATA_ROOT/${stage}_qqq_v4_1_fixed8.lmdb"
  feat="$HOME/train_data/quote_features_${stage}_fixed8_v8"
  if [[ ! -d "$feat/QQQ/regular/09:30-16:00/1min" ]]; then
    echo "missing features: $feat"
    exit 1
  fi
  if [[ ! -f "$lmdb/data.mdb" ]]; then
    "$PY" qqq_btc/tools/build_lmdb.py \
      --feature-root "$feat" \
      --config "$CONFIG" --symbol-map "$SYM" \
      --output "$lmdb" --symbols QQQ
  else
    echo "skip existing $lmdb"
  fi
done

echo "=== [2] V4.1 pretrain: fixed-8 数据, V4 config, 无 init-checkpoint ==="
mkdir -p "$CKPT_OUT"
"$PY" -m qqq_btc.model.train \
  --mode pretrain \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --train-lmdb "$LMDB_TRAIN" \
  --val-lmdbs "$LMDB_VAL" \
  --checkpoint-dir "$CKPT_OUT" \
  --epochs 20 \
  --device auto 2>&1 | tee "$CKPT_OUT/train.log"

echo "=== [3] test(4-6月) infer + strict replay ==="
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_OUT/best.pth" \
  --config "$CONFIG" \
  --feature-root "$FEAT_TEST" \
  --option-1m-root /mnt/s990/data/raw_1m/options_databento_fixed8_corrected \
  --output-dir "$EVAL_OUT" \
  --device auto

echo "=== [4] V4.1 vs V4 vs V8 对比 ==="
"$PY" - <<'PY'
import json
from pathlib import Path
import pandas as pd
from qqq_btc.common.event_replay import prepare_minute_frame
from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg

f = qcfg.REPLAY.position_frac
kw = dict(
    edge_col='net_edge', edge_q10_col=qcfg.EDGE_Q10_COL,
    call_edge_col=qcfg.CALL_EDGE_COL, put_edge_col=qcfg.PUT_EDGE_COL,
    put_gate_col=qcfg.PUT_GATE_COL,
)

def metrics(path):
    if not Path(path).exists():
        return None
    df = prepare_minute_frame(pd.read_parquet(path))
    r = run_strict_replay(df, qcfg.FILL_MODEL, qcfg.REPLAY, qcfg.EXIT_RAILS, **kw)
    ne = df['net_edge'].dropna()
    monthly = {}
    for t in r.trades:
        m = pd.to_datetime(t.entry_ts).month
        monthly.setdefault(m, []).append(t.net_return)
    monthly_out = {}
    for m, rets in monthly.items():
        eq = 1.0
        for nr in rets:
            eq *= 1 + f * nr
        monthly_out[int(m)] = {'return_pct': (eq - 1) * 100, 'n_trades': len(rets)}
    return {
        'total_return_pct': r.summary()['total_net_return'] * 100,
        'n_trades': len(r.trades),
        'monthly': monthly_out,
        'net_edge_ge_003_pct': float((ne >= 0.03).mean() * 100) if len(ne) else 0.0,
        'pred_std': float(ne.std()) if len(ne) else 0.0,
    }

models = {
    'V4 (dynamic)': '/tmp/qqq_btc_abcd_A_v4_on_v4/test_infer.parquet',
    'V4.1 (fixed-8)': '/tmp/qqq_btc_test_eval_v4_1_fixed8/test_infer.parquet',
    'V8 finetune': '/tmp/qqq_btc_abcd_C_v8ft_on_v8/test_infer.parquet',
    'V8 scratch': '/tmp/qqq_btc_abcd_D_v8scr_on_v8/test_infer.parquet',
}
rows = []
summary = {}
for name, path in models.items():
    m = metrics(path)
    if m is None:
        continue
    summary[name] = {**m, 'infer_path': path}
    for month, md in m['monthly'].items():
        rows.append({'model': name, 'month': month, **md})
    print(f"{name}: Q2={m['total_return_pct']:.2f}% trades={m['n_trades']} edge>=0.03={m['net_edge_ge_003_pct']:.1f}%")

out = Path('qqq_btc/results/v4_1_fixed8_vs_v4_summary.json')
out.write_text(json.dumps({
    'tag': 'v4_1_fixed8',
    'description': 'V4 code logic (28-dim pretrain) on fixed-8 data',
    'checkpoint': 'checkpoints_qqq_v4_1_fixed8/best.pth',
    'models': summary,
}, indent=2), encoding='utf-8')
pd.DataFrame(rows).to_csv('qqq_btc/results/v4_1_fixed8_vs_v4_monthly.csv', index=False)
print(f"wrote {out}")
PY

echo "done -> $CKPT_OUT/best.pth"
echo "eval -> $EVAL_OUT/replay_summary.json"
