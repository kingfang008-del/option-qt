#!/usr/bin/env bash
# TFT 协议编排：Base 冻结 → 近月微调 → July old_style 前向报告
# 文档：qqq_btc/docs/tft_base_ft_july_forward_protocol.md
#
# 用法:
#   bash qqq_btc/tools/run_tft_base_ft_july_forward.sh check|download_july|build_july|adapt|forward|all
#   SKIP_TRAIN=1 bash ... adapt|all
#   FORCE_BASE_RETRAIN=1 bash ... all   # 显式才重训 Base（很重）
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

PHASE="${1:-check}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
FORCE_BASE_RETRAIN="${FORCE_BASE_RETRAIN:-0}"

LOCK_MAP="${LOCK_MAP:-$HOME/train_data/locked_targets_map_old_style_trading_1dte_jul2026.parquet}"
RAW1S_JUL="${RAW1S_JUL:-/mnt/s990/data/raw_1s/dte1_options_old_style_jul2026}"
EXP_JUL="${EXP_JUL:-$HOME/train_data/july_v4_old_style}"
ANCHOR_CFG="${ANCHOR_CFG:-preprocess/CONFIG/anchor_qqq_1dte_4bucket.json}"
START_DATE="${START_DATE:-2026-07-01}"
END_DATE="${END_DATE:-2026-07-16}"

CKPT_V4="${CKPT_V4:-checkpoint/checkpoints_qqq_v4/best.pth}"
CKPT_FT="${CKPT_FT:-checkpoint/checkpoints_qqq_ft56_julw1/best.pth}"
FROZEN="${FROZEN:-qqq_btc/CONFIG/frozen_norm_qqq_daily.npz}"
CONFIG="${CONFIG:-qqq_btc/CONFIG/slow_feature_qqq_v4.json}"
RESULTS_ROOT="${RESULTS_ROOT:-qqq_btc/results/offline_live_aligned}"
READY_JSON="${READY_JSON:-qqq_btc/results/tft_base_ft_july_forward_ready.json}"

log() { echo "[$(date '+%F %T')] $*"; }

phase_check() {
  log "=== check protocol readiness ==="
  "$PY" - <<PY
import json, os
from pathlib import Path
import pandas as pd

repo = Path("$REPO")
lock = Path("$LOCK_MAP").expanduser()
raw1s = Path("$RAW1S_JUL").expanduser()
exp = Path("$EXP_JUL").expanduser()
ckpt_v4 = repo / "$CKPT_V4"
ckpt_ft = repo / "$CKPT_FT"
frozen = repo / "$FROZEN"
lmdb = Path.home() / "train_data/lmdb"
out = {
    "protocol": "base_ft_july_forward",
    "lock_kind": "old_style_foresight",
    "forbid": ["val=1-6+test=7", "open_4bucket_as_train_kpi"],
    "paths": {
        "lock_map": str(lock),
        "raw1s_july": str(raw1s),
        "exp_july": str(exp),
        "ckpt_v4": str(ckpt_v4),
        "ckpt_ft": str(ckpt_ft),
        "frozen_norm": str(frozen),
    },
    "gates": {},
    "july_lock_days": [],
}
out["gates"]["lock_map"] = lock.is_file()
out["gates"]["ckpt_v4"] = ckpt_v4.is_file()
out["gates"]["ckpt_ft"] = ckpt_ft.is_file()
out["gates"]["frozen_norm"] = frozen.is_file()
out["gates"]["lmdb_train_v4"] = (lmdb / "train_qqq_v4.lmdb/data.mdb").is_file()
out["gates"]["lmdb_val_v4"] = (lmdb / "val_qqq_v4.lmdb/data.mdb").is_file()
out["gates"]["lmdb_test_v4"] = (lmdb / "test_qqq_v4.lmdb/data.mdb").is_file()
out["gates"]["massive_key"] = bool(os.environ.get("MASSIVE_API_KEY"))
out["gates"]["polygon_key"] = bool(os.environ.get("POLYGON_API_KEY"))
out["gates"]["raw1s_july"] = raw1s.is_dir() and any(raw1s.rglob("*.parquet"))
feat = exp / "quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet"
out["gates"]["july_feat_raw"] = feat.is_file()
if lock.is_file():
    df = pd.read_parquet(lock)
    col = "date_str" if "date_str" in df.columns else "date"
    out["july_lock_days"] = sorted(df[col].astype(str).str[:10].unique().tolist())
    out["july_lock_n_contracts"] = int(len(df))
out["gates"]["ready_adapt"] = out["gates"]["ckpt_v4"] and out["gates"]["lmdb_train_v4"] is not False
# adapt needs bak features; soft check
bak = Path.home() / "train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00/1min"
out["gates"]["bak_may_jun"] = (bak / "2026-05.parquet").is_file() and (bak / "2026-06.parquet").is_file()
out["gates"]["ready_download"] = out["gates"]["lock_map"] and (
    out["gates"]["massive_key"] or out["gates"]["polygon_key"]
)
out["gates"]["ready_forward"] = out["gates"]["july_feat_raw"] and out["gates"]["ckpt_v4"] and out["gates"]["ckpt_ft"]
out["gates"]["ready_base_retrain"] = out["gates"]["lmdb_train_v4"] and out["gates"]["lmdb_val_v4"]
out["next"] = []
if not out["gates"]["raw1s_july"]:
    out["next"].append("download_july")
elif not out["gates"]["july_feat_raw"]:
    out["next"].append("build_july")
else:
    out["next"].append("forward")
out["next"].append("adapt (SKIP_TRAIN=1 ok if ckpt_ft exists)")
out["note"] = "Do NOT expand val to Jan-Jun; July is FORWARD_REPORT only."
path = repo / "$READY_JSON"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(out, indent=2, ensure_ascii=False))
print(f"\nwrote {path}")
PY
}

phase_download_july() {
  log "=== download July old_style 1s (RTH 09:30–16:00 for TFT) ==="
  [[ -f "$LOCK_MAP" ]] || { echo "missing lock map: $LOCK_MAP"; exit 1; }
  mkdir -p "$RAW1S_JUL"
  "$PY" preprocess/download/step2_polygon_second_sniper_v1.py \
    --target-map "$LOCK_MAP" \
    --output-dir "$RAW1S_JUL" \
    --symbols QQQ \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --window-start 09:30 \
    --window-end 16:00 \
    --allow-partial \
    --force
  log "download done → $RAW1S_JUL"
}

phase_build_july() {
  log "=== build July old_style month features ==="
  [[ -d "$RAW1S_JUL" ]] || { echo "missing raw1s: $RAW1S_JUL (run download_july)"; exit 1; }
  RAW1S="$RAW1S_JUL" \
  LOCK_MAP="$LOCK_MAP" \
  EXP_OVERRIDE="$EXP_JUL" \
  bash qqq_btc/tools/build_v4_old_lock_month.sh 2026-07
  log "build done → $EXP_JUL"
}

phase_adapt() {
  log "=== Adapt: FT May-Jun (honest KPI script) ==="
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    SKIP_TRAIN=1 bash qqq_btc/tools/train_ft56_julw1_honest_kpi.sh
  else
    bash qqq_btc/tools/train_ft56_julw1_honest_kpi.sh
  fi
}

phase_forward() {
  log "=== Forward: July old_style frozen+VX (V4 + FT56) ==="
  local feat_raw="$EXP_JUL/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet"
  [[ -f "$feat_raw" ]] || { echo "missing $feat_raw (run build_july)"; exit 1; }
  [[ -f "$CKPT_V4" ]] || { echo "missing $CKPT_V4"; exit 1; }
  [[ -f "$CKPT_FT" ]] || { echo "missing $CKPT_FT"; exit 1; }

  local tmp_root="$EXP_JUL/eval_feat_july_forward"
  local infer_v4="qqq_btc/results/v4_jul_forward_old_style_infer"
  local infer_ft="qqq_btc/results/ft56_jul_forward_old_style_infer"
  local opt1m="$EXP_JUL/options_1m_2026-07"
  rm -rf "$tmp_root"
  mkdir -p "$tmp_root/QQQ/regular/09:30-16:00/1min" "$tmp_root/QQQ/regular/09:30-16:00/5min" "$infer_v4" "$infer_ft"
  cp "$feat_raw" "$tmp_root/QQQ/regular/09:30-16:00/1min/"
  local feat5="$EXP_JUL/quote_features_raw/QQQ/regular/09:30-16:00/5min/2026-07.parquet"
  [[ -f "$feat5" ]] && cp "$feat5" "$tmp_root/QQQ/regular/09:30-16:00/5min/"

  for tag_ckpt_out in "v4|$CKPT_V4|$infer_v4" "ft56|$CKPT_FT|$infer_ft"; do
    IFS='|' read -r tag ckpt outdir <<<"$tag_ckpt_out"
    log "infer $tag"
    "$PY" qqq_btc/tools/eval_test_set.py \
      --checkpoint "$ckpt" \
      --config "$CONFIG" \
      --feature-root "$tmp_root" \
      --option-1m-root "$opt1m" \
      --call-bucket 2 --put-bucket 0 \
      --frozen-norm "$FROZEN" \
      --live-replay \
      --output-dir "$outdir" \
      --device cuda \
      --seed 42
  done

  "$PY" - <<PY
import json
from pathlib import Path
repo = Path("$REPO")
feat_raw = "$feat_raw"
opt1m = "$opt1m"
profiles = {
    "v4_vx_jul_forward_old_style_v1": {
        "schema_version": 1,
        "profile_id": "v4_vx_jul_forward_old_style_v1",
        "description": "FORWARD_REPORT: V4 + old_style foresight July + frozen + VX (protocol base_ft_july_forward)",
        "base_replay": "LIVE_REPLAY",
        "model": {"checkpoint": "$CKPT_V4"},
        "inputs": {
            "infer_by_month": {"2026-07": "qqq_btc/results/v4_jul_forward_old_style_infer/test_infer.parquet"},
            "raw1_by_month": {"2026-07": feat_raw},
            "opt1m_by_month": {"2026-07": opt1m},
        },
        "replay_overrides": {
            "edge_q10_floor": -0.2,
            "apply_put_entry_quantile": False,
            "next_day_put_quarantine_loss": -0.02,
            "next_day_put_quarantine_vx_slope_min": 0.06,
            "next_day_all_leg_defense_loss": -0.05,
            "next_day_all_leg_defense_position_frac": 0.125,
            "next_day_all_leg_defense_vx_slope_min": 0.06,
        },
        "selector": {
            "mode": "vx",
            "rule_profiles": "qqq_btc/CONFIG/rule_profiles.json",
            "vx_term_structure": "/mnt/s990/data/raw_1m/vix_futures_databento/vx_term_structure_1d.parquet",
            "spot_root": "~/train_data/spnq_train_resampled",
        },
        "execution": {
            "put_gate_mode": "vixy_z",
            "tick_exits": "off",
            "live_label_shift_sec": 60,
            "fill_spread_frac": 0.775,
            "execution_delay_bars": 0,
            "oms_signal_delay_bars": 0,
        },
        "features": {
            "slow_feature_config": "$CONFIG",
            "frozen_norm": "$FROZEN",
            "scope_label": "July_forward_old_style_foresight",
            "protocol": "base_ft_july_forward",
            "lock": "old_style_foresight",
        },
    },
    "ft56_vx_jul_forward_old_style_v1": None,
}
profiles["ft56_vx_jul_forward_old_style_v1"] = json.loads(json.dumps(profiles["v4_vx_jul_forward_old_style_v1"]))
ft = profiles["ft56_vx_jul_forward_old_style_v1"]
ft["profile_id"] = "ft56_vx_jul_forward_old_style_v1"
ft["description"] = "FORWARD_REPORT: FT56 + old_style foresight July + frozen + VX"
ft["model"]["checkpoint"] = "$CKPT_FT"
ft["inputs"]["infer_by_month"]["2026-07"] = "qqq_btc/results/ft56_jul_forward_old_style_infer/test_infer.parquet"
outdir = repo / "qqq_btc/CONFIG/strategy_profiles"
for pid, body in profiles.items():
    p = outdir / f"{pid}.json"
    p.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
    print("wrote", p)
PY

  for pid in v4_vx_jul_forward_old_style_v1 ft56_vx_jul_forward_old_style_v1; do
    log "replay live-aligned $pid"
    "$PY" qqq_btc/tools/replay_offline_live_aligned.py \
      --months 2026-07 \
      --strategy-profile "qqq_btc/CONFIG/strategy_profiles/${pid}.json" \
      --out-name "${pid}_offline" \
      --skip-build
  done

  log "forward summaries under $RESULTS_ROOT/*_jul_forward_old_style_v1_offline/"
}

phase_base_retrain() {
  if [[ "$FORCE_BASE_RETRAIN" != "1" ]]; then
    log "skip Base retrain (set FORCE_BASE_RETRAIN=1 to run; heavy)"
    return 0
  fi
  log "=== Base retrain (explicit) ==="
  # Prefer existing LMDB retrain only (no full feature rebuild) unless REBUILD_FEATURES=1
  if [[ "${REBUILD_FEATURES:-0}" == "1" ]]; then
    bash qqq_btc/tools/train_v4_old_v2_retrain.sh
  else
    local ckpt_out="${CKPT_BASE_OUT:-checkpoint/checkpoints_qqq_v4_protocol_retrain}"
    rm -rf "$ckpt_out"
    mkdir -p "$ckpt_out"
    "$PY" -m qqq_btc.model.train \
      --mode pretrain \
      --config "$CONFIG" \
      --data-root "$HOME/train_data/lmdb" \
      --train-lmdb train_qqq_v4.lmdb \
      --val-lmdbs val_qqq_v4.lmdb \
      --checkpoint-dir "$ckpt_out" \
      --epochs "${EPOCHS:-20}" \
      --batch-size 1024 \
      --num-workers 8 \
      --device cuda
    log "Base retrain → $ckpt_out/best.pth (test=4-6 still frozen; eval separately)"
  fi
}

case "$PHASE" in
  check) phase_check ;;
  download_july) phase_download_july ;;
  build_july) phase_build_july ;;
  adapt) phase_adapt ;;
  forward) phase_forward ;;
  base_retrain) FORCE_BASE_RETRAIN=1 phase_base_retrain ;;
  all)
    phase_check
    phase_base_retrain
    phase_download_july
    phase_build_july
    phase_adapt
    phase_forward
    phase_check
    ;;
  *)
    echo "usage: $0 check|download_july|build_july|adapt|forward|base_retrain|all"
    exit 2
    ;;
esac
