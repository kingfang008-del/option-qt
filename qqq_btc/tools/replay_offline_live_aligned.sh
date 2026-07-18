#!/usr/bin/env bash
# 离线 LIVE 对齐 replay 包装（默认 6+7 月，带版本 manifest）
#
# 用法:
#   bash qqq_btc/tools/replay_offline_live_aligned.sh
#   bash qqq_btc/tools/replay_offline_live_aligned.sh --months 2026-06,2026-07
#   bash qqq_btc/tools/replay_offline_live_aligned.sh --print-version
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# 若未显式指定 --out / --out-name，给一个可读的固定名（仍写 stamp 进 manifest）
has_out=0
for a in "$@"; do
  case "$a" in
    --out|--out-name|--print-version) has_out=1; break ;;
  esac
done

extra=()
if [[ "$has_out" -eq 0 ]]; then
  extra+=(--out-name "jun_jul_live_aligned")
fi

exec "$PY" -u "$REPO/qqq_btc/tools/replay_offline_live_aligned.py" "${extra[@]}" "$@"
