#!/usr/bin/env bash
# 兼容入口：转发到 restart_ft56_july_w1_stream_parity.sh
# （保留旧文件名，避免历史命令失效）
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "$REPO/qqq_btc/tools/restart_ft56_july_w1_stream_parity.sh" "$@"
