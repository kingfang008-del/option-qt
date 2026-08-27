#!/bin/bash
# Mag7 开盘前 → 实盘 一键流程（日历/新闻政策 + 会话启停）
#
# 流程：
#   1) sync-calendar（FOMC/财报 + hard_risk 新闻；大单只审计）
#   2) 打印今日禁入摘要（full-day / symbol）
#   3) 启动 Shadow/Paper/Live（复用 start_maga7_live_session.sh）
#
# 用法：
#   ./run_maga7_live_flow.sh                 # 默认：sync + 摘要 + start shadow
#   ./run_maga7_live_flow.sh preopen         # 只做 1+2（不启动）
#   ./run_maga7_live_flow.sh start paper --account DUxxx
#   ./run_maga7_live_flow.sh start live --account Uxxx --live-orders
#   ./run_maga7_live_flow.sh status|stop|sync|help
#
# 环境变量（可选）：
#   MAG7_NEWS_MODE=hard_risk|audit   默认 hard_risk（仅 CEO 可自动禁票）
#   MAG7_NO_NEWS=1                   跳过公司新闻拉取
#   MAG7_SKIP_SYNC=1                 跳过日历同步
#   MAG7_FLOW_NO_START=1             只 preopen
#   MAG7_ACCOUNT / MAG7_PROFILE 等   同 start_maga7_live_session.sh

set -euo pipefail

SHELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SHELL_DIR/../.." && pwd)"
LAUNCHER="$SHELL_DIR/start_maga7_live_session.sh"
CAL_JSON="$PROJECT_ROOT/maga7/CONFIG/event_calendar_live.json"
AUDIT_JSON="$PROJECT_ROOT/maga7/CONFIG/event_news_audit.json"
DEFAULT_PROFILE="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${CYAN}$*${NC}"; }
log_ok() { echo -e "${GREEN}$*${NC}"; }
log_warn() { echo -e "${YELLOW}$*${NC}"; }
log_err() { echo -e "${RED}$*${NC}"; }

usage() {
    cat <<EOF
Mag7 live flow（开盘前日历/新闻 + 会话）

Usage:
  $(basename "$0") [preopen|start|sync|status|stop|help] [mode] [launcher-options...]

Commands:
  (default) / start   sync-calendar → 今日摘要 → start_maga7_live_session.sh start …
  preopen             只 sync + 摘要（不启动会话）
  sync                只同步日历
  status|stop         转发给 start_maga7_live_session.sh
  help                本说明

Examples:
  $(basename "$0")                          # shadow 全流程
  $(basename "$0") preopen                  # 盘前准备
  $(basename "$0") start shadow
  $(basename "$0") start dry                 # :4001 live MD + dry OMS
  MAG7_ACCOUNT=DUxxx $(basename "$0") start paper --account DUxxx

  # 新闻完全不自动禁票：
  MAG7_NEWS_MODE=audit $(basename "$0") preopen

Policy:
  宏观 full-day 禁入 | 财报/CEO symbol 禁入 | 大单合作只审计+LLM
  新闻永不设定交易方向（见 maga7/common/event_news_policy.py）

Dash 审核（可选）：
  cd $PROJECT_ROOT && python dash/run.py   # → Event News
EOF
}

ny_date() {
    TZ=America/New_York date +%F
}

cmd_sync() {
    local start_d="${1:-}"
    local end_d="${2:-}"
    if [ "${MAG7_SKIP_SYNC:-0}" = "1" ]; then
        log_warn "MAG7_SKIP_SYNC=1 — skip sync-calendar"
        return 0
    fi
    log_info "=== [1/3] sync-calendar (news-mode=${MAG7_NEWS_MODE:-hard_risk}) ==="
    if [ -n "$start_d" ] && [ -n "$end_d" ]; then
        "$LAUNCHER" sync-calendar "$start_d" "$end_d"
    else
        # 默认：近 14 日 → +180 日（覆盖财报窗 + 近端新闻）
        local start end
        read -r start end < <(
            python - <<'PY'
from datetime import date, timedelta
t = date.today()
print((t - timedelta(days=14)).isoformat(), (t + timedelta(days=180)).isoformat())
PY
        )
        "$LAUNCHER" sync-calendar "$start" "$end"
    fi
    log_ok "calendar → $CAL_JSON"
    if [ -f "$AUDIT_JSON" ]; then
        log_ok "news audit → $AUDIT_JSON"
    fi
}

cmd_summary() {
    log_info "=== [2/3] today's blackout summary (NY=$(ny_date)) ==="
    cd "$PROJECT_ROOT"
    MAG7_EVENT_CALENDAR_PATH="$CAL_JSON" \
    MAG7_PROFILE="${MAG7_PROFILE:-$DEFAULT_PROFILE}" \
    python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from maga7.common.config import load_profile
from maga7.common.event_calendar import resolve_live_event_blackout
from maga7.common.event_news_policy import POLICY_SUMMARY_ZH, policy_from_profile

repo = Path(".").resolve()
profile_path = os.environ.get("MAG7_PROFILE") or (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
cal = os.environ.get("MAG7_EVENT_CALENDAR_PATH") or str(
    repo / "maga7/CONFIG/event_calendar_live.json"
)
os.environ["MAG7_EVENT_CALENDAR_PATH"] = cal

# NY trade date
try:
    from zoneinfo import ZoneInfo
    from datetime import datetime

    trade_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
except Exception:
    from datetime import date

    trade_date = date.today().isoformat()

prof = load_profile(profile_path)
pol = policy_from_profile(prof)
full, meta = resolve_live_event_blackout(prof, trade_date=trade_date)

print(f"policy: {pol['summary_zh']}")
print(f"mode:   company_news_mode={pol['company_news_mode']} "
      f"direction_from_news={pol['company_news_direction_from_news']} "
      f"llm_blackout={pol['company_news_use_llm_for_blackout']}")
print(f"trade_date (NY): {trade_date}")
print(f"calendar_file:   {cal} exists={Path(cal).is_file()}")
print(f"sources:         {meta.get('sources')}")
if meta.get("active_today_full") or meta.get("active_today"):
    print(f"TODAY: FULL-DAY BLACKOUT  dates⊃{trade_date}")
    print("       → OMS day_halted（全日不入场）")
elif meta.get("active_today_symbols"):
    print(f"TODAY: SYMBOL BLOCK {meta.get('active_today_symbols')}")
    print("       → 仅禁这些标的；其它 Mag7 照常")
else:
    print("TODAY: no live-file / preset hit for this session date")
    print("       （研究 preset 历史日仍在 profile 名单里，但不影响「今天」）")
sym = meta.get("symbol_blackout") or {}
if sym:
    # show nearby
    near = {d: v for d, v in sorted(sym.items()) if abs(
        (__import__("datetime").date.fromisoformat(d) -
         __import__("datetime").date.fromisoformat(trade_date)).days
    ) <= 14}
    if near:
        print(f"symbol_blackout (±14d): {near}")
print(f"preset full-day count: {len(meta.get('blackout_dates') or [])}")
print("---")
print("optional: python dash/run.py  → Event News（LLM 利好/空，不进黑名单）")
PY
}

cmd_start() {
    log_info "=== [3/3] start live session ==="
    # first arg may be mode; pass through everything
    if [ "$#" -eq 0 ]; then
        "$LAUNCHER" start shadow
    elif [ "$1" = "shadow" ] || [ "$1" = "dry" ] || [ "$1" = "paper" ] || [ "$1" = "live" ]; then
        "$LAUNCHER" start "$@"
    else
        # options only → default shadow
        "$LAUNCHER" start shadow "$@"
    fi
}

cmd_flow() {
    local mode_args=("$@")
    cmd_sync
    cmd_summary
    if [ "${MAG7_FLOW_NO_START:-0}" = "1" ]; then
        log_warn "MAG7_FLOW_NO_START=1 — stop after preopen"
        return 0
    fi
    cmd_start "${mode_args[@]}"
}

main() {
    if [ ! -x "$LAUNCHER" ]; then
        chmod +x "$LAUNCHER" 2>/dev/null || true
    fi
    local cmd="${1:-start}"
    case "$cmd" in
        help|-h|--help)
            usage
            ;;
        preopen)
            shift || true
            # optional sync date range
            if [ "${1:-}" = "--" ]; then shift || true; fi
            cmd_sync "$@"
            cmd_summary
            log_ok "preopen done. Start when ready:"
            log_info "  $0 start dry       # :4001 + dry OMS"
            log_info "  $0 start shadow    # :4002 + dry OMS"
            ;;
        sync|sync-calendar)
            shift || true
            cmd_sync "$@"
            ;;
        summary)
            cmd_summary
            ;;
        status|stop)
            "$LAUNCHER" "$cmd"
            ;;
        start)
            shift || true
            cmd_flow "$@"
            ;;
        shadow|dry|paper|live)
            # ./run_maga7_live_flow.sh dry|shadow …
            cmd_flow "$@"
            ;;
        *)
            # treat unknown as launcher passthrough after full flow with shadow
            log_warn "unknown command '$cmd' — running full flow as shadow with args"
            cmd_flow "$@"
            ;;
    esac
}

main "$@"
