#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
probe_alpha_stream_gap.py
=========================

针对 "alpha_logs 有缺口" 的 Redis 侧实锤探针.

工作原理:
  - `STREAM_TRADE_LOG` 的 Redis Stream ID 格式为 `<ms-ts>-<seq>`.
    ms-ts 就是 xadd 时的服务端毫秒墙钟, 因此可以直接用分钟窗口来 XRANGE.
  - 逐分钟扫描窗口, 统计该分钟里:
      - Stream 里实际保留的消息数 (分裂 ALPHA / OTHER)
      - Stream 里保留的 ALPHA 覆盖了多少个不同 symbol
  - 如果 PG 里 alpha_logs 这一分钟是 0, 而 Stream 里还有 ALPHA 消息 →
    说明被 DPS 漏消费了; 如果 Stream 里也是 0 → 说明 SE 没发 或 被 maxlen 裁掉.
  - 再取 XINFO STREAM 看 first-entry / last-entry 的 ID 覆盖范围, 判断
    maxlen 是否已经"吃掉"了目标窗口.

用法:
  python production/scripts/probe_alpha_stream_gap.py \
      --date 20260416 --start 09:45 --end 10:10
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, time as dt_time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "production" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from config import REDIS_CFG, STREAM_TRADE_LOG, NY_TZ  # type: ignore
from utils import serialization_utils as ser  # type: ignore
import redis  # type: ignore


def _parse_hhmm(s: str) -> dt_time:
    hh, mm = s.split(":")
    return dt_time(int(hh), int(mm))


def _window_ms(date_str: str, t_start: dt_time, t_end: dt_time):
    y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    s = NY_TZ.localize(datetime(y, m, d, t_start.hour, t_start.minute, 0))
    e = NY_TZ.localize(datetime(y, m, d, t_end.hour,   t_end.minute,   0))
    return int(s.timestamp() * 1000), int(e.timestamp() * 1000)


def _decode_payload(data: dict):
    try:
        raw = data.get(b'data') or data.get(b'pickle') or data.get(b'batch')
        if raw is None:
            return None
        return ser.unpack(raw)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",  default=None, help="YYYYMMDD, 默认今天 NY")
    ap.add_argument("--start", default="09:30")
    ap.add_argument("--end",   default="10:30")
    ap.add_argument("--db",    type=int, default=None, help="Redis DB index (default from config)")
    args = ap.parse_args()

    date_str = args.date or datetime.now(NY_TZ).strftime("%Y%m%d")
    t_s = _parse_hhmm(args.start)
    t_e = _parse_hhmm(args.end)
    start_ms, end_ms = _window_ms(date_str, t_s, t_e)

    db_idx = args.db if args.db is not None else REDIS_CFG['db']

    print("=" * 78)
    print(f"📡 Redis Stream 探针  stream={STREAM_TRADE_LOG}  db={db_idx}")
    print(f"   Window [{args.start}, {args.end}) NY on {date_str}")
    print(f"   ID range: {start_ms} → {end_ms}  (ms-ts)")
    print("=" * 78)

    r = redis.Redis(host=REDIS_CFG['host'], port=REDIS_CFG['port'], db=db_idx)

    # 1) Stream 基础信息
    try:
        info = r.xinfo_stream(STREAM_TRADE_LOG)
        xlen = info.get(b'length')
        first_id = info.get(b'first-entry')
        last_id  = info.get(b'last-entry')
        def _id_of(entry):
            if not entry:
                return None
            try:
                return entry[0].decode()
            except Exception:
                return str(entry[0])
        first_s = _id_of(first_id)
        last_s  = _id_of(last_id)
        def _ms_of(eid):
            if not eid:
                return None
            try:
                return int(eid.split('-')[0])
            except Exception:
                return None
        first_ms = _ms_of(first_s)
        last_ms_ = _ms_of(last_s)
        print(f"  XLEN={xlen}")
        print(f"  first-entry id = {first_s}   (ms={first_ms}  NY="
              f"{datetime.fromtimestamp(first_ms/1000, NY_TZ).strftime('%H:%M:%S') if first_ms else '-'})")
        print(f"  last-entry  id = {last_s}   (ms={last_ms_} NY="
              f"{datetime.fromtimestamp(last_ms_/1000, NY_TZ).strftime('%H:%M:%S') if last_ms_ else '-'})")

        if first_ms and first_ms > start_ms:
            lag_sec = (first_ms - start_ms) / 1000.0
            print(f"  ⚠️ Stream 保留最早消息晚于窗口起点 {lag_sec:.1f}s → "
                  f"窗口起点附近已被 MAXLEN 裁掉 (命中 '①')")
        elif first_ms and first_ms <= start_ms:
            print(f"  ✅ Stream 仍保留窗口起点之前的数据, 可以逐分钟 XRANGE 核对.")
    except Exception as e:
        print(f"  ❌ XINFO STREAM 失败: {e}")
        sys.exit(2)

    # 2) 消费者组
    try:
        groups = r.xinfo_groups(STREAM_TRADE_LOG)
        print("")
        print("— Consumer Groups —")
        for g in groups:
            name = (g.get(b'name') or b'').decode()
            last = (g.get(b'last-delivered-id') or b'').decode()
            pending = g.get(b'pending')
            lag = g.get(b'lag')
            last_ms = None
            try:
                last_ms = int(last.split('-')[0])
            except Exception:
                pass
            last_hm = datetime.fromtimestamp(last_ms/1000, NY_TZ).strftime('%H:%M:%S') if last_ms else '-'
            print(f"  group={name:<18} last-delivered={last} ({last_hm})  pending={pending}  lag={lag}")
    except Exception as e:
        print(f"  (xinfo groups failed: {e})")

    # 3) 分钟级 XRANGE
    print("")
    print("— 分钟级保留情况 —")
    print(f"  {'minute':<10} {'stream_cnt':>10} {'alpha_cnt':>10} {'alpha_symbols':>13}   sample_symbols")
    any_alpha_in_window = False
    cursor_ms = start_ms
    while cursor_ms < end_ms:
        next_ms = cursor_ms + 60_000
        try:
            msgs = r.xrange(STREAM_TRADE_LOG, min=cursor_ms, max=next_ms - 1)
        except Exception as e:
            print(f"  {datetime.fromtimestamp(cursor_ms/1000, NY_TZ).strftime('%H:%M')}   XRANGE error: {e}")
            cursor_ms = next_ms
            continue

        alpha_syms = set()
        other_cnt = 0
        for _mid, data in msgs:
            payload = _decode_payload(data)
            if isinstance(payload, dict) and payload.get('action') == 'ALPHA':
                alpha_syms.add(payload.get('symbol'))
            else:
                other_cnt += 1
        if alpha_syms:
            any_alpha_in_window = True

        minute_hm = datetime.fromtimestamp(cursor_ms/1000, NY_TZ).strftime('%H:%M')
        sample = sorted([s for s in alpha_syms if s])[:8]
        more = "" if len(alpha_syms) <= 8 else f" (+{len(alpha_syms)-8})"
        print(f"  {minute_hm:<10} {len(msgs):>10} {len(alpha_syms):>10} {len(alpha_syms):>13}   "
              f"{sample}{more}")
        cursor_ms = next_ms

    print("")
    print("=" * 78)
    if any_alpha_in_window:
        print("🧭 结论提示: Stream 里 [当前仍保留] 了 ALPHA 消息. 若 PG 里对应分钟为 0,")
        print("           则瓶颈在 DPS 消费侧 (先 xack 后 flush / 重启漏读 / 组位点异常).")
    else:
        print("🧭 结论提示: Stream 里 [完全没有] ALPHA 消息覆盖该窗口. 最可能的原因是:")
        print("           (a) MAXLEN=10000 已把这段裁掉 (对比 first-entry 的 NY 时间);")
        print("           (b) SE 在该窗口内根本没发 ALPHA (应查 SE 日志).")
    print("=" * 78)


if __name__ == "__main__":
    main()
