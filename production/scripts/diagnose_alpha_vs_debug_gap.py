#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
diagnose_alpha_vs_debug_gap.py
==============================

针对 "alpha_logs 有缺口, 但 debug_slow 完整" 这类问题做分钟级交叉诊断.

输出三张报表:
  1. 按分钟聚合每个表的行数 (整体密度)
  2. 每分钟缺失的 symbol 明细 (定位是哪几只股票 / 哪几分钟丢了)
  3. 提示最可能的根因 (基于写入路径的架构特征)

用法:
  python production/scripts/diagnose_alpha_vs_debug_gap.py
      --date 20260416 --start 09:30 --end 10:30

不传 --date 默认今天 (NY).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, time as dt_time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "production" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from config import PG_DB_URL, NY_TZ, TARGET_SYMBOLS  # type: ignore
import psycopg2  # type: ignore
import psycopg2.extras  # type: ignore


def _parse_hhmm(s: str) -> dt_time:
    hh, mm = s.split(":")
    return dt_time(int(hh), int(mm))


def _window_to_ts(date_str: str, start: dt_time, end: dt_time):
    y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    start_dt = NY_TZ.localize(datetime(y, m, d, start.hour, start.minute, 0))
    end_dt   = NY_TZ.localize(datetime(y, m, d, end.hour,   end.minute,   0))
    return float(start_dt.timestamp()), float(end_dt.timestamp()), start_dt, end_dt


def _minute_rows_alpha(c, start_ts, end_ts):
    c.execute(
        """
        SELECT (floor(ts / 60.0) * 60)::bigint AS minute_ts,
               symbol,
               COUNT(*) AS cnt
        FROM alpha_logs
        WHERE ts >= %s AND ts < %s
        GROUP BY minute_ts, symbol
        ORDER BY minute_ts, symbol
        """,
        (start_ts, end_ts),
    )
    return c.fetchall()


def _minute_rows_debug_slow(c, date_str, start_ts, end_ts):
    part = f"debug_slow_{date_str}"
    c.execute(
        f"""
        SELECT (floor(ts / 60.0) * 60)::bigint AS minute_ts,
               symbol,
               COUNT(*) AS cnt
        FROM {part}
        WHERE ts >= %s AND ts < %s
        GROUP BY minute_ts, symbol
        ORDER BY minute_ts, symbol
        """,
        (start_ts, end_ts),
    )
    return c.fetchall()


def _partition_exists(c, date_str: str) -> bool:
    c.execute("SELECT to_regclass(%s)", (f"public.debug_slow_{date_str}",))
    return c.fetchone()[0] is not None


def _alpha_partition_list(c, date_str: str):
    c.execute(
        """
        SELECT inhrelid::regclass::text
        FROM pg_inherits
        WHERE inhparent = 'alpha_logs'::regclass
        """
    )
    rows = [r[0] for r in c.fetchall()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",  default=None, help="YYYYMMDD, 默认今天 NY")
    ap.add_argument("--start", default="09:30", help="窗口起点 HH:MM")
    ap.add_argument("--end",   default="16:00", help="窗口终点 HH:MM")
    ap.add_argument("--symbols", default=None, help="逗号分隔, 默认 config.TARGET_SYMBOLS")
    args = ap.parse_args()

    date_str = args.date or datetime.now(NY_TZ).strftime("%Y%m%d")
    start_t  = _parse_hhmm(args.start)
    end_t    = _parse_hhmm(args.end)
    symbols  = args.symbols.split(",") if args.symbols else list(TARGET_SYMBOLS)
    sym_set  = set(symbols)
    expected_count = len(sym_set)

    start_ts, end_ts, start_dt, end_dt = _window_to_ts(date_str, start_t, end_t)

    print("=" * 78)
    print(f"📊 alpha_logs vs debug_slow 交叉诊断")
    print(f"   日期      : {date_str}  窗口 [{args.start}, {args.end}) NY")
    print(f"   Symbols   : {len(symbols)} 只 (config.TARGET_SYMBOLS)")
    print(f"   PG        : {PG_DB_URL.split('host=')[-1].split()[0]}")
    print("=" * 78)

    try:
        conn = psycopg2.connect(PG_DB_URL)
        c = conn.cursor()
    except Exception as e:
        print(f"❌ 无法连接 PostgreSQL: {e}")
        sys.exit(2)

    # 分区是否存在
    print("")
    print("— 分区检查 —")
    alpha_parts = _alpha_partition_list(c, date_str)
    print(f"  alpha_logs 子分区数量        : {len(alpha_parts)}")
    debug_slow_ok = _partition_exists(c, date_str)
    print(f"  debug_slow_{date_str} 分区是否存在 : {'✅' if debug_slow_ok else '❌'}")
    if not debug_slow_ok:
        print("  ⚠️ debug_slow 当日分区不存在, 后续 debug_slow 查询将跳过")

    # 拉数
    alpha_rows = _minute_rows_alpha(c, start_ts, end_ts)
    debug_rows = _minute_rows_debug_slow(c, date_str, start_ts, end_ts) if debug_slow_ok else []
    conn.close()

    # 聚合成 dict[minute_ts] -> set(symbols)
    def _to_minute_map(rows):
        m = {}
        for min_ts, sym, cnt in rows:
            m.setdefault(int(min_ts), {})[sym] = int(cnt)
        return m

    alpha_map = _to_minute_map(alpha_rows)
    debug_map = _to_minute_map(debug_rows)

    # 枚举窗口内每个整分钟
    minutes = []
    cur = int(start_ts)
    while cur < int(end_ts):
        minutes.append(cur)
        cur += 60

    # --------------- 报表 1: 分钟密度对比 ---------------
    print("")
    print("— 报表 1: 分钟级密度对比 (期望每分钟每只 symbol 一行) —")
    print(f"  {'minute (NY)':<20}  {'alpha':>6}  {'debug_slow':>11}  {'alpha_missing':>14}  {'debug_missing':>14}  {'overlap_loss':>14}")
    total_alpha_missing = 0
    total_debug_missing = 0
    total_overlap_loss  = 0
    suspicious_minutes = []
    for m_ts in minutes:
        a_syms = alpha_map.get(m_ts, {})
        d_syms = debug_map.get(m_ts, {})
        a_miss = sym_set - set(a_syms.keys())
        d_miss = sym_set - set(d_syms.keys())
        # overlap_loss = 本分钟在 debug_slow 里有但 alpha 里丢
        overlap_loss = (set(d_syms.keys()) & sym_set) - set(a_syms.keys())
        total_alpha_missing += len(a_miss)
        total_debug_missing += len(d_miss)
        total_overlap_loss  += len(overlap_loss)
        tag = ""
        if overlap_loss and not d_miss:
            tag = " ← ALPHA GAP"
            suspicious_minutes.append((m_ts, overlap_loss))
        elif a_miss and d_miss and set(a_miss) == set(d_miss):
            tag = " (both miss; upstream issue)"
        minute_ny = datetime.fromtimestamp(m_ts, NY_TZ).strftime("%H:%M")
        print(
            f"  {minute_ny:<20}  "
            f"{len(a_syms):>6}  "
            f"{len(d_syms):>11}  "
            f"{len(a_miss):>14}  "
            f"{len(d_miss):>14}  "
            f"{len(overlap_loss):>14}"
            f"{tag}"
        )

    print(f"\n  合计: alpha 缺失={total_alpha_missing}  debug_slow 缺失={total_debug_missing}  "
          f"仅 alpha 漏={total_overlap_loss}   (期望每分钟 {expected_count})")

    # --------------- 报表 2: 仅 alpha 漏的 symbol 明细 ---------------
    if suspicious_minutes:
        print("")
        print("— 报表 2: 具体[仅 alpha_logs 漏 / debug_slow 有]的 (分钟, symbol) —")
        for m_ts, syms in suspicious_minutes:
            minute_ny = datetime.fromtimestamp(m_ts, NY_TZ).strftime("%H:%M")
            sample = sorted(list(syms))[:12]
            more = "" if len(syms) <= 12 else f" ... (+{len(syms) - 12} more)"
            print(f"  {minute_ny}  缺 {len(syms):>2} 只: {sample}{more}")
    else:
        print("")
        print("✅ 报表 2: 没有 [仅 alpha_logs 漏] 的分钟 / symbol.")

    # --------------- 根因提示 ---------------
    print("")
    print("=" * 78)
    print("🧭 根因提示 (基于代码路径)")
    print("=" * 78)
    print(
        """
写入链路对比 (两条完全独立的路径):

  FCS (feature_compute_service_v8.py) ──(直连 PG)──► debug_slow     [同进程, 事务内写]
  SE  (signal_engine_v8.py)          ──(Redis Stream STREAM_TRADE_LOG, action=ALPHA)──►
       data_persistence_service_v8_pg.py (DPS) ──(批量 flush, 1s 一次)──► alpha_logs

因此:
  ① 若 debug_slow 完整而 alpha_logs 缺 → 一定出在 [SE → Redis → DPS → PG] 这条链上.
  ② 常见的具体原因 (按概率排序):
     (a) DPS 进程在缺口窗口里发生过重启 / 崩溃 / 被 kill -9.
         DPS 的主循环存在 "先 xack 再 flush" 的顺序 (execution 路径见 run()
         中 `xack` 后才 flush), 一旦进程被强杀, alpha_buffer 里
         尚未落库的消息就永久丢失.
     (b) SE 在缺口窗口内卡住 (GC / 死锁 / PG 探活阻塞), 没有把
         ALPHA 发到 STREAM_TRADE_LOG.
         ★ 关键区分: (a) 缺的是"分钟连续的一块", (b) 缺的是"同时段所有 symbol".
     (c) Redis Stream maxlen 裁剪. 若 STREAM_TRADE_LOG xadd 时 maxlen 设置
         过小 + DPS 消费慢, 历史消息会被 Redis 主动淘汰.
     (d) DPS flush 时 PG 瞬时拒绝 (如分区不存在), 代码会捕获后尝试建分区
         但本批数据已在 except 分支结束, 只等下一轮再重试 —— 如果期间
         DPS 被重启, 那一批也会丢.

排查建议 (按顺序):
  1) 搜 DPS 日志: 缺口窗口 (例如 09:45~09:50) 前后是否有
        "PG Persistence Service Started" 或 "Alpha Flush Error"
     前者说明 DPS 重启 → 命中 (a); 后者说明 flush 失败 → 命中 (d).
  2) 搜 SE 日志: 缺口窗口是否有 "Slow Compute"  / "Signal Compute" 长时间空档,
     或有 Python warning / traceback.
  3) 如果 "仅 alpha 漏" 的 symbol 全量 (≈30) 集中在连续几分钟 → (a) / (c) 最可信;
     如果是分散的少量 symbol → 上游 SE 单 symbol 跳过推理, 属 (b) 的子集.

修复方向:
  - DPS 改成 "先 flush 后 xack" (或 PEL 重放): 彻底杜绝 kill-9 丢数据.
  - SE 追加心跳 "alpha_emit_ts" 到 Redis monitor key, Dashboard 可直接看缺口.
  - STREAM_TRADE_LOG 的 maxlen 开大 (或干脆 MAXLEN ~ 一个交易日的容量).
"""
    )


if __name__ == "__main__":
    main()
