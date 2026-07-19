#!/usr/bin/env python3
"""本地：历史 Hunt 日对齐（Watchdog 候选 / stream / Scanner 缺口）。

三问（白话）：
  1) begin_day 算出来的 Hunt 候选，和 offline 成交是否同一票、同向、差不多同一时刻？
  2) stream 引擎在这些天能不能打出同样的 Hunt 单？
  3) 直播 Scanner 现在会不会注入 Hunt？（代码路径检查）
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.common.stream_engine import run_stream_replay
from maga7.common.watchdog import RegimeWatchdog

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
# Strong-window Hunt fills from prior L2 scoreboard (fallback if replay empty)
FALLBACK_HUNT_DATES = [
    "2026-05-07",
    "2026-05-15",
    "2026-05-28",
    "2026-06-02",
    "2026-06-03",
    "2026-06-11",
    "2026-06-24",
    "2026-06-26",
    "2026-07-01",
    "2026-07-02",
    "2026-07-13",
    "2026-07-16",
]


def _months(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start[:7], end[:7], freq="M")]


def _load_stock(prof: dict, start: str, end: str) -> dict[str, pd.DataFrame]:
    symbols = list(prof["symbols"])
    stock_root = Path(os.path.expanduser(prof["paths"]["stock_root"]))
    lb = (pd.Timestamp(start) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    months = _months(lb, end)
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols + ["QQQ"]:
        raw = load_stock_month_files(stock_root, sym, months)
        if raw is None or getattr(raw, "empty", True):
            continue
        out[sym] = attach_mf_features(raw)
    return out


def _scanner_injects_hunt() -> dict:
    """Static check: does live/scanner.py emit hunt fires?"""
    path = ROOT / "maga7" / "live" / "scanner.py"
    text = path.read_text()
    has_begin = "begin_day" in text and "hunt_armed" in text
    inject = (
        "def _schedule_hunts" in text
        and "def drain_hunts" in text
        and 'event_source": "hunt"' in text
    )
    return {
        "file": str(path.relative_to(ROOT)),
        "calls_begin_day_logs_armed": has_begin,
        "injects_hunt_candidates": bool(inject),
        "verdict": (
            "OK: Scanner 已 schedule/drain Hunt 并带 event_source=hunt"
            if inject
            else (
                "GAP: Scanner 只评估/打日志 hunt_armed，不把 Hunt 候选变成下单信号"
                if has_begin
                else "UNKNOWN"
            )
        ),
    }


def _key(date, sym, direction) -> str:
    return f"{date}|{str(sym).upper()}|{str(direction).upper()}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument(
        "--dates",
        default=",".join(FALLBACK_HUNT_DATES),
        help="comma dates to check (default=strong-window Hunt days)",
    )
    ap.add_argument("--out", default="maga7/results/watchdog/hunt_scanner_align")
    ap.add_argument(
        "--skip-stream",
        action="store_true",
        help="只做 begin_day 候选对齐 + Scanner 缺口，不跑 stream",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dates = [d.strip() for d in str(args.dates).split(",") if d.strip()]
    start, end = min(dates), max(dates)
    prof = load_profile(args.profile)
    symbols = list(prof["symbols"])

    print("loading stock for", start, "..", end)
    stock_by = _load_stock(prof, start, end)

    # Offline replay over the span covering all hunt dates
    print("offline replay...")
    p_off = copy.deepcopy(prof)
    p_off["date_range"] = {"start": start, "end": end}
    off = run_offline_replay(p_off, scheme="single")
    trades = off["trades"]
    if trades.empty or "event_source" not in trades.columns:
        hunt_off = trades.iloc[0:0]
    else:
        hunt_off = trades[trades["event_source"].astype(str) == "hunt"].copy()
    hunt_off.to_csv(out / "offline_hunt_trades.csv", index=False)
    print(f"  offline hunt fills: {len(hunt_off)}")

    # begin_day candidates for each date
    wd = RegimeWatchdog.from_profile(prof)
    cand_rows = []
    align_rows = []
    for d in dates:
        if wd is None:
            break
        dec = wd.begin_day(
            str(d),
            stock_by=stock_by,
            qqq_df=stock_by.get("QQQ"),
            symbols=symbols,
        )
        cands = list(getattr(wd, "hunt_candidates", None) or [])
        off_day = hunt_off[hunt_off["date"].astype(str) == str(d)] if len(hunt_off) else hunt_off
        off_keys = {
            _key(r.date, r.symbol, r.dir)
            for r in off_day.itertuples(index=False)
        } if len(off_day) else set()
        cand_keys = set()
        for hc in cands:
            row = {
                "date": str(d),
                "symbol": hc.symbol,
                "dir": hc.direction,
                "sig_ts": str(hc.sig_ts),
                "armed_until": str(hc.armed_until),
                "hunt_armed": bool(wd.hunt_armed),
                "watchdog_state": dec.state.value,
                "watchdog_reason": dec.reason,
            }
            cand_rows.append(row)
            cand_keys.add(_key(d, hc.symbol, hc.direction))
        # match: candidate key in offline hunt fills that day (fill may be subset)
        matched = sorted(cand_keys & off_keys)
        only_cand = sorted(cand_keys - off_keys)
        only_off = sorted(off_keys - cand_keys)
        align_rows.append(
            {
                "date": str(d),
                "hunt_armed": bool(wd.hunt_armed),
                "n_candidates": len(cands),
                "n_offline_hunt": int(len(off_day)),
                "matched": len(matched),
                "only_candidate": ",".join(only_cand),
                "only_offline": ",".join(only_off),
                "ok": bool(off_keys <= cand_keys) if off_keys else (len(cands) == 0),
                # offline fill ⊆ candidates (candidate may not fill due to gates)
                "note": "offline_subset_of_candidates" if off_keys <= cand_keys else "mismatch",
            }
        )
        print(
            f"  {d}: armed={wd.hunt_armed} cand={len(cands)} "
            f"off_hunt={len(off_day)} ok={align_rows[-1]['ok']}"
        )

    cand_df = pd.DataFrame(cand_rows)
    align_df = pd.DataFrame(align_rows)
    cand_df.to_csv(out / "begin_day_candidates.csv", index=False)
    align_df.to_csv(out / "begin_day_vs_offline.csv", index=False)

    # Stream parity on same window (Hunt-focused compare)
    stream_cmp = None
    if not args.skip_stream:
        print("stream replay (same window)...")
        p_st = copy.deepcopy(prof)
        p_st["date_range"] = {"start": start, "end": end}
        st = run_stream_replay(p_st, scheme="single")
        st_trades = st["trades"]
        st_trades.to_csv(out / "stream_trades.csv", index=False)
        if st_trades.empty or "event_source" not in st_trades.columns:
            hunt_st = st_trades.iloc[0:0]
        else:
            hunt_st = st_trades[st_trades["event_source"].astype(str) == "hunt"].copy()
        hunt_st.to_csv(out / "stream_hunt_trades.csv", index=False)

        def keys(df: pd.DataFrame) -> set[str]:
            if df is None or df.empty:
                return set()
            return {
                _key(r.date, r.symbol, r["dir"] if "dir" in df.columns else r.dir)
                for _, r in df.iterrows()
            }

        # also match entry tod
        def keys_tod(df: pd.DataFrame) -> set[str]:
            if df is None or df.empty:
                return set()
            outk = set()
            for _, r in df.iterrows():
                ts = pd.Timestamp(r["entry_ts"])
                if getattr(ts, "tzinfo", None) is not None:
                    ts = ts.tz_convert("America/New_York")
                outk.add(_key(r["date"], r["symbol"], r["dir"]) + f"|{ts.strftime('%H:%M')}")
            return outk

        ko, ks = keys(hunt_off), keys(hunt_st)
        kto, kts = keys_tod(hunt_off), keys_tod(hunt_st)
        stream_cmp = {
            "n_offline_hunt": len(hunt_off),
            "n_stream_hunt": len(hunt_st),
            "matched_sym_dir": len(ko & ks),
            "only_offline": sorted(ko - ks),
            "only_stream": sorted(ks - ko),
            "matched_sym_dir_tod": len(kto & kts),
            "only_offline_tod": sorted(kto - kts)[:20],
            "only_stream_tod": sorted(kts - kto)[:20],
            "ok_sym_dir": ko == ks,
            "ok_tod": kto == kts,
        }
        print(
            f"  stream hunt: off={len(hunt_off)} st={len(hunt_st)} "
            f"sym_dir_ok={stream_cmp['ok_sym_dir']} tod_ok={stream_cmp['ok_tod']}"
        )

    scanner = _scanner_injects_hunt()
    print("scanner:", scanner["verdict"])

    n_ok = int(align_df["ok"].sum()) if len(align_df) else 0
    n_days = int(len(align_df))
    summary = {
        "profile": args.profile,
        "dates": dates,
        "begin_day_ok_days": n_ok,
        "begin_day_n_days": n_days,
        "begin_day_all_ok": n_ok == n_days and n_days > 0,
        "stream": stream_cmp,
        "scanner": scanner,
        "plain": {
            "begin_day": (
                f"历史 {n_days} 个 Hunt 日里，{n_ok} 天 offline 成交票 ⊆ begin_day 候选"
            ),
            "stream": (
                "stream 与 offline Hunt 票/向一致"
                if stream_cmp and stream_cmp.get("ok_sym_dir")
                else (
                    "stream 与 offline Hunt 不一致或未跑"
                    if stream_cmp
                    else "未跑 stream"
                )
            ),
            "scanner": scanner["verdict"],
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    md = []
    md.append("# Hunt 历史日本地对齐\n\n")
    md.append(f"**日期：** 2026-07-19  \n**Profile：** `{args.profile}`\n\n")
    md.append("## 1) Watchdog begin_day 候选 vs offline Hunt 成交\n\n")
    md.append(
        f"- 检查天数：{n_days}  \n"
        f"- offline 成交 ⊆ 候选：{n_ok}/{n_days}  \n"
        f"- 全部通过：{'是' if summary['begin_day_all_ok'] else '否'}\n\n"
    )
    if len(align_df):
        md.append(align_df.to_markdown(index=False))
        md.append("\n\n")
    md.append("## 2) Stream vs offline Hunt\n\n")
    if stream_cmp:
        md.append(
            f"- offline Hunt：{stream_cmp['n_offline_hunt']}  \n"
            f"- stream Hunt：{stream_cmp['n_stream_hunt']}  \n"
            f"- 票+向一致：{'是' if stream_cmp['ok_sym_dir'] else '否'}  \n"
            f"- 票+向+时刻一致：{'是' if stream_cmp['ok_tod'] else '否'}  \n"
        )
        if stream_cmp["only_offline"]:
            md.append(f"- 仅 offline：{stream_cmp['only_offline']}\n")
        if stream_cmp["only_stream"]:
            md.append(f"- 仅 stream：{stream_cmp['only_stream']}\n")
        md.append("\n")
    else:
        md.append("（跳过）\n\n")
    md.append("## 3) Scanner 会不会下 Hunt？\n\n")
    md.append(f"- 文件：`{scanner['file']}`  \n")
    md.append(f"- 会 `begin_day` / 打 `hunt_armed` 日志：{scanner['calls_begin_day_logs_armed']}  \n")
    md.append(f"- 会把 Hunt 变成下单信号：{scanner['injects_hunt_candidates']}  \n")
    md.append(f"- **{scanner['verdict']}**\n\n")
    md.append("## 白话总判\n\n")
    if summary["begin_day_all_ok"] and stream_cmp and stream_cmp.get("ok_sym_dir"):
        md.append(
            "- 研究链路（offline / stream / begin_day）历史 Hunt 日对齐正常。  \n"
            "- **直播 Scanner 还没接 Hunt 注入**——开 Paper 前这是要补的工程洞，"
            "否则 Shadow 可能永远看不到 Hunt 单。  \n"
        )
    else:
        md.append(
            "- 候选或 stream 尚有不一致，见上表。  \n"
            f"- Scanner：{scanner['verdict']}  \n"
        )
    (out / "README.md").write_text("".join(md))
    print("\n" + "".join(md))
    print("wrote", out)
    return 0 if summary["begin_day_all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
