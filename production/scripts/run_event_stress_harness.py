#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
事件驱动压测脚手架（首版）

用途：
1) 对 s2 1s replay 进行多轮重复运行（同数据、不同扰动参数）
2) 自动汇总 alpha / trade / pnl 关键指标
3) 评估“确定性一致性 + 扰动鲁棒性”

示例：
python production/scripts/run_event_stress_harness.py \
  --start-date 20260310 --end-date 20260312 --runs 3 \
  --scenario baseline --scenario delayed_exec --scenario requote_tight
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Scenario:
    name: str
    extra_args: List[str]
    env_overrides: Dict[str, str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_replay_script() -> Path:
    return _repo_root() / "production" / "preprocess" / "backtest" / "second" / "s2_run_realtime_replay_sqlite_1s.py"


def _default_db_dir() -> Path:
    # 与 s2 脚本默认一致：~/quant_project/data/history_sqlite_1s
    return Path.home() / "quant_project" / "data" / "history_sqlite_1s"


def _parse_ymd(ymd: str) -> int:
    if len(ymd) != 8 or not ymd.isdigit():
        raise ValueError(f"Invalid date format: {ymd}, expected YYYYMMDD")
    return int(ymd)


def _list_target_dbs(db_dir: Path, start_date: str, end_date: str) -> List[Path]:
    s = _parse_ymd(start_date)
    e = _parse_ymd(end_date)
    if s > e:
        raise ValueError(f"start_date > end_date: {start_date} > {end_date}")

    out: List[Path] = []
    for p in sorted(db_dir.glob("market_*.db")):
        stem = p.stem
        # market_YYYYMMDD
        if len(stem) < 15:
            continue
        ymd = stem.split("_")[-1]
        if len(ymd) == 8 and ymd.isdigit():
            d = int(ymd)
            if s <= d <= e:
                out.append(p)
    return out


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?)",
        (table_name,),
    )
    return bool(cur.fetchone()[0])


def _table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    rows = cur.fetchall()
    return [str(r[1]) for r in rows]


def _sum_close_pnl(conn: sqlite3.Connection) -> float:
    if not _table_exists(conn, "trade_logs"):
        return 0.0
    cols = _table_columns(conn, "trade_logs")
    if "pnl" not in cols:
        return 0.0
    action_col = "action" if "action" in cols else None
    cur = conn.cursor()
    if action_col:
        cur.execute("SELECT COALESCE(SUM(CAST(pnl AS REAL)), 0.0) FROM trade_logs WHERE action='CLOSE'")
    else:
        cur.execute("SELECT COALESCE(SUM(CAST(pnl AS REAL)), 0.0) FROM trade_logs")
    row = cur.fetchone()
    return float(row[0] or 0.0)


def _count_rows(conn: sqlite3.Connection, table_name: str, where_sql: Optional[str] = None) -> int:
    if not _table_exists(conn, table_name):
        return 0
    cur = conn.cursor()
    if where_sql:
        cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {where_sql}")
    else:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    return int(cur.fetchone()[0] or 0)


def collect_sqlite_metrics(db_paths: List[Path]) -> Dict[str, float]:
    alpha_rows = 0
    trade_rows = 0
    trade_open = 0
    trade_close = 0
    pnl_close = 0.0

    for p in db_paths:
        conn = sqlite3.connect(str(p))
        try:
            alpha_rows += _count_rows(conn, "alpha_logs")
            trade_rows += _count_rows(conn, "trade_logs")
            if _table_exists(conn, "trade_logs"):
                cols = _table_columns(conn, "trade_logs")
                if "action" in cols:
                    trade_open += _count_rows(conn, "trade_logs", "action='OPEN'")
                    trade_close += _count_rows(conn, "trade_logs", "action='CLOSE'")
                pnl_close += _sum_close_pnl(conn)
        finally:
            conn.close()

    return {
        "alpha_rows": float(alpha_rows),
        "trade_rows": float(trade_rows),
        "trade_open": float(trade_open),
        "trade_close": float(trade_close),
        "pnl_close_sum": float(pnl_close),
    }


def _cv(values: List[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.0
    mean_v = statistics.fmean(values)
    if abs(mean_v) < 1e-12:
        return 0.0
    return statistics.pstdev(values) / abs(mean_v)


def _scenario_library() -> Dict[str, Scenario]:
    return {
        "baseline": Scenario(
            name="baseline",
            extra_args=[],
            env_overrides={},
        ),
        "delayed_exec": Scenario(
            name="delayed_exec",
            extra_args=[],
            env_overrides={
                "EXECUTION_DELAY_SECONDS": "2",
                "OMS_SIGNAL_DELAY_BARS": "1",
            },
        ),
        "requote_tight": Scenario(
            name="requote_tight",
            extra_args=[],
            env_overrides={
                "ENTRY_MAX_REQUOTE_SLIPPAGE_PCT": "0.005",
            },
        ),
        "stress_combo": Scenario(
            name="stress_combo",
            extra_args=[],
            env_overrides={
                "EXECUTION_DELAY_SECONDS": "2",
                "OMS_SIGNAL_DELAY_BARS": "1",
                "ENTRY_MAX_REQUOTE_SLIPPAGE_PCT": "0.005",
            },
        ),
    }


def run_one_replay(
    *,
    python_bin: str,
    replay_script: Path,
    start_date: str,
    end_date: str,
    common_args: List[str],
    scenario: Scenario,
    run_idx: int,
    log_dir: Path,
) -> Tuple[int, Path, float]:
    cmd = [
        python_bin,
        str(replay_script),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
    ] + common_args + scenario.extra_args

    env = os.environ.copy()
    env.update(scenario.env_overrides)

    log_path = log_dir / f"{scenario.name}.run{run_idx}.log"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    elapsed = time.time() - t0
    return int(proc.returncode), log_path, float(elapsed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Event-driven stress harness for 1s replay")
    parser.add_argument("--start-date", required=True, type=str, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, type=str, help="YYYYMMDD")
    parser.add_argument("--runs", type=int, default=3, help="repeats per scenario")
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="scenario name (can repeat): baseline/delayed_exec/requote_tight/stress_combo",
    )
    parser.add_argument("--python-bin", default=sys.executable, type=str)
    parser.add_argument("--replay-script", default=str(_default_replay_script()), type=str)
    parser.add_argument("--db-dir", default=str(_default_db_dir()), type=str)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--enable-oms", action="store_true")
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--parity-mode", action="store_true")
    parser.add_argument("--mirror-pg", action="store_true")
    args = parser.parse_args()

    replay_script = Path(args.replay_script).expanduser().resolve()
    db_dir = Path(args.db_dir).expanduser().resolve()
    if not replay_script.exists():
        print(f"[FATAL] replay script not found: {replay_script}")
        return 2
    if not db_dir.exists():
        print(f"[FATAL] db dir not found: {db_dir}")
        return 2

    target_dbs = _list_target_dbs(db_dir, args.start_date, args.end_date)
    if not target_dbs:
        print(f"[FATAL] no db files matched {args.start_date}~{args.end_date} in {db_dir}")
        return 2

    lib = _scenario_library()
    scenario_names = args.scenario or ["baseline", "delayed_exec", "stress_combo"]
    scenarios: List[Scenario] = []
    for name in scenario_names:
        if name not in lib:
            print(f"[FATAL] unknown scenario: {name}")
            return 2
        scenarios.append(lib[name])

    common_args: List[str] = []
    if args.skip_warmup:
        common_args.append("--skip-warmup")
    if args.enable_oms:
        common_args.append("--enable-oms")
    if args.turbo:
        common_args.append("--turbo")
    if args.parity_mode:
        common_args.append("--parity-mode")
    if args.mirror_pg:
        common_args.append("--mirror-pg")

    report_root = _repo_root() / "production" / "logs" / "event_stress"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = report_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] report dir: {out_dir}")
    print(f"[INFO] db count: {len(target_dbs)}")

    run_rows: List[Dict[str, object]] = []

    for sc in scenarios:
        print(f"\n[SCENARIO] {sc.name}")
        for i in range(1, args.runs + 1):
            print(f"  - run {i}/{args.runs} ...", end="", flush=True)
            rc, log_path, elapsed = run_one_replay(
                python_bin=args.python_bin,
                replay_script=replay_script,
                start_date=args.start_date,
                end_date=args.end_date,
                common_args=common_args,
                scenario=sc,
                run_idx=i,
                log_dir=out_dir,
            )
            metrics = collect_sqlite_metrics(target_dbs) if rc == 0 else {
                "alpha_rows": 0.0,
                "trade_rows": 0.0,
                "trade_open": 0.0,
                "trade_close": 0.0,
                "pnl_close_sum": 0.0,
            }
            row = {
                "scenario": sc.name,
                "run_idx": i,
                "return_code": rc,
                "elapsed_sec": round(elapsed, 3),
                "log_file": str(log_path),
                **metrics,
            }
            run_rows.append(row)
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f" {status} | alpha={int(metrics['alpha_rows'])} close={int(metrics['trade_close'])} pnl={metrics['pnl_close_sum']:.2f}")

    # aggregate
    agg: Dict[str, Dict[str, object]] = {}
    for sc in scenarios:
        rows = [r for r in run_rows if r["scenario"] == sc.name]
        ok_rows = [r for r in rows if int(r["return_code"]) == 0]
        alpha_vals = [float(r["alpha_rows"]) for r in ok_rows]
        close_vals = [float(r["trade_close"]) for r in ok_rows]
        pnl_vals = [float(r["pnl_close_sum"]) for r in ok_rows]

        agg[sc.name] = {
            "runs_total": len(rows),
            "runs_ok": len(ok_rows),
            "success_ratio": (len(ok_rows) / len(rows)) if rows else 0.0,
            "alpha_mean": statistics.fmean(alpha_vals) if alpha_vals else 0.0,
            "alpha_cv": _cv(alpha_vals),
            "trade_close_mean": statistics.fmean(close_vals) if close_vals else 0.0,
            "trade_close_cv": _cv(close_vals),
            "pnl_mean": statistics.fmean(pnl_vals) if pnl_vals else 0.0,
            "pnl_cv": _cv(pnl_vals),
        }

    # baseline drift
    baseline = agg.get("baseline")
    if baseline:
        b_pnl = float(baseline["pnl_mean"] or 0.0)
        b_close = float(baseline["trade_close_mean"] or 0.0)
        for sc_name, sc_agg in agg.items():
            pnl_mean = float(sc_agg["pnl_mean"] or 0.0)
            close_mean = float(sc_agg["trade_close_mean"] or 0.0)
            sc_agg["pnl_drift_vs_baseline_pct"] = ((pnl_mean - b_pnl) / abs(b_pnl) * 100.0) if abs(b_pnl) > 1e-9 else 0.0
            sc_agg["close_count_drift_vs_baseline_pct"] = ((close_mean - b_close) / abs(b_close) * 100.0) if abs(b_close) > 1e-9 else 0.0

    # write report json
    payload = {
        "meta": {
            "generated_at": stamp,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "runs_per_scenario": args.runs,
            "db_dir": str(db_dir),
            "replay_script": str(replay_script),
            "common_args": common_args,
            "scenario_names": [s.name for s in scenarios],
            "target_dbs": [str(p) for p in target_dbs],
        },
        "runs": run_rows,
        "aggregate": agg,
    }
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # write run csv
    csv_path = out_dir / "runs.csv"
    if run_rows:
        keys = list(run_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in run_rows:
                writer.writerow(r)

    # write aggregate csv
    agg_csv_path = out_dir / "aggregate.csv"
    with agg_csv_path.open("w", newline="", encoding="utf-8") as f:
        if agg:
            keys = ["scenario"] + list(next(iter(agg.values())).keys())
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for k, v in agg.items():
                row = {"scenario": k}
                row.update(v)
                writer.writerow(row)

    print("\n[SUMMARY]")
    for sc_name, sc_agg in agg.items():
        print(
            f"- {sc_name}: ok={sc_agg['runs_ok']}/{sc_agg['runs_total']} "
            f"trade_close_cv={sc_agg['trade_close_cv']:.4f} "
            f"pnl_cv={sc_agg['pnl_cv']:.4f} "
            f"pnl_mean={sc_agg['pnl_mean']:.2f}"
        )

    print(f"\n[REPORT] {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

