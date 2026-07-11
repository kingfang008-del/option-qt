#!/usr/bin/env python3
"""July W1: 4桶×2约 map + 盘中按股价选主交易合约。

设计
----
- 从现有 1DTE 4 约 map（每桶 1 约）扩展为 8 约：
  bucket0: PUT ATM 两档 (K0,K1)
  bucket1: PUT 更虚两档 (K2,K3)
  bucket2: CALL ATM 两档 (K0,K1)  ← 交易 CALL
  bucket3: CALL 更虚两档 (K2,K3)
- 盘中主约：同桶两约按当前 spot 选更贴近 ATM 的一侧；
  缺报价则用另一约（gapfill）。

用法
----
  # 只生成 map
  python qqq_btc/tools/july_w1_8contract_spot_switch.py build-map

  # 下载+聚合（需 DATABENTO_API_KEY）
  python qqq_btc/tools/july_w1_8contract_spot_switch.py download

  # 用已有 infer 重挂盘口并对比 replay
  python qqq_btc/tools/july_w1_8contract_spot_switch.py replay-compare
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NY = "America/New_York"
OCC_RE = re.compile(r"^(O:)?([A-Z]+)(\d{6})([CP])(\d{8})$")

BASE_MAP = Path.home() / "train_data/locked_targets_map_1dte_jul2026_w1.parquet"
OUT_MAP = Path.home() / "train_data/locked_targets_map_1dte_jul2026_w1_8contract.parquet"
STEP2_MAP = Path("/mnt/s990/data/v4_original_jul5/locked_maps/step2_jul2026_w1_1dte_8contract.parquet")
OUT_1S = Path("/mnt/s990/data/v4_original_jul5/databento_july_w1_8contract/raw_1s")
OUT_1M = Path("/mnt/s990/data/v4_original_jul5/databento_july_w1_8contract/raw_1m")
STOCK_1M = Path.home() / "train_data/spnq_train_resampled/QQQ/regular/09:30-16:00/1min/2026-07.parquet"
RESULTS = REPO / "qqq_btc/results/july_w1_8contract_spot_switch"
INFER_CANDIDATES = [
    REPO / "qqq_btc/results/ft56_julw1_with_vix/test_infer.parquet",
    REPO / "qqq_btc/results/v4_base_julw1_with_vix/test_infer.parquet",
]
BASE_1M_4 = Path.home() / "train_data/july_w1_v4_databento/options_1m_july_w1"


def parse_occ(sym: str) -> dict[str, Any]:
    m = OCC_RE.match(str(sym).strip())
    if not m:
        raise ValueError(f"bad OCC: {sym}")
    strike = int(m.group(5)) / 1000.0
    return {
        "root": m.group(2),
        "yymmdd": m.group(3),
        "cp": m.group(4),
        "strike": strike,
        "occ": f"O:{m.group(2)}{m.group(3)}{m.group(4)}{m.group(5)}",
    }


def make_occ(root: str, yymmdd: str, cp: str, strike: float) -> str:
    return f"O:{root}{yymmdd}{cp}{int(round(strike * 1000)):08d}"


def expand_day(g: pd.DataFrame) -> list[dict[str, Any]]:
    """4约 → 8约，并按 2/桶 重映射。"""
    puts = g[g["side"].str.upper() == "PUT"].sort_values("strike", ascending=False)
    calls = g[g["side"].str.upper() == "CALL"].sort_values("strike", ascending=True)
    if len(puts) < 2 or len(calls) < 2:
        raise ValueError(f"{g['date_str'].iloc[0]} need >=2 put and >=2 call")

    p0 = parse_occ(puts.iloc[0]["contract_symbol"])
    p1 = parse_occ(puts.iloc[1]["contract_symbol"])
    c0 = parse_occ(calls.iloc[0]["contract_symbol"])
    c1 = parse_occ(calls.iloc[1]["contract_symbol"])
    # QQQ 通常 $1 步长；用已有两档间距，否则默认 1
    p_step = abs(p0["strike"] - p1["strike"]) or 1.0
    c_step = abs(c1["strike"] - c0["strike"]) or 1.0
    p2_strike = p1["strike"] - p_step
    p3_strike = p1["strike"] - 2 * p_step
    c2_strike = c1["strike"] + c_step
    c3_strike = c1["strike"] + 2 * c_step

    spot = float(g["stock_close_at_lock"].iloc[0]) if "stock_close_at_lock" in g.columns else float("nan")
    base = g.iloc[0].to_dict()
    pairs = [
        (0, "PUT", [p0["strike"], p1["strike"]], p0),
        (1, "PUT", [p2_strike, p3_strike], p0),
        (2, "CALL", [c0["strike"], c1["strike"]], c0),
        (3, "CALL", [c2_strike, c3_strike], c0),
    ]
    rows: list[dict[str, Any]] = []
    for bucket, side, strikes, proto in pairs:
        cp = "P" if side == "PUT" else "C"
        # 开盘锁定时：更贴近 spot 的为 rank0
        scored = sorted(
            strikes,
            key=lambda k: abs(k - spot) if np.isfinite(spot) else k,
        )
        for rank, strike in enumerate(scored):
            occ = make_occ(proto["root"], proto["yymmdd"], cp, strike)
            rows.append(
                {
                    **{k: base.get(k) for k in (
                        "date_str", "symbol", "target_dte", "selected_dte",
                        "expiration", "stock_close_at_lock", "lock_timestamp",
                    )},
                    "contract_symbol": occ,
                    "bucket_id": int(bucket),
                    "side": side,
                    "tag": f"DTE1_{side}_B{bucket}_R{rank}",
                    "strike": float(strike),
                    "rank_in_bucket": int(rank),
                    "role_at_lock": "primary" if rank == 0 else "secondary",
                    "moneyness_at_lock": float(np.log(strike / spot)) if spot > 0 else float("nan"),
                }
            )
    return rows


def build_map() -> Path:
    src = pd.read_parquet(BASE_MAP)
    all_rows: list[dict[str, Any]] = []
    for _, g in src.groupby("date_str"):
        all_rows.extend(expand_day(g))
    out = pd.DataFrame(all_rows)
    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_MAP, index=False)
    STEP2_MAP.parent.mkdir(parents=True, exist_ok=True)
    step2 = out[["date_str", "contract_symbol", "bucket_id", "symbol"]].copy()
    step2.to_parquet(STEP2_MAP, index=False)
    report = {
        "source_map": str(BASE_MAP),
        "output_map": str(OUT_MAP),
        "step2_map": str(STEP2_MAP),
        "days": int(out["date_str"].nunique()),
        "rows": int(len(out)),
        "contracts_per_day": int(out.groupby("date_str").size().median()),
        "sample_2026-07-01": out[out.date_str == "2026-07-01"][
            ["bucket_id", "side", "strike", "role_at_lock", "contract_symbol"]
        ].to_dict(orient="records"),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "map_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return OUT_MAP


def download() -> None:
    if not OUT_MAP.exists():
        build_map()
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("请先 export DATABENTO_API_KEY=db-xxx")
    OUT_1S.mkdir(parents=True, exist_ok=True)
    OUT_1M.mkdir(parents=True, exist_ok=True)
    py = os.environ.get("PYTHON", sys.executable)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            py, str(REPO / "preprocess/download/step2_databento_second_sniper_v1.py"),
            "--target-map", str(STEP2_MAP),
            "--output-dir", str(OUT_1S),
            "--api-key", key,
            "--symbol", "QQQ",
            "--date-from", "2026-07-01",
            "--date-to", "2026-07-09",
            "--force",
            "--max-workers", "4",
        ],
        cwd=str(REPO),
        env=env,
        check=True,
    )
    subprocess.run(
        [
            py, str(REPO / "preprocess/download/step3_databento_aggregate_1s_to_1m.py"),
            "--input-dir", str(OUT_1S),
            "--output-dir", str(OUT_1M),
            "--symbol", "QQQ",
            "--date-from", "2026-07-01",
            "--date-to", "2026-07-09",
            "--force",
        ],
        cwd=str(REPO),
        env=env,
        check=True,
    )
    # sanity
    for p in sorted((OUT_1M / "QQQ").glob("*.parquet")):
        df = pd.read_parquet(p)
        print(p.name, "tickers", df.ticker.nunique(), "buckets", sorted(df.bucket_id.unique()))


def _load_stock_spot() -> pd.DataFrame:
    st = pd.read_parquet(STOCK_1M)
    st["timestamp"] = pd.to_datetime(st["timestamp"])
    if st["timestamp"].dt.tz is None:
        st["timestamp"] = st["timestamp"].dt.tz_localize(NY)
    else:
        st["timestamp"] = st["timestamp"].dt.tz_convert(NY)
    return st[["timestamp", "close"]].rename(columns={"close": "spot"}).sort_values("timestamp")


def select_bucket_quotes_by_spot(
    opt: pd.DataFrame,
    spot: pd.DataFrame,
    bucket_id: int,
    side: str,
) -> pd.DataFrame:
    """同桶多约：按 spot 选主约，缺报价用另一约。"""
    sub = opt[opt["bucket_id"] == bucket_id].copy()
    if sub.empty:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "ticker", "strike"])
    sub["timestamp"] = pd.to_datetime(sub["timestamp"])
    if sub["timestamp"].dt.tz is None:
        sub["timestamp"] = sub["timestamp"].dt.tz_localize(NY)
    else:
        sub["timestamp"] = sub["timestamp"].dt.tz_convert(NY)
    if "strike" not in sub.columns:
        sub["strike"] = sub["ticker"].map(lambda t: parse_occ(t)["strike"])
    merged = pd.merge_asof(
        sub.sort_values("timestamp"),
        spot.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("5min"),
    )
    # score: CALL 偏好 strike>=spot 且最接近；PUT 偏好 strike<=spot 且最接近
    spot_v = merged["spot"].to_numpy(dtype=float)
    strike = merged["strike"].to_numpy(dtype=float)
    if side.upper() == "CALL":
        ok = strike >= spot_v
        dist = np.where(ok, strike - spot_v, np.abs(strike - spot_v) + 1e3)
    else:
        ok = strike <= spot_v
        dist = np.where(ok, spot_v - strike, np.abs(strike - spot_v) + 1e3)
    merged["_dist"] = dist
    merged["_has_quote"] = (
        pd.to_numeric(merged["bid"], errors="coerce").fillna(0).gt(0)
        & pd.to_numeric(merged["ask"], errors="coerce").fillna(0).gt(0)
        & (pd.to_numeric(merged["ask"], errors="coerce") > pd.to_numeric(merged["bid"], errors="coerce"))
    )
    # 有报价优先，再按 dist
    merged["_rank"] = np.where(merged["_has_quote"], 0, 1) * 1e6 + merged["_dist"]
    picked = (
        merged.sort_values(["timestamp", "_rank"])
        .drop_duplicates("timestamp", keep="first")
    )
    return picked[["timestamp", "bid", "ask", "ticker", "strike"]].reset_index(drop=True)


def attach_exec_spot_switch(
    df: pd.DataFrame,
    option_root: Path,
    symbol: str = "QQQ",
    call_bucket: int = 2,
    put_bucket: int = 0,
    tolerance: str = "5min",
) -> pd.DataFrame:
    from qqq_btc.tools.eval_test_set import _align_ts, drop_embedded_exec_columns

    out = drop_embedded_exec_columns(df).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    spot = _load_stock_spot()
    ts_ny = out["timestamp"]
    if ts_ny.dt.tz is None:
        ts_ny = ts_ny.dt.tz_localize(NY)
    else:
        ts_ny = ts_ny.dt.tz_convert(NY)
    dates = ts_ny.dt.strftime("%Y-%m-%d").unique()
    call_parts, put_parts = [], []
    for day in dates:
        fp = option_root / symbol / f"{symbol}_{day}.parquet"
        if not fp.exists():
            continue
        opt = pd.read_parquet(fp)
        if "bucket_id" not in opt.columns:
            continue
        cq = select_bucket_quotes_by_spot(opt, spot, call_bucket, "CALL")
        pq = select_bucket_quotes_by_spot(opt, spot, put_bucket, "PUT")
        if len(cq):
            call_parts.append(cq.rename(columns={"bid": "exec_call_bid", "ask": "exec_call_ask",
                                                   "ticker": "exec_call_ticker", "strike": "exec_call_strike"}))
        if len(pq):
            put_parts.append(pq.rename(columns={"bid": "exec_put_bid", "ask": "exec_put_ask",
                                                  "ticker": "exec_put_ticker", "strike": "exec_put_strike"}))
    tol = pd.Timedelta(tolerance)
    out = out.sort_values("timestamp")
    for parts, cols in (
        (call_parts, ["exec_call_bid", "exec_call_ask", "exec_call_ticker", "exec_call_strike"]),
        (put_parts, ["exec_put_bid", "exec_put_ask", "exec_put_ticker", "exec_put_strike"]),
    ):
        if not parts:
            continue
        quotes = pd.concat(parts, ignore_index=True).sort_values("timestamp")
        quotes = quotes.drop_duplicates("timestamp", keep="last")
        out["timestamp"], quotes["timestamp"] = _align_ts(out["timestamp"], quotes["timestamp"])
        keep = ["timestamp"] + [c for c in cols if c in quotes.columns]
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            quotes[keep],
            on="timestamp",
            direction="backward",
            tolerance=tol,
        )
    for leg in ("call", "put"):
        b, a = f"exec_{leg}_bid", f"exec_{leg}_ask"
        if b in out.columns and a in out.columns:
            mid = (out[b] + out[a]) / 2.0
            out[f"exec_{leg}_spread_pct"] = np.where(
                (out[b] > 0) & (out[a] > out[b]),
                (out[a] - out[b]) / mid.replace(0, np.nan),
                np.nan,
            )
    return out


def _replay(df: pd.DataFrame) -> dict[str, Any]:
    from qqq_btc.common.event_replay import prepare_minute_frame
    from qqq_btc.common.replay_harness import run_strict_replay
    from qqq_btc.qqq import config as qcfg

    frame = prepare_minute_frame(df)
    r = run_strict_replay(
        frame,
        qcfg.FILL_MODEL,
        qcfg.REPLAY,
        qcfg.EXIT_RAILS,
        edge_col="net_edge",
        edge_q10_col=qcfg.EDGE_Q10_COL,
        call_edge_col=qcfg.CALL_EDGE_COL,
        put_edge_col=qcfg.PUT_EDGE_COL,
        put_gate_col=qcfg.PUT_GATE_COL,
    )
    s = r.summary(position_frac=qcfg.REPLAY.position_frac)
    trades = r.trades_frame()
    out = {
        "trades": s.get("trades"),
        "total_net_return": s.get("total_net_return"),
        "hit_rate": s.get("hit_rate"),
        "max_drawdown_mtm": s.get("max_drawdown_mtm"),
        "trades_by_leg": s.get("trades_by_leg"),
    }
    if len(trades):
        trades = trades.copy()
        trades["entry_ts"] = pd.to_datetime(trades["entry_ts"]).dt.tz_convert(NY)
        trades["date"] = trades["entry_ts"].dt.strftime("%Y-%m-%d")
        out["by_day"] = (
            trades.groupby("date")
            .agg(n=("net_return", "size"), sum_ret=("net_return", "sum"))
            .to_dict(orient="index")
        )
    return out


def replay_compare() -> None:
    from qqq_btc.tools.eval_test_set import attach_exec_quotes, drop_embedded_exec_columns

    infer_path = next((p for p in INFER_CANDIDATES if p.exists()), None)
    if infer_path is None:
        raise SystemExit("missing infer parquet for compare")
    if not (OUT_1M / "QQQ").exists():
        raise SystemExit(f"missing 8-contract 1m under {OUT_1M}; run download first")

    raw = pd.read_parquet(infer_path)
    # A: 原 4 约 + 旧 attach（keep last）
    a = attach_exec_quotes(drop_embedded_exec_columns(raw), BASE_1M_4, "QQQ", call_bucket=2, put_bucket=0)
    # B: 8 约 + spot 切主约
    b = attach_exec_spot_switch(raw, OUT_1M, "QQQ", call_bucket=2, put_bucket=0)
    # C: 8 约但旧 attach（同桶 keep last，无 spot 优化）作消融
    c = attach_exec_quotes(drop_embedded_exec_columns(raw), OUT_1M, "QQQ", call_bucket=2, put_bucket=0)

    results = {
        "infer": str(infer_path),
        "A_4contract_legacy_attach": _replay(a),
        "B_8contract_spot_primary": _replay(b),
        "C_8contract_legacy_attach": _replay(c),
    }
    # switch stats
    if "exec_call_ticker" in b.columns:
        results["spot_switch_stats"] = {
            "call_unique_tickers": int(b["exec_call_ticker"].nunique(dropna=True)),
            "put_unique_tickers": int(b["exec_put_ticker"].nunique(dropna=True)),
            "call_ticker_counts": b["exec_call_ticker"].value_counts(dropna=True).head(10).to_dict(),
            "put_ticker_counts": b["exec_put_ticker"].value_counts(dropna=True).head(10).to_dict(),
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "replay_compare.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    b.to_parquet(RESULTS / "infer_8contract_spot.parquet", index=False)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build-map", "download", "replay-compare", "all"])
    args = ap.parse_args()
    if args.cmd == "build-map":
        build_map()
    elif args.cmd == "download":
        download()
    elif args.cmd == "replay-compare":
        replay_compare()
    else:
        build_map()
        download()
        replay_compare()


if __name__ == "__main__":
    main()
