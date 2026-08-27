#!/usr/bin/env python3
"""Validate whether QQQ TFT edges help Mag7 smooth-UP launches (side-channel).

Does NOT train Mag7 TFT. Joins existing QQQ inference to Mag7 launches/trades:
  - discrimination vs y_allow (stock MFE/MAE label)
  - soft gates on stock UP trail120 PnL
  - compare vs raw QQQ price filters (already known weak)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.tools.run_smooth_impulse_stock_replay import MONTHS, SYMS, _equity

NY = "America/New_York"

EDGE_COLS = (
    "net_edge",
    "net_edge_q10",
    "call_net_edge",
    "put_net_edge",
    "spot_up_prob",
    "best_side_call_prob",
)


def _auc(y: np.ndarray, s: np.ndarray) -> float | None:
    m = np.isfinite(s) & np.isfinite(y)
    y, s = y[m].astype(float), s[m].astype(float)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    order = np.argsort(s)
    y = y[order]
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return None
    ranks = np.arange(1, len(y) + 1, dtype=float)
    return float((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _to_ny(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s)
    if getattr(t.dt, "tz", None) is None:
        # hybrid files are UTC-naive wall; apr_jun is NY-aware
        # try localize UTC then convert if typical UTC hours
        # safer: if hour>=13 mostly, treat as UTC
        sample = t.dt.hour.median()
        if sample >= 12:
            t = t.dt.tz_localize("UTC").dt.tz_convert(NY)
        else:
            t = t.dt.tz_localize(NY)
    else:
        t = t.dt.tz_convert(NY)
    return t


def load_qqq_edges(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists():
            print(f"[warn] missing {p}", flush=True)
            continue
        df = pd.read_parquet(p)
        keep = ["timestamp"] + [c for c in EDGE_COLS if c in df.columns]
        if "vix_level" in df.columns:
            keep.append("vix_level")
        if "spot_day_ret" in df.columns:
            keep.append("spot_day_ret")
        out = df[keep].copy()
        out["timestamp"] = _to_ny(out["timestamp"])
        out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
        out["source"] = p.parent.name
        frames.append(out)
        print(f"[edge] {p.name}: n={len(out)} {out.timestamp.min()} → {out.timestamp.max()}", flush=True)
    if not frames:
        raise SystemExit("no edge files")
    e = pd.concat(frames, ignore_index=True)
    e = e.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    # derived
    if "call_net_edge" in e.columns and "put_net_edge" in e.columns:
        e["call_minus_put"] = e["call_net_edge"] - e["put_net_edge"]
    return e.reset_index(drop=True)


def asof_join(events: pd.DataFrame, edges: pd.DataFrame, *, ts_col: str = "detect_ts") -> pd.DataFrame:
    ev = events.copy()
    ev["_ts"] = _to_ny(ev[ts_col])
    ev = ev.sort_values("_ts")
    # avoid clobbering event date/symbol columns
    drop_right = [c for c in ("date", "source") if c in edges.columns]
    eg = edges.drop(columns=drop_right).sort_values("timestamp")
    joined = pd.merge_asof(
        ev,
        eg,
        left_on="_ts",
        right_on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("2min"),
    )
    joined["edge_age_sec"] = (joined["_ts"] - joined["timestamp"]).dt.total_seconds()
    return joined


def scoreboard_disc(df: pd.DataFrame, ycol: str = "y_allow") -> pd.DataFrame:
    rows = []
    y = df[ycol].astype(float).to_numpy()
    base = float(np.nanmean(y))
    candidates = [
        c
        for c in list(EDGE_COLS)
        + ["call_minus_put", "vix_level", "spot_day_ret", "look_ret", "path_eff", "qqq_from_prev", "qqq_gap_open"]
        if c in df.columns
    ]
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce").to_numpy()
        auc = _auc(y, s)
        # top quintile precision
        m = np.isfinite(s)
        if m.sum() < 30:
            continue
        thr = float(np.nanquantile(s[m], 0.80))
        sel = m & (s >= thr)
        prec = float(y[sel].mean()) if sel.any() else None
        lift = (prec / base) if prec is not None and base > 0 else None
        rows.append(
            {
                "feature": c,
                "auc": auc,
                "base_allow": base,
                "top20_prec": prec,
                "top20_lift": lift,
                "top20_n": int(sel.sum()),
                "n": int(m.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("auc", ascending=False, na_position="last")


def gate_trades(trades: pd.DataFrame, joined: pd.DataFrame, *, col: str, q: float) -> dict:
    """Filter UP trades by edge quantile computed on joined launch universe same window."""
    if col not in joined.columns or trades.empty:
        return {"filter": f"{col}@q{q}", "n_trades": 0, "total_ret": 0.0, "error": "missing"}
    s = pd.to_numeric(joined[col], errors="coerce")
    thr = float(s.quantile(q))
    # map trade detect_ts -> edge
    jkey = joined.copy()
    jkey["_ts"] = _to_ny(jkey["detect_ts"])
    t = trades.copy()
    t["_ts"] = _to_ny(t["detect_ts"])
    t = t.sort_values("_ts")
    eg = jkey[["_ts", col]].dropna().sort_values("_ts")
    m = pd.merge_asof(t, eg, on="_ts", direction="backward", tolerance=pd.Timedelta("2min"))
    kept = m[pd.to_numeric(m[col], errors="coerce") >= thr].copy()
    eq = _equity(kept, frac=0.5) if not kept.empty else {"total_ret": 0, "maxdd": 0, "n_trades": 0, "trade_win": None, "avg_trade_ret": None}
    return {
        "filter": f"{col}>=q{q:.2f}",
        "thr": thr,
        "n_trades": eq.get("n_trades"),
        "total_ret": eq.get("total_ret"),
        "maxdd": eq.get("maxdd"),
        "trade_win": eq.get("trade_win"),
        "avg_trade_ret": eq.get("avg_trade_ret"),
        "coverage": float(len(kept) / max(len(trades), 1)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--edges",
        nargs="+",
        default=[
            str(ROOT / "qqq_btc/results/v4_apr_jun_causal5m_replay/test_infer.parquet"),
            str(ROOT / "qqq_btc/results/ft56_julw1_honest_infer_fixed5m_post_gatefix/test_infer.parquet"),
        ],
    )
    ap.add_argument(
        "--launches",
        default="/mnt/s990/data/maga7/results/smooth_bouncer_bakeoff_v1/dataset_up_mfe10.parquet",
    )
    ap.add_argument(
        "--trades",
        default="/mnt/s990/data/maga7/results/research_smooth_impulse_stock_may_jul/trades_up_trail120.csv",
    )
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-07-10")  # edge coverage ends ~Jul10
    ap.add_argument("--out", default="/mnt/s990/data/maga7/results/validate_qqq_tft_sidechannel_v1")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    edges = load_qqq_edges([Path(p) for p in args.edges])
    edges = edges[edges["date"].between(args.start, args.end)]

    launches = pd.read_parquet(args.launches)
    launches = launches[launches["date"].astype(str).between(args.start, args.end)].copy()
    # optional mfe15 label
    if "mfe" in launches.columns and "mae" in launches.columns:
        launches["y_allow_mfe15"] = ((launches["mfe"] >= 0.015) & (launches["mae"] <= 0.008)).astype(int)

    joined = asof_join(launches, edges)
    hit = float(joined["timestamp"].notna().mean())
    print(f"[join] launches={len(launches)} edge_hit={hit:.1%}", flush=True)
    joined.to_parquet(out / "launches_with_qqq_tft.parquet", index=False)

    # also attach qqq price context from stock file for baseline
    prof = load_profile(
        "maga7/CONFIG/strategy_profiles/"
        "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
    )
    root = Path(prof["_paths"]["stock_root"]).expanduser()
    qqq = attach_mf_features(load_stock_month_files(root, "QQQ", MONTHS))
    qqq = qqq[qqq["date"].astype(str).between(args.start, args.end)].copy()
    qqq["timestamp"] = _to_ny(qqq["timestamp"])
    qqq = qqq.sort_values("timestamp")
    # session open for from_open
    qqq["qqq_open"] = qqq.groupby("date")["close"].transform("first")
    qqq["qqq_from_open_px"] = qqq["close"] / qqq["qqq_open"] - 1.0
    j2 = pd.merge_asof(
        joined.sort_values("_ts"),
        qqq[["timestamp", "from_prev", "qqq_from_open_px", "close"]].rename(
            columns={"from_prev": "qqq_bar_from_prev", "close": "qqq_px"}
        ),
        left_on="_ts",
        right_on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("2min"),
        suffixes=("", "_qqqbar"),
    )
    # overwrite helper names
    if "qqq_from_open_px" in j2.columns:
        j2["qqq_from_open"] = j2["qqq_from_open_px"]

    disc10 = scoreboard_disc(j2.dropna(subset=["net_edge"]), "y_allow")
    disc10.to_csv(out / "discrimination_mfe10.csv", index=False)
    print("\n=== Discrimination y_allow (MFE≥1%) ===", flush=True)
    print(disc10.head(12).to_string(index=False), flush=True)

    if "y_allow_mfe15" in j2.columns:
        disc15 = scoreboard_disc(j2.dropna(subset=["net_edge"]), "y_allow_mfe15")
        disc15.to_csv(out / "discrimination_mfe15.csv", index=False)
        print("\n=== Discrimination y_allow (MFE≥1.5%) ===", flush=True)
        print(disc15.head(12).to_string(index=False), flush=True)

    # Trade gate scoreboard
    trades = pd.read_csv(args.trades)
    trades = trades[trades["date"].astype(str).between(args.start, args.end)].copy()
    # restrict to edge-covered dates
    edge_dates = set(edges["date"].astype(str))
    trades = trades[trades["date"].astype(str).isin(edge_dates)].copy()

    ung = _equity(trades, frac=0.5)
    gate_rows = [
        {
            "filter": "ungated",
            "thr": None,
            "n_trades": ung.get("n_trades"),
            "total_ret": ung.get("total_ret"),
            "maxdd": ung.get("maxdd"),
            "trade_win": ung.get("trade_win"),
            "avg_trade_ret": ung.get("avg_trade_ret"),
            "coverage": 1.0,
        }
    ]
    # price baselines on trades via asof qqq
    t_j = asof_join(trades.rename(columns={"detect_ts": "detect_ts"}), edges)
    # attach from_open
    t_j = pd.merge_asof(
        t_j.sort_values("_ts"),
        qqq[["timestamp", "qqq_from_open_px"]].sort_values("timestamp"),
        left_on="_ts",
        right_on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("2min"),
        suffixes=("", "_px"),
    )
    hard_masks: list[tuple[str, pd.Series]] = []
    if "qqq_from_open_px" in t_j.columns:
        hard_masks.append(("qqq_from_open>0", t_j["qqq_from_open_px"] > 0))
    if "call_minus_put" in t_j.columns:
        hard_masks.append(("call_minus_put>0", t_j["call_minus_put"] > 0))
    if "spot_up_prob" in t_j.columns:
        hard_masks.append(("spot_up_prob>0.4", t_j["spot_up_prob"] > 0.4))
    if "net_edge" in t_j.columns:
        hard_masks.append(("net_edge>0", t_j["net_edge"] > 0))
    if "call_net_edge" in t_j.columns:
        hard_masks.append(("call_net_edge>median", t_j["call_net_edge"] >= t_j["call_net_edge"].median()))

    for name, mask in hard_masks:
        kept = t_j.loc[mask.fillna(False)].copy()
        if "ret" not in kept.columns or "date" not in kept.columns:
            print(f"[warn] skip {name}: cols={list(kept.columns)[:12]}", flush=True)
            continue
        eq = (
            _equity(kept, frac=0.5)
            if not kept.empty
            else {"total_ret": 0, "maxdd": 0, "n_trades": 0, "trade_win": None, "avg_trade_ret": None}
        )
        gate_rows.append(
            {
                "filter": name,
                "thr": None,
                "n_trades": eq.get("n_trades"),
                "total_ret": eq.get("total_ret"),
                "maxdd": eq.get("maxdd"),
                "trade_win": eq.get("trade_win"),
                "avg_trade_ret": eq.get("avg_trade_ret"),
                "coverage": float(len(kept) / max(len(trades), 1)),
            }
        )

    for col in ["call_net_edge", "call_minus_put", "net_edge", "spot_up_prob", "net_edge_q10"]:
        if col not in j2.columns:
            continue
        for q in (0.50, 0.70, 0.80):
            # build thr from launch join, apply on trades
            s = pd.to_numeric(j2[col], errors="coerce")
            thr = float(s.quantile(q))
            kept = t_j[pd.to_numeric(t_j[col], errors="coerce") >= thr].copy()
            eq = _equity(kept, frac=0.5) if not kept.empty else {"total_ret": 0, "maxdd": 0, "n_trades": 0, "trade_win": None, "avg_trade_ret": None}
            gate_rows.append(
                {
                    "filter": f"{col}>=q{q:.2f}",
                    "thr": thr,
                    "n_trades": eq.get("n_trades"),
                    "total_ret": eq.get("total_ret"),
                    "maxdd": eq.get("maxdd"),
                    "trade_win": eq.get("trade_win"),
                    "avg_trade_ret": eq.get("avg_trade_ret"),
                    "coverage": float(len(kept) / max(len(trades), 1)),
                }
            )

    gdf = pd.DataFrame(gate_rows)
    gdf.to_csv(out / "trade_gates.csv", index=False)
    print("\n=== Stock UP trail120 gates (edge-covered dates) ===", flush=True)
    print(gdf.to_string(index=False), flush=True)

    # Incremental: AUC of look_ret alone vs look_ret + call_minus_put (rank-sum of sum of z)
    inc = {}
    if "look_ret" in j2.columns and "call_minus_put" in j2.columns:
        sub = j2.dropna(subset=["look_ret", "call_minus_put", "y_allow"])
        y = sub["y_allow"].to_numpy(float)
        z1 = (sub["look_ret"] - sub["look_ret"].mean()) / (sub["look_ret"].std() + 1e-9)
        z2 = (sub["call_minus_put"] - sub["call_minus_put"].mean()) / (sub["call_minus_put"].std() + 1e-9)
        inc = {
            "auc_look_ret": _auc(y, z1.to_numpy()),
            "auc_call_minus_put": _auc(y, z2.to_numpy()),
            "auc_look_plus_cmp": _auc(y, (z1 + z2).to_numpy()),
            "corr_look_cmp": float(z1.corr(z2)),
            "corr_cmp_qqq_from_open": float(sub["call_minus_put"].corr(sub["qqq_from_open"]))
            if "qqq_from_open" in sub.columns
            else None,
        }
        print("\n=== Incremental ===", flush=True)
        print(inc, flush=True)

    # Verdict
    best_auc_row = disc10.iloc[0].to_dict() if len(disc10) else {}
    tft_feats = disc10[disc10["feature"].isin(["call_net_edge", "call_minus_put", "net_edge", "spot_up_prob", "net_edge_q10"])]
    price_feats = disc10[disc10["feature"].isin(["qqq_from_open", "qqq_gap_open", "qqq_from_prev", "spot_day_ret"])]
    useful = False
    reason = []
    if len(tft_feats):
        max_auc = float(tft_feats["auc"].max())
        max_lift = float(tft_feats["top20_lift"].max()) if tft_feats["top20_lift"].notna().any() else 0.0
        reason.append(f"best TFT AUC={max_auc:.3f}, best top20 lift={max_lift:.2f}")
        # useful if AUC>=0.58 and lift>=1.15 OR gated ret beats ungated by >=0.5pp with n>=20
        g_tft = gdf[gdf["filter"].str.contains("call_minus_put|call_net_edge|spot_up|net_edge", regex=True)]
        ung_ret = float(ung.get("total_ret") or 0)
        beat = g_tft[(g_tft["n_trades"] >= 20) & (g_tft["total_ret"] >= ung_ret + 0.005)]
        if max_auc >= 0.58 and max_lift >= 1.15:
            useful = True
            reason.append("discrimination threshold met")
        if len(beat):
            useful = True
            reason.append(f"gate beat ungated: {beat.iloc[0]['filter']} ret={beat.iloc[0]['total_ret']:.3f}")
        if max_auc < 0.55 and len(beat) == 0:
            useful = False
            reason.append("no meaningful discrimination or PnL lift")

    summary = {
        "window": [args.start, args.end],
        "edge_hit_rate": hit,
        "n_launches": int(len(launches)),
        "n_trades": int(len(trades)),
        "ungated": ung,
        "incremental": inc,
        "best_disc_feature": best_auc_row,
        "useful": useful,
        "reason": reason,
        "note": "Side-channel only; Mag7 launches remain rule-generated.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    verdict = "USEFUL" if useful else "NOT USEFUL"
    report = [
        f"# QQQ TFT → Mag7 Side-channel Validation",
        "",
        f"**Verdict: `{verdict}`**",
        "",
        f"Window: {args.start} → {args.end} · edge hit {hit:.1%}",
        "",
        "## Reasons",
        "",
        *[f"- {r}" for r in reason],
        "",
        "## Discrimination (MFE≥1%)",
        "",
        "```",
        disc10.head(15).to_string(index=False),
        "```",
        "",
        "## Trade gates",
        "",
        "```",
        gdf.to_string(index=False),
        "```",
        "",
        "## Incremental",
        "",
        "```",
        json.dumps(inc, indent=2),
        "```",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(report))
    print(f"\nVERDICT: {verdict}", flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
