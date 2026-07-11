#!/usr/bin/env python3
"""0DTE vol-regime oracle: RV vs IV, skew, and theta-collect diagnostics.

This is intentionally model-free. It answers whether QQQ 0DTE has tradable
edge in volatility / skew / theta dimensions before any predictor or RL layer.

Data sources:
  - quote_options_bucketed_v7: ATM IV, skew, vw_theta (minute)
  - spnq_train_resampled 1m spot: realized vol
  - option_edge labels (optional): long-option PnL for theta/skew overlays
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NY = "America/New_York"
ANN_MIN = float(np.sqrt(252 * 390))


def summarize(values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return {"n": 0}
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "median": float(v.median()),
        "hit_rate": float((v > 0).mean()),
        "p10": float(v.quantile(0.10)),
        "p90": float(v.quantile(0.90)),
        "std": float(v.std(ddof=0)),
    }


def load_bucketed(bucketed_dir: Path, start: str, end: str) -> pd.DataFrame:
    frames = []
    for p in sorted(bucketed_dir.glob("*.parquet")):
        month = p.stem  # YYYY-MM
        if month < start[:7] or month > end[:7]:
            continue
        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no bucketed files under {bucketed_dir} for {start}..{end}")
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["timestamp"].dt.strftime("%Y-%m-%d") >= start) & (out["timestamp"].dt.strftime("%Y-%m-%d") <= end)]
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    out["date_str"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    out["month"] = out["timestamp"].dt.strftime("%Y-%m")
    out["tod_min"] = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    return out.reset_index(drop=True)


def load_spot_1m(spot_root: Path, start: str, end: str) -> pd.DataFrame:
    frames = []
    for p in sorted((spot_root / "QQQ/regular/09:30-16:00/1min").glob("*.parquet")):
        month = p.stem
        if month < start[:7] or month > end[:7]:
            continue
        df = pd.read_parquet(p, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no spot files for {start}..{end}")
    out = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    out = out[(out["timestamp"].dt.strftime("%Y-%m-%d") >= start) & (out["timestamp"].dt.strftime("%Y-%m-%d") <= end)]
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["log_ret"] = np.log(out["close"] / out["close"].shift(1))
    out["date_str"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    # causal within-day rolling RV (past window)
    for w in (5, 15, 30):
        out[f"rv_past_{w}m"] = (
            out.groupby("date_str")["log_ret"].rolling(w, min_periods=max(3, w // 2)).std().reset_index(level=0, drop=True)
            * ANN_MIN
        )
        # future RV: std of next w returns (oracle / label)
        fut = []
        for _, g in out.groupby("date_str", sort=False):
            lr = g["log_ret"].values
            arr = np.full(len(g), np.nan)
            for i in range(len(g) - w):
                window = lr[i + 1 : i + 1 + w]
                if np.isfinite(window).sum() >= max(3, w // 2):
                    arr[i] = np.nanstd(window) * ANN_MIN
            fut.append(pd.Series(arr, index=g.index))
        out[f"rv_fwd_{w}m"] = pd.concat(fut).sort_index()
    return out.reset_index(drop=True)


def load_option_edge_minute(label_dir: Path, symbol: str, start: str, end: str, horizons: list[int]) -> pd.DataFrame:
    """Best CALL/PUT ret per minute from option-edge labels (if available)."""
    files = sorted((label_dir / symbol).glob(f"{symbol}_*.parquet"))
    files = [p for p in files if start <= p.stem.replace(f"{symbol}_", "") <= end]
    if not files:
        return pd.DataFrame()
    frames = []
    for p in files:
        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    rows = []
    for ts, g in raw.groupby("timestamp", sort=False):
        row = {"timestamp": ts, "date_str": str(g["date_str"].iloc[0])}
        for h in horizons:
            col = f"ret_{h}m"
            if col not in g.columns:
                continue
            for side in ("CALL", "PUT"):
                sub = g[(g["side"] == side) & g[col].notna()]
                row[f"best_{side.lower()}_ret_{h}m"] = float(sub[col].max()) if not sub.empty else np.nan
            call_r = row.get(f"best_call_ret_{h}m", np.nan)
            put_r = row.get(f"best_put_ret_{h}m", np.nan)
            if pd.notna(call_r) and pd.notna(put_r):
                row[f"gap_{h}m"] = float(call_r) - float(put_r)
                # short-vol proxy: -max(call, put)  (selling the richer long side loses this)
                row[f"short_rich_ret_{h}m"] = -float(max(call_r, put_r))
                row[f"long_best_ret_{h}m"] = float(max(call_r, put_r))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def build_panel(
    bucketed: pd.DataFrame,
    spot: pd.DataFrame,
    edge: pd.DataFrame,
) -> pd.DataFrame:
    keep_b = [
        "timestamp",
        "date_str",
        "month",
        "tod_min",
        "stock_close",
        "options_struc_atm_iv",
        "options_struc_skew",
        "options_flow_skew",
        "options_vw_theta",
        "options_vw_iv",
        "options_vw_gamma",
        "options_iv_divergence",
        "options_pcr_volume",
    ]
    keep_b = [c for c in keep_b if c in bucketed.columns]
    panel = bucketed[keep_b].merge(
        spot[["timestamp"] + [c for c in spot.columns if c.startswith("rv_")]],
        on="timestamp",
        how="inner",
    )
    if not edge.empty:
        panel = panel.merge(edge.drop(columns=["date_str"], errors="ignore"), on="timestamp", how="left")

    atm = pd.to_numeric(panel["options_struc_atm_iv"], errors="coerce")
    for w in (5, 15, 30):
        fwd = pd.to_numeric(panel[f"rv_fwd_{w}m"], errors="coerce")
        past = pd.to_numeric(panel[f"rv_past_{w}m"], errors="coerce")
        panel[f"iv_minus_rv_fwd_{w}m"] = atm - fwd
        panel[f"rv_fwd_minus_iv_{w}m"] = fwd - atm
        panel[f"iv_minus_rv_past_{w}m"] = atm - past
        # oracle long-gamma when future RV > IV
        panel[f"long_gamma_edge_{w}m"] = fwd - atm
        # oracle short-vol / theta when IV > future RV
        panel[f"short_vol_edge_{w}m"] = atm - fwd
    return panel


def threshold_oracle(df: pd.DataFrame, score_col: str, edge_col: str, thresholds: list[float], higher_is_signal: bool = True) -> dict:
    work = df.dropna(subset=[score_col, edge_col]).copy()
    if work.empty:
        return {}
    out = {}
    for th in thresholds:
        if higher_is_signal:
            sub = work[work[score_col] >= th]
            tag = f">={th}"
        else:
            sub = work[work[score_col] <= th]
            tag = f"<={th}"
        out[tag] = {
            **summarize(sub[edge_col]),
            "trades_per_day": float(len(sub) / max(1, sub["date_str"].nunique())) if not sub.empty else 0.0,
            "coverage": float(len(sub) / len(work)),
        }
    return out


def top_frac_oracle(df: pd.DataFrame, score_col: str, edge_col: str, fracs: list[float], higher: bool = True) -> dict:
    work = df.dropna(subset=[score_col, edge_col]).copy()
    if work.empty:
        return {}
    out = {}
    for frac in fracs:
        n = max(1, int(len(work) * frac))
        top = work.nlargest(n, score_col) if higher else work.nsmallest(n, score_col)
        tag = f"top{int(frac * 100)}" if higher else f"bot{int(frac * 100)}"
        out[tag] = {
            **summarize(top[edge_col]),
            "trades_per_day": float(n / max(1, top["date_str"].nunique())),
            "score_mean": float(top[score_col].mean()),
        }
    return out


def evaluate_rv_iv(panel: pd.DataFrame) -> dict:
    result = {"overall": {}, "by_month": {}, "by_session": {}}
    for w in (5, 15, 30):
        spread = f"iv_minus_rv_fwd_{w}m"
        long_g = f"long_gamma_edge_{w}m"
        short_v = f"short_vol_edge_{w}m"
        block = {
            "atm_iv": summarize(panel["options_struc_atm_iv"]),
            "rv_fwd": summarize(panel[f"rv_fwd_{w}m"]),
            "iv_minus_rv_fwd": summarize(panel[spread]),
            # if IV systematically > future RV, short-vol has positive mean edge
            "short_vol_all": summarize(panel[short_v]),
            "long_gamma_all": summarize(panel[long_g]),
            "short_vol_when_iv_rich": top_frac_oracle(panel, spread, short_v, [0.2, 0.1, 0.05], higher=True),
            "long_gamma_when_iv_cheap": top_frac_oracle(panel, spread, long_g, [0.2, 0.1, 0.05], higher=False),
            "corr_iv_vs_fwd_rv": float(
                pd.to_numeric(panel["options_struc_atm_iv"], errors="coerce")
                .corr(pd.to_numeric(panel[f"rv_fwd_{w}m"], errors="coerce"))
            ),
        }
        result["overall"][f"h{w}m"] = block

        monthly = {}
        for mon, g in panel.groupby("month"):
            monthly[mon] = {
                "iv_minus_rv_mean": float(pd.to_numeric(g[spread], errors="coerce").mean()),
                "short_vol_top20_mean": top_frac_oracle(g, spread, short_v, [0.2]).get("top20", {}).get("mean"),
                "long_gamma_bot20_mean": top_frac_oracle(g, spread, long_g, [0.2], higher=False).get("bot20", {}).get("mean"),
                "n": int(len(g)),
            }
        result["by_month"][f"h{w}m"] = monthly

        # session buckets: open / midday / close
        sessions = {
            "open_0930_1030": (9 * 60 + 30, 10 * 60 + 30),
            "mid_1030_1430": (10 * 60 + 30, 14 * 60 + 30),
            "close_1430_1600": (14 * 60 + 30, 16 * 60),
        }
        sess = {}
        for name, (a, b) in sessions.items():
            g = panel[(panel["tod_min"] >= a) & (panel["tod_min"] < b)]
            sess[name] = {
                "iv_minus_rv_mean": float(pd.to_numeric(g[spread], errors="coerce").mean()),
                "short_vol_top20": top_frac_oracle(g, spread, short_v, [0.2]).get("top20", {}),
                "n": int(len(g)),
            }
        result["by_session"][f"h{w}m"] = sess
    return result


def evaluate_skew(panel: pd.DataFrame) -> dict:
    skew = pd.to_numeric(panel["options_struc_skew"], errors="coerce")
    flow = pd.to_numeric(panel.get("options_flow_skew"), errors="coerce") if "options_flow_skew" in panel.columns else None
    out = {
        "struc_skew": summarize(skew),
        "flow_skew": summarize(flow) if flow is not None else {"n": 0},
    }
    # Does high put skew predict put outperformance over next 5/15/30m RV asymmetry?
    # Use option-edge gap if available; else use signed spot move as weak proxy.
    for h in (5, 15, 30):
        gap_col = f"gap_{h}m" if f"gap_{h}m" in panel.columns else None
        if gap_col:
            # high skew (put rich) → expect PUT better → gap = call-put should be negative
            # oracle edge for PUT-side: -gap when skew high
            panel_tmp = panel.copy()
            panel_tmp["put_edge_proxy"] = -pd.to_numeric(panel_tmp[gap_col], errors="coerce")
            out[f"skew_vs_put_edge_{h}m"] = {
                "corr_skew_put_edge": float(skew.corr(panel_tmp["put_edge_proxy"])),
                "high_skew_top20_put_edge": top_frac_oracle(panel_tmp, "options_struc_skew", "put_edge_proxy", [0.2]).get("top20", {}),
                "low_skew_bot20_put_edge": top_frac_oracle(panel_tmp, "options_struc_skew", "put_edge_proxy", [0.2], higher=False).get("bot20", {}),
            }
        # skew vs future RV (does elevated skew precede vol expansion?)
        rv_col = f"rv_fwd_{h}m"
        out[f"skew_vs_rv_fwd_{h}m"] = {
            "corr": float(skew.corr(pd.to_numeric(panel[rv_col], errors="coerce"))),
            "high_skew_top20_rv": top_frac_oracle(panel, "options_struc_skew", rv_col, [0.2]).get("top20", {}),
        }
    return out


def evaluate_theta(panel: pd.DataFrame) -> dict:
    theta = pd.to_numeric(panel["options_vw_theta"], errors="coerce")
    # theta is typically negative for long options; more negative = faster decay for longs
    # short-vol / theta-collect wants: sell when |theta| large AND IV rich vs future RV
    panel = panel.copy()
    panel["abs_theta"] = (-theta).clip(lower=0)  # positive = decay speed for longs
    panel["theta_x_iv_rich_15m"] = panel["abs_theta"] * pd.to_numeric(panel["iv_minus_rv_fwd_15m"], errors="coerce").clip(lower=0)

    out = {
        "vw_theta": summarize(theta),
        "abs_theta": summarize(panel["abs_theta"]),
    }
    # close window only
    close = panel[(panel["tod_min"] >= 14 * 60 + 30) & (panel["tod_min"] < 16 * 60)].copy()
    out["close_window"] = {
        "n": int(len(close)),
        "abs_theta": summarize(close["abs_theta"]),
        "iv_minus_rv_15m": summarize(close["iv_minus_rv_fwd_15m"]),
        "short_vol_edge_15m": summarize(close["short_vol_edge_15m"]),
        "top20_abs_theta_short_vol_15m": top_frac_oracle(close, "abs_theta", "short_vol_edge_15m", [0.2]).get("top20", {}),
        "top20_theta_x_iv_rich_short_vol_15m": top_frac_oracle(close, "theta_x_iv_rich_15m", "short_vol_edge_15m", [0.2]).get("top20", {}),
    }
    # if option-edge available: short_rich_ret as crude short-premium proxy
    if "short_rich_ret_5m" in panel.columns:
        out["short_premium_proxy"] = {}
        for h in (5, 10):
            col = f"short_rich_ret_{h}m"
            if col not in panel.columns:
                continue
            out["short_premium_proxy"][f"h{h}m"] = {
                "all": summarize(panel[col]),
                "close_all": summarize(close[col]) if col in close.columns else {"n": 0},
                "close_top20_abs_theta": top_frac_oracle(close, "abs_theta", col, [0.2]).get("top20", {}),
                "close_top20_iv_rich": top_frac_oracle(close, "iv_minus_rv_fwd_15m", col, [0.2]).get("top20", {}),
            }
    # monthly close short-vol
    monthly = {}
    for mon, g in close.groupby("month"):
        monthly[mon] = {
            "short_vol_15m_mean": float(pd.to_numeric(g["short_vol_edge_15m"], errors="coerce").mean()),
            "top20_theta_short_vol": top_frac_oracle(g, "abs_theta", "short_vol_edge_15m", [0.2]).get("top20", {}),
            "n": int(len(g)),
        }
    out["close_by_month"] = monthly
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="0DTE RV-IV / skew / theta oracle")
    p.add_argument("--bucketed-dir", default=str(Path.home() / "train_data/quote_options_bucketed_v7/QQQ"))
    p.add_argument("--spot-root", default=str(Path.home() / "train_data/spnq_train_resampled"))
    p.add_argument("--label-dir", default=str(Path.home() / "train_data/option_edge_labels_0dte"))
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--start-date", default="2026-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--output", default="factor_lab/results/0dte_vol_oracle_2026H1.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bucketed = load_bucketed(Path(args.bucketed_dir).expanduser(), args.start_date, args.end_date)
    spot = load_spot_1m(Path(args.spot_root).expanduser(), args.start_date, args.end_date)
    edge = load_option_edge_minute(
        Path(args.label_dir).expanduser(), args.symbol, args.start_date, args.end_date, horizons=[1, 3, 5, 10]
    )
    panel = build_panel(bucketed, spot, edge)

    report = {
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "rows": int(len(panel)),
        "days": int(panel["date_str"].nunique()),
        "has_option_edge_labels": bool(not edge.empty),
        "rv_iv": evaluate_rv_iv(panel),
        "skew": evaluate_skew(panel),
        "theta": evaluate_theta(panel),
    }

    # compact verdict helpers
    verdict = {}
    for w in (5, 15, 30):
        blk = report["rv_iv"]["overall"][f"h{w}m"]
        verdict[f"h{w}m"] = {
            "iv_minus_rv_mean": blk["iv_minus_rv_fwd"].get("mean"),
            "short_vol_top20_mean": blk["short_vol_when_iv_rich"].get("top20", {}).get("mean"),
            "long_gamma_bot20_mean": blk["long_gamma_when_iv_cheap"].get("bot20", {}).get("mean"),
            "corr_iv_fwd_rv": blk["corr_iv_vs_fwd_rv"],
        }
    report["verdict"] = {
        "rv_iv": verdict,
        "skew_corr_put_edge_5m": report["skew"].get("skew_vs_put_edge_5m", {}).get("corr_skew_put_edge"),
        "close_short_vol_15m_mean": report["theta"]["close_window"]["short_vol_edge_15m"].get("mean"),
        "close_top20_theta_short_vol_15m": report["theta"]["close_window"]["top20_abs_theta_short_vol_15m"].get("mean"),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"panel rows={report['rows']} days={report['days']} edge_labels={report['has_option_edge_labels']}")
    print("\n=== RV vs IV verdict ===")
    for k, v in verdict.items():
        print(
            f"{k}: IV-RV={v['iv_minus_rv_mean']:.4f} "
            f"short_vol_top20={v['short_vol_top20_mean']} "
            f"long_gamma_bot20={v['long_gamma_bot20_mean']} "
            f"corr(IV,RVfwd)={v['corr_iv_fwd_rv']:.3f}"
        )
    print("\n=== Theta close window ===")
    cw = report["theta"]["close_window"]
    print(
        f"short_vol_15m_mean={cw['short_vol_edge_15m'].get('mean')} "
        f"top20_abs_theta={cw['top20_abs_theta_short_vol_15m'].get('mean')} "
        f"top20_theta_x_ivrich={cw['top20_theta_x_iv_rich_short_vol_15m'].get('mean')}"
    )
    if report["skew"].get("skew_vs_put_edge_5m"):
        s = report["skew"]["skew_vs_put_edge_5m"]
        print(
            f"\n=== Skew === corr(skew,put_edge_5m)={s.get('corr_skew_put_edge')} "
            f"high_skew_top20={s.get('high_skew_top20_put_edge', {}).get('mean')}"
        )
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
