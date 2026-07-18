#!/usr/bin/env python3
"""Diagnose V-reversals after Rule-A entry (Mag7+GOOGL causal baseline)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.replay import load_quotes, month_list, path_for_ticker, to_ny
from maga7.common.signals import attach_mf_features, load_stock_month_files

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        default=str(ROOT / "maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_v1.json"),
    )
    ap.add_argument(
        "--trades",
        default=str(
            ROOT / "maga7/results/replay_single_t30_rails_p20_mag7_googl_may_jul_delay60/trades.csv"
        ),
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "maga7/results/v_reversal_diagnosis_mag7_googl_may_jul"),
    )
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-13")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = load_profile(args.profile)
    trades = pd.read_csv(args.trades)
    paths = prof["_paths"]
    quote_root = paths["quote_1s_root"]
    start, end = args.start_date, args.end_date
    months = month_list(start, end)
    fill = FillSpec(entry_frac=0.8, exit_frac=0.8)

    stock_by: dict[str, pd.DataFrame] = {}
    for sym in prof["symbols"]:
        raw = load_stock_month_files(paths["stock_root"], sym, months)
        if raw.empty:
            continue
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]
        stock_by[sym] = attach_mf_features(raw, mf_window=10, vol_ma_window=20, mf_confirm_bars=3)

    quote_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def get_q(sym: str, date: str) -> pd.DataFrame:
        k = (sym, date)
        if k not in quote_cache:
            quote_cache[k] = load_quotes(quote_root, sym, date)
        return quote_cache[k]

    rows: list[dict] = []
    for _, tr in trades.iterrows():
        sym = str(tr["symbol"])
        date = str(tr["date"])
        ticker = str(tr["ticker"])
        direction = str(tr["dir"])
        entry_ts = to_ny(tr["entry_ts"])
        feature_ts = (
            to_ny(tr["feature_ts"])
            if pd.notna(tr.get("feature_ts"))
            else entry_ts - pd.Timedelta(minutes=1)
        )
        path = path_for_ticker(get_q(sym, date), ticker)
        if path is None or path.empty:
            continue
        # Cap path at T+30 for peak/MTM timing (do not use post-exit day path).
        hold_end = entry_ts + pd.Timedelta(minutes=30)
        after = path[(path["timestamp"] >= entry_ts) & (path["timestamp"] <= hold_end)].copy()
        if after.empty:
            continue
        entry = fill.buy(float(after.iloc[0]["bid"]), float(after.iloc[0]["ask"]))
        if not np.isfinite(entry) or entry <= 0:
            continue
        sell = fill.sell_series(
            after["bid"].astype(float).to_numpy(),
            after["ask"].astype(float).to_numpy(),
        )
        ts = [to_ny(x) for x in after["timestamp"].tolist()]
        mtm = sell / entry - 1.0

        def mtm_at(mins: float) -> float:
            tcut = entry_ts + pd.Timedelta(minutes=mins)
            idx = np.searchsorted([t.value for t in ts], tcut.value, side="right") - 1
            if idx < 0:
                return float("nan")
            return float(mtm[idx])

        peak_idx = int(np.nanargmax(mtm)) if len(mtm) else 0
        peak_ret = float(mtm[peak_idx])
        peak_min = (ts[peak_idx] - entry_ts).total_seconds() / 60.0
        exit_ts = to_ny(tr["exit_ts"])
        hold_min = (exit_ts - entry_ts).total_seconds() / 60.0
        final_ret = float(tr["ret"])
        m5, m10, m15, m20, m30 = [mtm_at(m) for m in (5, 10, 15, 20, 30)]
        fade_from_10 = (m10 - final_ret) if np.isfinite(m10) else float("nan")
        # V: early option peak then give-back vs realized trade ret
        early_peak = peak_min <= 12 and peak_ret >= 0.15
        gave_back = final_ret <= peak_ret - 0.25 or (early_peak and final_ret <= 0)
        v_rev_a = bool(early_peak and gave_back)
        v_rev_b = bool(
            np.isfinite(m10)
            and np.isfinite(m20)
            and (m10 - m20) >= 0.10
            and final_ret < m10 - 0.05
        )
        v_rev = v_rev_a or v_rev_b
        continuation = (not v_rev) and (final_ret >= 0.10 or peak_min >= 15)

        feat: dict = {}
        sdf = stock_by.get(sym)
        if sdf is not None:
            day = sdf[sdf["date"] == date].sort_values("timestamp")
            bar = day[day["timestamp"] <= feature_ts].tail(1)
            if len(bar):
                b = bar.iloc[0]
                feat = {
                    "mf10": float(b["mf10"]) if np.isfinite(b["mf10"]) else float("nan"),
                    "mf_short": float(b["mf_short"]) if np.isfinite(b["mf_short"]) else float("nan"),
                    "streak": float(b["streak_up"] if direction == "UP" else b["streak_dn"]),
                    "from_prev": float(b["from_prev"]),
                    "vol_z": float(b["vol_z"]) if np.isfinite(b["vol_z"]) else float("nan"),
                    "cum": float(b["cum"]),
                    "close": float(b["close"]),
                }
                hist = day[day["timestamp"] <= feature_ts].tail(12)
                nets = hist["net$"].to_numpy(dtype=float)
                if len(nets) >= 6:
                    feat["net_last3"] = float(np.sum(nets[-3:]))
                    feat["net_prev3"] = float(np.sum(nets[-6:-3]))
                    feat["net_accel"] = feat["net_last3"] - feat["net_prev3"]
                else:
                    feat["net_last3"] = feat["net_prev3"] = feat["net_accel"] = float("nan")
                if len(hist) >= 4:
                    mf = hist["mf10"].to_numpy(dtype=float)
                    feat["mf10_delta3"] = (
                        float(mf[-1] - mf[-4])
                        if np.isfinite(mf[-1]) and np.isfinite(mf[-4])
                        else float("nan")
                    )
                else:
                    feat["mf10_delta3"] = float("nan")
                if len(hist) >= 6:
                    feat["px_chg5"] = float(hist.iloc[-1]["close"] / hist.iloc[-6]["close"] - 1.0)
                else:
                    feat["px_chg5"] = float("nan")
                for mins in (5, 10, 15):
                    tvis = entry_ts + pd.Timedelta(minutes=mins) - pd.Timedelta(seconds=60)
                    bb = day[day["timestamp"] <= tvis].tail(1)
                    if not len(bb):
                        continue
                    fp = float(bb.iloc[0]["from_prev"])
                    mf = float(bb.iloc[0]["mf10"]) if np.isfinite(bb.iloc[0]["mf10"]) else float("nan")
                    feat[f"fp_p{mins}"] = fp
                    feat[f"mf_p{mins}"] = mf
                    if direction == "UP":
                        feat[f"adverse_fp_{mins}"] = float(fp < feat["from_prev"] - 0.005)
                        feat[f"mf_flip_{mins}"] = float(mf < 0) if np.isfinite(mf) else float("nan")
                    else:
                        feat[f"adverse_fp_{mins}"] = float(fp > feat["from_prev"] + 0.005)
                        feat[f"mf_flip_{mins}"] = float(mf > 0) if np.isfinite(mf) else float("nan")

        if feat:
            s = 1.0 if direction == "UP" else -1.0
            for src, dst in (
                ("mf10", "mf10_signed"),
                ("mf_short", "mf_short_signed"),
                ("net_accel", "net_accel_signed"),
                ("mf10_delta3", "mf10_delta3_signed"),
            ):
                v = feat.get(src, float("nan"))
                feat[dst] = float(v * s) if np.isfinite(v) else float("nan")

        rows.append(
            {
                "date": date,
                "symbol": sym,
                "dir": direction,
                "ticker": ticker,
                "ret": final_ret,
                "reason": tr["reason"],
                "hold_min": hold_min,
                "peak_ret": peak_ret,
                "peak_min": peak_min,
                "mtm5": m5,
                "mtm10": m10,
                "mtm15": m15,
                "mtm20": m20,
                "mtm30": m30,
                "fade_from_10": fade_from_10,
                "v_rev": v_rev,
                "v_rev_A": v_rev_a,
                "v_rev_B": v_rev_b,
                "continuation": continuation,
                "regime_qqq_fp": float(tr["regime_qqq_fp"])
                if pd.notna(tr.get("regime_qqq_fp"))
                else float("nan"),
                **feat,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "trade_path_features.csv", index=False)

    n = len(df)
    n_v = int(df["v_rev"].sum())
    n_c = int(df["continuation"].sum())
    summary = {
        "universe": "Mag7+GOOGL",
        "window": f"{start}..{end}",
        "n_trades": n,
        "n_v_reversal": n_v,
        "n_continuation": n_c,
        "v_rev_rate": n_v / n if n else 0,
        "v_rev_mean_ret": float(df.loc[df.v_rev, "ret"].mean()) if n_v else None,
        "cont_mean_ret": float(df.loc[df.continuation, "ret"].mean()) if n_c else None,
        "peak_min_p50_all": float(df["peak_min"].median()),
        "peak_min_p50_vrev": float(df.loc[df.v_rev, "peak_min"].median()) if n_v else None,
        "peak_min_p50_cont": float(df.loc[df.continuation, "peak_min"].median()) if n_c else None,
        "mean_fade_from_10_vrev": float(df.loc[df.v_rev, "fade_from_10"].mean()) if n_v else None,
    }

    entry_feats = [
        "streak",
        "from_prev",
        "vol_z",
        "mf10_signed",
        "mf_short_signed",
        "net_accel_signed",
        "mf10_delta3_signed",
        "px_chg5",
        "regime_qqq_fp",
    ]
    early_feats = ["adverse_fp_5", "mf_flip_5", "adverse_fp_10", "mf_flip_10", "mtm5", "mtm10"]

    def disc(feats: list[str]) -> list[dict]:
        out: list[dict] = []
        for f in feats:
            if f not in df.columns:
                continue
            a = df.loc[df.v_rev, f].dropna()
            b = df.loc[df.continuation, f].dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            md = float(a.mean() - b.mean())
            thr_candidates = np.nanpercentile(df[f].dropna(), [20, 30, 40, 50, 60, 70, 80])
            best = None
            lab = df[df.v_rev | df.continuation]
            for thr in thr_candidates:
                for op in ("ge", "le"):
                    pred = lab[f] >= thr if op == "ge" else lab[f] <= thr
                    if int(pred.sum()) == 0 or int((~pred).sum()) == 0:
                        continue
                    y = lab["v_rev"]
                    prec = float(y[pred].mean())
                    rec = float(pred[y].mean()) if y.sum() else 0.0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    if best is None or f1 > best["f1"]:
                        best = {
                            "thr": float(thr),
                            "op": op,
                            "prec": prec,
                            "rec": rec,
                            "f1": f1,
                            "n_flag": int(pred.sum()),
                        }
            out.append(
                {
                    "feature": f,
                    "mean_vrev": float(a.mean()),
                    "mean_cont": float(b.mean()),
                    "diff_v_minus_c": md,
                    "best_rule": best,
                }
            )
        return out

    entry_disc = disc(entry_feats)
    early_disc = disc(early_feats)

    uplift = {
        "mean_actual": float(df["ret"].mean()),
        "mean_if_exit_mtm10": float(df["mtm10"].dropna().mean()),
        "mean_if_peak_oracle": float(df["peak_ret"].mean()),
        "vrev_mean_actual": float(df.loc[df.v_rev, "ret"].mean()) if n_v else None,
        "vrev_mean_if_mtm10": float(df.loc[df.v_rev, "mtm10"].dropna().mean()) if n_v else None,
    }

    df["streak_bin"] = pd.cut(
        df["streak"], bins=[0, 8, 10, 12, 100], labels=["8", "9-10", "11-12", "13+"]
    )
    streak_tab = (
        df.groupby("streak_bin", observed=False)
        .agg(n=("v_rev", "size"), v_rate=("v_rev", "mean"), mean_ret=("ret", "mean"))
        .reset_index()
    )
    if df["net_accel_signed"].notna().sum() > 10:
        df["accel_bin"] = pd.qcut(
            df["net_accel_signed"].rank(method="first"), 3, labels=["weak", "mid", "strong"]
        )
        accel_tab = (
            df.groupby("accel_bin", observed=False)
            .agg(n=("v_rev", "size"), v_rate=("v_rev", "mean"), mean_ret=("ret", "mean"))
            .reset_index()
        )
    else:
        accel_tab = pd.DataFrame()

    df["short_aligned"] = df["mf_short_signed"] > 0
    short_tab = (
        df.groupby("short_aligned", observed=False)
        .agg(n=("v_rev", "size"), v_rate=("v_rev", "mean"), mean_ret=("ret", "mean"))
        .reset_index()
    )

    payload = {
        **summary,
        "uplift": uplift,
        "entry_discrimination": entry_disc,
        "early_discrimination": early_disc,
        "streak_tab": streak_tab.to_dict(orient="records"),
        "accel_tab": accel_tab.to_dict(orient="records") if len(accel_tab) else [],
        "short_tab": short_tab.to_dict(orient="records"),
        "label_def": {
            "v_rev_A": "peak<=12m & peak_ret>=15% & (final<=peak-25pp or final<=0)",
            "v_rev_B": "mtm10-mtm20>=10pp and final < mtm10-5pp",
            "continuation": "not V and (final>=10% or peak>=15m)",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    streak_tab.to_csv(out_dir / "streak_bins.csv", index=False)
    pd.DataFrame(entry_disc).to_csv(out_dir / "entry_feature_discrimination.csv", index=False)
    pd.DataFrame(early_disc).to_csv(out_dir / "early_feature_discrimination.csv", index=False)

    print(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
