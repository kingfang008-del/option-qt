#!/usr/bin/env python3
"""Quick feasibility check for QQQ 0DTE 1s triple-barrier burst detection.

This is a deliberately small, strict test:
  - sample on raw 1s option quotes
  - enter at future ask after a latency delay
  - exit at future bid
  - train a lightweight tabular classifier as an event detector
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


NY = "America/New_York"


def parse_side(ticker: object) -> str | None:
    m = re.search(r"\d{6}([CP])\d{8}$", str(ticker).replace("O:", ""))
    if not m:
        return None
    return "CALL" if m.group(1) == "C" else "PUT"


FEATURES = [
    "side_code",
    "bucket_id",
    "tod_frac",
    "ask",
    "mid",
    "spread_pct",
    "size_imbalance",
    "underlying",
    "moneyness",
    "mid_ret_1s",
    "mid_ret_3s",
    "mid_ret_5s",
    "mid_ret_10s",
    "spread_chg_3s",
    "imb_chg_3s",
    "under_ret_1s",
    "under_ret_3s",
    "under_ret_5s",
    "under_ret_10s",
    "under_rv_10s",
    "under_accel_3s",
]


@dataclass(frozen=True)
class BarrierConfig:
    horizon_s: int = 30
    latency_s: int = 1
    take_profit: float = 0.10
    stop_loss: float = -0.03
    commission_per_contract: float = 0.65
    max_spread_pct: float = 0.10
    min_ask: float = 0.20
    cooldown_s: int = 30
    max_trades_per_day: int = 20
    position_frac: float = 0.10


def first_barrier_return(
    bids: np.ndarray,
    entry_idx: int,
    entry_ask: float,
    horizon_s: int,
    tp: float,
    sl: float,
    cost_frac: float,
) -> tuple[int, float, str]:
    end = min(len(bids) - 1, entry_idx + horizon_s)
    last_ret = np.nan
    for j in range(entry_idx + 1, end + 1):
        bid = bids[j]
        if not np.isfinite(bid) or bid <= 0 or entry_ask <= 0:
            continue
        ret = bid / entry_ask - 1.0 - cost_frac
        last_ret = ret
        if ret >= tp:
            return j - entry_idx, float(ret), "TAKE_PROFIT"
        if ret <= sl:
            return j - entry_idx, float(ret), "STOP"
    if np.isfinite(last_ret):
        return max(0, end - entry_idx), float(last_ret), "TIME"
    return 0, np.nan, "NO_FUTURE"


def add_features_and_labels(day: pd.DataFrame, cfg: BarrierConfig) -> pd.DataFrame:
    df = day.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df["ticker"] = df["ticker"].astype(str).str.replace("O:", "", regex=False)
    df["side"] = df["ticker"].map(parse_side)
    df["side_code"] = df["side"].map({"PUT": -1.0, "CALL": 1.0})
    for c in ["bid", "ask", "bid_size", "ask_size", "price", "underlying", "strike", "bucket_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if ("underlying" not in df.columns or df["underlying"].isna().all()) and "price" in df.columns:
        df["underlying"] = pd.to_numeric(df["price"], errors="coerce")
    if "mid_price" in df.columns:
        df["mid"] = pd.to_numeric(df["mid_price"], errors="coerce")
    else:
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
    df["size_imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"]).replace(0, np.nan)
    df["moneyness"] = df["underlying"] / df["strike"].replace(0, np.nan) - 1.0
    df["tod_frac"] = (df["timestamp"].dt.hour * 3600 + df["timestamp"].dt.minute * 60 + df["timestamp"].dt.second - (9 * 3600 + 30 * 60)) / (6.5 * 3600)

    frames: list[pd.DataFrame] = []
    for _, g0 in df.sort_values(["ticker", "timestamp"]).groupby("ticker", sort=False):
        g = g0.drop_duplicates("timestamp", keep="last").copy().reset_index(drop=True)
        for w in (1, 3, 5, 10):
            g[f"mid_ret_{w}s"] = g["mid"] / g["mid"].shift(w) - 1.0
            g[f"under_ret_{w}s"] = g["underlying"] / g["underlying"].shift(w) - 1.0
        g["spread_chg_3s"] = g["spread_pct"] - g["spread_pct"].shift(3)
        g["imb_chg_3s"] = g["size_imbalance"] - g["size_imbalance"].shift(3)
        g["under_rv_10s"] = np.log(g["underlying"] / g["underlying"].shift(1)).rolling(10, min_periods=5).std()
        g["under_accel_3s"] = g["under_ret_3s"] - g["under_ret_3s"].shift(3)

        bids = g["bid"].to_numpy(dtype=float)
        asks = g["ask"].to_numpy(dtype=float)
        labels = np.zeros(len(g), dtype=np.int8)
        exit_rets = np.full(len(g), np.nan, dtype=float)
        bars = np.zeros(len(g), dtype=np.int16)
        reasons: list[str] = ["INVALID"] * len(g)

        for i in range(len(g)):
            entry_idx = i + cfg.latency_s
            if entry_idx >= len(g):
                continue
            entry_ask = asks[entry_idx]
            if not np.isfinite(entry_ask) or entry_ask <= 0:
                continue
            cost_frac = 2.0 * cfg.commission_per_contract / (entry_ask * 100.0)
            b, ret, reason = first_barrier_return(
                bids,
                entry_idx,
                entry_ask,
                cfg.horizon_s,
                cfg.take_profit,
                cfg.stop_loss,
                cost_frac,
            )
            if b <= 0 or not np.isfinite(ret):
                continue
            labels[i] = 1 if reason == "TAKE_PROFIT" else 0
            exit_rets[i] = ret
            bars[i] = b
            reasons[i] = reason

        g["label"] = labels
        g["exit_return"] = exit_rets
        g["bars_held"] = bars
        g["exit_reason"] = reasons
        frames.append(g)

    out = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    tradable = (
        out["side"].isin(["CALL", "PUT"])
        & out["exit_return"].notna()
        & (out["ask"] >= cfg.min_ask)
        & (out["bid"] > 0)
        & (out["spread_pct"] <= cfg.max_spread_pct)
        & out["bucket_id"].notna()
    )
    return out[tradable].copy()


def build_dataset(raw_root: Path, start: str, end: str, cfg: BarrierConfig) -> pd.DataFrame:
    files = sorted(raw_root.glob("QQQ_*.parquet"))
    files = [p for p in files if start <= p.stem.replace("QQQ_", "") <= end]
    frames = []
    for p in files:
        day = pd.read_parquet(p)
        if day.empty:
            continue
        labeled = add_features_and_labels(day, cfg)
        if labeled.empty:
            continue
        labeled["date_str"] = p.stem.replace("QQQ_", "")
        frames.append(labeled)
    if not frames:
        raise SystemExit(f"no 1s dataset for {start}..{end}")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def summarize_trades(trades: pd.DataFrame, cfg: BarrierConfig) -> dict:
    if trades.empty:
        return {**asdict(cfg), "trades": 0, "total_net_return": 0.0}
    r = pd.to_numeric(trades["exit_return"], errors="coerce").fillna(0.0)
    eq = (1.0 + cfg.position_frac * r).cumprod()
    dd = eq / eq.cummax() - 1.0
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return {
        **asdict(cfg),
        "trades": int(len(trades)),
        "total_net_return": float(eq.iloc[-1] - 1.0),
        "sum_net_return": float(r.sum()),
        "avg_net_return": float(r.mean()),
        "hit_rate": float((r > 0).mean()),
        "tp_rate": float((trades["exit_reason"] == "TAKE_PROFIT").mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "max_drawdown": float(dd.min()),
        "worst_trade": float(r.min()),
        "trades_by_side": trades["side"].value_counts().to_dict(),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    }


def replay_predictions(df: pd.DataFrame, proba: np.ndarray, threshold: float, cfg: BarrierConfig) -> tuple[dict, pd.DataFrame]:
    work = df.copy()
    work["proba"] = proba
    work = work.sort_values("timestamp").reset_index(drop=True)
    trades = []
    trades_by_day: dict[str, int] = {}
    last_ts_by_day: dict[str, pd.Timestamp] = {}
    for row in work.itertuples(index=False):
        if float(getattr(row, "proba")) < threshold:
            continue
        date_str = str(getattr(row, "date_str"))
        sig_ts = pd.Timestamp(getattr(row, "timestamp"))
        if trades_by_day.get(date_str, 0) >= cfg.max_trades_per_day:
            continue
        last_ts = last_ts_by_day.get(date_str)
        if last_ts is not None and sig_ts <= last_ts + pd.Timedelta(seconds=cfg.cooldown_s):
            continue
        trades.append(
            {
                "timestamp": sig_ts,
                "date_str": date_str,
                "month": sig_ts.strftime("%Y-%m"),
                "ticker": str(getattr(row, "ticker")),
                "side": str(getattr(row, "side")),
                "bucket_id": int(getattr(row, "bucket_id")),
                "proba": float(getattr(row, "proba")),
                "ask": float(getattr(row, "ask")),
                "spread_pct": float(getattr(row, "spread_pct")),
                "exit_return": float(getattr(row, "exit_return")),
                "bars_held": int(getattr(row, "bars_held")),
                "exit_reason": str(getattr(row, "exit_reason")),
            }
        )
        trades_by_day[date_str] = trades_by_day.get(date_str, 0) + 1
        last_ts_by_day[date_str] = sig_ts
    trades_df = pd.DataFrame(trades)
    return summarize_trades(trades_df, cfg), trades_df


def eval_probs(y: np.ndarray, p: np.ndarray) -> dict:
    out = {"n": int(len(y)), "pos_rate": float(np.mean(y))}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    for q in (0.90, 0.95, 0.975, 0.99):
        th = float(np.quantile(p, q))
        sel = p >= th
        out[f"top{int((1-q)*1000)/10:g}_n"] = int(sel.sum())
        out[f"top{int((1-q)*1000)/10:g}_pos_rate"] = float(np.mean(y[sel])) if sel.any() else 0.0
    return out


def fit_model(train: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[HistGradientBoostingClassifier, np.ndarray, np.ndarray]:
    tr = train.dropna(subset=FEATURES + ["label"]).copy()
    ev = eval_df.dropna(subset=FEATURES + ["label"]).copy()
    X = tr[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    y = tr["label"].astype(int).to_numpy()
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(X, y)
    p_train = model.predict_proba(X)[:, 1]
    p_eval = model.predict_proba(ev[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())[:, 1]
    return model, p_train, p_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/mnt/s990/data/raw_1s/dte0_options/QQQ")
    parser.add_argument("--train-start", default="2026-01-01")
    parser.add_argument("--train-end", default="2026-02-28")
    parser.add_argument("--val-start", default="2026-03-01")
    parser.add_argument("--val-end", default="2026-03-31")
    parser.add_argument("--test-start", default="2026-04-01")
    parser.add_argument("--test-end", default="2026-06-30")
    parser.add_argument("--horizon-s", type=int, default=30)
    parser.add_argument("--latency-s", type=int, default=1)
    parser.add_argument("--take-profit", type=float, default=0.10)
    parser.add_argument("--stop-loss", type=float, default=-0.03)
    parser.add_argument("--max-spread-pct", type=float, default=0.10)
    parser.add_argument("--min-ask", type=float, default=0.20)
    parser.add_argument("--output-dir", default="factor_lab/results/0dte_1s_triple_barrier")
    args = parser.parse_args()

    cfg = BarrierConfig(
        horizon_s=args.horizon_s,
        latency_s=args.latency_s,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        max_spread_pct=args.max_spread_pct,
        min_ask=args.min_ask,
    )
    root = Path(args.raw_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building datasets...")
    train = build_dataset(root, args.train_start, args.train_end, cfg)
    val = build_dataset(root, args.val_start, args.val_end, cfg)
    test = build_dataset(root, args.test_start, args.test_end, cfg)
    train_full = pd.concat([train, val], ignore_index=True)

    oracle = {}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        oracle[name] = {
            "rows": int(len(df)),
            "days": int(df["date_str"].nunique()),
            "pos_rate": float(df["label"].mean()),
            "avg_exit_return": float(df["exit_return"].mean()),
            "tp_rate": float((df["exit_reason"] == "TAKE_PROFIT").mean()),
            "by_side": df.groupby("side")["label"].mean().to_dict(),
            "by_bucket": {str(k): float(v) for k, v in df.groupby("bucket_id")["label"].mean().to_dict().items()},
        }

    print("training...")
    _, p_train, p_val = fit_model(train, val)
    train_eval = eval_probs(train.dropna(subset=FEATURES + ["label"])["label"].astype(int).to_numpy(), p_train)
    val_clean = val.dropna(subset=FEATURES + ["label"]).copy()
    val_eval = eval_probs(val_clean["label"].astype(int).to_numpy(), p_val)

    grid_rows = []
    best = None
    for q in (0.80, 0.85, 0.90, 0.95, 0.975, 0.99):
        threshold = float(np.quantile(p_val, q))
        summary, _ = replay_predictions(val_clean, p_val, threshold, cfg)
        score = summary.get("total_net_return", 0.0) - 0.5 * abs(summary.get("max_drawdown", 0.0))
        row = {"quantile": q, "threshold": threshold, "score": score, **summary}
        grid_rows.append(row)
        if summary["trades"] >= 10 and (best is None or row["score"] > best["score"]):
            best = row
    if best is None:
        best = max(grid_rows, key=lambda r: r["score"])

    final_model, _, p_test = fit_model(train_full, test)
    test_clean = test.dropna(subset=FEATURES + ["label"]).copy()
    test_eval = eval_probs(test_clean["label"].astype(int).to_numpy(), p_test)
    test_summary, test_trades = replay_predictions(test_clean, p_test, float(best["threshold"]), cfg)

    monthly = {}
    if not test_trades.empty:
        for mon, g in test_trades.groupby("month"):
            monthly[mon] = summarize_trades(g, cfg)

    payload = {
        "config": asdict(cfg),
        "features": FEATURES,
        "oracle": oracle,
        "learnability": {"train": train_eval, "val": val_eval, "test": test_eval},
        "selected_val": best,
        "test": test_summary,
        "test_monthly": monthly,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(grid_rows).to_csv(out_dir / "val_grid.csv", index=False)
    test_trades.to_parquet(out_dir / "test_trades.parquet", index=False)

    print("ORACLE", json.dumps(oracle, indent=2, default=str))
    print("LEARNABILITY", json.dumps(payload["learnability"], indent=2, default=str))
    print("SELECTED", json.dumps(best, indent=2, default=str))
    print("TEST", json.dumps(test_summary, indent=2, default=str))
    print("MONTHLY", json.dumps(monthly, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
