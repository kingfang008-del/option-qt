#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit TFT checkpoint predictions against LMDB labels.

This script answers a narrow question:

    Is the generated alpha bad because S0/TFT label mapping is wrong,
    or because the model does not generalize on the test LMDB labels?

It loads trading_tft_stock_embed.py, evaluates rank_score/logits_dir on an LMDB,
and reports IC/spread/direction accuracy against label_return_fwd/label_direction.
Optionally it also compares rank_score with alpha_logs from history_sqlite_1s.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
import torch
import zstandard as zstd


msgpack_numpy.patch()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "production/model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from trading_tft_stock_embed import AdvancedAlphaNet, UnifiedLMDBDataset, collate_fn, load_meta_info  # noqa: E402


def _read_lmdb_keys(lmdb_path: Path) -> list[bytes]:
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    dctx = zstd.ZstdDecompressor()
    with env.begin() as txn:
        keys_blob = txn.get(b"__keys__")
        if not keys_blob:
            raise RuntimeError(f"{lmdb_path} missing __keys__")
        keys = msgpack.unpackb(dctx.decompress(keys_blob), raw=False)
    env.close()
    return keys


def _read_lmdb_sample_labels(lmdb_path: Path, keys: list[bytes], sample_limit: int = 20000) -> pd.DataFrame:
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    dctx = zstd.ZstdDecompressor()
    rows: list[dict[str, Any]] = []
    inspect_keys = keys[:sample_limit] if sample_limit > 0 else keys
    label_names = [
        "label_return_fwd",
        "label_direction",
        "label_option_alpha_300s",
        "label_option_direction_300s",
        "label_option_best_net_ret_300s",
        "label_call_delayed_exec_ret_300s",
        "label_put_delayed_exec_ret_300s",
        "label_call_delayed_exec_ret_d60s_h300s",
        "label_put_delayed_exec_ret_d60s_h300s",
        "label_option_tradable_300s",
        "label_option_training_enabled",
        "label_option_source_has_alpha",
        "label_option_source_has_direction",
        "label_option_source_has_primary_exec",
        "label_option_source_has_matrix_primary",
        "label_option_source_missing_count",
    ]
    with env.begin() as txn:
        for key in inspect_keys:
            blob = txn.get(key)
            if not blob:
                continue
            sample = msgpack.unpackb(dctx.decompress(blob), raw=False)
            labels = sample.get("labels", {})
            symbol, raw_ts = _key_to_symbol_ts(key)
            epoch_ts = _lmdb_key_time_to_epoch_seconds(raw_ts)
            row = {"symbol": symbol, "raw_ts": raw_ts, "ts": epoch_ts, "date": _date_from_epoch(epoch_ts)}
            for name in label_names:
                row[name] = labels.get(name, np.nan)
            rows.append(row)
    env.close()
    return pd.DataFrame(rows)


def _key_to_symbol_ts(key: bytes) -> tuple[str, int]:
    text = key.decode("ascii") if isinstance(key, bytes) else str(key)
    symbol, ts = text.rsplit("_", 1)
    return symbol, int(ts)


def _load_alpha_logs(hist_dir: Path, dates: list[str]) -> pd.DataFrame:
    frames = []
    for date in dates:
        db = hist_dir / f"market_{date}.db"
        if not db.exists():
            continue
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        cols = pd.read_sql("PRAGMA table_info(alpha_logs)", conn)["name"].tolist()
        extra = []
        if "tradable_prob" in cols:
            extra.append("tradable_prob")
        if "edge_score" in cols:
            extra.append("edge_score")
        select_cols = "ts, symbol, alpha" + (", " + ", ".join(extra) if extra else "")
        df = pd.read_sql(f"SELECT {select_cols} FROM alpha_logs", conn)
        conn.close()
        if df.empty:
            continue
        df["date"] = date
        df["ts"] = df["ts"].astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _lmdb_key_time_to_epoch_seconds(ts: int) -> int:
    # S0 stores pandas int64 timestamps in nanoseconds.
    if abs(ts) > 10**17:
        return int(round(ts / 1_000_000_000))
    if abs(ts) > 10**14:
        return int(round(ts / 1_000_000))
    if abs(ts) > 10**11:
        return int(round(ts / 1_000))
    return int(ts)


def _date_from_epoch(ts: int) -> str:
    return pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").strftime("%Y%m%d")


def _filter_keys_by_dates(keys: list[bytes], dates: list[str]) -> list[bytes]:
    if not dates:
        return keys
    wanted = set(dates)
    out = []
    for key in keys:
        _, raw_ts = _key_to_symbol_ts(key)
        epoch_ts = _lmdb_key_time_to_epoch_seconds(raw_ts)
        if _date_from_epoch(epoch_ts) in wanted:
            out.append(key)
    return out


def _safe_spearman(group: pd.DataFrame, pred_col: str, label_col: str) -> float:
    if len(group) <= 3:
        return np.nan
    if group[pred_col].std() < 1e-12 or group[label_col].std() < 1e-12:
        return 0.0
    return group[pred_col].corr(group[label_col], method="spearman")


def _spread(df: pd.DataFrame, pred_col: str, label_col: str, q: float = 0.10) -> float:
    valid = df[[pred_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 20:
        return np.nan
    n = max(1, int(len(valid) * q))
    s = valid.sort_values(pred_col)
    return float(s.tail(n)[label_col].mean() - s.head(n)[label_col].mean())


def _cs_zscore(group: pd.DataFrame, col: str) -> pd.Series:
    values = pd.to_numeric(group[col], errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=group.index)
    return (values - values.mean()) / (std + 1e-6)


def _print_label_distribution(df: pd.DataFrame) -> None:
    print("\n[LMDB label 分布]")
    ret = df["label_return_fwd"]
    print(
        pd.DataFrame(
            [
                {
                    "n": len(df),
                    "mean%": ret.mean() * 100.0,
                    "median%": ret.median() * 100.0,
                    "std%": ret.std() * 100.0,
                    "nonzero%": (ret.abs() > 1e-12).mean() * 100.0,
                    "pos%": (ret > 0).mean() * 100.0,
                    "neg%": (ret < 0).mean() * 100.0,
                    "p10%": ret.quantile(0.10) * 100.0,
                    "p90%": ret.quantile(0.90) * 100.0,
                }
            ]
        )
        .round(4)
        .to_string(index=False)
    )
    print("direction dist:", df["label_direction"].value_counts(normalize=True).sort_index().mul(100).round(3).to_dict())


def evaluate_lmdb(
    lmdb_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    sample_limit: int,
    device_name: str,
    dates: list[str] | None = None,
) -> pd.DataFrame:
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    meta = load_meta_info()
    caps = {"stock": meta["max_stock_id"], "sector": meta["max_sector_id"], "dow": 7}

    device = torch.device(device_name)
    model = AdvancedAlphaNet(config, caps).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"checkpoint={checkpoint_path}")
    print(f"load_state missing={len(missing)} unexpected={len(unexpected)} best_ic={ckpt.get('best_ic', 'N/A') if isinstance(ckpt, dict) else 'N/A'}")

    ds = UnifiedLMDBDataset(str(lmdb_path), config)
    keys = _read_lmdb_keys(lmdb_path)
    keys = _filter_keys_by_dates(keys, dates or [])
    if sample_limit > 0:
        keys = keys[:sample_limit]
    if keys:
        ds.keys = keys
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    rows = []
    model.eval()
    with torch.no_grad():
        offset = 0
        for batch in loader:
            if not batch:
                continue
            x_stk, x_opt, static, target, ts_list = batch
            x_stk = x_stk.to(device)
            x_opt = x_opt.to(device)
            static = {k: v.to(device) for k, v in static.items()}
            out = model(x_stk, x_opt, static)
            pred = out["rank_score"].detach().cpu().numpy().reshape(-1)
            tradable_prob = (
                torch.sigmoid(out["tradable_logit"]).detach().cpu().numpy().reshape(-1)
                if "tradable_logit" in out
                else np.full_like(pred, np.nan, dtype=np.float64)
            )
            edge_score = (
                torch.sigmoid(out["edge_score"]).detach().cpu().numpy().reshape(-1)
                if "edge_score" in out
                else np.full_like(pred, np.nan, dtype=np.float64)
            )
            call_tradable_prob = (
                torch.sigmoid(out["call_tradable_logit"]).detach().cpu().numpy().reshape(-1)
                if "call_tradable_logit" in out
                else tradable_prob
            )
            put_tradable_prob = (
                torch.sigmoid(out["put_tradable_logit"]).detach().cpu().numpy().reshape(-1)
                if "put_tradable_logit" in out
                else tradable_prob
            )
            call_edge_score = (
                torch.sigmoid(out["call_edge_score"]).detach().cpu().numpy().reshape(-1)
                if "call_edge_score" in out
                else edge_score
            )
            put_edge_score = (
                torch.sigmoid(out["put_edge_score"]).detach().cpu().numpy().reshape(-1)
                if "put_edge_score" in out
                else edge_score
            )
            logits = out["logits_dir"].detach().cpu()
            pred_dir = torch.argmax(logits, dim=1).numpy()
            ret = target["return_fwd"].numpy()
            direction = target["direction"].numpy()
            tradable = target.get("tradable", torch.zeros_like(target["return_fwd"])).numpy()
            best_net = target.get("best_net_ret", torch.abs(target["return_fwd"])).numpy()
            aux_tradable = target.get("aux_tradable", target.get("tradable", torch.zeros_like(target["return_fwd"]))).numpy()
            aux_best_net = target.get("aux_best_net_ret", target.get("best_net_ret", torch.abs(target["return_fwd"]))).numpy()
            call_tradable = target.get("call_tradable", torch.zeros_like(target["return_fwd"])).numpy()
            put_tradable = target.get("put_tradable", torch.zeros_like(target["return_fwd"])).numpy()
            call_edge_ret = target.get("call_edge_ret", target.get("call_exec_ret", torch.zeros_like(target["return_fwd"]))).numpy()
            put_edge_ret = target.get("put_edge_ret", target.get("put_exec_ret", torch.zeros_like(target["return_fwd"]))).numpy()

            for i in range(len(pred)):
                # Dataset order is the LMDB key order.
                key = ds.keys[offset + i]
                symbol, raw_ts = _key_to_symbol_ts(key)
                epoch_ts = _lmdb_key_time_to_epoch_seconds(raw_ts)
                selected_is_call = pred[i] >= 0
                rows.append(
                    {
                        "symbol": symbol,
                        "raw_ts": raw_ts,
                        "ts": epoch_ts,
                        "date": _date_from_epoch(epoch_ts),
                        "pred_rank": float(pred[i]),
                        "tradable_prob": float(tradable_prob[i]),
                        "edge_score": float(edge_score[i]),
                        "call_tradable_prob": float(call_tradable_prob[i]),
                        "put_tradable_prob": float(put_tradable_prob[i]),
                        "call_edge_score": float(call_edge_score[i]),
                        "put_edge_score": float(put_edge_score[i]),
                        "selected_tradable_prob": float(call_tradable_prob[i] if selected_is_call else put_tradable_prob[i]),
                        "selected_edge_score": float(call_edge_score[i] if selected_is_call else put_edge_score[i]),
                        "pred_dir": int(pred_dir[i]),
                        "label_return_fwd": float(ret[i]),
                        "label_direction": int(direction[i]),
                        "label_tradable": float(max(tradable[i], aux_tradable[i])),
                        "label_best_net_ret": float(max(best_net[i], aux_best_net[i])),
                        "label_call_tradable": float(call_tradable[i]),
                        "label_put_tradable": float(put_tradable[i]),
                        "label_call_edge_ret": float(max(call_edge_ret[i], 0.0)),
                        "label_put_edge_ret": float(max(put_edge_ret[i], 0.0)),
                        "label_selected_tradable": float(call_tradable[i] if selected_is_call else put_tradable[i]),
                        "label_selected_edge_ret": float(max(call_edge_ret[i] if selected_is_call else put_edge_ret[i], 0.0)),
                    }
                )
            offset += len(pred)

    return pd.DataFrame(rows)


def report(df: pd.DataFrame, alpha_logs: pd.DataFrame | None = None) -> int:
    print("\n" + "=" * 92)
    print("TFT LMDB alpha mapping audit")
    print("=" * 92)
    print(f"samples={len(df)} dates={sorted(df['date'].unique())} symbols={sorted(df['symbol'].unique())}")

    _print_label_distribution(df)

    ic_by_ts = df.groupby("raw_ts").apply(lambda g: _safe_spearman(g, "pred_rank", "label_return_fwd"))
    overall_spearman = df["pred_rank"].corr(df["label_return_fwd"], method="spearman")
    overall_pearson = df["pred_rank"].corr(df["label_return_fwd"])
    spr = _spread(df, "pred_rank", "label_return_fwd")
    acc = (df["pred_dir"] == df["label_direction"]).mean()
    tradable_corr = (
        df["tradable_prob"].corr(df["label_tradable"], method="spearman")
        if "tradable_prob" in df and df["tradable_prob"].std() > 1e-12 and df["label_tradable"].std() > 1e-12
        else np.nan
    )
    edge_corr = (
        df["edge_score"].corr(df["label_best_net_ret"].clip(lower=0.0), method="spearman")
        if "edge_score" in df and df["edge_score"].std() > 1e-12 and df["label_best_net_ret"].std() > 1e-12
        else np.nan
    )
    selected_tradable_corr = (
        df["selected_tradable_prob"].corr(df["label_selected_tradable"], method="spearman")
        if "selected_tradable_prob" in df and df["selected_tradable_prob"].std() > 1e-12 and df["label_selected_tradable"].std() > 1e-12
        else np.nan
    )
    selected_edge_corr = (
        df["selected_edge_score"].corr(df["label_selected_edge_ret"], method="spearman")
        if "selected_edge_score" in df and df["selected_edge_score"].std() > 1e-12 and df["label_selected_edge_ret"].std() > 1e-12
        else np.nan
    )
    tradable_lift = np.nan
    if "tradable_prob" in df and df["tradable_prob"].notna().sum() >= 20:
        n_lift = max(1, int(len(df) * 0.10))
        top = df.sort_values("tradable_prob").tail(n_lift)["label_tradable"].mean()
        tradable_lift = float(top - df["label_tradable"].mean())
    edge_spread = _spread(df, "edge_score", "label_best_net_ret") if "edge_score" in df else np.nan

    print("\n[checkpoint -> LMDB label]")
    print(
        pd.DataFrame(
            [
                {
                    "overall_spearman": overall_spearman,
                    "overall_pearson": overall_pearson,
                    "cs_ic_mean": ic_by_ts.mean(),
                    "cs_ic_median": ic_by_ts.median(),
                    "spread_top_bottom": spr,
                    "dir_acc%": acc * 100.0,
                    "tradable_corr": tradable_corr,
                    "edge_corr": edge_corr,
                    "selected_tradable_corr": selected_tradable_corr,
                    "selected_edge_corr": selected_edge_corr,
                    "top_tradable_lift": tradable_lift,
                    "edge_spread": edge_spread,
                    "pred_std": df["pred_rank"].std(),
                    "label_std": df["label_return_fwd"].std(),
                }
            ]
        )
        .round(6)
        .to_string(index=False)
    )

    dec = df.copy()
    dec["pred_decile"] = pd.qcut(dec["pred_rank"].rank(method="first"), 10, labels=False) + 1
    decile = (
        dec.groupby("pred_decile")
        .agg(
            n=("label_return_fwd", "size"),
            pred_mean=("pred_rank", "mean"),
            label_mean=("label_return_fwd", "mean"),
            label_pos=("label_return_fwd", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )
    decile["label_mean%"] = decile["label_mean"] * 100.0
    decile["label_pos%"] = decile["label_pos"] * 100.0
    print("\n[pred_rank 十分桶 vs LMDB label]")
    print(decile[["pred_decile", "n", "pred_mean", "label_mean%", "label_pos%"]].round(5).to_string(index=False))

    if "tradable_prob" in df and df["tradable_prob"].notna().any():
        td = df.copy()
        td["tradable_decile"] = pd.qcut(td["tradable_prob"].rank(method="first"), 10, labels=False) + 1
        td_decile = (
            td.groupby("tradable_decile")
            .agg(
                n=("label_tradable", "size"),
                prob_mean=("tradable_prob", "mean"),
                tradable_rate=("label_tradable", "mean"),
                best_net_mean=("label_best_net_ret", "mean"),
                ret_mean=("label_return_fwd", "mean"),
            )
            .reset_index()
        )
        for col in ["tradable_rate", "best_net_mean", "ret_mean"]:
            td_decile[col + "%"] = td_decile[col] * 100.0
        print("\n[tradable_prob 十分桶 vs 可交易标签]")
        print(
            td_decile[
                ["tradable_decile", "n", "prob_mean", "tradable_rate%", "best_net_mean%", "ret_mean%"]
            ]
            .round(5)
            .to_string(index=False)
        )

    if "edge_score" in df and df["edge_score"].notna().any():
        ed = df.copy()
        ed["edge_decile"] = pd.qcut(ed["edge_score"].rank(method="first"), 10, labels=False) + 1
        edge_decile = (
            ed.groupby("edge_decile")
            .agg(
                n=("label_best_net_ret", "size"),
                edge_pred_mean=("edge_score", "mean"),
                best_net_mean=("label_best_net_ret", "mean"),
                tradable_rate=("label_tradable", "mean"),
                ret_mean=("label_return_fwd", "mean"),
            )
            .reset_index()
        )
        for col in ["best_net_mean", "tradable_rate", "ret_mean"]:
            edge_decile[col + "%"] = edge_decile[col] * 100.0
        print("\n[edge_score 十分桶 vs executable edge]")
        print(
            edge_decile[
                ["edge_decile", "n", "edge_pred_mean", "best_net_mean%", "tradable_rate%", "ret_mean%"]
            ]
            .round(5)
            .to_string(index=False)
        )

    if "selected_tradable_prob" in df and df["selected_tradable_prob"].notna().any():
        st = df.copy()
        st["selected_tradable_decile"] = pd.qcut(st["selected_tradable_prob"].rank(method="first"), 10, labels=False) + 1
        st_decile = (
            st.groupby("selected_tradable_decile")
            .agg(
                n=("label_selected_tradable", "size"),
                prob_mean=("selected_tradable_prob", "mean"),
                selected_tradable_rate=("label_selected_tradable", "mean"),
                selected_edge_mean=("label_selected_edge_ret", "mean"),
                ret_mean=("label_return_fwd", "mean"),
            )
            .reset_index()
        )
        for col in ["selected_tradable_rate", "selected_edge_mean", "ret_mean"]:
            st_decile[col + "%"] = st_decile[col] * 100.0
        print("\n[selected side tradable_prob 十分桶 vs 实际开仓侧标签]")
        print(
            st_decile[
                ["selected_tradable_decile", "n", "prob_mean", "selected_tradable_rate%", "selected_edge_mean%", "ret_mean%"]
            ]
            .round(5)
            .to_string(index=False)
        )

    if "selected_edge_score" in df and df["selected_edge_score"].notna().any():
        se = df.copy()
        se["selected_edge_decile"] = pd.qcut(se["selected_edge_score"].rank(method="first"), 10, labels=False) + 1
        se_decile = (
            se.groupby("selected_edge_decile")
            .agg(
                n=("label_selected_edge_ret", "size"),
                edge_pred_mean=("selected_edge_score", "mean"),
                selected_edge_mean=("label_selected_edge_ret", "mean"),
                selected_tradable_rate=("label_selected_tradable", "mean"),
                ret_mean=("label_return_fwd", "mean"),
            )
            .reset_index()
        )
        for col in ["selected_edge_mean", "selected_tradable_rate", "ret_mean"]:
            se_decile[col + "%"] = se_decile[col] * 100.0
        print("\n[selected side edge_score 十分桶 vs 实际开仓侧 edge]")
        print(
            se_decile[
                ["selected_edge_decile", "n", "edge_pred_mean", "selected_edge_mean%", "selected_tradable_rate%", "ret_mean%"]
            ]
            .round(5)
            .to_string(index=False)
        )

    if alpha_logs is not None and not alpha_logs.empty:
        alpha_cols = ["date", "symbol", "ts", "alpha"]
        for col in ["tradable_prob", "edge_score"]:
            if col in alpha_logs.columns:
                alpha_cols.append(col)
        merged = df.merge(alpha_logs[alpha_cols], on=["date", "symbol", "ts"], how="inner", suffixes=("", "_log"))
        print("\n[LMDB checkpoint pred vs history_sqlite alpha_logs]")
        if merged.empty:
            print("  ⚠️ 没有匹配到 alpha_logs；检查 LMDB key timestamp 与 alpha_logs ts 是否同一口径。")
        else:
            merged = merged.copy()
            merged["pred_cs_z"] = merged.groupby(["date", "ts"], group_keys=False).apply(
                lambda g: _cs_zscore(g, "pred_rank")
            )
            ts_rank_corr = merged.groupby(["date", "ts"]).apply(
                lambda g: _safe_spearman(g, "pred_rank", "alpha")
            )
            print(f"matched={len(merged)}/{len(df)}")
            print(
                pd.DataFrame(
                    [
                        {
                            "pred_vs_alpha_spearman": merged["pred_rank"].corr(merged["alpha"], method="spearman"),
                            "pred_vs_alpha_pearson": merged["pred_rank"].corr(merged["alpha"]),
                            "pred_cs_z_vs_alpha_spearman": merged["pred_cs_z"].corr(merged["alpha"], method="spearman"),
                            "pred_cs_z_vs_alpha_pearson": merged["pred_cs_z"].corr(merged["alpha"]),
                            "per_ts_rank_corr_mean": ts_rank_corr.mean(),
                            "per_ts_rank_corr_median": ts_rank_corr.median(),
                            "mean_abs_diff": (merged["pred_rank"] - merged["alpha"]).abs().mean(),
                            "mean_abs_diff_cs_z": (merged["pred_cs_z"] - merged["alpha"]).abs().mean(),
                            "p99_abs_diff": (merged["pred_rank"] - merged["alpha"]).abs().quantile(0.99),
                            "p99_abs_diff_cs_z": (merged["pred_cs_z"] - merged["alpha"]).abs().quantile(0.99),
                        }
                    ]
                )
                .round(6)
                .to_string(index=False)
            )
            extra_rows = []
            if "tradable_prob_log" in merged.columns:
                extra_rows.append({
                    "field": "selected_tradable_prob",
                    "checkpoint_vs_alpha_logs_spearman": merged["selected_tradable_prob"].corr(merged["tradable_prob_log"], method="spearman"),
                    "checkpoint_vs_alpha_logs_pearson": merged["selected_tradable_prob"].corr(merged["tradable_prob_log"]),
                    "mean_abs_diff": (merged["selected_tradable_prob"] - merged["tradable_prob_log"]).abs().mean(),
                    "p99_abs_diff": (merged["selected_tradable_prob"] - merged["tradable_prob_log"]).abs().quantile(0.99),
                })
            if "edge_score_log" in merged.columns:
                extra_rows.append({
                    "field": "selected_edge_score",
                    "checkpoint_vs_alpha_logs_spearman": merged["selected_edge_score"].corr(merged["edge_score_log"], method="spearman"),
                    "checkpoint_vs_alpha_logs_pearson": merged["selected_edge_score"].corr(merged["edge_score_log"]),
                    "mean_abs_diff": (merged["selected_edge_score"] - merged["edge_score_log"]).abs().mean(),
                    "p99_abs_diff": (merged["selected_edge_score"] - merged["edge_score_log"]).abs().quantile(0.99),
                })
            if extra_rows:
                print("\n[checkpoint aux heads vs history_sqlite alpha_logs]")
                print(pd.DataFrame(extra_rows).round(6).to_string(index=False))

    print("\n[定位判断]")
    label_nonzero = (df["label_return_fwd"].abs() > 1e-12).mean()
    if label_nonzero < 0.005:
        print("  ❌ LMDB 固定训练标签 label_return_fwd 几乎全 0。")
        print("     这不是模型泛化问题，而是 S0 输出的 LMDB 没有映射到 option executable label，或正在审计旧 LMDB。")
        print("     请重新生成 test_quote_alpha.lmdb，并确认 USE_OPTION_EXEC_LABEL=1 与 source_has_alpha=100%。")
    elif abs(spr) < 1e-4 or (pd.notna(ic_by_ts.mean()) and ic_by_ts.mean() < 0.01):
        print("  ❌ checkpoint 在 LMDB label 上也没有排序能力：更像模型/OOS 泛化问题，不是 SE zscore。")
    else:
        print("  ✅ checkpoint 对 LMDB label 有排序能力。若 alpha_logs 无 edge，优先查 replay 推理路径/特征输入/归一化。")

    if acc < 0.40:
        print("  ⚠️ logits_dir 与 label_direction 准确率偏低；但当前 SE 只用 rank_score，方向头没有进入交易。")
    if pd.notna(tradable_corr) and tradable_corr > 0.20:
        print("  ✅ tradable_prob 对可交易样本有排序能力，可以接入策略层二级过滤。")
    if pd.notna(edge_corr) and edge_corr > 0.20:
        print("  ✅ edge_score 对可成交净收益有排序能力，可以作为开仓强度/过滤信号。")
    print("  注意：如果 rank_score 有排序能力，但按 sign 交易没有，可能需要用 logits_dir 决定 CALL/PUT，而不是 rank_score 正负。")
    return 0


def report_raw_lmdb_labels(raw_df: pd.DataFrame) -> None:
    print("\n[Raw LMDB label source audit]")
    if raw_df.empty:
        print("  ⚠️ 无法读取 raw LMDB labels")
        return
    cols = [
        "label_return_fwd",
        "label_option_alpha_300s",
        "label_option_best_net_ret_300s",
        "label_call_delayed_exec_ret_300s",
        "label_put_delayed_exec_ret_300s",
        "label_option_tradable_300s",
        "label_option_training_enabled",
        "label_option_source_has_alpha",
        "label_option_source_has_direction",
        "label_option_source_has_primary_exec",
        "label_option_source_has_matrix_primary",
        "label_option_source_missing_count",
    ]
    rows = []
    for col in cols:
        if col not in raw_df.columns:
            continue
        s = pd.to_numeric(raw_df[col], errors="coerce")
        rows.append(
            {
                "label": col,
                "non_null%": s.notna().mean() * 100.0,
                "nonzero%": (s.fillna(0.0).abs() > 1e-12).mean() * 100.0,
                "mean": s.mean(),
                "p10": s.quantile(0.10),
                "p90": s.quantile(0.90),
            }
        )
    print(pd.DataFrame(rows).round(6).to_string(index=False))
    if "label_direction" in raw_df.columns:
        print(
            "label_direction dist:",
            pd.to_numeric(raw_df["label_direction"], errors="coerce")
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(3)
            .to_dict(),
        )
    if "label_option_direction_300s" in raw_df.columns:
        print(
            "label_option_direction_300s dist:",
            pd.to_numeric(raw_df["label_option_direction_300s"], errors="coerce")
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(3)
            .to_dict(),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", type=Path, default=Path("/mnt/s990/data/h5_unified_overlap_id/test_quote_alpha.lmdb"))
    parser.add_argument("--config", type=Path, default=Path.home() / "notebook/train/slow_feature.json")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_advanced_alpha/advanced_alpha_best.pth"))
    parser.add_argument("--hist-dir", type=Path, default=Path.home() / "quant_project/data/history_sqlite_1s")
    parser.add_argument("--dates", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    dates = [x.strip() for x in args.dates.split(",") if x.strip()] if args.dates else []

    all_keys = _read_lmdb_keys(args.lmdb)
    audit_keys = _filter_keys_by_dates(all_keys, dates)
    if args.sample_limit > 0:
        audit_keys = audit_keys[: args.sample_limit]
    raw_labels = _read_lmdb_sample_labels(args.lmdb, audit_keys, sample_limit=50000)

    df = evaluate_lmdb(
        lmdb_path=args.lmdb,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        device_name=args.device,
        dates=dates,
    )
    report_raw_lmdb_labels(raw_labels)
    alpha_logs = None
    if dates:
        alpha_logs = _load_alpha_logs(args.hist_dir, dates)
    return report(df, alpha_logs)


if __name__ == "__main__":
    raise SystemExit(main())
