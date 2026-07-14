#!/usr/bin/env python3
"""由标准 VX 月合约日线构造因果期限结构序列。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/mnt/s990/data/raw_1m/vix_futures_databento")


def _causal_z(series: pd.Series, window: int = 63, min_periods: int = 20) -> pd.Series:
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=min_periods).mean()
    std = prior.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def build_term_structure(bars: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["ts_event"], utc=True).dt.normalize()
    bars["symbol"] = bars["symbol"].astype(str)
    bars = bars.drop_duplicates(["date", "symbol"], keep="last")

    definitions = definitions.copy()
    definitions["symbol"] = definitions["symbol"].astype(str)
    definitions["expiration"] = pd.to_datetime(
        definitions["expiration"], utc=True, errors="coerce"
    ).dt.normalize()
    expiry = (
        definitions.dropna(subset=["expiration"])
        .sort_values("ts_event")
        .drop_duplicates("symbol", keep="last")
        .set_index("symbol")["expiration"]
    )
    bars["expiration"] = bars["symbol"].map(expiry)
    bars = bars.dropna(subset=["expiration", "close"])
    bars["dte"] = (bars["expiration"] - bars["date"]).dt.days
    # 到期日早盘结算，日频 selector 不使用 dte=0 合约。
    bars = bars.loc[bars["dte"] > 0].copy()

    rows: list[dict[str, object]] = []
    for day, group in bars.groupby("date", sort=True):
        active = group.sort_values(["expiration", "symbol"]).head(2)
        if len(active) < 2:
            continue
        front, second = active.iloc[0], active.iloc[1]
        d1, d2 = int(front["dte"]), int(second["dte"])
        f1, f2 = float(front["close"]), float(second["close"])
        if d2 <= d1 or f1 <= 0 or f2 <= 0:
            continue
        w2 = float(np.clip((30.0 - d1) / (d2 - d1), 0.0, 1.0))
        w1 = 1.0 - w2
        rows.append(
            {
                "date": day,
                "vx1_symbol": front["symbol"],
                "vx2_symbol": second["symbol"],
                "vx1_expiration": front["expiration"],
                "vx2_expiration": second["expiration"],
                "vx1_dte": d1,
                "vx2_dte": d2,
                "vx1_close": f1,
                "vx2_close": f2,
                "vx1_volume": int(front["volume"]),
                "vx2_volume": int(second["volume"]),
                "vx_curve_slope": f2 / f1 - 1.0,
                "vx_curve_slope_30d": np.log(f2 / f1) * 30.0 / (d2 - d1),
                "vx_cm30_close": w1 * f1 + w2 * f2,
                "vx_cm30_w1": w1,
                "vx_cm30_w2": w2,
            }
        )

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["vx1_ret_1d"] = out["vx1_close"].pct_change(fill_method=None)
    out["vx_cm30_ret_1d"] = out["vx_cm30_close"].pct_change(fill_method=None)
    out["vx1_level_z63"] = _causal_z(np.log(out["vx1_close"]))
    out["vx_cm30_level_z63"] = _causal_z(np.log(out["vx_cm30_close"]))
    out["vx_curve_slope_z63"] = _causal_z(out["vx_curve_slope"])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build VX1/VX2 constant-maturity features")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()
    bars_path = root / "vx_standard_ohlcv_1d.parquet"
    definitions_path = root / "vx_standard_definitions.parquet"
    output = args.output or root / "vx_term_structure_1d.parquet"
    bars = pd.read_parquet(bars_path)
    definitions = pd.read_parquet(definitions_path)
    result = build_term_structure(bars, definitions)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(
        f"wrote {output} rows={len(result):,} "
        f"range={result['date'].min()}..{result['date'].max()}"
    )
    print(
        f"missing_z63={int(result['vx_curve_slope_z63'].isna().sum())} "
        f"front_symbols={result['vx1_symbol'].nunique()}"
    )


if __name__ == "__main__":
    main()
