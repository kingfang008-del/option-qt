#!/usr/bin/env python3
"""Mag7 earnings-AH ATM straddle scoreboard (research).

For each AH earnings name:
  buy ATM call+put on AH eve RTH close, mark next session open / +30m / +60m.

Measures:
  - stock overnight gap vs priced expected move (straddle/spot)
  - straddle mid return (captures IV crush + move jointly)
  - optional day_iv crush when available

Missing option days are kept as rows with status=missing_data (Jul 29/30).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import to_ny

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

# Curated Mag7 AH events (Finnhub Jul validation + manual NVDA).
DEFAULT_EVENTS: list[dict[str, str]] = [
    {"date": "2026-05-20", "symbol": "NVDA", "source": "manual"},
    {"date": "2026-07-22", "symbol": "GOOGL", "source": "finnhub"},
    {"date": "2026-07-22", "symbol": "TSLA", "source": "finnhub"},
    {"date": "2026-07-29", "symbol": "META", "source": "finnhub"},
    {"date": "2026-07-29", "symbol": "MSFT", "source": "finnhub"},
    {"date": "2026-07-30", "symbol": "AAPL", "source": "finnhub"},
    {"date": "2026-07-30", "symbol": "AMZN", "source": "finnhub"},
]

BUCKET_PUT = 0
BUCKET_CALL = 2


@dataclass
class LegQuote:
    mid: float | None
    source: str
    ts: str | None = None


def _norm_ticker(t: str) -> str:
    s = str(t).strip()
    if s.startswith("O:"):
        return s
    return f"O:{s}"


def _bare_ticker(t: str) -> str:
    s = str(t).strip()
    return s[2:] if s.startswith("O:") else s


def _load_lock(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    df["date_str"] = df["date_str"].astype(str)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["bucket_id"] = df["bucket_id"].astype(int)
    df["front_dte"] = df["front_dte"].astype(int)
    return df


def _pick_atm_straddle(
    lock: pd.DataFrame, *, symbol: str, date: str, prefer_dte: tuple[int, ...]
) -> dict[str, Any] | None:
    day = lock[(lock["symbol"] == symbol) & (lock["date_str"] == date)]
    if day.empty:
        return None
    for dte in prefer_dte:
        sub = day[day["front_dte"] == int(dte)]
        put = sub[sub["bucket_id"] == BUCKET_PUT]
        call = sub[sub["bucket_id"] == BUCKET_CALL]
        if put.empty or call.empty:
            continue
        # Same strike only — skip broken 0DTE ATM mismatches (e.g. GOOGL 07-22).
        for _, pr in put.iterrows():
            cr = call[np.isclose(call["strike"].astype(float), float(pr["strike"]))]
            if cr.empty:
                continue
            c0 = cr.iloc[0]
            return {
                "front_dte": int(dte),
                "strike": float(pr["strike"]),
                "lock_spot": float(pr["lock_spot"]),
                "put": _norm_ticker(pr["contract_symbol"]),
                "call": _norm_ticker(c0["contract_symbol"]),
                "put_tag": str(pr.get("tag") or ""),
                "call_tag": str(c0.get("tag") or ""),
            }
    return None


def _next_session(stock_root: Path, symbol: str, ah_date: str) -> str | None:
    """Next calendar file after ah_date under stock_1s_root."""
    folder = stock_root / symbol
    if not folder.is_dir():
        return None
    dates = sorted(
        p.stem.split("_")[-1]
        for p in folder.glob(f"{symbol}_????-??-??.parquet")
        if p.stem.split("_")[-1] > ah_date
    )
    return dates[0] if dates else None


def _stock_close_open(
    stock_root: Path, symbol: str, eve: str, nxt: str
) -> tuple[float | None, float | None]:
    def _rth_ohlc(date: str) -> pd.DataFrame | None:
        p = stock_root / symbol / f"{symbol}_{date}.parquet"
        if not p.is_file():
            return None
        df = pd.read_parquet(p)
        ts = "timestamp" if "timestamp" in df.columns else "ts"
        df[ts] = pd.to_datetime(df[ts], utc=True).dt.tz_convert("America/New_York")
        df = df[
            (df[ts].dt.time >= pd.Timestamp("09:30").time())
            & (df[ts].dt.time < pd.Timestamp("16:00").time())
        ]
        if df.empty:
            return None
        return df.sort_values(ts)

    a = _rth_ohlc(eve)
    b = _rth_ohlc(nxt)
    if a is None or b is None:
        return None, None
    return float(a.iloc[-1]["close"]), float(b.iloc[0]["open"] if "open" in b.columns else b.iloc[0]["close"])


def _series_to_ny(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, utc=True)
    return ts.dt.tz_convert("America/New_York")


def _read_quote_1s(root: Path, symbol: str, date: str) -> pd.DataFrame | None:
    p = root / symbol / f"{symbol}_{date}.parquet"
    if not p.is_file():
        return None
    df = pd.read_parquet(p)
    df = df.copy()
    df["timestamp"] = _series_to_ny(df["timestamp"])
    df["ticker_norm"] = df["ticker"].map(_norm_ticker)
    return df


def _read_option_1m(root: Path, symbol: str, date: str) -> pd.DataFrame | None:
    p = root / symbol / f"{symbol}_{date}.parquet"
    if not p.is_file():
        return None
    df = pd.read_parquet(p)
    df = df.copy()
    df["timestamp"] = _series_to_ny(df["timestamp"])
    df["ticker_norm"] = df["ticker"].map(_norm_ticker)
    return df


def _mid_asof(
    *,
    quote: pd.DataFrame | None,
    opt1m: pd.DataFrame | None,
    ticker: str,
    asof: pd.Timestamp,
    mode: str,
) -> LegQuote:
    """mode: last_le | first_ge."""
    t = _norm_ticker(ticker)
    # Prefer quote mid when present; fall back to option_1m close.
    for src_name, frame, col in (
        ("quote_1s", quote, "mid_price"),
        ("option_1m", opt1m, "c"),
    ):
        if frame is None or frame.empty:
            continue
        sub = frame[frame["ticker_norm"] == t]
        if sub.empty:
            continue
        if mode == "last_le":
            win = sub[sub["timestamp"] <= asof]
            if win.empty:
                continue
            row = win.iloc[-1]
        else:
            win = sub[sub["timestamp"] >= asof]
            if win.empty:
                continue
            row = win.iloc[0]
        try:
            px = float(row[col])
        except (TypeError, ValueError, KeyError):
            continue
        if not np.isfinite(px) or px <= 0:
            continue
        return LegQuote(px, src_name, str(to_ny(row["timestamp"])))
    return LegQuote(None, "missing", None)


def _iv_asof(day_iv_root: Path, symbol: str, date: str, ticker: str, asof: pd.Timestamp) -> float | None:
    p = day_iv_root / symbol / f"{symbol}_{date}.parquet"
    if not p.is_file():
        return None
    df = pd.read_parquet(p)
    if "iv" not in df.columns or "ticker" not in df.columns:
        return None
    df = df.copy()
    df["timestamp"] = _series_to_ny(df["timestamp"])
    bare = _bare_ticker(ticker)
    sub = df[df["ticker"].astype(str).map(_bare_ticker) == bare]
    if sub.empty:
        return None
    win = sub[sub["timestamp"] <= asof]
    if win.empty:
        win = sub[sub["timestamp"] >= asof]
        if win.empty:
            return None
        row = win.iloc[0]
    else:
        row = win.iloc[-1]
    try:
        iv = float(row["iv"])
    except (TypeError, ValueError):
        return None
    return iv if np.isfinite(iv) and iv > 0 else None


def _load_events_from_json(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return list(DEFAULT_EVENTS)
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = raw.get("events") if isinstance(raw, dict) else raw
    out: list[dict[str, str]] = []
    for e in events or []:
        if not isinstance(e, dict):
            continue
        tag = str(e.get("tag") or "")
        sym = str(e.get("symbol") or "").strip().upper()
        d = str(e.get("date") or "").strip()
        if not d or not sym:
            continue
        if "earnings" not in tag.lower() and tag not in {"nvda_earnings_ah"}:
            continue
        if tag.endswith("_bmo") or "bmo" in tag.lower():
            continue  # AH scoreboard only
        out.append({"date": d, "symbol": sym, "source": str(e.get("source") or "")})
    # Always include manual NVDA if calendar JSON is Jul-only.
    have = {(r["date"], r["symbol"]) for r in out}
    for e in DEFAULT_EVENTS:
        if (e["date"], e["symbol"]) not in have:
            out.append(e)
    out.sort(key=lambda x: (x["date"], x["symbol"]))
    return out


def evaluate_event(
    *,
    symbol: str,
    ah_date: str,
    source: str,
    lock: pd.DataFrame,
    paths: dict[str, Any],
    prefer_dte: tuple[int, ...],
) -> dict[str, Any]:
    stock_root = Path(paths["stock_1s_root"]).expanduser()
    quote_root = Path(paths["quote_1s_root"]).expanduser()
    opt1m_root = Path(paths["option_1m_root"]).expanduser()
    day_iv_root = Path(paths.get("day_iv_root") or "").expanduser()

    nxt = _next_session(stock_root, symbol, ah_date)
    row: dict[str, Any] = {
        "ah_date": ah_date,
        "next_date": nxt,
        "symbol": symbol,
        "source": source,
        "status": "ok",
    }
    if nxt is None:
        row["status"] = "missing_next_session"
        return row

    pick = _pick_atm_straddle(lock, symbol=symbol, date=ah_date, prefer_dte=prefer_dte)
    if pick is None:
        row["status"] = "missing_lock_atm"
        return row
    row.update(
        {
            "front_dte": pick["front_dte"],
            "strike": pick["strike"],
            "lock_spot": pick["lock_spot"],
            "put": pick["put"],
            "call": pick["call"],
        }
    )

    eve_close, nxt_open = _stock_close_open(stock_root, symbol, ah_date, nxt)
    row["stock_eve_close"] = eve_close
    row["stock_next_open"] = nxt_open
    if eve_close and nxt_open and eve_close > 0:
        row["stock_gap"] = nxt_open / eve_close - 1.0
        row["stock_gap_abs"] = abs(row["stock_gap"])
    else:
        row["stock_gap"] = None
        row["stock_gap_abs"] = None

    q_eve = _read_quote_1s(quote_root, symbol, ah_date)
    q_nxt = _read_quote_1s(quote_root, symbol, nxt)
    o_eve = _read_option_1m(opt1m_root, symbol, ah_date)
    o_nxt = _read_option_1m(opt1m_root, symbol, nxt)

    eve_asof = pd.Timestamp(f"{ah_date} 15:59:00", tz="America/New_York")
    marks = {
        "eve_close": (eve_asof, "last_le", q_eve, o_eve),
        "next_open": (pd.Timestamp(f"{nxt} 09:30:00", tz="America/New_York"), "first_ge", q_nxt, o_nxt),
        "next_p30": (pd.Timestamp(f"{nxt} 10:00:00", tz="America/New_York"), "first_ge", q_nxt, o_nxt),
        "next_p60": (pd.Timestamp(f"{nxt} 10:30:00", tz="America/New_York"), "first_ge", q_nxt, o_nxt),
    }

    legs: dict[str, dict[str, LegQuote]] = {}
    for name, (asof, mode, q, o) in marks.items():
        legs[name] = {
            "call": _mid_asof(quote=q, opt1m=o, ticker=pick["call"], asof=asof, mode=mode),
            "put": _mid_asof(quote=q, opt1m=o, ticker=pick["put"], asof=asof, mode=mode),
        }

    for name, pair in legs.items():
        c, p = pair["call"].mid, pair["put"].mid
        row[f"{name}_call"] = c
        row[f"{name}_put"] = p
        row[f"{name}_call_src"] = pair["call"].source
        row[f"{name}_put_src"] = pair["put"].source
        if c is not None and p is not None:
            row[f"{name}_straddle"] = float(c) + float(p)
        else:
            row[f"{name}_straddle"] = None
            if name in {"eve_close", "next_open"} and row["status"] == "ok":
                row["status"] = "missing_option_quote"

    eve_s = row.get("eve_close_straddle")
    spot = eve_close or pick["lock_spot"]
    if eve_s and spot and float(spot) > 0:
        row["em_pct"] = float(eve_s) / float(spot)
    else:
        row["em_pct"] = None

    if row.get("stock_gap_abs") is not None and row.get("em_pct") is not None:
        row["move_vs_em"] = float(row["stock_gap_abs"]) / float(row["em_pct"]) if row["em_pct"] else None
    else:
        row["move_vs_em"] = None

    for name in ("next_open", "next_p30", "next_p60"):
        s = row.get(f"{name}_straddle")
        if eve_s and s is not None and float(eve_s) > 0:
            row[f"straddle_ret_{name}"] = float(s) / float(eve_s) - 1.0
        else:
            row[f"straddle_ret_{name}"] = None

    # IV crush (when day_iv present for both sessions).
    if day_iv_root.is_dir():
        iv_eve_c = _iv_asof(day_iv_root, symbol, ah_date, pick["call"], eve_asof)
        iv_eve_p = _iv_asof(day_iv_root, symbol, ah_date, pick["put"], eve_asof)
        iv_nxt_c = _iv_asof(
            day_iv_root,
            symbol,
            nxt,
            pick["call"],
            pd.Timestamp(f"{nxt} 09:35:00", tz="America/New_York"),
        )
        iv_nxt_p = _iv_asof(
            day_iv_root,
            symbol,
            nxt,
            pick["put"],
            pd.Timestamp(f"{nxt} 09:35:00", tz="America/New_York"),
        )
        row["iv_eve_call"] = iv_eve_c
        row["iv_eve_put"] = iv_eve_p
        row["iv_next_call"] = iv_nxt_c
        row["iv_next_put"] = iv_nxt_p
        if None not in (iv_eve_c, iv_eve_p, iv_nxt_c, iv_nxt_p):
            eve_iv = 0.5 * (float(iv_eve_c) + float(iv_eve_p))
            nxt_iv = 0.5 * (float(iv_nxt_c) + float(iv_nxt_p))
            row["iv_eve_avg"] = eve_iv
            row["iv_next_avg"] = nxt_iv
            row["iv_crush"] = nxt_iv / eve_iv - 1.0 if eve_iv > 0 else None
        else:
            row["iv_eve_avg"] = None
            row["iv_next_avg"] = None
            row["iv_crush"] = None
    return row


def _summarize(df: pd.DataFrame) -> dict[str, Any]:
    ok = df[df["status"] == "ok"].copy()
    out: dict[str, Any] = {
        "n_events": int(len(df)),
        "n_ok": int(len(ok)),
        "n_missing": int((df["status"] != "ok").sum()),
        "missing": df.loc[
            df["status"] != "ok",
            [c for c in ("ah_date", "symbol", "status") if c in df.columns],
        ].to_dict("records"),
    }
    if ok.empty:
        out["verdict"] = "INSUFFICIENT_DATA"
        return out
    for col in (
        "stock_gap_abs",
        "em_pct",
        "move_vs_em",
        "straddle_ret_next_open",
        "straddle_ret_next_p30",
        "straddle_ret_next_p60",
        "iv_crush",
    ):
        s = pd.to_numeric(ok[col], errors="coerce") if col in ok.columns else pd.Series(dtype=float)
        out[col] = {
            "n": int(s.notna().sum()),
            "mean": float(s.mean()) if s.notna().any() else None,
            "median": float(s.median()) if s.notna().any() else None,
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
            "pct_pos": float((s > 0).mean()) if s.notna().any() else None,
        }
    # Heuristic research gate (tiny n): need more crush/no-move samples.
    open_rets = pd.to_numeric(ok["straddle_ret_next_open"], errors="coerce").dropna()
    if len(open_rets) < 3:
        out["verdict"] = "RESEARCH_ONLY_SMALL_N"
    elif float((open_rets > 0).mean()) >= 0.6 and float(open_rets.mean()) > 0:
        out["verdict"] = "EDGE_HINT_NEED_MORE_SAMPLE"
    else:
        out["verdict"] = "NO_CLEAR_EDGE_OR_CRUSH_RISK"
    out["note"] = (
        "Long AH straddle wins only when |gap| >> priced EM after IV crush. "
        "Do not promote sleeve from TSLA-like winners alone."
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument(
        "--calendar",
        default=str(ROOT / "maga7/results/event_calendar_finnhub_jul2026.json"),
        help="JSON with earnings_ah events; NVDA manual always merged",
    )
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/earnings_ah_straddle_scoreboard_v1",
    )
    ap.add_argument(
        "--prefer-dte",
        default="2,1,0",
        help="Comma list; first dte with matching ATM C/P strike wins",
    )
    args = ap.parse_args()

    prof = load_profile(args.profile)
    paths = dict(prof.get("paths") or {})
    lock_path = Path(paths["open_locked_map"]).expanduser()
    lock = _load_lock(lock_path)
    prefer = tuple(int(x) for x in str(args.prefer_dte).split(",") if str(x).strip() != "")
    events = _load_events_from_json(Path(args.calendar) if args.calendar else None)

    rows = [
        evaluate_event(
            symbol=e["symbol"],
            ah_date=e["date"],
            source=e.get("source") or "",
            lock=lock,
            paths=paths,
            prefer_dte=prefer,
        )
        for e in events
    ]
    df = pd.DataFrame(rows)
    summary = _summarize(df)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(
        df.to_json(orient="records", indent=2), encoding="utf-8"
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "verdict.json").write_text(
        json.dumps(
            {
                "verdict": summary.get("verdict"),
                "n_ok": summary.get("n_ok"),
                "n_missing": summary.get("n_missing"),
                "note": summary.get("note"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    if not df.empty:
        cols = [
            c
            for c in [
                "ah_date",
                "symbol",
                "status",
                "front_dte",
                "strike",
                "stock_gap",
                "em_pct",
                "move_vs_em",
                "straddle_ret_next_open",
                "straddle_ret_next_p30",
                "iv_crush",
            ]
            if c in df.columns
        ]
        print(df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
