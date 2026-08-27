from __future__ import annotations

import pandas as pd

from maga7.tools.run_earnings_ah_straddle_scoreboard import (
    _pick_atm_straddle,
    _summarize,
)


def test_pick_atm_skips_mismatched_zero_dte():
    lock = pd.DataFrame(
        [
            {
                "symbol": "GOOGL",
                "date_str": "2026-07-22",
                "front_dte": 0,
                "bucket_id": 0,
                "strike": 330.0,
                "lock_spot": 348.0,
                "contract_symbol": "O:GOOGL260722P00330000",
                "tag": "bad_p",
            },
            {
                "symbol": "GOOGL",
                "date_str": "2026-07-22",
                "front_dte": 0,
                "bucket_id": 2,
                "strike": 390.0,
                "lock_spot": 348.0,
                "contract_symbol": "O:GOOGL260722C00390000",
                "tag": "bad_c",
            },
            {
                "symbol": "GOOGL",
                "date_str": "2026-07-22",
                "front_dte": 2,
                "bucket_id": 0,
                "strike": 347.5,
                "lock_spot": 348.0,
                "contract_symbol": "O:GOOGL260724P00347500",
                "tag": "ok_p",
            },
            {
                "symbol": "GOOGL",
                "date_str": "2026-07-22",
                "front_dte": 2,
                "bucket_id": 2,
                "strike": 347.5,
                "lock_spot": 348.0,
                "contract_symbol": "O:GOOGL260724C00347500",
                "tag": "ok_c",
            },
        ]
    )
    pick = _pick_atm_straddle(
        lock, symbol="GOOGL", date="2026-07-22", prefer_dte=(2, 1, 0)
    )
    assert pick is not None
    assert pick["front_dte"] == 2
    assert pick["strike"] == 347.5


def test_summarize_small_n_verdict():
    df = pd.DataFrame(
        [
            {
                "status": "ok",
                "stock_gap_abs": 0.01,
                "em_pct": 0.05,
                "move_vs_em": 0.2,
                "straddle_ret_next_open": -0.4,
                "straddle_ret_next_p30": -0.4,
                "straddle_ret_next_p60": -0.4,
                "iv_crush": -0.15,
            },
            {
                "status": "ok",
                "stock_gap_abs": 0.09,
                "em_pct": 0.06,
                "move_vs_em": 1.5,
                "straddle_ret_next_open": 0.7,
                "straddle_ret_next_p30": 1.0,
                "straddle_ret_next_p60": 1.2,
                "iv_crush": None,
            },
        ]
    )
    s = _summarize(df)
    assert s["n_ok"] == 2
    assert s["verdict"] == "RESEARCH_ONLY_SMALL_N"
