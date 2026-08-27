#!/usr/bin/env python3
"""Post-fill WAVE_ABORT vs baseline / pre-fill hard path confirm.

Windows: May–Jul (→07-21), Jan–Mar, Jul21 single-day.
Research only — does not promote into peer3_v1.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

PEER3 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
OUT_V1 = Path("/mnt/s990/data/maga7/results/wave_abort_postfill_v1")
OUT = Path("/mnt/s990/data/maga7/results/wave_abort_postfill_v1")

WINDOWS = {
    "may_jul": ("2026-05-01", "2026-07-21"),
    "jan_mar": ("2026-01-02", "2026-03-31"),
    "jul21": ("2026-07-21", "2026-07-21"),
}

WAVE_ABORT = {
    "enabled": True,
    "thr_pos": 0.0015,
    "thr_neg": -0.003,
    "max_wait_seconds": 300,
    "revoke_seconds": 1800,
    "on_timeout": "abort",
}

WAVE_HARD_PREFILL = {
    "enabled": True,
    "thr_pos": 0.0015,
    "thr_neg": -0.003,
    "max_wait_seconds": 300,
    "tod_start": "09:30",
    "tod_end": "16:00",
    "on_timeout": "block",
    "delay_on_pos": False,
}

VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "wave_abort": {"wave_abort": WAVE_ABORT},
    "wave_hard_prefill": {"stock_path_confirm": WAVE_HARD_PREFILL},
    "wave_abort_timeout_allow": {
        "wave_abort": {**WAVE_ABORT, "on_timeout": "allow"},
    },
}

# Knife-2: cool the hot revoke (keep Jul21, lift May–Jul retain).
KNIFE2: dict[str, dict[str, Any]] = {
    "baseline": {},
    "hot_rev1800": {"wave_abort": WAVE_ABORT},  # §8b reference
    "rev600": {"wave_abort": {**WAVE_ABORT, "revoke_seconds": 600}},
    "rev900": {"wave_abort": {**WAVE_ABORT, "revoke_seconds": 900}},
    "asym_r5": {
        "wave_abort": {
            **WAVE_ABORT,
            "thr_neg_revoke": -0.005,
            "revoke_seconds": 1800,
        }
    },
    "asym_r5_rev900": {
        "wave_abort": {
            **WAVE_ABORT,
            "thr_neg_revoke": -0.005,
            "revoke_seconds": 900,
        }
    },
    "rev_opt0": {
        "wave_abort": {
            **WAVE_ABORT,
            "revoke_opt_mtm_max": 0.0,
            "revoke_seconds": 1800,
        }
    },
    "rev600_opt0": {
        "wave_abort": {
            **WAVE_ABORT,
            "revoke_seconds": 600,
            "revoke_opt_mtm_max": 0.0,
        }
    },
    "asym_r5_opt0_rev900": {
        "wave_abort": {
            **WAVE_ABORT,
            "thr_neg_revoke": -0.005,
            "revoke_opt_mtm_max": 0.0,
            "revoke_seconds": 900,
        }
    },
    "no_revoke": {
        "wave_abort": {**WAVE_ABORT, "allow_revoke": False},
    },
}

# Knife-2b: timeout is the TP killer (~300s); keep revoke for Jul21.
KNIFE2B: dict[str, dict[str, Any]] = {
    "baseline": {},
    "hot_abort_timeout": {"wave_abort": WAVE_ABORT},
    "allow_to_rev1800": {
        "wave_abort": {**WAVE_ABORT, "on_timeout": "allow", "revoke_seconds": 1800},
    },
    "allow_to_rev600": {
        "wave_abort": {**WAVE_ABORT, "on_timeout": "allow", "revoke_seconds": 600},
    },
    "allow_to_asym_r5_rev900": {
        "wave_abort": {
            **WAVE_ABORT,
            "on_timeout": "allow",
            "thr_neg_revoke": -0.005,
            "revoke_seconds": 900,
        }
    },
    "wait600_abort": {
        "wave_abort": {**WAVE_ABORT, "max_wait_seconds": 600, "revoke_seconds": 1800},
    },
    "wait900_abort": {
        "wave_abort": {**WAVE_ABORT, "max_wait_seconds": 900, "revoke_seconds": 1800},
    },
    "allow_to_only_neg_rev": {
        # soft confirm: never timeout-abort; only adverse / revoke
        "wave_abort": {
            **WAVE_ABORT,
            "on_timeout": "allow",
            "revoke_seconds": 1800,
            "thr_neg": -0.003,
        }
    },
}

CLOCK_REASONS = {"T+30", "T+45", "TIME", "HOLD", "HOLD_EXTEND", "EXTEND"}


def _trade_tail(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty or "ret" not in trades.columns:
        return {
            "worst": None,
            "n_le_25": 0,
            "left_tail_sum": 0.0,
            "worst_trade": None,
            "n_wave_abort": 0,
            "n_clock": 0,
            "clock_share": None,
            "reasons": {},
        }
    r = pd.to_numeric(trades["ret"], errors="coerce")
    vc = {str(k): int(v) for k, v in trades["reason"].value_counts().items()} if "reason" in trades.columns else {}
    n = int(sum(vc.values())) if vc else len(trades)
    n_clock = sum(v for k, v in vc.items() if k in CLOCK_REASONS or str(k).startswith("T+"))
    worst_i = int(r.idxmin()) if len(r) and r.notna().any() else None
    worst_trade = None
    if worst_i is not None:
        row = trades.loc[worst_i]
        worst_trade = {
            "date": str(row.get("date", "")),
            "symbol": str(row.get("symbol", "")),
            "reason": str(row.get("reason", "")),
            "ret": float(row["ret"]),
        }
    return {
        "worst": float(r.min()) if r.notna().any() else None,
        "n_le_25": int((r <= -0.25).sum()),
        "left_tail_sum": float(r[r <= -0.15].sum()) if r.notna().any() else 0.0,
        "worst_trade": worst_trade,
        "n_wave_abort": int(vc.get("WAVE_ABORT", 0)),
        "n_clock": n_clock,
        "clock_share": float(n_clock / n) if n else None,
        "reasons": vc,
    }


def run_one(window: str, variant: str, overlay: dict) -> dict[str, Any]:
    start, end = WINDOWS[window]
    prof = deepcopy(load_profile(PEER3))
    prof["date_range"] = {"start": start, "end": end}
    trade = prof.setdefault("trade", {})
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(trade.get(k), dict):
            trade[k] = {**trade[k], **v}
        else:
            trade[k] = v
    print(f"=== {window} / {variant} {start}..{end} ===", flush=True)
    result = run_offline_replay(prof, scheme="single")
    summary, trades, daily = result["summary"], result["trades"], result.get("daily")
    tag = OUT / window / f"replay__{variant}"
    tag.mkdir(parents=True, exist_ok=True)
    (tag / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    trades.to_csv(tag / "trades.csv", index=False)
    if daily is not None and len(daily):
        daily.to_csv(tag / "daily.csv", index=False)
    tail = _trade_tail(trades)
    row = {
        "window": window,
        "variant": variant,
        "start": start,
        "end": end,
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "n_path_block": int(summary.get("n_stock_path_confirm_block") or 0),
        "n_path_ok": int(summary.get("n_stock_path_confirm_ok") or 0),
        "worst": tail["worst"],
        "n_le_25": tail["n_le_25"],
        "left_tail_sum": tail["left_tail_sum"],
        "worst_trade": tail["worst_trade"],
        "n_wave_abort": tail["n_wave_abort"],
        "n_clock": tail["n_clock"],
        "clock_share": tail["clock_share"],
        "reasons": tail["reasons"],
    }
    print(
        f"  ret={row['total_ret']:+.3f} n={row['n_trades']} "
        f"≤-25%={row['n_le_25']} worst={row['worst']} "
        f"WAVE_ABORT={row['n_wave_abort']} clock_share={row['clock_share']}",
        flush=True,
    )
    return row


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default="may_jul,jan_mar,jul21")
    ap.add_argument("--variants", default="baseline,wave_abort,wave_hard_prefill")
    ap.add_argument(
        "--mode",
        choices=("v1", "knife2", "knife2b"),
        default="v1",
        help="v1=prefill; knife2=cool revoke; knife2b=timeout-allow (→ v2b)",
    )
    ap.add_argument(
        "--reuse-baseline",
        action="store_true",
        help="copy baseline summary/trades from wave_abort_postfill_v1 when present",
    )
    args = ap.parse_args()
    if args.mode == "knife2":
        catalog = KNIFE2
        OUT = Path("/mnt/s990/data/maga7/results/wave_abort_postfill_v2")
    elif args.mode == "knife2b":
        catalog = KNIFE2B
        OUT = Path("/mnt/s990/data/maga7/results/wave_abort_postfill_v2b")
    else:
        catalog = VARIANTS
    if args.mode in {"knife2", "knife2b"} and args.variants == "baseline,wave_abort,wave_hard_prefill":
        variants = ["baseline"] + [v for v in catalog if v != "baseline"]
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    wins = [w.strip() for w in args.windows.split(",") if w.strip()]
    for v in variants:
        if v not in catalog:
            raise SystemExit(f"unknown variant {v}; choose from {list(catalog)}")

    scoreboard: list[dict] = []
    base_rets: dict[str, float] = {}
    for window in wins:
        for variant in variants:
            if (
                args.reuse_baseline
                and variant == "baseline"
                and (OUT_V1 / window / "replay__baseline" / "summary.json").exists()
            ):
                import shutil

                src = OUT_V1 / window / "replay__baseline"
                dst = OUT / window / "replay__baseline"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.resolve() != src.resolve():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                summary = json.loads((dst / "summary.json").read_text(encoding="utf-8"))
                trades = pd.read_csv(dst / "trades.csv")
                tail = _trade_tail(trades)
                row = {
                    "window": window,
                    "variant": "baseline",
                    "start": WINDOWS[window][0],
                    "end": WINDOWS[window][1],
                    "total_ret": float(summary["total_ret"]),
                    "maxdd": float(summary["maxdd"]),
                    "n_trades": int(summary["n_trades"]),
                    "n_path_block": int(summary.get("n_stock_path_confirm_block") or 0),
                    "n_path_ok": int(summary.get("n_stock_path_confirm_ok") or 0),
                    "worst": tail["worst"],
                    "n_le_25": tail["n_le_25"],
                    "left_tail_sum": tail["left_tail_sum"],
                    "worst_trade": tail["worst_trade"],
                    "n_wave_abort": tail["n_wave_abort"],
                    "n_clock": tail["n_clock"],
                    "clock_share": tail["clock_share"],
                    "reasons": tail["reasons"],
                    "ret_retention": 1.0,
                }
                base_rets[window] = row["total_ret"]
                print(f"=== {window} / baseline (reused v1) ===", flush=True)
                scoreboard.append(row)
                continue
            row = run_one(window, variant, catalog[variant])
            if variant == "baseline":
                base_rets[window] = row["total_ret"]
            b = base_rets.get(window)
            if b is not None and (1 + b) != 0:
                row["ret_retention"] = float((1 + row["total_ret"]) / (1 + b))
            scoreboard.append(row)

    (OUT / "scoreboard.json").write_text(
        json.dumps(scoreboard, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if k not in {"reasons", "worst_trade"}
            }
            | {
                "worst_trade": json.dumps(r.get("worst_trade"), default=str),
                "reasons": json.dumps(r.get("reasons"), default=str),
            }
            for r in scoreboard
        ]
    ).to_csv(OUT / "scoreboard.csv", index=False)

    # Jul21 AMD focus + dual-window verdict sketch
    jul21 = [r for r in scoreboard if r["window"] == "jul21"]
    may = {r["variant"]: r for r in scoreboard if r["window"] == "may_jul"}
    jan = {r["variant"]: r for r in scoreboard if r["window"] == "jan_mar"}
    verdict = {
        "out": str(OUT),
        "jul21": jul21,
        "may_jul_wave_abort_retain": (may.get("wave_abort") or {}).get("ret_retention"),
        "jan_mar_wave_abort_n_le_25": (jan.get("wave_abort") or {}).get("n_le_25"),
        "may_jul_baseline_n_le_25": (may.get("baseline") or {}).get("n_le_25"),
        "may_jul_wave_abort_n_le_25": (may.get("wave_abort") or {}).get("n_le_25"),
        "note": (
            "Pass bar (research): Jul21 AMD not T+30≤-25%; "
            "may_jul retain not ≪0.5 if tails improve; "
            "jan_mar ≤-25% not worse. No peer3_v1 promote."
        ),
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print(json.dumps(verdict, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
