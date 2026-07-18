#!/usr/bin/env python3
"""离线 LIVE 对齐 replay（可追溯版本）。

把原先散落的 inline 回放固化成单一入口：
  - 配方默认 = 当前生产倾向：LIVE_REPLAY + VX selector + PUT quarantine
  - 每次 run 写入 run_dir/manifest.json（git / 脚本 / gate / infer 路径与 mtime）
  - summary.json 内嵌同一 provenance，便于事后对拍

用法:
  # 默认 6+7 月
  python qqq_btc/tools/replay_offline_live_aligned.py

  # 指定月份 / 输出目录
  python qqq_btc/tools/replay_offline_live_aligned.py \\
    --months 2026-06,2026-07 \\
    --out qqq_btc/results/jun_jul2026_offline_live_aligned

  # 只打印版本信息不跑
  python qqq_btc/tools/replay_offline_live_aligned.py --print-version
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qqq_btc.common.rule_profiles import load_rule_profiles
from qqq_btc.common.strategy_profile import (
    DEFAULT_PROFILE,
    StrategyProfile,
    load_strategy_profile,
    materialize_replay_cfg,
    profile_path,
    profile_snapshot,
    resolve_profile_value_path,
)
from qqq_btc.qqq import config as qcfg
from qqq_btc.tools.match_week_regime import load_symbol_1m
from qqq_btc.tools.replay_regime_profiles_apr_jul import (
    DEFAULT_VX_TERM,
    SPOT_ROOT,
    ensure_month,
    run_month,
)

SCRIPT = Path(__file__).resolve()
DEFAULT_MONTHS = "2026-06,2026-07"
DEFAULT_OUT = REPO / "qqq_btc/results/offline_live_aligned"
DEFAULT_PUT_QUARANTINE_LOSS = -0.02
DEFAULT_PUT_QUARANTINE_VX_SLOPE_MIN = 0.06


def _git(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *cmd],
            cwd=REPO,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    meta: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }
    # 小文件可 hash；大 parquet 只记 size+mtime，避免拖慢
    if st.st_size <= 8 * 1024 * 1024:
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        meta["sha256_16"] = h
    return meta


def _live_replay_gates() -> dict[str, Any]:
    cfg = qcfg.LIVE_REPLAY
    keys = [
        "edge_q10_floor",
        "session_entry_start_bar",
        "session_entry_end_bar",
        "min_dual_leg_edge_gap",
        "put_early_cross_confirm_end_bar",
        "put_early_cross_confirm_edge_gap_max",
        "next_day_put_quarantine_loss",
        "next_day_put_quarantine_vix_z_max",
        "next_day_put_quarantine_vx_slope_min",
        "next_day_all_leg_defense_position_frac",
        "next_day_all_leg_defense_account_loss",
        "vixy_open_shock_min_dual_leg_edge_gap",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        if hasattr(cfg, k):
            out[k] = getattr(cfg, k)
    return out


def collect_provenance(
    *,
    months: list[str],
    out_dir: Path,
    selector_source: str,
    put_quarantine_loss: float | None,
    put_quarantine_vx_slope_min: float | None,
    put_quarantine_vix_z_max: float | None,
    vx_term: Path,
    spot_root: Path,
    skip_build: bool,
    strategy_profile: StrategyProfile,
    paths_by_month: dict[str, dict[str, Path]] | None = None,
) -> dict[str, Any]:
    dirty = _git(["status", "--porcelain"])
    tracked = [
        "qqq_btc/tools/replay_offline_live_aligned.py",
        "qqq_btc/tools/replay_regime_profiles_apr_jul.py",
        "qqq_btc/qqq/config.py",
        "qqq_btc/common/entry_decision.py",
        "qqq_btc/common/replay_harness.py",
        "qqq_btc/common/rule_profiles.py",
        "qqq_btc/CONFIG/rule_profiles.json",
    ]
    file_versions = {p: _git(["log", "-1", "--format=%h %ci %s", "--", p]) for p in tracked}

    month_inputs: dict[str, Any] = {}
    if paths_by_month:
        for ym, paths in paths_by_month.items():
            month_inputs[ym] = {
                "infer": _file_meta(paths["infer"]),
                "raw1": _file_meta(paths["raw1"]),
                "exp": str(paths.get("exp", "")),
                "opt1m": str(paths.get("opt1m", "")),
            }

    return {
        "tool": "replay_offline_live_aligned",
        "tool_path": str(SCRIPT.relative_to(REPO)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(REPO),
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "commit_short": _git(["rev-parse", "--short", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit_msg": _git(["log", "-1", "--format=%s"]),
            "commit_time": _git(["log", "-1", "--format=%ci"]),
            "dirty": bool(dirty),
            "dirty_files": dirty.splitlines()[:40] if dirty else [],
        },
        "python": sys.version.split()[0],
        "env_python": os.environ.get("PYTHON") or sys.executable,
        "recipe": {
            "name": strategy_profile.data.get("description")
            or "LIVE_REPLAY + selector + cross-day defenses",
            "strategy_profile_id": strategy_profile.profile_id,
            "strategy_profile_sha256": strategy_profile.sha256,
            "selector_source": selector_source,
            "put_quarantine_loss": put_quarantine_loss,
            "put_quarantine_vx_slope_min": put_quarantine_vx_slope_min,
            "put_quarantine_vix_z_max": put_quarantine_vix_z_max,
            "skip_build": skip_build,
        },
        "strategy_profile": profile_snapshot(strategy_profile),
        "live_replay_gates": strategy_profile.replay_overrides,
        "rule_profiles": str((REPO / "qqq_btc/CONFIG/rule_profiles.json").resolve()),
        "vx_term_structure": _file_meta(vx_term),
        "spot_root": str(spot_root),
        "months": months,
        "out_dir": str(out_dir),
        "source_file_git": file_versions,
        "month_inputs": month_inputs,
    }


def _mdd_from_daily(daily: list[dict[str, Any]]) -> float:
    peak = 1.0
    mdd = 0.0
    for d in daily or []:
        e = 1.0 + float(d["cum_acct25"])
        peak = max(peak, e)
        mdd = min(mdd, e / peak - 1.0)
    return float(mdd)


def run_one_month(
    ym: str,
    *,
    vixy: pd.DataFrame,
    qqq: pd.DataFrame,
    profiles: dict[str, Any],
    calendar: list[date],
    vx: pd.DataFrame | None,
    selector_source: str,
    put_quarantine_loss: float | None,
    put_quarantine_vx_slope_min: float | None,
    put_quarantine_vix_z_max: float | None,
    base_replay_cfg,
    strategy_profile: StrategyProfile,
    skip_build: bool,
) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = ensure_month(ym, skip_build=skip_build)
    infer_by_month = strategy_profile.inputs.get("infer_by_month") or {}
    raw1_by_month = strategy_profile.inputs.get("raw1_by_month") or {}
    opt1m_by_month = strategy_profile.inputs.get("opt1m_by_month") or {}
    if ym in infer_by_month:
        paths["infer"] = resolve_profile_value_path(
            strategy_profile, infer_by_month[ym]
        )
    if ym in raw1_by_month:
        paths["raw1"] = resolve_profile_value_path(
            strategy_profile, raw1_by_month[ym]
        )
    if ym in opt1m_by_month:
        paths["opt1m"] = resolve_profile_value_path(
            strategy_profile, opt1m_by_month[ym]
        )
    if not paths["infer"].exists() or not paths["raw1"].exists():
        raise FileNotFoundError(
            f"profile inputs missing for {ym}: "
            f"infer={paths['infer']} raw1={paths['raw1']}"
        )
    print(f"\n===== {ym} =====", flush=True)
    print(f"infer {paths['infer']}", flush=True)
    print(f"raw1  {paths['raw1']}", flush=True)
    if paths.get("opt1m"):
        print(f"opt1m {paths['opt1m']}", flush=True)
    x = run_month(
        ym,
        paths,
        vixy,
        qqq,
        profiles,
        calendar,
        vx,
        selector_source=selector_source,
        put_quarantine_loss=put_quarantine_loss,
        put_quarantine_vix_z_max=put_quarantine_vix_z_max,
        put_quarantine_vx_slope_min=put_quarantine_vx_slope_min,
        base_replay_cfg=base_replay_cfg,
    )
    r = x["regime_daily_switch"]
    b = x["baseline_TREND_PUT_OK"]
    scope = str(
        (strategy_profile.features or {}).get("scope_label")
        or {
            "2026-04": "April_v4_old_lock",
            "2026-05": "May_v4_old_lock",
            "2026-06": "June_v4_old_lock",
            "2026-07": "July_W1_honest_openwin",
        }.get(ym, ym)
    )
    month_summary = {
        "ym": ym,
        "scope": scope,
        "infer": str(paths["infer"]),
        "raw1": str(paths["raw1"]),
        "n_days": x["n_days"],
        "profile_day_counts": x["profile_day_counts"],
        "regime": {
            "acct25_pct": round(100.0 * float(r["acct25"]), 2),
            "mdd_pct": round(100.0 * _mdd_from_daily(r.get("daily") or []), 2),
            "trades": r["trades"],
            "hit_pct": round(100.0 * float(r["hit"]), 1),
            "legs": r.get("legs"),
            "daily": r.get("daily"),
            "early4_min_cum": r.get("early4_min_cum"),
            "segments": r.get("segments"),
        },
        "baseline_TREND_PUT_OK": {
            "acct25_pct": round(100.0 * float(b["acct25"]), 2),
            "trades": b["trades"],
            "hit_pct": round(100.0 * float(b["hit"]), 1),
            "legs": b.get("legs"),
            "daily": b.get("daily"),
        },
        "delta_regime_vs_baseline_pp": round(float(x["delta_regime_vs_baseline_pp"]), 2),
        "day_profiles": {str(k): v for k, v in x["day_profiles"].items()},
    }
    print(
        f"  regime  acct25={month_summary['regime']['acct25_pct']:+.2f}% "
        f"trades={month_summary['regime']['trades']} "
        f"mdd={month_summary['regime']['mdd_pct']:.2f}% "
        f"profiles={month_summary['profile_day_counts']}",
        flush=True,
    )
    print(
        f"  base    acct25={month_summary['baseline_TREND_PUT_OK']['acct25_pct']:+.2f}% "
        f"trades={month_summary['baseline_TREND_PUT_OK']['trades']} "
        f"delta={month_summary['delta_regime_vs_baseline_pp']:+.2f}pp",
        flush=True,
    )
    traded = {str(d.get("date"))[:10] for d in (r.get("daily") or [])}
    for drow in r.get("daily") or []:
        print(
            f"    {drow['date']} n={drow['n']} "
            f"day={drow['day_acct25']*100:+.2f}% "
            f"cum={drow['cum_acct25']*100:+.2f}% {drow.get('legs')}",
            flush=True,
        )
    # 有 profile 但无成交的日也打印，避免误以为数据缺失（如 07-10）
    for day in sorted(month_summary.get("day_profiles") or {}):
        if day not in traded:
            print(f"    {day} n=0 day=+0.00% (no trades)", flush=True)
    return paths, month_summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Offline LIVE-aligned regime replay with version provenance"
    )
    ap.add_argument("--months", default=DEFAULT_MONTHS, help="comma-separated YYYY-MM")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output dir (default: results/offline_live_aligned/<stamp>)",
    )
    ap.add_argument(
        "--out-name",
        default=None,
        help="fixed subdirectory name under results/offline_live_aligned/",
    )
    ap.add_argument(
        "--strategy-profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="shared offline/stream strategy profile JSON",
    )
    ap.add_argument("--selector-source", choices=("off", "vx", "vixy"), default=None)
    ap.add_argument(
        "--put-quarantine-loss",
        type=float,
        default=None,
    )
    ap.add_argument(
        "--put-quarantine-vx-slope-min",
        type=float,
        default=None,
    )
    ap.add_argument("--put-quarantine-vix-z-max", type=float, default=None)
    ap.add_argument("--vx-term-structure", type=Path, default=None)
    ap.add_argument("--skip-build", action="store_true", default=True)
    ap.add_argument("--build", action="store_true", help="allow ensure_month rebuild")
    ap.add_argument(
        "--lookback-start",
        default="2026-02-01",
        help="spot lookback start for regime features",
    )
    ap.add_argument(
        "--lookback-end",
        default="2026-07-31",
        help="spot lookback end for regime features",
    )
    ap.add_argument(
        "--print-version",
        action="store_true",
        help="print provenance JSON and exit",
    )
    args = ap.parse_args()

    months = [m.strip() for m in args.months.split(",") if m.strip()]
    skip_build = False if args.build else bool(args.skip_build)
    strategy_profile = load_strategy_profile(args.strategy_profile)
    assert strategy_profile is not None
    base_replay_cfg = materialize_replay_cfg(strategy_profile)
    selector_source = (
        args.selector_source
        or str(strategy_profile.selector.get("mode") or "vx")
    )
    put_quarantine_loss = (
        args.put_quarantine_loss
        if args.put_quarantine_loss is not None
        else base_replay_cfg.next_day_put_quarantine_loss
    )
    put_quarantine_vx_slope_min = (
        args.put_quarantine_vx_slope_min
        if args.put_quarantine_vx_slope_min is not None
        else base_replay_cfg.next_day_put_quarantine_vx_slope_min
    )
    put_quarantine_vix_z_max = (
        args.put_quarantine_vix_z_max
        if args.put_quarantine_vix_z_max is not None
        else base_replay_cfg.next_day_put_quarantine_vix_z_max
    )
    vx_term_structure = (
        args.vx_term_structure
        or profile_path(strategy_profile, "selector", "vx_term_structure")
        or DEFAULT_VX_TERM
    )
    spot_root = (
        profile_path(strategy_profile, "selector", "spot_root") or SPOT_ROOT
    )

    if args.out is not None:
        out_dir = args.out if args.out.is_absolute() else (REPO / args.out)
    elif args.out_name:
        out_dir = DEFAULT_OUT / args.out_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        commit = _git(["rev-parse", "--short", "HEAD"]) or "nogit"
        out_dir = DEFAULT_OUT / f"{stamp}_{commit}"

    if args.print_version:
        prov = collect_provenance(
            months=months,
            out_dir=out_dir,
            selector_source=selector_source,
            put_quarantine_loss=put_quarantine_loss,
            put_quarantine_vx_slope_min=put_quarantine_vx_slope_min,
            put_quarantine_vix_z_max=put_quarantine_vix_z_max,
            vx_term=vx_term_structure,
            spot_root=spot_root,
            skip_build=skip_build,
            strategy_profile=strategy_profile,
        )
        print(json.dumps(prov, indent=2, ensure_ascii=False, default=str))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_rule_profiles()
    start = date.fromisoformat(args.lookback_start)
    end = date.fromisoformat(args.lookback_end)
    print(
        f"profile={strategy_profile.profile_id} sha256={strategy_profile.sha256[:16]}",
        flush=True,
    )
    print(f"loading spot lookback {start}..{end} from {spot_root}", flush=True)
    vixy = load_symbol_1m(spot_root, "VIXY", start, end)
    qqq = load_symbol_1m(spot_root, "QQQ", start, end)
    calendar = sorted(
        pd.to_datetime(vixy.timestamp, utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date.unique()
        .tolist()
    )
    vx = (
        pd.read_parquet(vx_term_structure)
        if vx_term_structure.exists()
        else None
    )
    if vx is None and selector_source == "vx":
        raise FileNotFoundError(f"VX term missing: {vx_term_structure}")

    paths_by_month: dict[str, dict[str, Path]] = {}
    months_out: dict[str, Any] = {}
    for ym in months:
        paths, month_summary = run_one_month(
            ym,
            vixy=vixy,
            qqq=qqq,
            profiles=profiles,
            calendar=calendar,
            vx=vx,
            selector_source=selector_source,
            put_quarantine_loss=put_quarantine_loss,
            put_quarantine_vx_slope_min=put_quarantine_vx_slope_min,
            put_quarantine_vix_z_max=put_quarantine_vix_z_max,
            base_replay_cfg=base_replay_cfg,
            strategy_profile=strategy_profile,
            skip_build=skip_build,
        )
        paths_by_month[ym] = paths
        months_out[ym] = month_summary

    provenance = collect_provenance(
        months=months,
        out_dir=out_dir,
        selector_source=selector_source,
        put_quarantine_loss=put_quarantine_loss,
        put_quarantine_vx_slope_min=put_quarantine_vx_slope_min,
        put_quarantine_vix_z_max=put_quarantine_vix_z_max,
        vx_term=vx_term_structure,
        spot_root=spot_root,
        skip_build=skip_build,
        strategy_profile=strategy_profile,
        paths_by_month=paths_by_month,
    )

    summary = {
        "recipe": provenance["recipe"]["name"],
        "gates": {
            **provenance["live_replay_gates"],
            "put_quarantine_loss": put_quarantine_loss,
            "put_quarantine_vx_slope_min": put_quarantine_vx_slope_min,
            "put_quarantine_vix_z_max": put_quarantine_vix_z_max,
            "selector_source": selector_source,
        },
        "months": months_out,
        "headline": {
            ym: {
                "acct25_pct": months_out[ym]["regime"]["acct25_pct"],
                "trades": months_out[ym]["regime"]["trades"],
                "mdd_pct": months_out[ym]["regime"]["mdd_pct"],
                "profiles": months_out[ym]["profile_day_counts"],
                "delta_vs_baseline_pp": months_out[ym]["delta_regime_vs_baseline_pp"],
            }
            for ym in months
        },
        "provenance": provenance,
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    # 稳定别名：方便文档引用「最新一次」
    latest = DEFAULT_OUT / "LATEST"
    latest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.resolve())
    except OSError:
        (DEFAULT_OUT / "LATEST_PATH.txt").write_text(str(out_dir.resolve()) + "\n")

    print("\n=== headline ===", flush=True)
    print(json.dumps(summary["headline"], indent=2, ensure_ascii=False), flush=True)
    print(f"\nwrote {out_dir / 'summary.json'}", flush=True)
    print(f"wrote {out_dir / 'manifest.json'}", flush=True)
    print(
        f"git={provenance['git'].get('commit_short')} dirty={provenance['git'].get('dirty')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
