#!/usr/bin/env python3
"""Parity Board：扫描 strategy profile、对拍结果与 catalog 配方。"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO / "qqq_btc" / "results"
PROFILES_DIR = REPO / "qqq_btc" / "CONFIG" / "strategy_profiles"
CATALOG_PATH = REPO / "qqq_btc" / "CONFIG" / "parity_board_catalog.json"
OFFLINE_ROOT = RESULTS_ROOT / "offline_live_aligned"

SUMMARY_NAMES = ("stream_summary_paired.json", "summary.txt", "gates_status.json")
MANIFEST_NAME = "manifest.json"
RESOLVED_NAME = "strategy_profile.resolved.json"


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def resolve_repo_path(raw: str | Path | None) -> Optional[Path]:
    if raw in (None, ""):
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = REPO / path
    return path.resolve()


@dataclass(frozen=True)
class ParityRecipe:
    recipe_id: str
    title: str
    priority: int
    strategy_profile: Optional[Path]
    stream_script: Optional[Path]
    offline_cmd: str
    stream_cmd: str
    offline_result: Optional[Path]
    result_name_prefixes: tuple[str, ...]
    baseline_acct25_pct: Optional[float]
    baseline_trades: Optional[int]
    docs: tuple[str, ...]
    notes: str


@dataclass
class StreamRun:
    path: Path
    mtime: float
    name: str
    manifest: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    resolved: dict[str, Any] = field(default_factory=dict)
    recipe_id: str = ""
    artifacts: list[str] = field(default_factory=list)

    @property
    def profile_id(self) -> str:
        return str(
            self.manifest.get("strategy_profile_id")
            or self.resolved.get("profile_id")
            or ""
        )

    @property
    def profile_sha(self) -> str:
        return str(
            self.manifest.get("strategy_profile_sha256")
            or self.resolved.get("profile_sha256")
            or ""
        )

    @property
    def parity_status(self) -> str:
        return str(self.summary.get("parity_status") or "UNKNOWN")

    @property
    def acct25(self) -> Optional[float]:
        val = self.summary.get("acct25")
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None

    @property
    def trades(self) -> Optional[int]:
        val = self.summary.get("trades")
        if val is None:
            return None
        try:
            return int(val)
        except Exception:
            return None


def load_catalog(path: Path | None = None) -> list[ParityRecipe]:
    catalog = _read_json(path or CATALOG_PATH) or {}
    recipes: list[ParityRecipe] = []
    for raw in catalog.get("recipes") or []:
        if not isinstance(raw, dict):
            continue
        recipes.append(
            ParityRecipe(
                recipe_id=str(raw.get("recipe_id") or ""),
                title=str(raw.get("title") or raw.get("recipe_id") or ""),
                priority=int(raw["priority"]) if raw.get("priority") is not None else 99,
                strategy_profile=resolve_repo_path(raw.get("strategy_profile")),
                stream_script=resolve_repo_path(raw.get("stream_script")),
                offline_cmd=str(raw.get("offline_cmd") or ""),
                stream_cmd=str(raw.get("stream_cmd") or ""),
                offline_result=resolve_repo_path(raw.get("offline_result")),
                result_name_prefixes=tuple(
                    str(x) for x in (raw.get("result_name_prefixes") or []) if x
                ),
                baseline_acct25_pct=(
                    float(raw["baseline_acct25_pct"])
                    if raw.get("baseline_acct25_pct") is not None
                    else None
                ),
                baseline_trades=(
                    int(raw["baseline_trades"])
                    if raw.get("baseline_trades") is not None
                    else None
                ),
                docs=tuple(str(x) for x in (raw.get("docs") or []) if x),
                notes=str(raw.get("notes") or ""),
            )
        )
    recipes.sort(key=lambda r: (r.priority, r.recipe_id))
    return recipes


def list_strategy_profiles() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not PROFILES_DIR.is_dir():
        return rows
    for path in sorted(PROFILES_DIR.glob("*.json")):
        data = _read_json(path) or {}
        payload = path.read_bytes()
        rows.append(
            {
                "profile_id": str(data.get("profile_id") or path.stem),
                "path": _rel(path),
                "abs_path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "description": str(data.get("description") or ""),
                "tick_exits": str((data.get("execution") or {}).get("tick_exits") or ""),
                "selector": str((data.get("selector") or {}).get("mode") or ""),
                "put_gate": str((data.get("execution") or {}).get("put_gate_mode") or ""),
                "mtime": path.stat().st_mtime,
            }
        )
    return rows


def _load_stream_run(path: Path) -> Optional[StreamRun]:
    if not path.is_dir():
        return None
    summary = _read_json(path / "stream_summary_paired.json") or {}
    manifest = _read_json(path / MANIFEST_NAME) or {}
    if not summary and not manifest:
        return None
    # Prefer honest / strategy-profile runs; still allow older stream summaries.
    if not summary and "strategy_profile_id" not in manifest and "mode" not in manifest:
        return None
    gates = _read_json(path / "gates_status.json") or {}
    if not gates and isinstance(summary.get("gates"), dict):
        gates = dict(summary["gates"])
    resolved = _read_json(path / RESOLVED_NAME) or {}
    artifacts = sorted(
        p.name
        for p in path.iterdir()
        if p.is_file()
        and (
            p.suffix in {".json", ".txt", ".csv", ".log", ".pkl"}
            or p.name.startswith("fill_audit_")
            or p.name.startswith("stream_")
            or p.name.startswith("signals_")
            or p.name.startswith("feat_parity_")
        )
    )
    return StreamRun(
        path=path,
        mtime=path.stat().st_mtime,
        name=path.name,
        manifest=manifest,
        summary=summary,
        gates=gates,
        resolved=resolved,
        artifacts=artifacts,
    )


def discover_stream_runs(*, limit: int = 80) -> list[StreamRun]:
    if not RESULTS_ROOT.is_dir():
        return []
    candidates: list[Path] = []
    for child in RESULTS_ROOT.iterdir():
        if not child.is_dir():
            continue
        if (child / "stream_summary_paired.json").is_file() or (
            child / MANIFEST_NAME
        ).is_file():
            candidates.append(child)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    runs: list[StreamRun] = []
    for path in candidates:
        run = _load_stream_run(path)
        if run is None:
            continue
        runs.append(run)
        if len(runs) >= limit:
            break
    return runs


def match_recipe(run: StreamRun, recipes: list[ParityRecipe]) -> Optional[ParityRecipe]:
    # Prefer exact profile_id so shared result prefixes don't steal production runs.
    if run.profile_id:
        for recipe in recipes:
            if not recipe.strategy_profile:
                continue
            data = _read_json(recipe.strategy_profile) or {}
            ids = {
                recipe.strategy_profile.stem,
                str(data.get("profile_id") or ""),
            }
            if run.profile_id in ids:
                return recipe
    for recipe in recipes:
        for prefix in recipe.result_name_prefixes:
            if run.name == prefix or run.name.startswith(prefix):
                return recipe
    return None


def annotate_runs(
    runs: list[StreamRun], recipes: list[ParityRecipe]
) -> list[StreamRun]:
    for run in runs:
        recipe = match_recipe(run, recipes)
        run.recipe_id = recipe.recipe_id if recipe else ""
    return runs


def offline_headline(offline_dir: Path | None) -> dict[str, Any]:
    if offline_dir is None or not offline_dir.is_dir():
        return {}
    summary = _read_json(offline_dir / "summary.json") or {}
    headline = summary.get("headline")
    if isinstance(headline, dict) and headline:
        # Prefer July W1 if present.
        if "2026-07" in headline and isinstance(headline["2026-07"], dict):
            row = dict(headline["2026-07"])
            row["month"] = "2026-07"
            row["source"] = _rel(offline_dir / "summary.json")
            return row
        first_key = next(iter(headline))
        if isinstance(headline[first_key], dict):
            row = dict(headline[first_key])
            row["month"] = first_key
            row["source"] = _rel(offline_dir / "summary.json")
            return row
    months = summary.get("months")
    if isinstance(months, dict):
        for ym, payload in months.items():
            regime = (payload or {}).get("regime") if isinstance(payload, dict) else None
            if isinstance(regime, dict) and "acct25_pct" in regime:
                return {
                    "month": ym,
                    "acct25_pct": regime.get("acct25_pct"),
                    "trades": regime.get("trades"),
                    "mdd_pct": regime.get("mdd_pct"),
                    "source": _rel(offline_dir / "summary.json"),
                }
    manifest = _read_json(offline_dir / "manifest.json") or {}
    return {
        "profile_id": manifest.get("strategy_profile_id"),
        "profile_sha256": manifest.get("strategy_profile_sha256"),
        "source": _rel(offline_dir),
    }


def gate_pass_labels(gates: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    g1 = gates.get("gate1_raw")
    g2 = gates.get("gate2_norm")
    if isinstance(g1, dict):
        out["gate1"] = "PASS" if g1.get("pass") else "FAIL"
    elif g1 is True:
        out["gate1"] = "PASS"
    elif g1 is False:
        out["gate1"] = "FAIL"
    if isinstance(g2, dict):
        out["gate2"] = "PASS" if g2.get("pass") else "FAIL"
    elif g2 is True:
        out["gate2"] = "PASS"
    elif g2 is False:
        out["gate2"] = "FAIL"
    g3 = gates.get("gate3_trade_allowed")
    if g3 is True:
        out["gate3"] = "PASS"
    elif g3 is False:
        out["gate3"] = "FAIL"
    return out


def trades_frame_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary.get("trades_detail") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "date": item.get("date"),
                "leg": item.get("leg"),
                "entry_sb": item.get("entry_sb"),
                "exit_sb": item.get("exit_sb"),
                "reason": str(item.get("reason") or "").replace("QQQ_BTC_", ""),
                "net_return": item.get("net_return"),
            }
        )
    return rows


def daily_acct_rows(summary: dict[str, Any], position_frac: float = 0.25) -> list[dict[str, Any]]:
    by_day: dict[str, list[float]] = defaultdict(list)
    for item in summary.get("trades_detail") or []:
        if not isinstance(item, dict):
            continue
        day = str(item.get("date") or "")[:10]
        try:
            net = float(item.get("net_return"))
        except Exception:
            continue
        by_day[day].append(net)
    rows: list[dict[str, Any]] = []
    cum = 1.0
    for day in sorted(by_day):
        eq = 1.0
        for net in by_day[day]:
            eq *= 1.0 + position_frac * net
        cum *= eq
        rows.append(
            {
                "date": day,
                "n": len(by_day[day]),
                "day_acct25_pct": (eq - 1.0) * 100.0,
                "cum_acct25_pct": (cum - 1.0) * 100.0,
            }
        )
    return rows


def tail_text(path: Path, *, max_bytes: int = 24_000) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            data = fh.read()
        else:
            data = fh.read()
    text = data.decode("utf-8", errors="replace")
    if size > max_bytes:
        return f"... truncated ({size} bytes) ...\n" + text
    return text


def recipe_card_state(
    recipe: ParityRecipe, runs: list[StreamRun]
) -> dict[str, Any]:
    matched = [r for r in runs if r.recipe_id == recipe.recipe_id]
    latest = matched[0] if matched else None
    profile = _read_json(recipe.strategy_profile) if recipe.strategy_profile else {}
    offline = offline_headline(recipe.offline_result)
    baseline_acct = recipe.baseline_acct25_pct
    if baseline_acct is None and offline.get("acct25_pct") is not None:
        try:
            baseline_acct = float(offline["acct25_pct"])
        except Exception:
            baseline_acct = None
    delta_pp = None
    if latest and latest.acct25 is not None and baseline_acct is not None:
        delta_pp = latest.acct25 * 100.0 - baseline_acct
    return {
        "recipe_id": recipe.recipe_id,
        "title": recipe.title,
        "profile_id": str(profile.get("profile_id") or ""),
        "profile_path": _rel(recipe.strategy_profile) if recipe.strategy_profile else "",
        "stream_script": _rel(recipe.stream_script) if recipe.stream_script else "",
        "latest_run": latest.name if latest else "",
        "latest_path": _rel(latest.path) if latest else "",
        "parity_status": latest.parity_status if latest else "NO_RUN",
        "acct25_pct": (latest.acct25 * 100.0) if latest and latest.acct25 is not None else None,
        "trades": latest.trades if latest else None,
        "baseline_acct25_pct": baseline_acct,
        "baseline_trades": recipe.baseline_trades or offline.get("trades"),
        "delta_pp": delta_pp,
        "notes": recipe.notes,
        "n_runs": len(matched),
    }
