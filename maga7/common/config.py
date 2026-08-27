"""Load maga7 strategy profile JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parents[1]
_REPO = _PKG.parent
DEFAULT_PROFILE = _PKG / "CONFIG" / "mf10_top2_v1.json"

# Keep heavy / live artifacts off the git tree (same disk family as stock_1s).
DEFAULT_RESULTS_DIR = Path("/mnt/s990/data/maga7/results")
DEFAULT_LIVE_SESSIONS_DIR = Path("/mnt/s990/data/maga7/live_sessions")


def expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).expanduser()


def _as_abs_path(value: Any, *, relative_to: Path = _REPO) -> Path:
    path = expand(value)
    if not path.is_absolute():
        path = (relative_to / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_results_dir(paths: dict[str, Any] | None = None) -> Path:
    """Offline / research results root.

    Priority: ``MAG7_RESULTS_DIR`` → profile ``paths.results_dir`` → s990 default.
    """
    env = os.environ.get("MAG7_RESULTS_DIR", "").strip()
    if env:
        return _as_abs_path(env)
    paths = paths or {}
    if paths.get("results_dir") is not None:
        return _as_abs_path(paths["results_dir"])
    return DEFAULT_RESULTS_DIR.resolve()


def resolve_live_sessions_dir(paths: dict[str, Any] | None = None) -> Path:
    """Live session artifacts (tape / locks / oms / scanner).

    Priority: ``MAG7_LIVE_SESSIONS_DIR`` → profile ``paths.live_sessions_dir``
    → s990 default (not nested under the git repo).
    """
    env = os.environ.get("MAG7_LIVE_SESSIONS_DIR", "").strip()
    if env:
        return _as_abs_path(env)
    paths = paths or {}
    if paths.get("live_sessions_dir") is not None:
        return _as_abs_path(paths["live_sessions_dir"])
    return DEFAULT_LIVE_SESSIONS_DIR.resolve()


def load_profile(path: str | Path | None = None) -> dict[str, Any]:
    p = expand(path) if path else DEFAULT_PROFILE
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["_profile_path"] = str(p)
    raw_paths = dict(cfg.get("paths") or {})
    resolved: dict[str, Any] = {}
    for key, value in raw_paths.items():
        if isinstance(value, str) and (value.startswith("~") or "/" in value):
            resolved[key] = expand(value)
        else:
            resolved[key] = value

    resolved["results_dir"] = resolve_results_dir(resolved)
    resolved["live_sessions_dir"] = resolve_live_sessions_dir(resolved)
    cfg["_paths"] = resolved
    return cfg
