"""Load maga7 strategy profile JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = _PKG / "CONFIG" / "mf10_top2_v1.json"


def expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).expanduser()


def load_profile(path: str | Path | None = None) -> dict[str, Any]:
    p = expand(path) if path else DEFAULT_PROFILE
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["_profile_path"] = str(p)
    paths = cfg.get("paths") or {}
    cfg["_paths"] = {k: expand(v) if isinstance(v, str) and ("/" in v or v.startswith("~")) else v for k, v in paths.items()}
    # results_dir relative to repo
    rd = cfg["_paths"].get("results_dir", "maga7/results")
    if not Path(str(rd)).is_absolute():
        cfg["_paths"]["results_dir"] = (_PKG.parent / str(rd)).resolve()
    return cfg
