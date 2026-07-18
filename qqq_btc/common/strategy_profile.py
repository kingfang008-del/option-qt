#!/usr/bin/env python3
"""离线 replay 与流式交易共用的策略配方解析器。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

from qqq_btc.common.replay_types import ReplayConfig
from qqq_btc.qqq import config as qcfg

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = (
    REPO
    / "qqq_btc/CONFIG/strategy_profiles/v4_vx_live_aligned_v1.json"
)
PROFILE_ENV = "QQQ_BTC_STRATEGY_PROFILE"


@dataclass(frozen=True)
class StrategyProfile:
    path: Path
    data: dict[str, Any]
    sha256: str

    @property
    def profile_id(self) -> str:
        return str(self.data["profile_id"])

    @property
    def replay_overrides(self) -> dict[str, Any]:
        return dict(self.data.get("replay_overrides") or {})

    @property
    def selector(self) -> dict[str, Any]:
        return dict(self.data.get("selector") or {})

    @property
    def execution(self) -> dict[str, Any]:
        return dict(self.data.get("execution") or {})

    @property
    def features(self) -> dict[str, Any]:
        return dict(self.data.get("features") or {})

    @property
    def model(self) -> dict[str, Any]:
        return dict(self.data.get("model") or {})

    @property
    def inputs(self) -> dict[str, Any]:
        return dict(self.data.get("inputs") or {})


def _resolve_path(raw: str | Path, *, profile_dir: Optional[Path] = None) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    repo_path = (REPO / path).resolve()
    if repo_path.exists() or profile_dir is None:
        return repo_path
    return (profile_dir / path).resolve()


def resolve_profile_path(path: str | Path | None = None) -> Path:
    raw = path or os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE
    return _resolve_path(raw)


@lru_cache(maxsize=16)
def _load_strategy_profile_cached(resolved_str: str) -> StrategyProfile:
    resolved = Path(resolved_str)
    payload = resolved.read_bytes()
    data = json.loads(payload)
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported strategy profile schema: {resolved}")
    if not str(data.get("profile_id") or "").strip():
        raise ValueError(f"strategy profile missing profile_id: {resolved}")
    overrides = dict(data.get("replay_overrides") or {})
    allowed = {f.name for f in fields(ReplayConfig)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(
            f"unknown ReplayConfig fields in {resolved}: {', '.join(unknown)}"
        )
    return StrategyProfile(
        path=resolved,
        data=data,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def load_strategy_profile(
    path: str | Path | None = None,
    *,
    required: bool = True,
) -> Optional[StrategyProfile]:
    resolved = resolve_profile_path(path)
    if not resolved.exists():
        if required:
            raise FileNotFoundError(f"strategy profile not found: {resolved}")
        return None
    return _load_strategy_profile_cached(str(resolved))


def load_active_strategy_profile() -> Optional[StrategyProfile]:
    """仅在显式设置 env 时加载；保持旧调用方兼容。"""
    raw = os.environ.get(PROFILE_ENV, "").strip()
    return load_strategy_profile(raw) if raw else None


def materialize_replay_cfg(
    profile: StrategyProfile,
    base_cfg: Optional[ReplayConfig] = None,
) -> ReplayConfig:
    if base_cfg is None:
        base_name = str(profile.data.get("base_replay") or "LIVE_REPLAY")
        if base_name not in ("REPLAY", "LIVE_REPLAY"):
            raise ValueError(f"unsupported base_replay={base_name!r}")
        base_cfg = getattr(qcfg, base_name)
    return replace(base_cfg, **profile.replay_overrides)


def profile_path(
    profile: StrategyProfile,
    section: str,
    key: str,
) -> Optional[Path]:
    value = (profile.data.get(section) or {}).get(key)
    if value in (None, ""):
        return None
    return _resolve_path(value, profile_dir=profile.path.parent)


def resolve_profile_value_path(
    profile: StrategyProfile,
    value: str | Path,
) -> Path:
    return _resolve_path(value, profile_dir=profile.path.parent)


def profile_snapshot(profile: StrategyProfile) -> dict[str, Any]:
    cfg = materialize_replay_cfg(profile)
    return {
        "profile_id": profile.profile_id,
        "profile_path": str(profile.path),
        "profile_sha256": profile.sha256,
        "schema_version": profile.data["schema_version"],
        "base_replay": profile.data.get("base_replay"),
        "resolved_replay_config": asdict(cfg),
        "selector": profile.selector,
        "execution": profile.execution,
        "features": profile.features,
        "model": profile.model,
        "inputs": profile.inputs,
    }


def shell_environment(profile: StrategyProfile) -> dict[str, str]:
    """返回非 ReplayConfig 的流式环境；ReplayConfig 本身由 governor 直接物化。"""
    selector = profile.selector
    execution = profile.execution
    features = profile.features
    model = profile.model
    env: dict[str, str] = {
        PROFILE_ENV: str(profile.path),
        "QQQ_BTC_USE_LIVE_REPLAY": (
            "1" if profile.data.get("base_replay") == "LIVE_REPLAY" else "0"
        ),
    }
    mapping: tuple[tuple[Mapping[str, Any], str, str], ...] = (
        (selector, "mode", "QQQ_BTC_RULE_PROFILE_SELECTOR"),
        (selector, "vx_term_structure", "QQQ_BTC_VX_TERM_STRUCTURE"),
        (selector, "spot_root", "QQQ_BTC_SPOT_ROOT"),
        (execution, "put_gate_mode", "QQQ_BTC_PUT_GATE_MODE"),
        (execution, "tick_exits", "QQQ_BTC_TICK_EXITS"),
        (execution, "live_label_shift_sec", "QQQ_BTC_LIVE_LABEL_SHIFT_SEC"),
        (execution, "fill_spread_frac", "BACKTEST_OPT_FILL_SPREAD_FRAC"),
        (execution, "execution_delay_bars", "EXECUTION_DELAY_BARS"),
        (execution, "oms_signal_delay_bars", "OMS_SIGNAL_DELAY_BARS"),
        (features, "slow_feature_config", "SLOW_FEATURE_CONFIG"),
        (features, "frozen_norm", "FCS_FROZEN_NORM_PATH"),
        (features, "honest_feature_root", "HONEST_FEAT_ROOT"),
        (model, "checkpoint", "CKPT"),
    )
    path_keys = {
        "vx_term_structure",
        "spot_root",
        "slow_feature_config",
        "frozen_norm",
        "honest_feature_root",
        "checkpoint",
    }
    for section, key, env_key in mapping:
        value = section.get(key)
        if value in (None, ""):
            continue
        if key in path_keys:
            value = _resolve_path(value, profile_dir=profile.path.parent)
        env[env_key] = str(value)
    return env


def _main() -> None:
    ap = argparse.ArgumentParser(description="Inspect/export a QQQ-BTC strategy profile")
    ap.add_argument("--profile", type=Path, default=None)
    ap.add_argument("--shell-env", action="store_true")
    ap.add_argument("--snapshot", action="store_true")
    args = ap.parse_args()
    profile = load_strategy_profile(args.profile)
    assert profile is not None
    if args.shell_env:
        for key, value in shell_environment(profile).items():
            quoted = shlex.quote(value)
            if key == PROFILE_ENV:
                print(f"export {key}={quoted}")
            else:
                # 调用方显式 env 是 emergency override；profile 只填未设置项。
                print(
                    f"if [[ -z ${{{key}+x}} ]]; then "
                    f"export {key}={quoted}; fi"
                )
        return
    print(
        json.dumps(
            profile_snapshot(profile),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    _main()
