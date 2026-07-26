"""Narrow-expert registry under Watchdog / Router (not a second baseline).

Spine stays research_baseline Rule-A. This module only loads the catalog used to
track which risk/entry morphs are ACCEPT / REJECT / QUOTE_REJECT, so promotions
stay auditable and HF sleeves do not silently become default arms.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = "maga7/CONFIG/narrow_experts/catalog_v1.json"

# Entry morphs may only be wired into live/scanner when status is one of these.
PROMOTE_ENTRY_STATUSES = frozenset({"ACCEPT_RESEARCH", "ACCEPT_LIVE"})
# Risk overlays already on the spine (degrade/halt) use ACCEPT_RESEARCH.
ACTIVE_STATUSES = frozenset({"ACCEPT_RESEARCH", "ACCEPT_LIVE"})


@dataclass(frozen=True)
class NarrowExpert:
    expert_id: str
    kind: str
    lane: str
    status: str
    enabled_on_spine: bool
    description: str
    raw: dict[str, Any]

    @property
    def may_promote_entry(self) -> bool:
        return self.kind == "entry" and self.status in PROMOTE_ENTRY_STATUSES

    @property
    def is_active_on_spine(self) -> bool:
        return bool(self.enabled_on_spine) and self.status in ACTIVE_STATUSES


@dataclass(frozen=True)
class NarrowExpertCatalog:
    version: str
    updated: str
    spine: str
    discipline: tuple[str, ...]
    experts: dict[str, NarrowExpert]
    path: Path

    def get(self, expert_id: str) -> NarrowExpert | None:
        return self.experts.get(str(expert_id))

    def by_status(self, status: str) -> list[NarrowExpert]:
        want = str(status)
        return [e for e in self.experts.values() if e.status == want]

    def entry_candidates(self) -> list[NarrowExpert]:
        """Entry morphs not yet on spine (research queue)."""
        return [
            e
            for e in self.experts.values()
            if e.kind == "entry" and not e.enabled_on_spine
        ]

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for e in self.experts.values():
            counts[e.status] = int(counts.get(e.status, 0)) + 1
        on_spine = [e.expert_id for e in self.experts.values() if e.is_active_on_spine]
        return {
            "version": self.version,
            "updated": self.updated,
            "spine": self.spine,
            "n_experts": len(self.experts),
            "status_counts": counts,
            "active_on_spine": on_spine,
            "entry_queue": [e.expert_id for e in self.entry_candidates()],
        }


def _as_expert(expert_id: str, raw: dict[str, Any]) -> NarrowExpert:
    return NarrowExpert(
        expert_id=str(expert_id),
        kind=str(raw.get("kind") or "unknown"),
        lane=str(raw.get("lane") or ""),
        status=str(raw.get("status") or "RESEARCH_ONLY"),
        enabled_on_spine=bool(raw.get("enabled_on_spine", False)),
        description=str(raw.get("description") or ""),
        raw=dict(raw),
    )


def load_narrow_expert_catalog(
    path: str | Path | None = None,
) -> NarrowExpertCatalog:
    """Load catalog JSON. ``path`` may be repo-relative or absolute."""
    p = Path(path) if path else Path(DEFAULT_CATALOG)
    if not p.is_absolute():
        p = (_REPO / p).resolve()
    else:
        p = p.resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    experts_raw = data.get("experts") if isinstance(data.get("experts"), dict) else {}
    experts = {
        str(k): _as_expert(str(k), v if isinstance(v, dict) else {})
        for k, v in experts_raw.items()
    }
    disc = data.get("discipline") if isinstance(data.get("discipline"), list) else []
    return NarrowExpertCatalog(
        version=str(data.get("version") or ""),
        updated=str(data.get("updated") or ""),
        spine=str(data.get("spine") or ""),
        discipline=tuple(str(x) for x in disc),
        experts=experts,
        path=p,
    )


def catalog_from_profile(profile: dict[str, Any] | None) -> NarrowExpertCatalog | None:
    """Read ``profile.narrow_experts.catalog_path`` when the block exists."""
    if not isinstance(profile, dict):
        return None
    block = profile.get("narrow_experts")
    if not isinstance(block, dict):
        return None
    path = block.get("catalog_path") or DEFAULT_CATALOG
    return load_narrow_expert_catalog(path)


def assert_entry_promotable(expert_id: str, catalog: NarrowExpertCatalog) -> None:
    """Raise if an entry morph is not cleared for wiring into live/scanner."""
    ex = catalog.get(expert_id)
    if ex is None:
        raise ValueError(f"unknown narrow expert: {expert_id}")
    if not ex.may_promote_entry:
        raise ValueError(
            f"narrow expert {expert_id!r} status={ex.status!r} "
            f"is not promotable as an entry arm (need {sorted(PROMOTE_ENTRY_STATUSES)})"
        )
