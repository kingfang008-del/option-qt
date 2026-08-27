"""Gate-fail degrade state for TFT deploy bundle (outer control, not model)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

Mode = Literal["NORMAL", "NO_NEW", "HALT"]
DEFAULT_STATE = Path(__file__).resolve().parents[1] / "results/deploy_runtime/degrade_state.json"


@dataclass
class DegradeState:
    mode: Mode = "NORMAL"
    reason: str = ""
    since_utc: str | None = None
    until_utc: str | None = None
    last_gate: dict[str, Any] | None = None

    def allows_new_entries(self) -> bool:
        self._expire_if_needed()
        return self.mode == "NORMAL"

    def _expire_if_needed(self) -> None:
        if not self.until_utc:
            return
        until = datetime.fromisoformat(self.until_utc)
        if datetime.now(timezone.utc) >= until:
            self.mode = "NORMAL"
            self.reason = "ttl_expired"
            self.until_utc = None


def load_state(path: Path | str | None = None) -> DegradeState:
    p = Path(path or DEFAULT_STATE).expanduser()
    if not p.is_file():
        return DegradeState()
    raw = json.loads(p.read_text(encoding="utf-8"))
    st = DegradeState(
        mode=str(raw.get("mode") or "NORMAL"),  # type: ignore[arg-type]
        reason=str(raw.get("reason") or ""),
        since_utc=raw.get("since_utc"),
        until_utc=raw.get("until_utc"),
        last_gate=raw.get("last_gate"),
    )
    if st.mode not in {"NORMAL", "NO_NEW", "HALT"}:
        st.mode = "HALT"
    st._expire_if_needed()
    return st


def save_state(state: DegradeState, path: Path | str | None = None) -> Path:
    p = Path(path or DEFAULT_STATE).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "mode": state.mode,
        "reason": state.reason,
        "since_utc": state.since_utc,
        "until_utc": state.until_utc,
        "last_gate": state.last_gate,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return p


def apply_gate_result(
    *,
    gate1_pass: bool | None = None,
    gate2_pass: bool | None = None,
    gate3_pass: bool | None = None,
    unexplained_day_loss: bool = False,
    manual_kill: bool = False,
    path: Path | str | None = None,
) -> DegradeState:
    """Update degrade mode from Gate results (bundle rules)."""
    now = datetime.now(timezone.utc)
    st = load_state(path)
    st.last_gate = {
        "gate1_pass": gate1_pass,
        "gate2_pass": gate2_pass,
        "gate3_pass": gate3_pass,
        "unexplained_day_loss": unexplained_day_loss,
        "manual_kill": manual_kill,
        "checked_utc": now.isoformat(),
    }
    if manual_kill:
        st.mode, st.reason = "HALT", "manual_kill"
        st.since_utc, st.until_utc = now.isoformat(), None
    elif gate3_pass is False or unexplained_day_loss:
        st.mode, st.reason = "HALT", "gate3_fail_or_unexplained_day_loss"
        st.since_utc = now.isoformat()
        st.until_utc = (now + timedelta(hours=48)).isoformat()
    elif gate1_pass is False or gate2_pass is False:
        which = "gate1_fail" if gate1_pass is False else "gate2_fail"
        st.mode, st.reason = "NO_NEW", which
        st.since_utc = now.isoformat()
        st.until_utc = (now + timedelta(hours=24)).isoformat()
    else:
        # all provided gates passed (or None=not checked)
        if gate1_pass is True and gate2_pass is True and (gate3_pass in (True, None)):
            st.mode, st.reason = "NORMAL", "gates_ok"
            st.since_utc, st.until_utc = now.isoformat(), None
    save_state(st, path)
    return st
