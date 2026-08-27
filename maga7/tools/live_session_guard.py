#!/usr/bin/env python3
"""Independent Mag7 process/manifest guard with durable and optional webhook alerts."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _manifest(live_root: Path, session_id: str) -> Path | None:
    matches = list(live_root.glob(f"*/{session_id}/manifest.json"))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _read(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _alert(log_dir: Path, session_id: str, payload: dict[str, Any]) -> None:
    payload = {"session_id": session_id, "alerted_at": time.time(), **payload}
    alert_dir = log_dir / "alerts"
    alert_dir.mkdir(parents=True, exist_ok=True)
    path = alert_dir / f"{session_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    webhook = str(os.environ.get("MAG7_ALERT_WEBHOOK") or "").strip()
    if webhook:
        request = urllib.request.Request(
            webhook,
            data=json.dumps(payload, ensure_ascii=True).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5):
                pass
        except Exception as exc:
            payload["webhook_error"] = str(exc)
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


def run(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    live_root = Path(args.live_root)
    stop_marker = log_dir / "live_session.stop_requested"
    stale_alerted = False
    while _alive(args.pid):
        document = _read(_manifest(live_root, args.session_id))
        updated = float(document.get("updated_at") or 0.0)
        state = str(document.get("state") or "")
        if state == "RUNNING" and updated and time.time() - updated > args.stale_sec:
            if not stale_alerted:
                _alert(
                    log_dir,
                    args.session_id,
                    {
                        "kind": "MANIFEST_STALE",
                        "pid": args.pid,
                        "age_sec": time.time() - updated,
                        "positions": ((document.get("oms") or {}).get("positions")),
                    },
                )
                stale_alerted = True
        else:
            stale_alerted = False
        time.sleep(args.interval_sec)

    time.sleep(2.0)
    if stop_marker.exists():
        return
    document = _read(_manifest(live_root, args.session_id))
    state = str(document.get("state") or "")
    if state not in {"DONE", "STOPPED", "PREPARED"}:
        _alert(
            log_dir,
            args.session_id,
            {
                "kind": "PROCESS_EXITED",
                "pid": args.pid,
                "manifest_state": state or "missing",
                "manifest_error": document.get("error"),
                "positions": ((document.get("oms") or {}).get("positions")),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--live-root", required=True)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--stale-sec", type=float, default=20.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
