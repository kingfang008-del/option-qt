#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launch the qqq_btc Streamlit dashboard."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    dash = repo / "qqq_btc" / "dashboard" / "qqq_btc_dash.py"
    port = os.environ.get("QQQ_BTC_DASH_PORT", "8502")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dash),
        "--server.port",
        str(port),
        "--server.address",
        os.environ.get("QQQ_BTC_DASH_HOST", "0.0.0.0"),
    ]
    return subprocess.call(cmd, cwd=str(repo))


if __name__ == "__main__":
    raise SystemExit(main())
