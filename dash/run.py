#!/usr/bin/env python3
"""Launch the repository-wide Streamlit control plane."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    app = Path(__file__).resolve().with_name("app.py")
    host = os.environ.get("OPTION_QT_DASH_HOST", "127.0.0.1")
    port = os.environ.get("OPTION_QT_DASH_PORT", "8501")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        host,
        "--server.port",
        port,
    ]
    return subprocess.call(cmd, cwd=repo)


if __name__ == "__main__":
    raise SystemExit(main())
