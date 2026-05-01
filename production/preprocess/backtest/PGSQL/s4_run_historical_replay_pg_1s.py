#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""兼容入口：等价于 s4_run_historical_replay_s2_1s.py --use-pgsql [参数...]"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    s2 = Path(__file__).resolve().parent.parent / "second" / "s4_run_historical_replay_s2_1s.py"
    if not s2.is_file():
        sys.stderr.write(f"missing: {s2}\n")
        raise SystemExit(2)
    raise SystemExit(subprocess.call([sys.executable, str(s2), "--use-pgsql", *sys.argv[1:]]))
