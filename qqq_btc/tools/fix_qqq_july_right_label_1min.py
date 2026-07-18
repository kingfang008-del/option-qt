#!/usr/bin/env python3
"""兼容入口：转发到 qqq_btc.common.bar_label_convention（W1 右标签）。"""
from __future__ import annotations

import sys

from qqq_btc.common.bar_label_convention import main

if __name__ == "__main__":
    # 默认修 QQQ 2026-07（旧脚本行为）；可继续传 --scan/--fix/--start/--end
    argv = list(sys.argv[1:])
    if not any(a in {"--scan", "--fix"} for a in argv):
        argv = ["--fix", "--symbols", "QQQ", "--start", "2026-07-01", "--end", "2026-07-31", *argv]
    raise SystemExit(main(argv))
