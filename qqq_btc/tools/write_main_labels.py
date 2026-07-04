#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对已切分/归一化的特征目录写入主标签(process_labels_file)。

主标签列:
  label_direction / label_event / label_volatility / label_return_fwd
  label_return_fwd_gross / label_return_fwd_net / label_direction_net
  label_execution_cost

只用 OHLC 算三重障碍与净收益，可在 rolling_norm 之后对 train/val/test 直接写。
仅处理 **/1min/*.parquet（LMDB 只读 1min 标签）。

用法:
  python qqq_btc/tools/write_main_labels.py \\
      --feature-roots ~/train_data/quote_features_train,~/train_data/quote_features_val,~/train_data/quote_features_test \\
      --config qqq_btc/CONFIG/slow_feature_qqq_v2.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from preprocess.ask_bid.feature_merge_option_raw import process_labels_file  # noqa: E402

logger = logging.getLogger("qqq_btc.write_main_labels")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write main labels for LMDB")
    parser.add_argument(
        "--feature-roots",
        required=True,
        help="逗号分隔目录，如 quote_features_train,quote_features_val",
    )
    parser.add_argument(
        "--config",
        default="qqq_btc/CONFIG/slow_feature_qqq_v2.json",
    )
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(Path(args.config), "r", encoding="utf-8") as f:
        config = json.load(f)

    tasks: list[Path] = []
    for root_s in args.feature_roots.split(","):
        root = Path(root_s.strip()).expanduser()
        if not root.is_dir():
            logger.warning("skip missing root: %s", root)
            continue
        found = sorted(root.glob("**/1min/*.parquet"))
        logger.info("%s: %d 个 1min 文件", root, len(found))
        tasks.extend(found)

    if not tasks:
        raise SystemExit("未找到任何 1min parquet")

    worker = partial(process_labels_file, config=config)
    ok = err = skip = 0
    with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        for res in tqdm(pool.map(worker, tasks), total=len(tasks), desc="write labels"):
            status = res.get("status") if isinstance(res, dict) else "unknown"
            if status == "success":
                ok += 1
            elif status and status.startswith("skip"):
                skip += 1
            else:
                err += 1
                logger.error("%s", res)

    logger.info("done: success=%d skip=%d error=%d", ok, skip, err)
    if err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
