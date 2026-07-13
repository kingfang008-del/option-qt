#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qqq_btc Redis 高仿真 —— 一键启动 FCS + Signal + OMS,并按秒推送 SQLite 1s 数据。

生产同形四进程:
  1. feature_compute_service_v8  (消费 fused_market_stream)
  2. run_live_signal_qqq.py      (消费 unified_inference_stream)
  3. run_live_exec_qqq.py        (消费 orch_trade_signals)
  4. redis_fused_pitcher_1s.py   (写入 fused_market_stream)

用法:
    # 推荐: conda activate ibkr 后运行,或直接用包装脚本
    conda activate ibkr
    python qqq_btc/tools/run_qqq_btc_redis_sim.py --date 20260202 --speed 1.0

    # 包装脚本会自动使用 anaconda3/envs/ibkr/bin/python
    bash qqq_btc/tools/run_qqq_btc_redis_sim.sh --date 20260202 --speed 1.0

    # 仅发球 (栈已手动启动)
    python qqq_btc/tools/run_qqq_btc_redis_sim.py --pitcher-only --date 20260202 --speed 1.0

    # 压测: 无 sync 屏障 + 无限速
    python qqq_btc/tools/run_qqq_btc_redis_sim.py --date 20260202 --no-sync --speed inf

验收 (dry-run 一天后):
    python qqq_btc/tools/signal_diff_day.py --date 2026-02-02
    python qqq_btc/tools/shadow_parity_report.py
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_BASELINE = _REPO / "New_Pro" / "baseline_qqq"
_ENV_FILE = _BASELINE / "config" / "minimal_stack.env"

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_BASELINE) not in sys.path:
    sys.path.insert(0, str(_BASELINE))

import baseline_paths  # noqa: E402,F401

from qqq_btc.tools.redis_fused_pitcher_1s import (  # noqa: E402
    DEFAULT_OPTION_ROOT,
    FusedPitcher1s,
    RawParquetPitcher1s,
    create_pitcher,
    init_replay_redis,
    set_replay_start_ts,
    _iso_from_yyyymmdd,
    _normalize_yyyymmdd,
    _redis_client,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [REDIS_SIM] - %(message)s",
)
logger = logging.getLogger("RedisSim")

_IBKR_PY = Path.home() / "anaconda3" / "envs" / "ibkr" / "bin" / "python"


def _resolve_python() -> str:
    """子进程统一走 ibkr conda(FCS 依赖 psycopg2 等生产栈包)。"""
    explicit = os.environ.get("PYTHON", "").strip()
    if explicit and Path(explicit).is_file():
        return explicit
    if _IBKR_PY.is_file():
        return str(_IBKR_PY)
    return sys.executable


def _ensure_ibkr_python() -> None:
    """若当前解释器不是 ibkr,自动 re-exec 到 ibkr python。"""
    target = _resolve_python()
    if Path(target).resolve() == Path(sys.executable).resolve():
        return
    if not Path(target).is_file():
        logger.warning("ibkr python not found at %s; using %s", target, sys.executable)
        return
    logger.info("Re-launching with ibkr python: %s", target)
    os.execv(target, [target, *sys.argv])


def _load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def _build_child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    # minimal_stack.env 只补缺省；shell/step2 已 export 的键（如 DELAY=1）不被覆盖
    for key, val in _load_env_file(_ENV_FILE).items():
        env.setdefault(key, val)
    env.setdefault("RUN_MODE", "REALTIME_DRY")
    env.setdefault("QQQ_BTC_LIVE", "1")
    env.setdefault("SKIP_DEEP_WARMUP", "1")
    env.setdefault("REDIS_STREAM_SIM", "1")
    env.setdefault("OMS_MOCK_IBKR", "1")
    v4 = _REPO / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v4.json"
    v2 = _REPO / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v2.json"
    default_slow = v4 if v4.exists() else v2
    if default_slow.exists():
        env.setdefault("SLOW_FEATURE_CONFIG", str(default_slow))
    env.setdefault("RECALC_GREEKS", "1")
    env.setdefault("FCS_STATE_BACKEND", "none")
    default_frozen = _REPO / "qqq_btc" / "CONFIG" / "frozen_norm_qqq_daily.npz"
    if default_frozen.exists():
        env.setdefault("FCS_FROZEN_NORM_PATH", str(default_frozen))
    repo = str(_REPO)
    baseline = str(_BASELINE)
    pp = env.get("PYTHONPATH", "")
    for p in (repo, baseline):
        if p not in pp.split(os.pathsep):
            pp = f"{p}{os.pathsep}{pp}" if pp else p
    env["PYTHONPATH"] = pp
    if extra:
        env.update(extra)
    return env


class _ProcGroup:
    def __init__(self) -> None:
        self._procs: list[tuple[str, subprocess.Popen]] = []

    def add(self, name: str, cmd: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.Popen:
        logger.info("Starting %s: %s", name, " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._procs.append((name, proc))
        threading.Thread(target=self._pipe_reader, args=(name, proc), daemon=True).start()
        return proc

    @staticmethod
    def _pipe_reader(name: str, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            sys.stdout.write(f"[{name}] {line}")
            sys.stdout.flush()

    def stop_all(self) -> None:
        for name, proc in reversed(self._procs):
            if proc.poll() is None:
                logger.info("Stopping %s (pid=%s)", name, proc.pid)
                try:
                    proc.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
        time.sleep(1.0)
        for name, proc in reversed(self._procs):
            if proc.poll() is None:
                proc.terminate()
        for name, proc in reversed(self._procs):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Killing %s", name)
                proc.kill()


def _wait_se_oms_alive(group: _ProcGroup, *, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok = all(p.poll() is None for _, p in group._procs[1:])
        if not ok:
            return False
        time.sleep(1.0)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="qqq_btc Redis stream high-fidelity sim")
    parser.add_argument("--date", type=str, required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--speed", type=float, default=1.0, help="1.0=realtime 1Hz")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--pitcher-only", action="store_true", help="Only run 1s pitcher")
    parser.add_argument("--keep-stack", action="store_true", help="Do not stop FCS/SE/OMS after pitcher")
    parser.add_argument("--no-reset-redis", action="store_true")
    parser.add_argument("--skip-deep-warmup", action="store_true", default=True)
    parser.add_argument(
        "--deep-warmup",
        action="store_true",
        help="启用 FCS PG 深预热(SKIP_DEEP_WARMUP=0);仅流目标日时配合 REPLAY_START_TS",
    )
    parser.add_argument(
        "--warmup-from-date",
        type=str,
        default=None,
        help="流式预热起点 YYYYMMDD(默认=--date 单日);对拍时与 replay parquet 最早日对齐",
    )
    parser.add_argument("--db-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--python", type=str, default=None, help="Override python for FCS/SE/OMS")
    parser.add_argument("--use-ibkr", action="store_true", help="启用真实 IBKR 连接(默认 Mock 成交,无 Gateway 日志)")
    parser.add_argument(
        "--source",
        choices=("auto", "raw", "sqlite"),
        default="raw",
        help="发球机数据源: raw=options_databento_v3(默认)",
    )
    parser.add_argument("--option-root", type=str, default=str(DEFAULT_OPTION_ROOT))
    parser.add_argument(
        "--greek-parity",
        action="store_true",
        help="发球机仅注入分钟 Greeks/IV + RECALC_GREEKS=0; 盘口仍用 raw_1s",
    )
    parser.add_argument(
        "--greek-root",
        type=str,
        default=str(Path.home() / "train_data/quote_options_day_iv"),
    )
    parser.add_argument(
        "--frozen-norm",
        type=str,
        default=None,
        help="FCS_FROZEN_NORM_PATH: 离线 rolling_norm 导出的 .npz",
    )
    parser.add_argument(
        "--no-frozen-norm",
        action="store_true",
        help="禁用冻结归一化,走 FCS RollingWindowNormalizer(与离线 rolling 同族)",
    )
    parser.add_argument(
        "--trade-from-date",
        type=str,
        default=None,
        help="禁止早于该日的新开仓 YYYYMMDD(用于 June 预热 + July 交易连续流)",
    )
    parser.add_argument(
        "--rolling-norm-seed",
        type=str,
        default=None,
        help="FCS_ROLLING_NORM_SEED_PATH: 预填 rolling buffer(不冻结)。"
        "对齐 +37.7% 金标请用已归一化 bak June "
        "(_bak_pre4c/quote_features_test_QQQ/.../2026-06.parquet)，勿用 raw",
    )
    parser.add_argument("--fcs-wait", type=float, default=45.0, help="Seconds to wait after FCS start")
    parser.add_argument(
        "--max-session-bars",
        type=int,
        default=None,
        help="仅流最后一个交易日的前 N 个 session bar(快速首小时诊断)",
    )
    args = parser.parse_args()

    _ensure_ibkr_python()
    py = args.python or _resolve_python()
    logger.info("Using python: %s", py)

    os.environ.setdefault("RUN_MODE", "REALTIME_DRY")
    mock_fill = "0" if args.use_ibkr else "1"
    extra_env: dict[str, str] = {
        "RUN_MODE": "REALTIME_DRY",
        "QQQ_BTC_LIVE": "1",
        "SKIP_DEEP_WARMUP": "0" if args.deep_warmup else ("1" if args.skip_deep_warmup else "0"),
        "REDIS_STREAM_SIM": "1",
        "OMS_MOCK_IBKR": mock_fill,
    }
    if args.greek_parity:
        extra_env["GREEK_PARITY_MODE"] = "1"
        extra_env["RECALC_GREEKS"] = "0"
        # 诊断上界：让 FCS 在分钟 commit 时也使用 minute IV parquet
        # 重建 Greeks/spread/imbalance，避免只在 pitcher 侧补 Greeks 后仍由
        # FCS 使用 raw 1s 的盘口 size 计算 imbalance。
        extra_env["FCS_MINUTE_PARITY_INJECT"] = "1"
    if args.frozen_norm and args.no_frozen_norm:
        raise SystemExit("不能同时指定 --frozen-norm 与 --no-frozen-norm")
    if args.frozen_norm:
        extra_env["FCS_FROZEN_NORM_PATH"] = str(Path(args.frozen_norm).expanduser())
    # 对齐离线 merge_1m_5m(asof backward)：1m 时刻 T 使用 timestamp<=T 的 5m 桶
    # (含 T 本身的完整 5m bar)。默认 lag=1 的「仅已完成上一桶」适合实盘因果，
    # 但与训练特征 merge 口径不一致，会导致 poc_deviation 等 5m 慢变量系统性偏离。
    extra_env.setdefault("FCS_5M_COMPLETED_BUCKET_ONLY", "0")
    extra_env.setdefault("FCS_PREFER_EXTERNAL_5M", "1")
    # live rolling 与离线逐 bar 刷新更接近
    extra_env.setdefault("FCS_NORMALIZER_STATS_UPDATE_INTERVAL", "1")

    from config import get_redis_db  # noqa: E402

    args.date = _normalize_yyyymmdd(args.date)
    if args.end_date:
        args.end_date = _normalize_yyyymmdd(args.end_date)
    if args.warmup_from_date:
        args.warmup_from_date = _normalize_yyyymmdd(args.warmup_from_date)
    else:
        args.warmup_from_date = args.date
    if args.trade_from_date:
        args.trade_from_date = _normalize_yyyymmdd(args.trade_from_date)
        extra_env["QQQ_BTC_TRADE_FROM_DATE"] = args.trade_from_date
    if getattr(args, "rolling_norm_seed", None):
        seed_path = str(Path(args.rolling_norm_seed).expanduser())
        extra_env["FCS_ROLLING_NORM_SEED_PATH"] = seed_path
        # 种子只取严格早于交易日起的行,避免灌入 July
        extra_env["FCS_ROLLING_NORM_SEED_BEFORE"] = args.trade_from_date or args.date
        extra_env.setdefault("FCS_ROLLING_NORM_SEED_SYMBOL", "QQQ")

    # put_gate 因果 VIXY 预热:只灌入严格早于流起点的 1m close
    vixy_before = args.warmup_from_date or args.date
    extra_env["QQQ_BTC_VIXY_SEED_BEFORE"] = vixy_before
    child_env = _build_child_env(extra_env)
    os.environ["QQQ_BTC_VIXY_SEED_BEFORE"] = vixy_before
    if args.trade_from_date:
        os.environ["QQQ_BTC_TRADE_FROM_DATE"] = args.trade_from_date
    if getattr(args, "rolling_norm_seed", None):
        os.environ["FCS_ROLLING_NORM_SEED_PATH"] = child_env["FCS_ROLLING_NORM_SEED_PATH"]
        os.environ["FCS_ROLLING_NORM_SEED_BEFORE"] = child_env["FCS_ROLLING_NORM_SEED_BEFORE"]
    if args.no_frozen_norm:
        child_env.pop("FCS_FROZEN_NORM_PATH", None)
        os.environ.pop("FCS_FROZEN_NORM_PATH", None)
    elif "FCS_FROZEN_NORM_PATH" in child_env:
        os.environ["FCS_FROZEN_NORM_PATH"] = child_env["FCS_FROZEN_NORM_PATH"]

    stream_start = args.warmup_from_date if not args.deep_warmup else args.date
    stream_end = args.end_date or args.date

    logger.info(
        "qqq_btc redis sim | RUN_MODE=%s redis_db=%d target=%s stream=[%s..%s] speed=%s sync=%s mock_ibkr=%s source=%s greek_parity=%s frozen_norm=%s deep_warmup=%s trade_from=%s",
        child_env.get("RUN_MODE"),
        get_redis_db(),
        args.date,
        stream_start,
        stream_end,
        args.speed,
        not args.no_sync,
        child_env.get("OMS_MOCK_IBKR"),
        args.source,
        args.greek_parity,
        child_env.get("FCS_FROZEN_NORM_PATH", "") or ("OFF" if args.no_frozen_norm else ""),
        args.deep_warmup,
        getattr(args, "trade_from_date", None) or "",
    )

    opt_probe = Path(args.option_root) / "QQQ" / f"QQQ_{_iso_from_yyyymmdd(args.date)}.parquet"
    sqlite_db = Path.home() / "quant_project/data/history_sqlite_1s" / f"market_{args.date}.db"
    if args.source == "sqlite" and not sqlite_db.exists():
        logger.error("sqlite 1s 不存在: %s", sqlite_db)
        return 1
    if args.source in ("raw", "auto") and not opt_probe.exists() and not sqlite_db.exists():
        logger.error("raw option 与 sqlite 均不存在: %s | %s", opt_probe, sqlite_db)
        return 1
    r = _redis_client()
    run_id = init_replay_redis(r, reset=not args.no_reset_redis)
    replay_start_ts = set_replay_start_ts(r, stream_start)
    # FCS 子进程 env 是 copy,必须显式写入 REPLAY_START_TS 供 Deep Warmup 截断
    child_env["REPLAY_START_TS"] = str(replay_start_ts)
    os.environ["REPLAY_START_TS"] = str(replay_start_ts)
    logger.info("REPLAY_START_TS=%s (stream_start=%s)", replay_start_ts, stream_start)

    group = _ProcGroup()
    try:
        if not args.pitcher_only:
            fcs = group.add(
                "FCS",
                [py, str(_BASELINE / "DAO" / "feature_compute_service_v8.py")],
                cwd=_BASELINE,
                env=child_env,
            )
            time.sleep(args.fcs_wait)
            if fcs.poll() is not None:
                logger.error("FCS failed to start")
                return 1

            se_cmd = [py, str(_REPO / "qqq_btc" / "tools" / "run_live_signal_qqq.py")]
            if args.checkpoint:
                ckpt = str(Path(args.checkpoint).expanduser().resolve())
                se_cmd.extend(["--checkpoint", ckpt])
            group.add("SE", se_cmd, cwd=_BASELINE, env=child_env)

            group.add(
                "OMS",
                [py, str(_REPO / "qqq_btc" / "tools" / "run_live_exec_qqq.py")],
                cwd=_BASELINE,
                env=child_env,
            )

            if not _wait_se_oms_alive(group):
                logger.error("SE/OMS failed to stay alive")
                return 1
            logger.info("Stack up: FCS + SE + OMS (waiting 3s for consumer groups)...")
            time.sleep(3.0)

        pitcher = create_pitcher(
            args.source,
            option_root=Path(args.option_root),
            greek_root=Path(args.greek_root),
            greek_parity=args.greek_parity,
            run_id=run_id,
        )
        total = pitcher.run(
            start_date=stream_start,
            end_date=stream_end,
            speed_factor=args.speed,
            sync_mode=not args.no_sync,
            max_session_bars=args.max_session_bars,
        )
        logger.info(
            "Simulation complete | ticks=%d | target=%s | stream=[%s..%s] | audit=~/quant_project/shadow/signals_%s-%s-%s.csv",
            total,
            args.date,
            stream_start,
            stream_end,
            args.date[:4],
            args.date[4:6],
            args.date[6:8],
        )

        if args.pitcher_only or args.keep_stack:
            logger.info("Stack left running (--pitcher-only / --keep-stack). Ctrl+C to stop.")
            while True:
                time.sleep(3600)
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    finally:
        if not args.pitcher_only and not args.keep_stack:
            group.stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
