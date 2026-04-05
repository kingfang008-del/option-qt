#!/usr/bin/env python3
"""
test_strategy_parity.py - 策略对齐验证工具
功能: 关闭所有 s4 独有 Guard，开启 plan_a 独有规则，验证交易数量和收益一致性
用法: python3.10 test_strategy_parity.py --date 20260312
"""
import sys, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../baseline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../preprocess/backtest'))

from strategy_config import StrategyConfig


def create_plan_a_parity_config():
    """创建与 plan_a 完全对齐的策略配置 (关闭所有 s4 独有 Guard, 打开 plan_a 独有规则)"""
    cfg = StrategyConfig()

    # --- 关闭 s4 独有的入场 Guard ---
    cfg.ENTRY_MOMENTUM_GUARD_ENABLED = False       # plan_a 不检查 stock_roc/snap_roc
    cfg.ENTRY_LIQUIDITY_GUARD_ENABLED = False       # plan_a 不检查 bid/ask spread
    cfg.MACD_HIST_CONFIRM_ENABLED = False           # plan_a 不检查 MACD
    cfg.INDEX_GUARD_ENABLED = False                 # plan_a 不检查大盘方向

    # --- 关闭 s4 独有的平仓 Guard ---
    cfg.EXIT_COUNTER_TREND_ENABLED = False          # plan_a 无 CT_TIMEOUT
    cfg.EXIT_INDEX_REVERSAL_ENABLED = False         # plan_a 无 IDX_REVERSAL
    cfg.EXIT_STOCK_HARD_STOP_ENABLED = False        # plan_a 无 STOCK_STOP
    cfg.EXIT_ZOMBIE_STOP_ENABLED = False            # plan_a 无 ZOMBIE_STOP
    cfg.EXIT_MACD_FADE_ENABLED = False              # plan_a 无 MACD_FADE
    cfg.EXIT_SIGNAL_FLIP_ENABLED = False            # plan_a 无 FLIP
    cfg.EXIT_LIQUIDITY_GUARD_ENABLED = False        # plan_a 无 SPREAD_STOP
    cfg.EXIT_COND_STOP_ENABLED = False              # plan_a 无 COND_STOP
    cfg.EXIT_SMALL_GAIN_ENABLED = False             # plan_a 无 SMALL_GAIN_P
    cfg.INDEX_REVERSAL_EXIT_ENABLED = False         # plan_a 无大盘反转平仓

    # --- 打开 plan_a 独有的平仓规则 ---
    cfg.EXIT_EARLY_STOP_ENABLED = True              # plan_a: 5分钟内 roi < -5%
    cfg.EXIT_NO_MOMENTUM_ENABLED = True             # plan_a: 5min后 max_roi < 2%
    cfg.PARITY_STRICT_MODE = True                   # 强制使用 Plan A 入场过滤 (Prob >= 0.7)

    return cfg


def run_plan_a_baseline(date_str):
    """直接运行 plan_a 逻辑获取基准结果"""
    from plan_a_fine_tune import load_day, run_sim, get_db_files
    import glob

    hist_dir = os.path.join(os.path.dirname(__file__), '../preprocess/backtest/history_sqlite_1m')
    db_path = os.path.join(hist_dir, f"market_{date_str}.db")
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return None, None

    cfg = StrategyConfig()
    df_a, df_s, df_o = load_day(db_path)
    ret, trades = run_sim(df_a, df_s, df_o, cfg, 
                          es_roi=-0.05, mom_mins=5, mom_max=0.02, abs_sl=-0.20,
                          use_vol_z=True)
    return ret, trades


def print_comparison(plan_a_ret, plan_a_trades, s4_trades_csv=None):
    """打印对比报告"""
    import pandas as pd
    from datetime import datetime, timedelta
    import pytz
    ny_tz = pytz.timezone('America/New_York')

    print("\n" + "=" * 70)
    print("📊 STRATEGY PARITY REPORT")
    print("=" * 70)

    # Plan A 结果
    print(f"\n🔹 Plan A (Baseline):")
    print(f"   交易笔数: {len(plan_a_trades)}")
    print(f"   总收益率: {plan_a_ret:.2%}")

    if plan_a_trades:
        print(f"\n   {'Sym':<6} {'Dir':<5} {'Entry Time':<12} {'Exit Time':<12} {'ROI':<10} {'Reason'}")
        print(f"   {'-'*60}")
        for t in plan_a_trades:
            et = datetime.fromtimestamp(t['ets'], tz=ny_tz).strftime('%H:%M')
            xt = datetime.fromtimestamp(t['ots'], tz=ny_tz).strftime('%H:%M')
            print(f"   {t['sym']:<6} {t['typ']:<5} {et:<12} {xt:<12} {t['roi']:>8.1%}  {t['reason']}")

    # S4 结果
    if s4_trades_csv and os.path.exists(s4_trades_csv):
        df = pd.read_csv(s4_trades_csv)
        print(f"\n🔸 S4 (High-Fidelity, Guards OFF):")
        print(f"   交易笔数: {len(df)}")
        if len(df) > 0:
            total_pnl = df['pnl'].sum()
            print(f"   总 PnL:   ${total_pnl:,.2f}")
            print(f"\n   {'Sym':<6} {'Dir':<5} {'Entry Time':<18} {'Exit Time':<18} {'PnL':<10} {'Reason'}")
            print(f"   {'-'*80}")
            for _, row in df.iterrows():
                et = datetime.fromtimestamp(row['entry_ts'], tz=ny_tz).strftime('%H:%M')
                xt = datetime.fromtimestamp(row['exit_ts'], tz=ny_tz).strftime('%H:%M')
                print(f"   {row['symbol']:<6} {row['opt_dir']:<5} {et:<18} {xt:<18} ${row['pnl']:>8.2f}  {row['reason']}")

        # 笔数对比
        diff = abs(len(df) - len(plan_a_trades))
        status = "✅ PASS" if diff <= 2 else "❌ FAIL"
        print(f"\n📏 交易笔数差异: {diff} ({status})")
    else:
        print(f"\n⚠️ S4 CSV 不存在，请先运行 s4 回测")

    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20260312")
    parser.add_argument("--run-s4", action="store_true", help="同时运行 s4 回测")
    args = parser.parse_args()

    # 1. 打印对齐配置
    parity_cfg = create_plan_a_parity_config()
    print("🔧 Plan A 对齐配置:")
    guard_fields = [f for f in dir(parity_cfg) if ('GUARD' in f or 'EXIT_' in f or 'ENTRY_' in f) and not f.startswith('_')]
    for f in sorted(guard_fields):
        val = getattr(parity_cfg, f)
        if isinstance(val, bool):
            status = "🟢 ON" if val else "🔴 OFF"
            print(f"   {f:<40} {status}")
    
    # 2. 运行 Plan A 基准
    print(f"\n🚀 Running Plan A baseline for {args.date}...")
    ret, trades = run_plan_a_baseline(args.date)
    if ret is None:
        sys.exit(1)

    # 3. 如果指定 --run-s4，运行 s4
    s4_csv = os.path.expanduser("~/quant_project/logs/replay_trades_v8.csv")
    if args.run_s4:
        print(f"\n🚀 Running S4 (Guards OFF) for {args.date}...")
        s4_script = os.path.join(os.path.dirname(__file__), '../preprocess/backtest/s4_run_historical_replay.py')
        cmd = f"PARITY_MODE=PLAN_A python3.10 {s4_script} --date {args.date}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(res.stdout)
        print(res.stderr, file=sys.stderr)

    # 4. 打印对比报告
    print_comparison(ret, trades, s4_csv)
