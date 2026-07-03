import os, sys, json, pandas as pd, numpy as np, re
from pathlib import Path

# Path setup
BT_DIR = os.path.dirname(os.path.abspath(__file__))
PROD_DIR = os.path.dirname(os.path.dirname(BT_DIR))
BASE_DIR = os.path.join(PROD_DIR, "baseline")
MODEL_DIR = os.path.join(PROD_DIR, "model")
REPLAY_DIR = os.path.join(PROD_DIR, "history_replay")

# 设置环境变量，确保 SignalEngineV8 能找到生产模块
os.environ["PYTHONPATH"] = f"{PROD_DIR}:{BASE_DIR}:{MODEL_DIR}:{REPLAY_DIR}:{os.environ.get('PYTHONPATH', '')}"

PLAN_A_SCRIPT = os.path.join(BT_DIR, "plan_a_fine_tune.py")
S4_REPLAY_SCRIPT = os.path.join(BT_DIR, "s4_run_historical_replay.py")
CSV_LOG = Path.home() / "quant_project/logs/replay_trades_v8.csv"

def parse_plan_a_log(log_path):
    """从 plan_a_fine_tune.py 的 stdout 中解析交易表格"""
    trades = []
    with open(log_path, 'r') as f:
        in_table = False
        for line in f:
            if "DETAILED TRADE EXPLORER" in line:
                in_table = True
                continue
            if in_table and "|" in line:
                # 示例: 3/13     ADBE     put   14:21    15:00    |    5.53    6.15 |    11.2% EOD
                parts = line.split("|")
                if len(parts) >= 3:
                    left = parts[0].split()
                    if len(left) >= 5:
                        date, sym, typ, in_time, out_time = left[0], left[1], left[2], left[3], left[4]
                        roi_parts = parts[2].split()
                        roi = roi_parts[0]
                        reason = roi_parts[1]
                        trades.append({
                            'date': date, 'symbol': sym, 'typ': typ.upper(), 
                            'in_time': in_time, 'out_time': out_time, 'roi': roi, 'reason': reason
                        })
    return pd.DataFrame(trades)

def verify_parity(date_str="20260312"):
    print(f"🚀 Starting BT Parity Audit for {date_str}...")
    
    # 1. 运行 plan_a_fine_tune.py
    log_a = "/tmp/parity_plan_a.txt"
    print("Running FAST simulator...")
    os.system(f"python3.10 {PLAN_A_SCRIPT} --date {date_str} > {log_a} 2>&1")
    df_a = parse_plan_a_log(log_a)
    
    # 2. 运行 s4_run_historical_replay.py
    print("Running HIGH-Fidelity simulator...")
    os.system(f"python3.10 {S4_REPLAY_SCRIPT} --date {date_str} > /tmp/parity_s4.txt 2>&1")
    
    if not CSV_LOG.exists():
        print("Error: s4 CSV output not found!")
        return
        
    df_s4 = pd.read_csv(CSV_LOG)
    def to_ny_time(ts_str):
        try:
            import pytz
            dt = pd.to_datetime(ts_str)
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)
            ny_tz = pytz.timezone('America/New_York')
            ny_dt = dt.astimezone(ny_tz)
            return ny_dt.strftime('%H:%M')
        except: return ts_str
        
    from datetime import datetime, timedelta
    df_s4['in_time'] = df_s4['entry_ts'].apply(lambda ts: (datetime.fromtimestamp(ts) - timedelta(hours=13)).strftime('%H:%M'))
    df_s4['out_time'] = df_s4['exit_ts'].apply(lambda ts: (datetime.fromtimestamp(ts) - timedelta(hours=13)).strftime('%H:%M'))
    df_s4['typ'] = df_s4['opt_dir']
    
    # 🚀 [Fix] 确保所有 Key 都没有空格影响 Merge
    for df in [df_a, df_s4]:
        for col in ['symbol', 'typ', 'in_time']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
    
    # 3. 交叉比对
    print("\n" + "="*50)
    print(f"📊 PARITY DIFF REPORT: {date_str}")
    print("="*50)
    
    # 打印汇总
    print(f"FAST (plan_a): {len(df_a)} trades")
    print(f"HIGH (s4_bus): {len(df_s4)} trades")
    
    # 寻找不匹配项
    merged = pd.merge(df_a, df_s4, on=['symbol', 'typ', 'in_time'], how='outer', suffixes=('_a', '_s4'))
    
    mismatches = merged[merged['out_time_a'] != merged['out_time_s4']]
    if mismatches.empty:
        print("✅ SUCCESS: All trade entry/exit times are IDENTICAL.")
    else:
        print(f"⚠️ FOUND {len(mismatches)} MISMATCHES:")
        # 🚀 [Fix] 使用合并后的列名
        cols = ['symbol', 'in_time', 'out_time_a', 'out_time_s4']
        if 'reason_a' in merged.columns: cols.append('reason_a')
        if 'reason_s4' in merged.columns: cols.append('reason_s4')
        print(mismatches[cols])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20260312", help="Date in YYYYMMDD format")
    args = parser.parse_args()
    verify_parity(args.date)
