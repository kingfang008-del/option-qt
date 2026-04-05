import subprocess
import re
import pandas as pd

dates = ["20260102", "20260105", "20260106", "20260107", "20260108", "20260109"]
results = []

print(f"🚀 Starting Weekly Batch Backtest (Delay = 1 bar)...")

for d in dates:
    print(f"📅 Running {d}...")
    cmd = ["/usr/local/bin/python3.10", "/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/s4_run_historical_replay_stable.py", "--date", d]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    last_pnl = 0
    last_trades = 0
    last_wr = 0
    
    for line in process.stdout:
        # 寻找 V8 引擎最新的表格化总结输出
        if "Net realized" in line:
            # 💰 总实现盈亏 (Net realized):  $+39,450.60  或  $-1,550.40
            match = re.search(r"Net realized\):\s+\$([+?\-?\d,.]+)", line)
            if match: last_pnl = float(match.group(1).replace(",", "").replace("+", ""))
            
        elif "Total Trades" in line:
            # 📈 交易总笔数 (Total Trades):  89
            match = re.search(r"Total Trades\):\s+(\d+)", line)
            if match: last_trades = int(match.group(1))
            
        elif "Win Rate" in line:
            # 🎯 胜率 (Win Rate):           51.69%
            match = re.search(r"Win Rate\):\s+([\d.]+)%", line)
            if match: last_wr = float(match.group(1))
    
    process.wait()
    results.append({
        "Date": d[:4]+"-"+d[4:6]+"-"+d[6:],
        "PnL": last_pnl,
        "Trades": last_trades,
        "WinRate": last_wr
    })

df = pd.DataFrame(results)
df["Return"] = df["PnL"] / 50000.0
df["CumPnL"] = df["PnL"].cumsum()
df["Equity"] = 50000 + df["CumPnL"]

print("\n" + "="*60)
print("📊 WEEKLY BATCH BACKTEST PERFORMANCE (1-BAR DELAY)")
print("="*60)
print(df.to_string(index=False))
print("="*60)
print(f"💰 Total Net PnL: ${df['PnL'].sum():,.2f}")
print(f"📈 Total Trades:  {df['Trades'].sum()}")
print(f"🎯 Average WR:    {df['WinRate'].mean():.2f}%")
print("="*60)
