import pandas as pd
import psycopg2
from pathlib import Path
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'baseline'))
from config import PG_DB_URL

def diagnostic_report(date_str):
    # 1. 加载 S4 的黄金基准 (CSV)
    # The file might be in different directories depending on where s4 was run. Let's look for it.
    s4_csv_paths = [
        Path(f"replay_trades_s2_pg_1s.csv"),
        Path.home() / "quant_project/logs" / "replay_trades_s2_pg_1s.csv",
        Path("/Users/fangshuai/Documents/GitHub/option-qt") / "replay_trades_s2_pg_1s.csv"
    ]
    s4_csv = next((p for p in s4_csv_paths if p.exists()), None)
    
    if not s4_csv:
        print(f"❌ 找不到 S4 基准 CSV: {s4_csv_paths[0]} 等路径下均未发现。请确保刚刚成功运行了 s4 并生成了文件。")
        return
    
    df_s4 = pd.read_csv(s4_csv)
    df_s4['source'] = 'S4_BASELINE'
    
    # 2. 加载 Standalone 的 PG 记录
    print(f"📡 正在从 PG 读取 trade_logs_backtest (按照 {date_str} 过滤)...")
    conn = psycopg2.connect(PG_DB_URL)
    query = f"SELECT symbol, ts, action, qty, price, details_json FROM trade_logs_backtest WHERE action = 'SELL'"
    df_pg_raw = pd.read_sql(query, conn)
    
    # 这里需要解析 details_json 提取 entry_ts 和 reason
    import json
    def parse_details(row):
        try:
            d = json.loads(row['details_json'])
            return pd.Series([float(d.get('entry_ts', 0)), d.get('reason', 'UNKNOWN'), float(d.get('pnl', 0.0))])
        except:
            return pd.Series([0.0, 'ERROR', 0.0])
            
    df_pg_sell = df_pg_raw.copy()
    if df_pg_sell.empty:
        print(f"❌ PostgreSQL 中没有任何交易记录，请确认 Standalone 执行完毕。")
        return
        
    df_pg_sell[['entry_ts', 'reason', 'pnl']] = df_pg_sell.apply(parse_details, axis=1)
    df_pg_sell['source'] = 'STANDALONE_PG'

    # Filter PG trades for the specific date if needed based on UTC timestamps
    # 3. 核心对比逻辑
    print(f"\n📈 统计摘要:")
    print(f"- S4 基准成交笔数: {len(df_s4)}")
    print(f"- Standalone 实盘笔数: {len(df_pg_sell)}")
    
    # 找寻第一笔分歧
    # 以 symbol + entry_ts 作为主键对齐
    merged = pd.merge(
        df_s4[['symbol', 'entry_ts', 'exit_ts', 'pnl', 'reason']], 
        df_pg_sell[['symbol', 'entry_ts', 'ts', 'pnl', 'reason']], 
        on=['symbol', 'entry_ts'], 
        how='outer', 
        suffixes=('_s4', '_pg')
    ).sort_values('entry_ts')

    print("\n🔍 正在追踪分歧点...")
    diff_found = False
    for _, row in merged.iterrows():
        # 情况 A: Standalone 多出了单子 (Ghost Trade)
        if pd.isna(row['exit_ts_s4']):
            print(f"🚨 [多余交易] {row['symbol']} @ EntryTS: {row['entry_ts']:.0f} | Standalone 莫名其妙开仓了，S4 没动。原因: {row['reason_pg']}")
            diff_found = True
            break
        # 情况 B: Standalone 漏掉了单子
        if pd.isna(row['ts_pg']):
            print(f"🚨 [缺失交易] {row['symbol']} @ EntryTS: {row['entry_ts']:.0f} | S4 已成交，但 Standalone 没反应。 S4原因: {row['reason_s4']}")
            diff_found = True
            break
        # 情况 C: 提前离场 (Exit Timing Diff)
        if abs(row['exit_ts_s4'] - row['ts_pg']) > 1:
            print(f"🚨 [离场时间差] {row['symbol']} | S4离场: {row['exit_ts_s4']:.0f} | PG离场: {row['ts_pg']:.0f} | 相差: {row['ts_pg']-row['exit_ts_s4']:.1f}s")
            print(f"   => S4原因: {row['reason_s4']} | PG原因: {row['reason_pg']}")
            diff_found = True
            break

    if not diff_found:
        print("✅ 没发现明显信号触发时点分歧，系统已经完全同频！请通过打印的 PnL 检查是否有微弱的定价手续费误差。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    diagnostic_report(args.date)
