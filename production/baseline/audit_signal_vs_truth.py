import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path
import os

# ================= 配置 =================
# 注意：请根据实际运行后的路径调整 CSV 路径
CSV_PATH = Path.home() / "quant_project/logs/alpha_audit.csv"
DB_PATH = Path("/home/kingfang007/quant_project/data/history_sqlite_1m/market_20260102.db")

def audit_signals():
    if not CSV_PATH.exists():
        print(f"❌ 找不到审计日志: {CSV_PATH}")
        print("请先运行一次 S4 回测以生成日志。")
        return

    print(f"📖 读取审计日志: {CSV_PATH}")
    df_alpha = pd.read_csv(CSV_PATH)
    
    print(f"🗄️ 连接数据库: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    
    # 获取数据库中的所有价格数据（用于计算未来收益）
    print("📥 正在从数据库提取行情真值...")
    df_truth = pd.read_sql("SELECT ts, symbol, close FROM market_bars_1m", conn)
    conn.close()

    # 转换时间戳为整数方便比对
    df_alpha['timestamp'] = df_alpha['timestamp'].astype(int)
    df_truth['ts'] = df_truth['ts'].astype(int)

    results = []
    timestamps = sorted(df_alpha['timestamp'].unique())
    if not timestamps:
        print(f"⚠️ 审计日志 {CSV_PATH} 中没有有效的信号记录。请确认 S4 回测已正确运行。")
        return
    
    print(f"🔍 开始审计 {len(timestamps)} 个时间点的信号质量...")
    
    for ts in timestamps:
        # 1. 提取当前时刻的 Alpha 预测
        batch_alpha = df_alpha[df_alpha['timestamp'] == ts].copy()
        
        # 2. 寻找 5 分钟后的价格 (T + 300)
        ts_future = ts + 300
        batch_future = df_truth[df_truth['ts'] == ts_future][['symbol', 'close']].rename(columns={'close': 'close_future'})
        
        if batch_future.empty:
            continue
            
        # 合并真值
        batch = pd.merge(batch_alpha, batch_future, on='symbol', how='inner')
        if batch.empty:
            continue
            
        # 计算 5 分钟实际收益率
        batch['true_roc_5m'] = (batch['close_future'] - batch['price']) / batch['price']
        
        # 统计指标
        # A. 相关性 (Alpha vs True ROC)
        corr = batch['alpha'].corr(batch['true_roc_5m'])
        
        # B. Top 3 Alpha 的命中情况
        top_alpha = batch.nlargest(3, 'alpha')
        top_truth = batch.nlargest(3, 'true_roc_5m')
        
        hit_top = len(set(top_alpha['symbol']) & set(top_truth['symbol']))
        captured_roc = top_alpha['true_roc_5m'].mean()
        
        results.append({
            'ts': ts,
            'corr': corr,
            'hit_top_3': hit_top,
            'avg_top_alpha_roc': captured_roc,
            'best_possible_roc': top_truth['true_roc_5m'].mean()
        })

    if not results:
        print("⚠️ 未能匹配到任何有效的 5 分钟未来收益数据。请检查 CSV 和 DB 的时间戳范围。")
        print(f"   - 信号时间范围: {min(timestamps)} ~ {max(timestamps)}")
        print(f"   - 数据库时间范围: {df_truth['ts'].min()} ~ {df_truth['ts'].max()}")
        return

    # 生成最终报告
    report = pd.DataFrame(results)
    print("\n" + "="*50)
    print("📊 SIGNAL VS TRUTH AUDIT REPORT")
    print("="*50)
    print(f"平均各分钟 Alpha 相关性 (Correlation): {report['corr'].mean():.4f}")
    print(f"Top 3 Alpha 捕获的平均 5m 收益率: {report['avg_top_alpha_roc'].mean()*100:.4f}%")
    print(f"理论最大平均 5m 收益率 (Oracle): {report['best_possible_roc'].mean()*100:.4f}%")
    print(f"信号捕获率 (Capture Efficiency): {(report['avg_top_alpha_roc'].mean() / report['best_possible_roc'].mean())*100:.2f}%")
    print(f"Top 3 命中数 (Hit Count): {report['hit_top_3'].sum()} / {len(report)*3}")
    print("="*50)
    
    # 重新聚合一张全量表 (用于后续 Threshold Scan)
    all_rows = []
    for ts in timestamps:
        ts_future = ts + 300
        batch_alpha = df_alpha[df_alpha['timestamp'] == ts]
        batch_future = df_truth[df_truth['ts'] == ts_future][['symbol', 'close']].rename(columns={'close': 'close_future'})
        if batch_future.empty: continue
        batch = pd.merge(batch_alpha, batch_future, on='symbol', how='inner')
        batch['true_roc_5m'] = (batch['close_future'] - batch['price']) / batch['price']
        all_rows.append(batch)
    
    if all_rows:
        df_full = pd.concat(all_rows)
        # 🚀 [优化] 使用 Pivot Table 进行向量化计算，确保时间戳对齐
        print("📊 正在准备向量化时间序列对齐...")
        truth_pivot = df_truth.pivot(index='ts', columns='symbol', values='close')
        # 去除列名两端可能的空格，防止匹配失败
        truth_pivot.columns = [str(c).strip() for c in truth_pivot.columns]
        
        # 调试：检查对齐信息
        print(f"   - 数据库起始时间: {min(truth_pivot.index)}")
        print(f"   - 信号起始时间: {min(df_full['timestamp'])}")
        print(f"   - 数据库 Symbol 样例: {list(truth_pivot.columns)[:5]}")
        
        # 计算每个 (ts, symbol) 的 1 分钟前价格
        def get_prev_price(row):
            ts_now_aligned = (int(row['timestamp']) // 60) * 60
            ts_p = ts_now_aligned - 60
            sym = str(row['symbol']).strip()
            
            try:
                if sym in truth_pivot.columns:
                    if ts_p in truth_pivot.index:
                        return truth_pivot.at[ts_p, sym]
                    # 如果找不到前一分钟，使用 5 分钟前累计 ROC (CSV 中已带)
                    return row['price'] / (1.0 + row['roc_5m'])
            except:
                pass
            return row['price']

        df_full['price_prev_1m'] = df_full.apply(get_prev_price, axis=1)
        df_full['prev_roc_1m'] = (df_full['price'] - df_full['price_prev_1m']) / (df_full['price_prev_1m'] + 1e-9)

        # 调试输出前 5 条样本
        print("\n🔍 调试：前 5 条信号对齐样本:")
        print(df_full[['timestamp', 'symbol', 'price', 'price_prev_1m', 'prev_roc_1m']].head(5))

        print("\n" + "="*65)
        print(f"📊 CONFIDENCE THRESHOLD SCAN (Event Prob)")
        print("="*65)
        print(f"{'Prob >':<8} | {'Trades':<8} | {'Prev ROC':<10} | {'Next ROC':<10} | {'Win Rate':<8}")
        print("-" * 65)
        
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
        for p_limit in thresholds:
            subset = df_full[df_full['event_prob'] > p_limit]
            if subset.empty: continue
            
            avg_prev = subset['prev_roc_1m'].mean() * 100
            avg_next = subset['true_roc_5m'].mean() * 100
            win_rate = (subset['true_roc_5m'] > 0).mean() * 100
            print(f"{p_limit:<8.1f} | {len(subset):<8} | {avg_prev:<9.4f}% | {avg_next:<9.4f}% | {win_rate:<7.1f}%")
        
        # 🚀 [NEW] 趋势确认分析 (Trend Confirmation)
        # 过滤：既要由于预兆 (Alpha 高)，又要已经启动 (ROC_1m > 0)，但还未过热 (ROC_5m < 0.4%)
        print("\n" + "="*50)
        print(f"📈 TREND CONFIRMATION (Alpha>1.0 & ROC_1m>0.0% & ROC_5m<0.3%)")
        print("="*50)
        
        # 注意：由于 prev_roc_1m 之前 debug 时可能还有对齐问题，这里我们综合判断
        subset_trend = df_full[(df_full['alpha'] > 1.0) & (df_full['roc_5m'] > 0) & (df_full['roc_5m'] < 0.003)]
        if not subset_trend.empty:
            print(f"趋势确认交易数: {len(subset_trend)}")
            print(f"平均 5m 收益 (Next ROC): {subset_trend['true_roc_5m'].mean()*100:.4f}%")
            print(f"胜率 (Win Rate): {(subset_trend['true_roc_5m'] > 0).mean()*100:.1f}%")
        else:
            print("⚠️ 信号太少，无法进行趋势分析。")

        # 🚀 [NEW] 黄金门控可行性分析 (Golden Gate Feasibility)
        # 目的：排查为什么回测 0 交易
        print("\n" + "="*50)
        print(f"🔬 GOLDEN GATE FEASIBILITY (Prob>0.7 vs ROC_5m)")
        print("="*50)
        subset_p7 = df_full[df_full['event_prob'] > 0.7]
        if not subset_p7.empty:
            print(f"Prob>0.7 总信号数: {len(subset_p7)}")
            print(f"ROC_5m 均值: {subset_p7['roc_5m'].mean()*100:.4f}%")
            print(f"ROC_5m 最小值: {subset_p7['roc_5m'].min()*100:.4f}%")
            print(f"ROC_5m 最大值: {subset_p7['roc_5m'].max()*100:.4f}%")
            
            # 统计通过门控的数量
            passed = subset_p7[(abs(subset_p7['roc_5m']) > 0.0001) & (abs(subset_p7['roc_5m']) < 0.003)]
            print(f"能通过 Trend Gate (0.01%<|ROC|<0.3%) 的信号数: {len(passed)}")
            if len(passed) > 0:
                print(f"通过后平均 5m 收益: {passed['true_roc_5m'].mean()*100:.4f}%")
        else:
            print("⚠️ 没有 Prob > 0.7 的信号。")
        print("="*50)

        # 🚀 [NEW] 长周期收益分析 (30m Horizon)
        print("\n" + "="*50)
        print(f"⏳ LONG-HORIZON AUDIT (Next 30m Return)")
        print("="*50)
        all_30m = []
        for ts in timestamps:
            ts_30 = ts + 1800
            batch_alpha = df_alpha[df_alpha['timestamp'] == ts]
            batch_30 = df_truth[df_truth['ts'] == ts_30][['symbol', 'close']].rename(columns={'close': 'close_30m'})
            if batch_30.empty: continue
            batch = pd.merge(batch_alpha, batch_30, on='symbol', how='inner')
            batch['roc_30m'] = (batch['close_30m'] - batch['price']) / batch['price']
            all_30m.append(batch)
        
        if all_30m:
            df_30 = pd.concat(all_30m)
            top_30 = df_30[df_30['event_prob'] > 0.7]
            print(f"30m 持有交易数 (Prob>0.7): {len(top_30)}")
            print(f"平均 30m 收益: {top_30['roc_30m'].mean()*100:.4f}%")
            print(f"30m 胜率: {(top_30['roc_30m'] > 0).mean()*100:.1f}%")
        print("="*50)

if __name__ == "__main__":
    audit_signals()
