import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm.asyncio import tqdm
from thetadata import ThetaClient, OptionReqType, OptionRight, DateRange, DataType

# ================= 配置参数 =================
SYMBOL = "SPY"
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 1, 31)

# --- 三维裁剪防线 (3D Pruning) ---
MAX_DTE = 45          # 期限裁剪：只下载 45 天内到期的合约 (砍掉长线期权)
STRIKE_PCT = 0.05     # 深度裁剪：只下载距离正股价格 ±5% 的行权价 (砍掉极端深度虚值/实值)
# 时间裁剪：在后面的 Pandas 处理中过滤 RTH (09:30 - 16:00)

OUTPUT_DIR = "./data_lake/options_1m_quotes"
# ============================================

async def fetch_daily_options(client: ThetaClient, current_date: datetime):
    """拉取单日符合裁剪条件的期权分钟级盘口数据"""
    date_range = DateRange(current_date, current_date)
    
    # 1. 获取当日正股(Underlying)的收盘价，作为动态锚定点
    try:
        # 假设这里获取正股日线数据来确定 ATM (平值) 价格
        # (实际 ThetaData API 中可用 DataType.OHLC 获取正股，此处简化为模拟价或前一日收盘价)
        stock_req = await client.get_hist_stock(
            req=OptionReqType.EOD, 
            root=SYMBOL, 
            date_range=date_range
        )
        if stock_req.empty: return None
        anchor_price = stock_req['close'].iloc[-1]
    except Exception as e:
        print(f"[{current_date.date()}] 获取正股基准价失败: {e}")
        return None

    # 2. 动态计算合法的行权价区间 (Strike Pruning)
    min_strike = anchor_price * (1 - STRIKE_PCT)
    max_strike = anchor_price * (1 + STRIKE_PCT)

    # 3. 获取当日所有存活的期权合约列表 (Exp, Strike, Right)
    routing_data = await client.get_expirations(root=SYMBOL)
    valid_contracts = []
    
    for exp_date in routing_data:
        # 期限裁剪 (DTE Pruning)
        dte = (exp_date - current_date.date()).days
        if 0 <= dte <= MAX_DTE:
            strikes = await client.get_strikes(root=SYMBOL, exp=exp_date)
            for strike in strikes:
                if min_strike <= strike <= max_strike:
                    valid_contracts.append((exp_date, strike, OptionRight.CALL))
                    valid_contracts.append((exp_date, strike, OptionRight.PUT))

    if not valid_contracts:
        return None

    # 4. 并发拉取合法合约的 1 分钟 Quote (包含 Bid/Ask)
    # ThetaData 允许按分钟聚合 Quote，极大地减小了体积
    tasks = []
    for exp, strike, right in valid_contracts:
        # 请求 1 分钟级别的 Quote
        task = client.get_hist_option(
            req=OptionReqType.QUOTE_1M, # 取决于具体 ThetaData API 版本，通常有 1M 聚合
            root=SYMBOL,
            exp=exp,
            strike=strike,
            right=right,
            date_range=date_range
        )
        tasks.append(task)

    # 限制并发量，防止压垮本地 Terminal 内存
    chunk_size = 50
    daily_dfs = []
    for i in range(0, len(tasks), chunk_size):
        chunk_results = await asyncio.gather(*tasks[i:i+chunk_size], return_exceptions=True)
        for df in chunk_results:
            if isinstance(df, pd.DataFrame) and not df.empty:
                daily_dfs.append(df)

    if not daily_dfs: return None
    
    daily_df = pd.concat(daily_dfs, ignore_index=True)
    return daily_df

def clean_and_build_features(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗与公允特征生成"""
    # 1. 时间裁剪 (Time Pruning) - 仅保留美东正常交易时间 (RTH)
    # 假设返回的 df 有 'datetime' 列
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.between_time('09:30', '16:00')
    df.reset_index(inplace=True)

    # 2. 剔除脏数据 (Bid/Ask 为 0 或倒挂的数据)
    df = df[(df['bid'] > 0) & (df['ask'] > 0) & (df['ask'] >= df['bid'])]

    # 3. 核心合成：计算公允特征 (Mid-Price 与 Spread)
    df['mid_price'] = (df['bid'] + df['ask']) / 2.0
    df['spread_pct'] = (df['ask'] - df['bid']) / df['mid_price']

    # 4. 计算微观结构特征 (订单簿失衡 Volume Imbalance)
    # 这对你未来的高频预测模型是个杀手级特征
    df['bid_size'] = df['bid_size'].astype(float)
    df['ask_size'] = df['ask_size'].astype(float)
    df['size_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-9)

    return df

async def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 初始化 ThetaData 客户端 (需要本地后台运行 Theta Terminal)
    client = ThetaClient()
    await client.connect()

    current_date = START_DATE
    dates_to_fetch = []
    while current_date <= END_DATE:
        if current_date.weekday() < 5: # 排除周末
            dates_to_fetch.append(current_date)
        current_date += timedelta(days=1)

    print(f"🚀 开始极速拉取 {SYMBOL} 期权 Quote 数据，共 {len(dates_to_fetch)} 个交易日...")

    for dt in tqdm(dates_to_fetch):
        try:
            raw_df = await fetch_daily_options(client, dt)
            if raw_df is not None and not raw_df.empty:
                
                # 执行清洗与特征融合
                clean_df = clean_and_build_features(raw_df)
                
                # 按照日期分区保存为 Parquet (极高压缩比)
                date_str = dt.strftime("%Y%m%d")
                save_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}_{date_str}.parquet")
                
                # 保留原始 bid/ask，供 orchestrator 回测模拟撮合使用
                clean_df.to_parquet(save_path, engine='pyarrow', compression='snappy')
                
        except Exception as e:
            print(f"❌ 处理日期 {dt.date()} 时发生错误: {e}")

    # 优雅关闭连接
    await client.close()
    print("✅ 数据下载与清洗完毕！")

if __name__ == "__main__":
    asyncio.run(main())