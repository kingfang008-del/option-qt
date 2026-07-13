import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import os
import concurrent.futures
import multiprocessing

# 配置全局日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入你写好的选约逻辑
ANCHOR_CONFIG = {
    # --- Front (近月) ---
    # 实盘逻辑: today + 7~13 days
    'FRONT_TARGET_DTE': 9,        # 理想锚点 (取中间值)
    'FRONT_MIN_DTE': 5,           # 最小允许
    'FRONT_MAX_DTE': 16,          # 最大允许
    
    # --- Next (次月) ---
    # 实盘逻辑: Front + 28 days -> 也就是 35~42 DTE 左右
    'NEXT_TARGET_DTE': 37,        # 理想锚点 (9 + 28)
    'NEXT_MIN_DTE': 25,           # 最小允许 (必须与 Front 拉开差距)
    'NEXT_MAX_DTE': 60,           # 最大允许
    
    # 选合约时的 Delta 参考标准
    'ATM_CENTER': 0.50,
    'OTM_CENTER': 0.25 
}
 
def get_daily_locked_contracts(df):
    """
    【核心逻辑：日内锁定选合约 - 实盘锚点对齐版】
    参考实盘逻辑: 
    1. Front = Today + ~1 week
    2. Next  = Front + ~4 weeks (28 days)
    确保 Front 和 Next 有足够的时间间隔，从而计算出正确的 Term Structure。
    """
    # 1. 辅助列
    df['date_str'] = df['timestamp'].dt.date.astype(str)
    df['abs_delta'] = df['delta'].abs()
    
    # 2. 预筛选 (宽泛范围)
    mask_dte = (df['dte'] >= 2) & (df['dte'] <= 90)
    candidates = df[mask_dte].copy()
    
    if candidates.empty: return None

    locked_map = [] 
    
    # 按天遍历
    for date_val, daily_group in candidates.groupby('date_str'):
        # 获取该天所有可选的 DTE
        available_dtes = daily_group['dte'].unique()
        if len(available_dtes) == 0: continue
        
        # --- A. 确定当天的 Front DTE 和 Next DTE (核心算法) ---
        
        # 1. 寻找 Front (目标: 9 DTE)
        # 逻辑: 找绝对距离最近的
        front_target = ANCHOR_CONFIG['FRONT_TARGET_DTE']
        front_options = [d for d in available_dtes if ANCHOR_CONFIG['FRONT_MIN_DTE'] <= d <= ANCHOR_CONFIG['FRONT_MAX_DTE']]
        
        if not front_options:
            # 兜底: 如果没有完美的 7-14 天，就找所有 DTE 中最小的那个（但不能是末日轮 < 3）
            valid_mins = [d for d in available_dtes if d >= 3]
            if not valid_mins: continue
            selected_front_dte = min(valid_mins)
        else:
            # 选离 Target 最近的
            selected_front_dte = min(front_options, key=lambda x: abs(x - front_target))
            
        # 2. 寻找 Next (目标: Front + 28)
        # 逻辑: 实盘是 Front Date + 28 days
        next_target = selected_front_dte + 28
        
        # 搜索范围: [Front+20, Front+45] 确保拉开差距
        min_next = selected_front_dte + 20
        max_next = selected_front_dte + 50
        
        next_options = [d for d in available_dtes if min_next <= d <= max_next]
        
        if not next_options:
            # 兜底: 只要比 Front 大 15 天以上就行
            fallbacks = [d for d in available_dtes if d > selected_front_dte + 15]
            if not fallbacks:
                # 实在没有次月合约，为了防止报错，Next = Front (Term Structure = 0)
                selected_next_dte = selected_front_dte
            else:
                # 选最小的那个（离 Front 最近的远期）
                selected_next_dte = min(fallbacks)
        else:
            # 选离 Target 最近的
            selected_next_dte = min(next_options, key=lambda x: abs(x - next_target))

        # ---------------------------------------------------------
        
        # 预计算 Volume Ranks
        volume_ranks = daily_group.groupby('contract_symbol')['volume'].sum()
        
        # 定义 6 个目标桶
        targets = [
            (0, True, False, 0.50), # Front Put ATM
            (1, True, False, 0.25), # Front Put OTM
            (2, True,  True, 0.50), # Front Call ATM
            (3, True,  True, 0.25), # Front Call OTM
            (4, False, False, 0.50), # Next Put ATM
            (5, False,  True, 0.50)  # Next Call ATM
        ]
        
        for b_id, is_front, is_call, target_delta in targets:
            # 1. 确定 DTE
            target_dte = selected_front_dte if is_front else selected_next_dte
            
            # 2. 筛选
            type_str = 'Call' if is_call else 'Put'
            
            # 组合掩码 (DTE + Type)
            # 注意: 这里使用严格的 DTE 匹配，因为我们已经在上面选好了具体的 DTE
            mask = (daily_group['dte'] == target_dte) & \
                   (daily_group['contract_type'].astype(str).str.upper().apply(lambda x: x.startswith(type_str[0].upper())))
            
            subset = daily_group[mask]
            if subset.empty: continue
            
            # 3. Delta 筛选 (寻找最接近 Target Delta 的)
            # 复用实盘逻辑: 找 Spot +/- 20% (对应 Delta 也就是找 Target +/- 0.15 左右)
            subset = subset.copy()
            subset['delta_dist'] = (subset['abs_delta'] - target_delta).abs()
            
            # 优先选 Delta 准的 (< 0.15 偏差)
            delta_candidates = subset[subset['delta_dist'] < 0.15]
            
            if delta_candidates.empty:
                # 如果没有满足条件的，直接选 Delta 偏差最小的那一个
                best_ticker = subset.sort_values('delta_dist').iloc[0]['contract_symbol']
            else:
                # 都有满足条件的，选最接近目标 Delta 的
                best_ticker = delta_candidates.sort_values('delta_dist').iloc[0]['contract_symbol']
            
            locked_map.append({
                'date_str': date_val,
                'contract_symbol': best_ticker,
                'bucket_id': b_id
            })
    
    return pd.DataFrame(locked_map)


def process_single_file(args):
    """
    [独立的 Worker 进程]
    负责单个 Parquet 文件的加载、时区清洗、DTE 计算和锁定逻辑。
    返回: (成功提取的 DataFrame, 报错信息字符串)
    """
    file_path, sym = args
    
    if "high_features" in file_path.name:
        return None, None
        
    try:
        df = pd.read_parquet(file_path)
        if df.empty: 
            return None, None
        
        # 1. 兼容列名
        rename_map = {'expiration_date': 'expiration', 'strike_price': 'strike', 'ticker': 'contract_symbol'}
        df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
        
        # 2. 🌟 强制时间清洗与时区对齐
        for col in ['timestamp', 'expiration']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            else:
                df[col] = df[col].dt.tz_convert('America/New_York')
        
        # 3. 🌟 安全计算 DTE
        df['dte'] = (df['expiration'].dt.normalize() - df['timestamp'].dt.normalize()).dt.days.fillna(-1).astype(int)

        # 4. 核心：执行选约算法
        locked_df = get_daily_locked_contracts(df)
        
        if locked_df is not None and not locked_df.empty:
            locked_df['symbol'] = sym
            return locked_df, None
            
    except Exception as e:
        return None, f"🚨 [报错] 处理文件 {file_path.name} 时发生错误: {str(e)}"
        
    return None, None


import argparse

def main():
    parser = argparse.ArgumentParser(description="Build Target Map with Date Filtering")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    RAW_DIR = Path("/home/kingfang007/train_data/nq_options_day_iv/")
    OUTPUT_FILE = Path("/home/kingfang007/train_data/locked_targets_map.parquet")
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        from config import TARGET_SYMBOLS
    except ImportError:
        logger.error("❌ 无法导入 TARGET_SYMBOLS，请检查 config.py 是否存在。")
        return
        
    logger.info(f"📡 开启多进程雷达：扫描历史数据 [{args.start_date or 'Earliest'} -> {args.end_date or 'Latest'}]...")
    
    # 1. 收集所有任务
    tasks = []
    logger.info("正在扫描本地文件系统构建任务队列...")
    for sym in TARGET_SYMBOLS:
        src_dir = RAW_DIR / sym
        if not src_dir.exists(): 
            continue
        
        files = list(src_dir.glob(f"{sym}_*.parquet"))
        for p in files:
            # 文件名格式如: AAPL_2026-03-18.parquet
            try:
                file_date_str = p.stem.split('_')[-1]
                if args.start_date and file_date_str < args.start_date: continue
                if args.end_date and file_date_str > args.end_date: continue
            except Exception:
                pass
                
            tasks.append((p, sym))
            
    total_tasks = len(tasks)
    if total_tasks == 0:
        logger.error("❌ 未找到任何待处理的 Parquet 文件。")
        return
        
    logger.info(f"📦 成功构建任务队列，总计 {total_tasks} 个日切片文件。")

    # 2. 开启多进程池并发处理
    # 自动获取当前机器的 CPU 核心数，你也可以手动指定如 max_workers=8 或 16
    cpu_cores = multiprocessing.cpu_count()
    # 建议保留 1-2 个核心给系统，防止机器卡死
    workers = max(1, cpu_cores - 2) 
    logger.info(f"🚀 启动 ProcessPoolExecutor，并发进程数: {workers}")
    
    all_targets = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # 使用 executor.map 可以高效批量处理并保持相对顺序
        # 通过 tqdm 包装实现平滑进度条
        for result_df, error_msg in tqdm(executor.map(process_single_file, tasks), total=total_tasks, desc="⚡ 全局处理进度"):
            if error_msg:
                tqdm.write(error_msg)
            if result_df is not None:
                all_targets.append(result_df)

    # 3. 合并与保存结果
    if all_targets:
        logger.info("正在合并所有结果并落盘...")
        final_map = pd.concat(all_targets, ignore_index=True)
        final_map.to_parquet(OUTPUT_FILE, compression='zstd', index=False)
        logger.info(f"🎉 雷达扫描大功告成！全市场共锁定 {len(final_map):,} 个需要精准下载的合约，已保存至 {OUTPUT_FILE}")
    else:
        logger.error("❌ 任务执行完毕，但未找到任何符合条件的合约。")

if __name__ == "__main__":
    main()