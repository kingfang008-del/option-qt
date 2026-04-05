import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import json
# 导入你模型定义文件中的类
# 假设你的模型文件名是 trading_tft_stock_embed_new.py
from trading_tft_stock_embed_new import AdvancedAlphaNet, UnifiedLMDBDataset, collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Backtest")

class OutOfSampleDataset(UnifiedLMDBDataset):
    """
    扩展原有的 Dataset，支持按日期过滤样本
    """
    def __init__(self, db_path, config, start_date, end_date, seq_len=30):
        super().__init__(db_path, config, seq_len=seq_len)
        
        # 将日期转换为 Unix 时间戳 (纳秒/毫秒级，需根据你 LMDB 存储精度匹配)
        # s0 脚本中使用的是 NY 时间转 int64
        s_ts = pd.Timestamp(start_date, tz='America/New_York').value
        e_ts = pd.Timestamp(end_date, tz='America/New_York').value
        
        filtered_keys = []
        for k in self.keys:
            # 假设 key 格式为 b"SYMBOL_TIMESTAMP"
            _, ts_part = k.decode('ascii').rsplit('_', 1)
            ts = int(ts_part)
            if s_ts <= ts <= e_ts:
                filtered_keys.append(k)
        
        self.keys = filtered_keys
        logger.info(f"📅 日期筛选完成: {start_date} 至 {end_date} | 样本数: {len(self.keys)}")

def run_oos_backtest(model_path, db_path, config_path, start_date, end_date, device='cuda'):
    # 1. 加载配置和模型
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 这里的 caps 需要根据你训练时的逻辑确定，通常包含 stock_id 的最大值
    # 建议直接在模型保存时把 caps 存进 state_dict 或者从数据库实时获取
    caps = {'stock': 10826, 'sector': 20}
    
    model = AdvancedAlphaNet(config, caps).to(device)
   # --- 替换原有的加载逻辑 ---
    checkpoint = torch.load(model_path, map_location=device)
    
    # 自动识别保存格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logger.info("检测到封装格式，从 'model_state_dict' 加载...")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            logger.info("检测到封装格式，从 'state_dict' 加载...")
        else:
            # 如果字典里没有特定的键，可能整个字典就是 state_dict
            state_dict = checkpoint
            logger.info("将整个字典视为 state_dict 加载...")
    else:
        state_dict = checkpoint
        logger.info("直接从 checkpoint 加载 state_dict...")

    model.load_state_dict(state_dict)
    # -----------------------

    model.eval()
    logger.info(f"✅ 模型加载成功: {model_path}")

    # 2. 准备数据集
    ds = OutOfSampleDataset(db_path, config, start_date, end_date)
    dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 3. 推理阶段
    all_results = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="Inference"):
            if not batch: continue
            x_stk, x_opt, s, t, ts = batch
            
            out = model(x_stk.to(device), x_opt.to(device), {k:v.to(device) for k,v in s.items()})
            
            # 记录结果
            res = pd.DataFrame({
                'timestamp': ts,
                'pred_gamma': out['pred_gamma'].cpu().numpy(),
                'true_gamma': t['gamma_pnl_std'].numpy(),
                'pred_spread': out['pred_spread'].cpu().numpy(),
                'true_spread': t['iv_rv_spread'].numpy()
            })
            all_results.append(res)

    df = pd.concat(all_results)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # 4. IC 核心验证逻辑
    def calc_ic(group, p_col, t_col):
        if len(group) < 5: return np.nan
        return group[p_col].corr(group[t_col], method='spearman')

    # 按日计算 IC
    daily_ic = df.groupby('date').apply(lambda x: pd.Series({
        'gamma_ic': calc_ic(x, 'pred_gamma', 'true_gamma'),
        'spread_ic': calc_ic(x, 'pred_spread', 'true_spread')
    }))

    # 5. 打印报表
    print("\n" + "="*40)
    print(f"📊 样本外 IC 验证报告 ({start_date} - {end_date})")
    print("-" * 40)
    for col in ['gamma_ic', 'spread_ic']:
        ic_mean = daily_ic[col].mean()
        ic_std = daily_ic[col].std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        print(f"[{col.upper()}]")
        print(f"  Mean IC: {ic_mean:.4f}")
        print(f"  Std  IC: {ic_std:.4f}")
        print(f"  IC IR:   {ic_ir:.4f}")
        print(f"  Win Rate (IC>0): {(daily_ic[col] > 0).mean():.2%}")
    print("="*40 + "\n")

    # 6. 简易 P&L 模拟 (验证 Gamma 信号的盈利潜力)
    # 策略：每天做多 pred_gamma 最强的 Top 10% 标的
    df['gamma_rank'] = df.groupby('date')['pred_gamma'].rank(pct=True)
    strategy_returns = df[df['gamma_rank'] > 0.90].groupby('date')['true_gamma'].mean()
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    daily_ic['gamma_ic'].cumsum().plot(title="Cumulative Gamma IC")
    plt.subplot(1, 2, 2)
    strategy_returns.cumsum().plot(title="Simulated Strategy Returns (Top 10% Gamma)")
    plt.tight_layout()
    plt.savefig("backtest_results.png")
    plt.show()

    return daily_ic, df

if __name__ == "__main__":
    # 配置回测参数
    MODEL_FILE = "/home/kingfang007/quant_project/checkpoints_option_alpha/advanced_alpha_best.pth"
    LMDB_DATA = "/mnt/s990/data/h5_unified_overlap_id/test_quote_alpha.lmdb"
    CONFIG_JSON = "/home/kingfang007/notebook/train/slow_feature.json"
    
    # 指定样本外日期 (确保 LMDB 中包含这些日期)
    START = "2026-01-01"
    END   = "2026-03-18"
    
    run_oos_backtest(MODEL_FILE, LMDB_DATA, CONFIG_JSON, START, END)