from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class StrategyConfig:
    # ================= 1. Capital Management =================
    # 从 config.py 迁移，确保回测与生产资金逻辑一致
    INITIAL_ACCOUNT: float = 50000.0
    MAX_POSITIONS: int = 4
    POSITION_RATIO: float = 0.25
    MAX_TRADE_CAP: float = 150000.0
    GLOBAL_EXPOSURE_LIMIT: float = 0.90
    COMMISSION_PER_CONTRACT: float = 0.65

    # ================= 2. Trading Session =================
    # 升级为字符串格式，适配 1s/1m 数据
    START_TIME: str = "09:35:00"
    NO_ENTRY_TIME: str = "15:30:00"
    CLOSE_TIME: str = "15:40:00"
    
    # 兼容老代码的整型字段
    START_HOUR: int = 9
    START_MINUTE: int = 35           
    NO_ENTRY_HOUR: int = 15
    NO_ENTRY_MINUTE: int = 30
    CLOSE_HOUR: int = 15
    CLOSE_MINUTE: int = 40           
    
    # ================= 3. Entry Thresholds =================
    VOL_MIN_Z: float = 0           
    VOL_MAX_Z: float = 4.0           
    ALPHA_ENTRY_THRESHOLD: float = 0.8   
    ALPHA_ENTRY_STRICT: float = 2      
    MIN_CS_ALPHA_Z: float = 0.5           
    
    # ================= 4. Momentum & Trend =================
    STOCK_MOMENTUM_TOLERANCE: float = 0.001  
    MIN_LAST_SNAP_ROC: float = 0.0001
    MAX_SNAP_ROC_LIMIT: float = 0.008
    
    MIN_TREND_ROC: float = 0.0001
    MAX_TREND_ROC: float = 0.0030

    # ================= 5. Signal Logic (Rolling) =================
    # 统一控制 Alpha 滚动窗口
    ROLLING_WINDOW_MINS: int = 30
    CORR_THRESHOLD: float = -0.1
    
    # ================= 6. Risk & Event =================
    STOCK_HARD_STOP_TIGHT: float = 0.002 
    STOCK_HARD_STOP_LOOSE: float = 0.004
    STOCK_HARD_STOP_EVENT: float = 0.008  
    EVENT_PROB_THRESHOLD: float = 0.7     
    EVENT_HODL_MINS: int = 30              
    
    # 熔断与冷却 (统一各处不一致的定义)
    COOLDOWN_MINUTES: int = 60
    CIRCUIT_BREAKER_THRESHOLD: int = 3
    CIRCUIT_BREAKER_MINUTES: int = 30
    MIN_OPTION_PRICE: float = 1.0

    # ================= 7. Liquidity =================
    MAX_SPREAD_PCT_ENTRY: float = 0.07    
    MAX_SPREAD_PCT_EXIT: float = 0.2 
    MAX_SPREAD_DIVERGENCE: float = 0.02
    
    # ================= 8. Exit & Stop Loss =================
    STOP_LOSS: float = -0.20         
    ABSOLUTE_STOP_LOSS: float = -0.20  # Plan A 推荐值
    TIME_STOP_MINS: int = 120         # Plan A 推荐值: 4小时，给赢家时间
    TIME_STOP_ROI: float = 0.0
    ALPHA_FLIP_THRESHOLD: float = 0.8
    HIGH_CONFIDENCE_THRESHOLD: float = 1.3
    
    # ================= 9. Plan A: Smart Stop-Loss (Grid-Optimized) =================
    EARLY_STOP_MINS: int = 5
    EARLY_STOP_ROI: float = -0.05
    NO_MOMENTUM_MINS: int = 5
    NO_MOMENTUM_MIN_MAX_ROI: float = 0.02

    # ================= 10. Execution Parameters =================
    SLIPPAGE_PCT: float = 0.001
    LIMIT_BUFFER_ENTRY: float = 1.03
    LIMIT_BUFFER_EXIT: float = 0.97
    ORDER_TIMEOUT_SECONDS: int = 30
    ORDER_MAX_RETRIES: int = 3

    # ================= 11. Profit Guards (Universal Ladder) =================
    # [TIGHT] Default production ladder
    LADDER_TIGHT: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.12, 0.10), (0.25, 0.20), (0.35, 0.30), (0.50, 0.40), 
        (0.75, 0.60), (1.00, 0.85), (2.00, 1.75), (4.50, 3.80)
    ])
    
    # [WIDE] Optional 'Runner-Friendly' ladder for high-conviction signals
    LADDER_WIDE: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.30, 0.20), (0.50, 0.40), (0.80, 0.65), (1.50, 1.25), 
        (2.50, 2.10), (5.00, 4.20)
    ])
    
    FLASH_PROTECT_TRIGGER: float = 0.05
    FLASH_PROTECT_EXIT: float = 0.02
    TRAILING_TRIGGER_ROI: float = 5.50
    TRAILING_KEEP_RATIO: float = 0.92
    COUNTER_TREND_PROTECT_TRIGGER: float = 0.25
    COUNTER_TREND_PROTECT_EXIT: float = 0.10
    MACD_FADE_MIN_ROI: float = 0.08
    
    # ================= 12. Dynamic Strategy Logic =================
    DYNAMIC_LADDER_ENABLED: bool = True
    HIGH_ALPHA_WIDE_THRESHOLD: float = 2.5
    
    # ================= 13. Inactivity & Small Gain =================
    ZOMBIE_EXIT_MINS: int = 20
    COUNTER_TREND_MAX_MINS: int = 5
    INDEX_REVERSAL_EXIT_ENABLED: bool = True
    SMALL_GAIN_THRESHOLD: float = 0.08
    SMALL_GAIN_MINS: int = 15
    SMALL_GAIN_LOCKED_ROI: float = 0.04
    
    # ================= 14. MACD & Slow Bull =================
    MACD_HIST_CONFIRM_ENABLED: bool = False
    MACD_HIST_THRESHOLD: float = 0.05
    SLOW_BULL_CHANNEL_ENABLED: bool = False
    SLOW_BULL_MAX_VOL_Z: float = 0.5
    SLOW_BULL_ALPHA_THRESHOLD: float = 0.75
    SLOW_BULL_MACD_THRESHOLD: float = 0.02
    SLOW_BULL_MIN_INDEX_ROC: float = 0.0005
    
    # ================= 15. Market & Logic Guards =================
    INDEX_GUARD_ENABLED: bool = True
    INDEX_GUARD_SHORT_BLOCK_ENABLED: bool = False
    INDEX_ROC_THRESHOLD: float = -0.01

    # ================= 16. Guard Switches (Parity Testing) =================
    # --- 开仓 Guard 开关 (s4 独有，plan_a 没有) ---
    ENTRY_MOMENTUM_GUARD_ENABLED: bool = False     # stock_roc / snap_roc 方向过滤
    ENTRY_LIQUIDITY_GUARD_ENABLED: bool = True     # bid/ask 价差守卫
    # MACD_HIST_CONFIRM_ENABLED 已在 §14      # MACD 柱确认
    # INDEX_GUARD_ENABLED 已在 §15             # 大盘方向守卫

    # --- 平仓 Guard 开关 (s4 独有，plan_a 没有) ---
    EXIT_COUNTER_TREND_ENABLED: bool = True        # 逆势超时平仓 (CT_TIMEOUT)
    EXIT_INDEX_REVERSAL_ENABLED: bool = True       # 大盘反转平仓 (IDX_REVERSAL)
    EXIT_STOCK_HARD_STOP_ENABLED: bool = True      # 正股硬止损 (STOCK_STOP)
    EXIT_ZOMBIE_STOP_ENABLED: bool = True          # 僵尸持仓平仓 (ZOMBIE_STOP)
    EXIT_MACD_FADE_ENABLED: bool = True            # MACD衰退平仓 (MACD_FADE)
    EXIT_SIGNAL_FLIP_ENABLED: bool = True          # Alpha反向平仓 (FLIP)
    EXIT_LIQUIDITY_GUARD_ENABLED: bool = True      # 出场价差守卫 (SPREAD_STOP)
    EXIT_COND_STOP_ENABLED: bool = True            # 条件止损 (COND_STOP)
    EXIT_SMALL_GAIN_ENABLED: bool = True           # 小利锁定平仓 (SMALL_GAIN_P)

    # --- plan_a 核心逻辑开关 (s4 需补充) ---
    EXIT_EARLY_STOP_ENABLED: bool = False          # plan_a: 5分钟内 roi < -5% 
    EXIT_NO_MOMENTUM_ENABLED: bool = False         # plan_a: 5分钟后 max_roi < 2%
    PARITY_STRICT_MODE: bool = False               # 开启此模式将强制使用严格的 1.45 阈值并禁用事件放行逻辑
    # ================= 15. Market Regime Guard (Choppiness Filter) =================
    REGIME_GUARD_ENABLED: bool = True
    REGIME_REVERSAL_THRESHOLD: int = 5
    REGIME_WINDOW_MINS: int = 30
    REGIME_REVERSAL_PERCENT: float = 0.0015
