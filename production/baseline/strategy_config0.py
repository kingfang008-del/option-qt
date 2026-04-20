from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class StrategyConfig:
    """
    StrategyConfig0 - 对应 StrategyCoreV0 (V15 像素级复刻版) 的参数集
    抽象方式与最新的 StrategyConfig 一致，但参数值回退到 V0 版本。
    """
    
    # ================= 1. Capital Management =================
    # 默认沿用基准配置，V0 核心主要专注信号逻辑
    INITIAL_ACCOUNT: float = 50000.0
    MAX_POSITIONS: int = 4
    POSITION_RATIO: float = 0.25
    MAX_TRADE_CAP: float = 150000.0
    GLOBAL_EXPOSURE_LIMIT: float = 0.90
    COMMISSION_PER_CONTRACT: float = 0.65

    # ================= 2. Trading Session =================
    # V0 逻辑: 9:45 开始，15:30 禁入，15:40 离场
    START_TIME: str = "09:45:00"
    NO_ENTRY_TIME: str = "15:30:00"
    CLOSE_TIME: str = "15:40:00"
    
    START_HOUR: int = 9
    START_MINUTE: int = 45           
    NO_ENTRY_HOUR: int = 15
    NO_ENTRY_MINUTE: int = 30
    CLOSE_HOUR: int = 15
    CLOSE_MINUTE: int = 40           
    
    # ================= 3. Entry Thresholds =================
    # V0 逻辑: VOL_MIN 为 -1 (更宽松)，ALPHA_ENTRY 为 0.85
    VOL_MIN_Z: float = -1          
    VOL_MAX_Z: float = 4.0           
    ALPHA_ENTRY_THRESHOLD: float = 0.8
    ALPHA_ENTRY_STRICT: float = 1.2
    MIN_CS_ALPHA_Z: float = 0.5           
    
    # ================= 4. Momentum & Trend =================
    STOCK_MOMENTUM_TOLERANCE: float = 0.001  
    MIN_LAST_SNAP_ROC: float = 0.0001
    MAX_SNAP_ROC_LIMIT: float = 0.01          # V0 为 0.01 (1%)
    
    MIN_TREND_ROC: float = 0.0001
    MAX_TREND_ROC: float = 0.0030

    # ================= 5. Signal Logic (Rolling) =================
    ROLLING_WINDOW_MINS: int = 30
    CORR_THRESHOLD: float = -0.1
    
    # ================= 6. Risk & Event =================
    # V0 核心版本中对 Stock Hard Stop 的定义
    # [默认值] 平静市使用 0.003 / 0.005
    STOCK_HARD_STOP_TIGHT: float = 0.003
    STOCK_HARD_STOP_LOOSE: float = 0.005
    # [中间值] 轻度洗盘时先温和收紧，避免一刀切
    STOCK_HARD_STOP_TIGHT_MIXED: float = 0.0022
    STOCK_HARD_STOP_LOOSE_MIXED: float = 0.0040
    # [波动值] 横盘洗盘 / VIXY 扰动明显时收紧到 0.0015 / 0.003
    STOCK_HARD_STOP_TIGHT_VOLATILE: float = 0.0015
    STOCK_HARD_STOP_LOOSE_VOLATILE: float = 0.0030
    STOCK_HARD_STOP_EVENT: float = 0.008  
    EVENT_PROB_THRESHOLD: float = 0.7     
    EVENT_HODL_MINS: int = 30              
    
    COOLDOWN_MINUTES: int = 60
    CIRCUIT_BREAKER_THRESHOLD: int = 3
    CIRCUIT_BREAKER_MINUTES: int = 30
    MIN_OPTION_PRICE: float = 1.0

    # ================= 7. Liquidity =================
    MAX_SPREAD_PCT_ENTRY: float = 0.1         # V0 准入点差 10%
    MAX_SPREAD_PCT_EXIT: float = 0.2 
    MAX_SPREAD_DIVERGENCE: float = 0.02
    
    # ================= 8. Exit & Stop Loss =================
    # V0 止损较紧: -10% 常规，-15% 绝对
    STOP_LOSS: float = -0.10         
    ABSOLUTE_STOP_LOSS: float = -0.15
    TIME_STOP_MINS: int = 30          # V0 只有 30 分钟窗口
    TIME_STOP_ROI: float = 0.05       # 30 分钟内若未达到 5%，则可能离场
    ALPHA_FLIP_THRESHOLD: float = 0.8
    HIGH_CONFIDENCE_THRESHOLD: float = 1.2
    
    # ================= 9. Plan A: Smart Stop-Loss (Old Style) =================
    # V0 并不原生支持 Plan A 的 Grid 搜索，此处为兼容性占位
    EARLY_STOP_MINS: int = 5
    EARLY_STOP_ROI: float = -0.05
    NO_MOMENTUM_MINS: int = 5
    NO_MOMENTUM_MIN_MAX_ROI: float = 0.02

    # ================= 10. Execution Parameters =================
    SLIPPAGE_PCT: float = 0.001
    LIMIT_BUFFER_ENTRY: float = 1.03
    LIMIT_BUFFER_EXIT: float = 0.97
    ORDER_TIMEOUT_SECONDS: int = 5
    ORDER_MAX_RETRIES: int = 3

    # ================= 11. Profit Guards (Universal Ladder) =================
    # 旧版 V0 的第一档是 15% 才开始保利润，很多单到不了这里就被回撤吃掉。
    # 这里改成和 strategy_config.py 一致的 ladder 写法，并把第一档前移。
    LADDER_TIGHT: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.08, 0.05),
        (0.12, 0.08),
        (0.20, 0.15),
        (0.35, 0.28),
        (0.50, 0.40),
        (0.75, 0.60),
        (1.00, 0.85),
        (1.50, 1.30),
        (2.00, 1.75),
        (4.50, 3.80),
    ])
    LADDER_WIDE: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.15, 0.08),
        (0.30, 0.20),
        (0.50, 0.38),
        (0.80, 0.65),
        (1.50, 1.25),
        (2.50, 2.10),
        (5.00, 4.20),
    ])
    FLASH_PROTECT_TRIGGER: float = 0.05
    FLASH_PROTECT_EXIT: float = 0.02
    TRAILING_TRIGGER_ROI: float = 5.50
    TRAILING_KEEP_RATIO: float = 0.92
    COUNTER_TREND_PROTECT_TRIGGER: float = 0.25
    COUNTER_TREND_PROTECT_EXIT: float = 0.10
    MACD_FADE_MIN_ROI: float = 0.03            # V0 动能衰减门槛极低 (3%)
    
    # ================= 12. Dynamic Strategy Logic =================
    DYNAMIC_LADDER_ENABLED: bool = False      # V0 为固定阶梯
    HIGH_ALPHA_WIDE_THRESHOLD: float = 2.5
    
    # ================= 13. Inactivity & Small Gain =================
    ZOMBIE_EXIT_MINS: int = 20
    COUNTER_TREND_MAX_MINS: int = 5
    INDEX_REVERSAL_EXIT_ENABLED: bool = True
    SMALL_GAIN_THRESHOLD: float = 0.08
    SMALL_GAIN_MINS: int = 15
    SMALL_GAIN_LOCKED_ROI: float = 0.04
    
    # ================= 14. MACD & Slow Bull =================
    MACD_HIST_CONFIRM_ENABLED: bool = True
    MACD_HIST_THRESHOLD: float = 0.05
    SLOW_BULL_CHANNEL_ENABLED: bool = False
    SLOW_BULL_MAX_VOL_Z: float = 0.5
    SLOW_BULL_ALPHA_THRESHOLD: float = 0.75
    SLOW_BULL_MACD_THRESHOLD: float = 0.02
    SLOW_BULL_MIN_INDEX_ROC: float = 0.0005
    
    INDEX_GUARD_ENABLED: bool = True
    INDEX_GUARD_SHORT_BLOCK_ENABLED: bool = True
    INDEX_ROC_THRESHOLD: float = -0.01

    # ================= 16. Market Regime Guard (Choppiness Filter) =================
    REGIME_GUARD_ENABLED: bool = False
    REGIME_ENTRY_GUARD_ENABLED: bool = True
    REGIME_ADAPTIVE_STOCK_STOP_ENABLED: bool = True
    REGIME_REVERSAL_THRESHOLD: int = 6         # 30分钟内 > 5次 0.15% 反转即拦截
    REGIME_WINDOW_MINS: int = 30
    REGIME_REVERSAL_PERCENT: float = 0.0015     # 0.15% 反转阈值
    REGIME_VIXY_ROC_THRESHOLD: float = 0.003   # VIXY 5分钟正向跳升超过 0.3% 时标记波动候选
    REGIME_REQUIRE_NEUTRAL_INDEX_FOR_ENTRY_GUARD: bool = True  # 只有大盘方向不清楚时才启用 regime 入场拦截
    REGIME_MIXED_SCORE_THRESHOLD: float = 0.60
    REGIME_VOLATILE_SCORE_THRESHOLD: float = 1.00
    REGIME_BAND_ENTER_CONFIRM_BARS: int = 2
    REGIME_BAND_EXIT_CONFIRM_BARS: int = 4

    # ================= 17. Guard Switches (V0 Context) =================
    ENTRY_MOMENTUM_GUARD_ENABLED: bool = True
    ENTRY_LIQUIDITY_GUARD_ENABLED: bool = True
    
    EXIT_COUNTER_TREND_ENABLED: bool = True
    EXIT_INDEX_REVERSAL_ENABLED: bool = True
    EXIT_STOCK_HARD_STOP_ENABLED: bool = True
    EXIT_ZOMBIE_STOP_ENABLED: bool = True
    EXIT_MACD_FADE_ENABLED: bool = True
    EXIT_SIGNAL_FLIP_ENABLED: bool = True
    EXIT_LIQUIDITY_GUARD_ENABLED: bool = True
    EXIT_COND_STOP_ENABLED: bool = True
    EXIT_SMALL_GAIN_ENABLED: bool = True

    # ================= 18. 1s 高频平仓确认 =================
    # 仅用于秒级回测/高频执行路径，避免被秒级盘口噪声过早打出场
    EXIT_CONFIRM_SECONDS_1S: int = 8
    EXIT_CONFIRM_REASON_PREFIXES: Tuple[str, ...] = (
        "HARD_STOP",
        "COND_STOP",
        "TRAILING_",
        "STEP_PROT_",
        "FLASH_PROT_",
        "PROTECT_COUNTER",
        "TIME_STOP",
        "SMALL_GAIN_",
        "MACD_FADE",
        "STOCK_STOP",
        "ZOMBIE_STOP",
        "SPREAD_STOP",
    )
    
    PARITY_STRICT_MODE: bool = True           # V0 通常代表严格基准模式
