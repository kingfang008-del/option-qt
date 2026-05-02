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
    MAX_POSITIONS: int = 2
    POSITION_RATIO: float = 0.5
    MAX_TRADE_CAP: float = 150000.0
    GLOBAL_EXPOSURE_LIMIT: float = 0.90
    COMMISSION_PER_CONTRACT: float = 0.65

    # ================= 2. Trading Session =================
    # V0 逻辑: 9:45 开始，15:30 禁入，15:40 离场
    START_TIME: str = "09:45:00"
    NO_ENTRY_TIME: str = "15:30:00"
    CLOSE_TIME: str = "15:50:00"
    
    START_HOUR: int = 9
    START_MINUTE: int = 45
    NO_ENTRY_HOUR: int = 15
    NO_ENTRY_MINUTE: int = 30
    CLOSE_HOUR: int = 15
    CLOSE_MINUTE: int = 50
    
    # ================= 3. Entry Thresholds =================
    # V0 逻辑: VOL_MIN 为 -1 (更宽松)，ALPHA_ENTRY 为 0.85
    VOL_MIN_Z: float = -1          
    VOL_MAX_Z: float = 4.0           
    ALPHA_ENTRY_THRESHOLD: float = 0.4
    ALPHA_ENTRY_STRICT: float = 1.2
    MIN_CS_ALPHA_Z: float = 0.5           
    
    # ================= 4. Momentum & Trend =================
    STOCK_MOMENTUM_TOLERANCE: float = 0.001  
    # 强趋势标的经常在 alpha 已经很强时出现 1m 微回撤；
    # 这里允许 -3bp 的瞬时回撤，只拦明显逆向，而不是要求每分钟必须单边延续。
    MIN_LAST_SNAP_ROC: float = -0.0003
    MAX_SNAP_ROC_LIMIT: float = 0.01          # V0 为 0.01 (1%)
    # 入场排序从单纯 alpha/iv 改为利润潜力导向:
    # - alpha 非线性加权，提高真正强信号的排名
    # - alpha 已是全市场横截面 z-score，IV 不再作为排序主除数
    ENTRY_RANK_ALPHA_POWER: float = 1.35
    ENTRY_RANK_IV_PENALTY_POWER: float = 0.5
    ENTRY_RANK_HIGH_ALPHA_FLOOR: float = 1.20
    ENTRY_RANK_HIGH_ALPHA_BONUS_SCALE: float = 0.35
    ENTRY_RANK_HIGH_ALPHA_MAX_BONUS: float = 0.50
    ENTRY_RANK_ROC_ABS_SCALE: float = 100.0
    ENTRY_RANK_STOCK_ROC_SCALE: float = 120.0
    ENTRY_RANK_STOCK_ROC_MAX_BONUS: float = 0.35
    ENTRY_RANK_SNAP_ROC_SCALE: float = 200.0
    ENTRY_RANK_SNAP_ROC_MAX_BONUS: float = 0.30
    ENTRY_RANK_MACD_SCALE: float = 8.0
    ENTRY_RANK_MACD_MAX_BONUS: float = 0.30
    # 趋势质量只做温和降权/小幅加分: alpha 仍是主轴，避免高 alpha 但正股路径来回织布的候选排太前。
    ENTRY_RANK_TREND_QUALITY_ENABLED: bool = True
    ENTRY_RANK_TREND_WINDOW_MINS: int = 30
    ENTRY_RANK_TREND_MIN_OBS: int = 16
    ENTRY_RANK_TREND_NET_TARGET: float = 0.012
    ENTRY_RANK_TREND_QUALITY_FLOOR: float = 0.25
    ENTRY_RANK_TREND_QUALITY_BOOST: float = 0.06
    ENTRY_RANK_TREND_QUALITY_PENALTY: float = 0.04
    ENTRY_RANK_TREND_MIN_MULT: float = 0.96
    ENTRY_RANK_TREND_MAX_MULT: float = 1.06
    # 对“信号强 + 方向确认完整”的候选做通用优先级提升，而不是硬编码某几个标的。
    ENTRY_PRIORITY_RESERVED_SLOTS: int = 1
    ENTRY_PRIORITY_ALPHA_FLOOR: float = 0.9
    ENTRY_PRIORITY_BOOST: float = 0.80
    ENTRY_PRIORITY_STOCK_ROC_FLOOR: float = 0.0002
    ENTRY_PRIORITY_STOCK_BONUS: float = 0.25
    ENTRY_PRIORITY_SNAP_ROC_FLOOR: float = 0.0
    ENTRY_PRIORITY_SNAP_BONUS: float = 0.15
    ENTRY_PRIORITY_MACD_FLOOR: float = 0.01
    ENTRY_PRIORITY_MACD_BONUS: float = 0.20
    ENTRY_PRIORITY_MIN_CONFIRMATIONS: int = 2
    # CALL/PUT 分池排序：避免强负 alpha PUT 与慢涨 CALL 在同一个 abs(alpha) 池里抢光入场名额。
    ENTRY_DIRECTION_SPLIT_POOL_ENABLED: bool = True

    # ================= 4b. Trend Hunter Core =================
    # 启用方式: STRATEGY_CORE_VERSION=TREND。该核心把 TFT 作为雷达/确认器，
    # 主交易条件改为“大盘 + 个股 + MACD + 路径效率”的趋势状态。
    TREND_CORE_ALLOW_SHORT: bool = True
    TREND_CORE_BLOCK_VOLATILE_REGIME: bool = True
    TREND_CORE_ALLOW_MIXED_REGIME: bool = True
    TREND_CORE_MIN_ALPHA_ABS: float = 0.35
    TREND_CORE_ALPHA_ALIGN_MIN_ABS: float = 0.80
    TREND_CORE_MIN_INDEX_ROC: float = 0.00015
    TREND_CORE_MIN_STOCK_ROC: float = 0.00045
    TREND_CORE_MIN_SNAP_ROC: float = -0.00012
    TREND_CORE_MIN_MACD_HIST: float = 0.010
    TREND_CORE_WINDOW_MINS: int = 30
    TREND_CORE_MIN_OBS: int = 16
    TREND_CORE_MIN_NET: float = 0.0040
    TREND_CORE_MIN_EFFICIENCY: float = 0.22
    TREND_CORE_MIN_R2: float = 0.08
    TREND_CORE_STRONG_NET: float = 0.010
    TREND_CORE_SCORE_ALPHA_WEIGHT: float = 0.35
    TREND_CORE_SCORE_TREND_WEIGHT: float = 1.00
    TREND_CORE_SCORE_MOMENTUM_WEIGHT: float = 0.65

    # Trend core exits: 入场可以等确认，出错必须快。ROI 口径沿用 OMS 当前可成交/公平价。
    # 与 STOP_LOSS / ABSOLUTE_STOP_LOSS 在 strategy_core_trend 中取 min(更负) 合并，默认与 V0 一致。
    TREND_EXIT_STOP_LOSS: float = -0.10
    TREND_EXIT_ABSOLUTE_STOP_LOSS: float = -0.15
    TREND_EXIT_STOCK_ADVERSE_ROC: float = 0.0040
    TREND_EXIT_SNAP_BREAK: float = 0.0010
    TREND_EXIT_MACD_BREAK: float = 0.010
    TREND_EXIT_INDEX_BREAK_MIN_MINS: float = 1.0
    TREND_EXIT_NO_PROGRESS_MINS: float = 3.0
    TREND_EXIT_NO_PROGRESS_ROI: float = 0.00
    TREND_EXIT_TIME_STOP_MINS: float = 15.0
    TREND_EXIT_TIME_STOP_ROI: float = 0.05
    TREND_EXIT_MAX_HOLD_MINS: float = 30.0
    TREND_EXIT_PROTECT_TRIGGER: float = 0.12
    TREND_EXIT_PROTECT_FLOOR: float = 0.04
    # 峰值 ROI 达 trigger 后，若当前 ROI 低于 max_roi * keep 则平仓（与 V0 TRAILING_KEEP_RATIO 同语义）。
    # keep=0.88：从峰值回撤约 12% 期权 ROI 才触发 trail 卖压，与下方 getattr 默认一致。
    TREND_EXIT_TRAIL_TRIGGER: float = 0.22
    TREND_EXIT_TRAIL_KEEP: float = 0.80
    
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
    MIN_OPTION_PRICE: float = 2.0

    # ================= 7. Liquidity =================
    MAX_SPREAD_PCT_ENTRY: float = 0.10        # 兼容保留：未区分方向时的默认准入点差
    MAX_SPREAD_PCT_ENTRY_CALL: float = 0.08   # V0 做多(CALL)开仓点差上限 8%
    MAX_SPREAD_PCT_ENTRY_PUT: float = 0.10    # V0 做空(PUT)开仓点差上限 10%
    MAX_SPREAD_PCT_EXIT: float = 0.2 
    MAX_SPREAD_DIVERGENCE: float = 0.02
    
    # ================= 8. Exit & Stop Loss =================
    # V0 止损较紧: -10% 常规，-15% 绝对
    STOP_LOSS: float = -0.10         
    ABSOLUTE_STOP_LOSS: float = -0.15
    MID_TIME_STOP_MINS: int = 15      # 15 分钟未走强则提前离场
    MID_TIME_STOP_ROI: float = 0.05   # 15 分钟时当前收益若仍低于 5%，则离场
    TIME_STOP_MINS: int = 30          # V0 的长期时间止损窗口
    TIME_STOP_ROI: float = 0.05       # 30 分钟时当前收益若仍低于 5%，则离场
    ALPHA_FLIP_THRESHOLD: float = 0.8
    HIGH_CONFIDENCE_THRESHOLD: float = 1.2
    
    # ================= 9. Plan A: Smart Stop-Loss (Old Style) =================
    # V0 并不原生支持 Plan A 的 Grid 搜索，此处为兼容性占位
    EARLY_STOP_MINS: int = 5
    EARLY_STOP_ROI: float = -0.05
    NO_MOMENTUM_MINS: int = 5
    NO_MOMENTUM_MIN_MAX_ROI: float = 0.02

    # ================= 10. Execution Parameters =================
    SLIPPAGE_PCT: float = 0.002
    LIMIT_BUFFER_ENTRY: float = 1.03
    LIMIT_BUFFER_EXIT: float = 0.97
    ORDER_TIMEOUT_SECONDS: int = 3
    ORDER_MAX_RETRIES: int = 3
    EXIT_ORDER_MAX_RETRIES: int = 10
    EXIT_UNFILLED_RETRY_FRAMES: int = 3
    # 平仓快速重报: 节奏放慢到 IB 端能稳定 ack 的节拍 (cancel→ack ~150ms / new→ack ~150ms)。
    # 0.25s 间隔在实盘会因为 cancel 还没 ack 就再下单, 触发 IB 拒绝或重复挂单。
    # 普通 fast_requote 以低滑点为优先: 挂在 bid 附近等待成交, 不主动跌破 bid。
    # 真正风险止损由 STOP_EXIT_FAST_* 接管, 会使用更激进的价格和 MKT fallback。
    EXIT_FAST_REQUOTE_MODE_ENABLED: bool = True
    EXIT_FAST_REQUOTE_MAX_SECONDS: float = 3.0
    EXIT_FAST_REQUOTE_INTERVAL_SECONDS: float = 0.40
    EXIT_FAST_REQUOTE_CANCEL_SETTLE_SECONDS: float = 0.20
    EXIT_FAST_REQUOTE_INITIAL_BID_OFFSET: float = 0.0
    EXIT_FAST_REQUOTE_STEP: float = 0.01
    EXIT_FAST_REQUOTE_BASE_DISCOUNT: float = 0.03
    EXIT_FAST_REQUOTE_DISCOUNT: float = 0.01
    EXIT_FAST_REQUOTE_MIN_BID_RATIO: float = 0.97
    EXIT_FAST_REQUOTE_MAX_ABS_DISCOUNT: float = 0.05
    # 止损专用快速模式: 节奏比通用 fast_requote 更紧凑 (0.5s/次, 6 次), 价格更激进。
    # 触底后强制升级到 MKT 兜底, 避免亏损一直扩大。
    STOP_EXIT_FAST_MODE_ENABLED: bool = True
    STOP_EXIT_FAST_MAX_SECONDS: float = 3.0
    STOP_EXIT_FAST_INTERVAL_SECONDS: float = 0.50
    STOP_EXIT_FAST_CANCEL_SETTLE_SECONDS: float = 0.30
    STOP_EXIT_FAST_INITIAL_BID_OFFSET: float = 0.01
    STOP_EXIT_FAST_REQUOTE_STEP: float = 0.03
    STOP_EXIT_FAST_BASE_DISCOUNT: float = 0.06
    STOP_EXIT_FAST_REQUOTE_DISCOUNT: float = 0.03
    STOP_EXIT_FAST_MIN_BID_RATIO: float = 0.90
    STOP_EXIT_FAST_MAX_ABS_DISCOUNT: float = 0.15
    STOP_EXIT_FAST_MKT_FALLBACK_ENABLED: bool = True
    STOP_EXIT_FAST_MKT_FALLBACK_WAIT_SECONDS: float = 2.0
    STOP_EXIT_FAST_FLOOR_STREAK_THRESHOLD: int = 2
    # 兼容旧字段(已被 INTERVAL_SECONDS / MAX_SECONDS 取代, 保留以防外部读取)
    STOP_EXIT_FAST_MAX_RETRIES: int = 6
    STOP_EXIT_FAST_WAIT_SECONDS: int = 1
    # 入场快速重报: 与平仓节奏对齐, 让追价跟得上 ask 上跳。
    ENTRY_FAST_REQUOTE_MODE_ENABLED: bool = True
    ENTRY_FAST_REQUOTE_MAX_SECONDS: float = 3.0
    ENTRY_FAST_REQUOTE_INTERVAL_SECONDS: float = 0.40
    ENTRY_FAST_REQUOTE_CANCEL_SETTLE_SECONDS: float = 0.20
    # 冰山子单刷新 quote 开关: 每个 chunk 起点都重新读 bid/ask, 避免锁死旧价。
    ENTRY_ICEBERG_REFRESH_QUOTE_PER_CHUNK: bool = True
    # IBKR TWS API hard limit is commonly 50 msg/s; keep headroom for callbacks/manual actions.
    # 35 留足余量给 marketData/accountUpdate/手工操作; high priority 约 40 msg/s,
    # 并由 MAX_MESSAGES_PER_SECOND 硬封顶，避免贴近 IBKR 50 msg/s pacing 限制。
    IBKR_API_MAX_MESSAGES_PER_SECOND: int = 35
    IBKR_API_PACING_WINDOW_SECONDS: float = 1.0
    IBKR_API_PACING_SAFETY_SLEEP: float = 0.02
    IBKR_API_HIGH_PRIORITY_BOOST: float = 1.15
    IBKR_API_HIGH_PRIORITY_MAX_MESSAGES_PER_SECOND: int = 45

    # ================= 11. Profit Guards (Universal Ladder) =================
    # 旧版 V0 的第一档是 15% 才开始保利润，很多单到不了这里就被回撤吃掉。
    # 这里改成和 strategy_config.py 一致的 ladder 写法，并把第一档前移。
    # TREND 中端入场、峰值往往不大：收紧 (trigger, floor) 间距，减少从峰值回吐过多才离场。
    # 微利档：峰值曾 ≥5% 则当前净利跌破 2% 平仓（填补 5%～8% 峰值之间无阶梯空白）。
    LADDER_TIGHT: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.05, 0.02),
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
    COUNTER_TREND_MAX_MINS: int = 10
    INDEX_REVERSAL_EXIT_ENABLED: bool = True
    SMALL_GAIN_THRESHOLD: float = 0.08
    SMALL_GAIN_MINS: int = 15
    SMALL_GAIN_LOCKED_ROI: float = 0.04
    
    # ================= 14. MACD & Slow Bull =================
    MACD_HIST_CONFIRM_ENABLED: bool = True
    # 原 0.05 对高价慢趋势标的过严，容易把稳定爬升全部挡掉。
    # 保留方向确认，但降到 0.015，让慢涨行情能进入 OMS 后续风控。
    MACD_HIST_THRESHOLD: float = 0.015
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

    # ================= 18. Exit Frequency Control =================
    # 分钟级平仓信号模式:
    # True  -> 仅分钟级策略链路产生平仓信号; 秒级只做执行层成交推进
    # False -> 允许秒级风控链路(_process_exits/_process_fast_fused_tick)直接触发平仓
    EXIT_SIGNAL_MINUTE_ONLY: bool = True

    # 仅在启用秒级平仓判定时生效: 当前 OMS 主链路默认 minute-only，
    # 因此这组参数主要保留给归档/实验性 1s exit 路径使用。
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
