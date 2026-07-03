import os
from dataclasses import dataclass, field
from typing import List, Tuple

try:
    from config import V0_PROFIT_HYBRID_CONTINUOUS_ENABLED as _CFG_V0_PROFIT_HYBRID_CONTINUOUS
except Exception:  # pragma: no cover
    _CFG_V0_PROFIT_HYBRID_CONTINUOUS = False


@dataclass
class StrategyConfig:
    """
    StrategyConfig0 - 对应 StrategyCoreV0 (V15 像素级复刻版) 的参数集
    抽象方式与最新的 StrategyConfig 一致，但参数值回退到 V0 版本。
    """
    
    # ================= 1. Capital Management =================
    # QQQ-only：单仓；与 config.MAX_POSITIONS 对齐
    try:
        from config import MAX_POSITIONS as _CFG_MAX_POS
    except Exception:  # pragma: no cover
        _CFG_MAX_POS = 1
    INITIAL_ACCOUNT: float = 50000.0
    MAX_POSITIONS: int = int(_CFG_MAX_POS)
    POSITION_RATIO: float = 0.90 if MAX_POSITIONS <= 1 else 0.25
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
    # QQQ net_edge 绝对阈值（非截面 z-score；alpha_z 字段在 absolute 模式下即 net_edge）
    VOL_MIN_Z: float = -1
    VOL_MAX_Z: float = 4.0
    ALPHA_ENTRY_THRESHOLD: float = 0.015
    ALPHA_ENTRY_STRICT: float = 0.030
    MIN_CS_ALPHA_Z: float = 0.0  # 单标的：禁用截面 alpha 门槛

    # 单标的 QQQ：不做 CALL/PUT 分池抢名额、不做 priority 预留槽
    ENTRY_DIRECTION_SPLIT_POOL_ENABLED: bool = False
    ENTRY_PRIORITY_RESERVED_SLOTS: int = 0
    # net_edge 量级下的入场排序微调（仅当同帧多候选时生效；QQQ-only 通常 bypass）
    ENTRY_RANK_ALPHA_POWER: float = 1.0
    ENTRY_RANK_IV_PENALTY_POWER: float = 0.0
    ENTRY_RANK_HIGH_ALPHA_FLOOR: float = 0.020
    ENTRY_PRIORITY_ALPHA_FLOOR: float = 0.018
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
    # 与 TREND_CORE_WINDOW_MINS 对齐：过长窗口会把 net 憋成大波段后才放行，高 beta 标的体感「确诊太晚」
    ENTRY_RANK_TREND_WINDOW_MINS: int = 12
    ENTRY_RANK_TREND_MIN_OBS: int = 8
    ENTRY_RANK_TREND_NET_TARGET: float = 0.009
    ENTRY_RANK_TREND_QUALITY_FLOOR: float = 0.25
    ENTRY_RANK_TREND_QUALITY_BOOST: float = 0.06
    ENTRY_RANK_TREND_QUALITY_PENALTY: float = 0.04
    ENTRY_RANK_TREND_MIN_MULT: float = 0.96
    ENTRY_RANK_TREND_MAX_MULT: float = 1.06
    ENTRY_PRIORITY_BOOST: float = 0.80
    ENTRY_PRIORITY_STOCK_ROC_FLOOR: float = 0.0002
    ENTRY_PRIORITY_STOCK_BONUS: float = 0.25
    ENTRY_PRIORITY_SNAP_ROC_FLOOR: float = 0.0
    ENTRY_PRIORITY_SNAP_BONUS: float = 0.15
    ENTRY_PRIORITY_MACD_FLOOR: float = 0.01
    ENTRY_PRIORITY_MACD_BONUS: float = 0.20
    ENTRY_PRIORITY_MIN_CONFIRMATIONS: int = 2

    # 快通道门控（options_vw_spread / options_iv_momentum，来自 FCS payload）
    FAST_GATE_ENABLED: bool = True
    FAST_GATE_SPREAD_MAX: float = 0.12
    FAST_GATE_IV_MOMENTUM_ABS_MAX: float = 0.50

    # ================= 4. Momentum & Trend =================
    STOCK_MOMENTUM_TOLERANCE: float = 0.001
    MIN_LAST_SNAP_ROC: float = -0.0003
    MAX_SNAP_ROC_LIMIT: float = 0.01

    # ================= 4b. Trend Hunter Core =================
    # 启用方式: STRATEGY_CORE_VERSION=TREND。该核心把 TFT 作为雷达/确认器，
    # 主交易条件改为“大盘 + 个股 + MACD + 路径效率”的趋势状态。
    TREND_CORE_ALLOW_SHORT: bool = True
    TREND_CORE_BLOCK_VOLATILE_REGIME: bool = True
    TREND_CORE_ALLOW_MIXED_REGIME: bool = True
    TREND_CORE_MIN_ALPHA_ABS: float = 0.015
    TREND_CORE_ALPHA_ALIGN_MIN_ABS: float = 0.025
    TREND_CORE_MIN_INDEX_ROC: float = 0.00015
    # 5m ROC / snap：略放宽，避免仅靠更长单边才把 roc_5m 推过阈值
    TREND_CORE_MIN_STOCK_ROC: float = 0.00028
    TREND_CORE_MIN_SNAP_ROC: float = -0.00015
    # MACD 柱默认 0.01 偏滞后，略降以便趋势早期对齐仍能下单
    TREND_CORE_MIN_MACD_HIST: float = 0.007
    # 窗口缩短：net = (窗口首尾价差)，30m 易等来「已经涨了一段」的形态；18m 更偏早期顺势
    TREND_CORE_WINDOW_MINS: int = 12
    TREND_CORE_MIN_OBS: int = 8
    # 窗口内需累积的方向性净值（小数）；0.004≈0.4%，偏高 Beta 要等很久；略降提高先手、噪声亦略升
    TREND_CORE_MIN_NET: float = 0.0022
    TREND_CORE_MIN_EFFICIENCY: float = 0.18
    TREND_CORE_MIN_R2: float = 0.06
    TREND_CORE_STRONG_NET: float = 0.008
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
    MIN_OPTION_PRICE: float = float(os.environ.get("MIN_OPTION_PRICE", "0.25"))

    # ================= 7. Liquidity =================
    MAX_SPREAD_PCT_ENTRY: float = 0.05        # 兼容保留：未区分方向时的默认准入点差
    MAX_SPREAD_PCT_ENTRY_CALL: float = 0.05   # V0 做多(CALL)开仓点差上限
    MAX_SPREAD_PCT_ENTRY_PUT: float = 0.09    # PUT 点差略放宽（跌日流动性更差）
    MAX_SPREAD_PCT_EXIT: float = 0.2 
    MAX_SPREAD_DIVERGENCE: float = 0.02
    
    # ================= 8. Exit & Stop Loss =================
    # 默认 (= SWING) ；SCALP/SWING 覆盖见 §8b（exec_profile.py 按 pos.exec_mode 解析）
    STOP_LOSS: float = -0.08
    ABSOLUTE_STOP_LOSS: float = -0.12
    MID_TIME_STOP_MINS: int = 12
    MID_TIME_STOP_ROI: float = 0.03
    TIME_STOP_MINS: int = 25
    TIME_STOP_ROI: float = 0.05

    # ================= 8b. Exec Profile Exit Overlays (Path A / Path C) =================
    # Path A — 0DTE 短打
    SCALP_STOP_LOSS: float = -0.10
    SCALP_ABSOLUTE_STOP_LOSS: float = -0.15
    SCALP_MID_TIME_STOP_MINS: int = 6
    SCALP_MID_TIME_STOP_ROI: float = 0.02
    SCALP_TIME_STOP_MINS: int = 8
    SCALP_TIME_STOP_ROI: float = 0.03
    SCALP_TRAILING_TRIGGER_ROI: float = 0.12
    SCALP_TRAILING_KEEP_RATIO: float = 0.75
    SCALP_FLASH_PROTECT_TRIGGER: float = 0.10
    SCALP_FLASH_PROTECT_EXIT: float = 0.02
    SCALP_NO_MOMENTUM_MINS: int = 4
    SCALP_NO_MOMENTUM_MIN_MAX_ROI: float = 0.015
    SCALP_ZOMBIE_EXIT_MINS: int = 7

    # Path C default leg — 1DTE 趋势
    SWING_STOP_LOSS: float = -0.08
    SWING_ABSOLUTE_STOP_LOSS: float = -0.12
    SWING_MID_TIME_STOP_MINS: int = 12
    SWING_MID_TIME_STOP_ROI: float = 0.03
    SWING_TIME_STOP_MINS: int = 25
    SWING_TIME_STOP_ROI: float = 0.05
    SWING_TRAILING_TRIGGER_ROI: float = 0.18
    SWING_TRAILING_KEEP_RATIO: float = 0.80
    SWING_FLASH_PROTECT_TRIGGER: float = 0.14
    SWING_FLASH_PROTECT_EXIT: float = 0.04
    SWING_NO_MOMENTUM_MINS: int = 8
    SWING_NO_MOMENTUM_MIN_MAX_ROI: float = 0.02
    SWING_ZOMBIE_EXIT_MINS: int = 20

    # auto_hybrid 升级 0DTE 的门控
    HYBRID_SCALP_MIN_NET_EDGE: float = 0.030
    HYBRID_SCALP_MAX_SPREAD: float = 0.08
    HYBRID_SCALP_SESSION_END_HOUR: int = 14
    HYBRID_SCALP_SESSION_END_MINUTE: int = 30

    # ================= 8c. Multi-Band Roll (dislocation → trend → epic) =================
    MULTI_BAND_MAX_LEGS_PER_DAY: int = 3
    MULTI_BAND_ROLL_COOLDOWN_MINS: int = 8
    MULTI_BAND2_PRICE_FLOOR: float = 0.85
    MULTI_BAND3_PRICE_FLOOR: float = 2.00
    MULTI_BAND_SWING_DTE: int = 1
    MULTI_BAND1_MIN_NET_EDGE: float = 0.012
    MULTI_BAND1_MIN_SNAP_ROC: float = 0.0008
    MULTI_BAND1_MAX_SPREAD: float = 0.18
    MULTI_BAND3_MIN_STOCK_ROC: float = 0.0004
    MULTI_BAND_DISLOCATION_ENTRY_ENABLED: bool = True

    # ================= 19. Bidirectional (Phase 2) =================
    try:
        from config import BIDIRECTIONAL_ENABLED as _CFG_BIDIR
    except Exception:  # pragma: no cover
        _CFG_BIDIR = True
    BIDIRECTIONAL_ENABLED: bool = bool(_CFG_BIDIR)
    BIDIRECTIONAL_DISLOCATION_ENTRY_ENABLED: bool = True
    BIDIRECTIONAL_DISLOC_MAX_PRICE: float = 0.85
    BIDIRECTIONAL_DISLOC_MIN_NET_EDGE: float = 0.012
    BIDIRECTIONAL_DISLOC_MIN_SNAP_ROC: float = 0.0008
    BIDIRECTIONAL_DISLOC_MAX_SPREAD: float = 0.18
    BIDIRECTIONAL_INDEX_GUARD_ENABLED: bool = True
    BIDIRECTIONAL_PUT_BLOCK_INDEX_ROC: float = 0.0015
    BIDIRECTIONAL_PUT_MACD_RELAX: float = 0.70
    BIDIRECTIONAL_PUT_MOMENTUM_RELAX: float = 1.50
    BIDIRECTIONAL_PUT_ALPHA_RELAX: float = 0.85
    MACD_HIST_THRESHOLD_PUT: float = 0.010
    INDEX_ROC_THRESHOLD_PUT: float = -0.008

    # ================= 20. Minimal Stack (2-path entry) =================
    try:
        from config import V0_SIMPLE_ENTRY_ENABLED as _CFG_SIMPLE_ENTRY
    except Exception:  # pragma: no cover
        _CFG_SIMPLE_ENTRY = True
    V0_SIMPLE_ENTRY_ENABLED: bool = bool(_CFG_SIMPLE_ENTRY)
    # 趋势腿：仅 alpha + 大盘护栏 + 波动率；跳过 MACD / 5m+snp 动量套件
    SIMPLE_TREND_SKIP_MACD: bool = True
    SIMPLE_TREND_SKIP_MOMENTUM: bool = True
    SIMPLE_TREND_MIN_INDEX_ROC: float = 0.00015

    # Band1 — 低价错价快打（5min 内离场，underlying 止损优先）
    BAND1_STOP_LOSS: float = -0.12
    BAND1_ABSOLUTE_STOP_LOSS: float = -0.18
    BAND1_MID_TIME_STOP_MINS: int = 4
    BAND1_MID_TIME_STOP_ROI: float = 0.15
    BAND1_TIME_STOP_MINS: int = 5
    BAND1_TIME_STOP_ROI: float = 0.20
    BAND1_TRAILING_TRIGGER_ROI: float = 0.25
    BAND1_TRAILING_KEEP_RATIO: float = 0.70
    BAND1_FLASH_PROTECT_TRIGGER: float = 0.18
    BAND1_FLASH_PROTECT_EXIT: float = 0.08
    BAND1_NO_MOMENTUM_MINS: int = 3
    BAND1_NO_MOMENTUM_MIN_MAX_ROI: float = 0.05
    BAND1_ZOMBIE_EXIT_MINS: int = 5

    # Band2 — 趋势确认（12–20min）
    BAND2_STOP_LOSS: float = -0.10
    BAND2_ABSOLUTE_STOP_LOSS: float = -0.14
    BAND2_MID_TIME_STOP_MINS: int = 10
    BAND2_MID_TIME_STOP_ROI: float = 0.05
    BAND2_TIME_STOP_MINS: int = 18
    BAND2_TIME_STOP_ROI: float = 0.08
    BAND2_TRAILING_TRIGGER_ROI: float = 0.20
    BAND2_TRAILING_KEEP_RATIO: float = 0.78
    BAND2_FLASH_PROTECT_TRIGGER: float = 0.12
    BAND2_FLASH_PROTECT_EXIT: float = 0.05
    BAND2_NO_MOMENTUM_MINS: int = 6
    BAND2_NO_MOMENTUM_MIN_MAX_ROI: float = 0.02
    BAND2_ZOMBIE_EXIT_MINS: int = 15

    # Band3 — 尾段 epic（可持到 30min+，宽 trailing）
    BAND3_STOP_LOSS: float = -0.08
    BAND3_ABSOLUTE_STOP_LOSS: float = -0.12
    BAND3_MID_TIME_STOP_MINS: int = 15
    BAND3_MID_TIME_STOP_ROI: float = 0.04
    BAND3_TIME_STOP_MINS: int = 30
    BAND3_TIME_STOP_ROI: float = 0.06
    BAND3_TRAILING_TRIGGER_ROI: float = 0.15
    BAND3_TRAILING_KEEP_RATIO: float = 0.82
    BAND3_FLASH_PROTECT_TRIGGER: float = 0.10
    BAND3_FLASH_PROTECT_EXIT: float = 0.04
    BAND3_NO_MOMENTUM_MINS: int = 10
    BAND3_NO_MOMENTUM_MIN_MAX_ROI: float = 0.02
    BAND3_ZOMBIE_EXIT_MINS: int = 22
    ALPHA_FLIP_THRESHOLD: float = 0.012
    HIGH_CONFIDENCE_THRESHOLD: float = 0.025
    
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
    # OMS 秒级 tight-exit：原为开仓后延迟盈利阶梯 / FLASH；默认 0 = 不再延迟。
    # 亏损侧 ABSOLUTE_STOP_LOSS / STOP_LOSS 仍按秒级评估。
    TIGHT_1S_ENTRY_PROTECT_SECONDS: float = 0.0
    # OMS 平仓保护期：只挡 TIME/SPREAD/ZOMBIE 等非风险、非利润保护退出；
    # STEP/FLASH/TRAILING 等利润保护不再等待 60s。
    NON_URGENT_EXIT_PROTECT_SECONDS: float = 60.0
    # 正股瞬时涨跌与持仓方向相反（CALL 遇正股下跌 tick、PUT 遇正股上涨 tick），连续 N 秒触发秒级平仓。
    TIGHT_1S_DIR_OPP_CONSEC_SECONDS: int = 5
    # >0 时仅当 |ΔS/S| 超过该阈值才算反向；0 = 任一反向 tick 计数。
    TIGHT_1S_DIR_OPP_MIN_REL_MOVE: float = 0.0
    # 峰值 ROI≥该值后启用 TRAILING_EPIC（与 LADDER 同口径：0.05=5%）。原 5.50 易被当成 550%，几乎永不触发。
    TRAILING_TRIGGER_ROI: float = 0.22
    TRAILING_KEEP_RATIO: float = 0.92
    # 与 config.V0_PROFIT_HYBRID_CONTINUOUS_ENABLED 同步；False 时跳过 TRAILING_EPIC（仅阶梯等离散档 + 其它 exit）
    V0_PROFIT_HYBRID_CONTINUOUS_ENABLED: bool = _CFG_V0_PROFIT_HYBRID_CONTINUOUS
    COUNTER_TREND_PROTECT_TRIGGER: float = 0.25
    COUNTER_TREND_PROTECT_EXIT: float = 0.10
    MACD_FADE_MIN_ROI: float = 0.03            # V0 动能衰减门槛极低 (3%)
    
    # ================= 12. Dynamic Strategy Logic =================
    DYNAMIC_LADDER_ENABLED: bool = False      # V0 为固定阶梯
    HIGH_ALPHA_WIDE_THRESHOLD: float = 0.040
    
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
    MACD_HIST_THRESHOLD: float = 0.012
    SLOW_BULL_CHANNEL_ENABLED: bool = False
    SLOW_BULL_MAX_VOL_Z: float = 0.5
    SLOW_BULL_ALPHA_THRESHOLD: float = 0.012
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
