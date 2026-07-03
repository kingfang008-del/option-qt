from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional


class InstrumentKind(str, Enum):
    STOCK = "stock"
    OPTION = "option"
    PERPETUAL = "perpetual"
    FUTURE = "future"
    UNKNOWN = "unknown"


class QuoteSourceKind(str, Enum):
    UNKNOWN = "unknown"
    CURRENT_BATCH = "current_batch"
    LATCHED_PREV_SECOND = "latched_prev_second"
    REPLAY_FEED = "replay_feed"
    LIVE_FEED = "live_feed"
    EXECUTION_BACKFILL = "execution_backfill"
    SYNTHETIC = "synthetic"


class PositionSide(str, Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _coerce_enum(enum_cls, value, default):
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except Exception:
        return default


def _finish_validation(name: str, errors: List[str], strict: bool) -> List[str]:
    if strict and errors:
        raise ValueError(f"{name} validation failed: {'; '.join(errors)}")
    return errors


@dataclass(frozen=True)
class InstrumentTraits:
    """Instrument-level semantics kept separate from strategy facts.

    字段约定：
    - `symbol`: 交易标识，不带时间语义
    - `instrument_kind`: 资产类别，决定后续扩展字段和风险语义
    - `quote_currency` / `settlement_currency`: 报价币种与结算币种
    - `contract_multiplier`: 1 张或 1 手合约对应的名义乘数
    - `min_price_increment` / `qty_step`: 最小价格跳动与数量步长
    - `supports_long` / `supports_short`: 资产是否允许该方向持仓
    - `has_expiry` / `has_strike` / `has_iv`: 是否存在期权专属语义
    - `mark_price_required` / `funding_rate_supported`: 永续等衍生品扩展语义
    - `metadata`: 非关键补充字段，不能承载核心契约
    """

    # 基础标识
    symbol: str
    instrument_kind: InstrumentKind

    # 结算与撮合基础属性
    quote_currency: str = "USD"
    settlement_currency: str = "USD"
    contract_multiplier: float = 1.0
    min_price_increment: float = 0.01
    qty_step: float = 1.0

    # 可交易方向
    supports_long: bool = True
    supports_short: bool = False

    # 期权 / 衍生品专属语义开关
    has_expiry: bool = False
    has_strike: bool = False
    has_iv: bool = False
    mark_price_required: bool = False
    funding_rate_supported: bool = False

    # 仅用于补充上下文，禁止替代正式 schema 字段
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def stock(cls, symbol: str, **overrides: Any) -> "InstrumentTraits":
        return cls.from_dict({
            "symbol": symbol,
            "instrument_kind": InstrumentKind.STOCK.value,
            "supports_short": True,
            **overrides,
        })

    @classmethod
    def option(cls, symbol: str, **overrides: Any) -> "InstrumentTraits":
        return cls.from_dict({
            "symbol": symbol,
            "instrument_kind": InstrumentKind.OPTION.value,
            "contract_multiplier": 100.0,
            "supports_short": False,
            "has_expiry": True,
            "has_strike": True,
            "has_iv": True,
            **overrides,
        })

    @classmethod
    def perpetual(cls, symbol: str, **overrides: Any) -> "InstrumentTraits":
        return cls.from_dict({
            "symbol": symbol,
            "instrument_kind": InstrumentKind.PERPETUAL.value,
            "supports_short": True,
            "mark_price_required": True,
            "funding_rate_supported": True,
            "min_price_increment": 0.1,
            "qty_step": 0.001,
            **overrides,
        })

    def validate(self, strict: bool = False) -> List[str]:
        errors: List[str] = []
        if not self.symbol:
            errors.append("symbol is required")
        if not self.quote_currency:
            errors.append("quote_currency is required")
        if not self.settlement_currency:
            errors.append("settlement_currency is required")
        if not _is_finite_number(self.contract_multiplier) or float(self.contract_multiplier) <= 0.0:
            errors.append("contract_multiplier must be > 0")
        if not _is_finite_number(self.min_price_increment) or float(self.min_price_increment) <= 0.0:
            errors.append("min_price_increment must be > 0")
        if not _is_finite_number(self.qty_step) or float(self.qty_step) <= 0.0:
            errors.append("qty_step must be > 0")
        if not self.supports_long and not self.supports_short:
            errors.append("at least one trading side must be supported")
        return _finish_validation("InstrumentTraits", errors, strict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "instrument_kind": self.instrument_kind.value,
            "quote_currency": self.quote_currency,
            "settlement_currency": self.settlement_currency,
            "contract_multiplier": float(self.contract_multiplier),
            "min_price_increment": float(self.min_price_increment),
            "qty_step": float(self.qty_step),
            "supports_long": bool(self.supports_long),
            "supports_short": bool(self.supports_short),
            "has_expiry": bool(self.has_expiry),
            "has_strike": bool(self.has_strike),
            "has_iv": bool(self.has_iv),
            "mark_price_required": bool(self.mark_price_required),
            "funding_rate_supported": bool(self.funding_rate_supported),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InstrumentTraits":
        return cls(
            symbol=str(payload.get("symbol", "") or ""),
            instrument_kind=_coerce_enum(
                InstrumentKind,
                payload.get("instrument_kind", InstrumentKind.UNKNOWN.value),
                InstrumentKind.UNKNOWN,
            ),
            quote_currency=str(payload.get("quote_currency", "USD") or "USD"),
            settlement_currency=str(payload.get("settlement_currency", "USD") or "USD"),
            contract_multiplier=float(payload.get("contract_multiplier", 1.0) or 1.0),
            min_price_increment=float(payload.get("min_price_increment", 0.01) or 0.01),
            qty_step=float(payload.get("qty_step", 1.0) or 1.0),
            supports_long=bool(payload.get("supports_long", True)),
            supports_short=bool(payload.get("supports_short", False)),
            has_expiry=bool(payload.get("has_expiry", False)),
            has_strike=bool(payload.get("has_strike", False)),
            has_iv=bool(payload.get("has_iv", False)),
            mark_price_required=bool(payload.get("mark_price_required", False)),
            funding_rate_supported=bool(payload.get("funding_rate_supported", False)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class DecisionQuoteSnapshot:
    """Tradable quote snapshot seen by strategy at decision time.

    字段约定：
    - `quote_ts`: 策略真正看到这份快照的事件时刻
    - `last_price`: 最新成交价或最近参考价，不保证可成交
    - `best_bid` / `best_ask`: 决策时刻盘口最优价
    - `bid_size` / `ask_size`: 最优价上的可见深度
    - `mark_price` / `index_price`: 永续等资产可选的估值/指数锚点
    - `contract_id`: 标的或具体合约标识，股票可为空
    - `source_kind`: 说明这份决策快照来自当前批次、上一秒锁存还是 replay
    - `metadata`: 非关键扩展字段
    """

    # 标识与时间
    symbol: str
    instrument_kind: InstrumentKind
    quote_ts: float

    # 可交易报价
    last_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0

    # 衍生品补充锚点
    mark_price: Optional[float] = None
    index_price: Optional[float] = None

    # 合约来源信息
    contract_id: str = ""
    venue: str = ""
    source_kind: QuoteSourceKind = QuoteSourceKind.UNKNOWN
    contract_multiplier: float = 1.0

    # 补充字段，不参与核心 validate 逻辑
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mid_price(self) -> float:
        if self.best_bid > 0.0 and self.best_ask > 0.0:
            return (self.best_bid + self.best_ask) / 2.0
        if self.mark_price is not None and self.mark_price > 0.0:
            return float(self.mark_price)
        return float(self.last_price or 0.0)

    @property
    def has_book(self) -> bool:
        return self.best_bid > 0.0 and self.best_ask > 0.0

    def validate(self, strict: bool = False) -> List[str]:
        errors: List[str] = []
        if not self.symbol:
            errors.append("symbol is required")
        if not _is_finite_number(self.quote_ts) or float(self.quote_ts) <= 0.0:
            errors.append("quote_ts must be > 0")
        for name in ("last_price", "best_bid", "best_ask", "bid_size", "ask_size", "contract_multiplier"):
            value = getattr(self, name)
            if not _is_finite_number(value):
                errors.append(f"{name} must be finite")
            elif float(value) < 0.0:
                errors.append(f"{name} must be >= 0")
        if self.best_bid > 0.0 and self.best_ask > 0.0 and self.best_ask < self.best_bid:
            errors.append("best_ask must be >= best_bid")
        if self.mark_price is not None:
            if not _is_finite_number(self.mark_price) or float(self.mark_price) < 0.0:
                errors.append("mark_price must be finite and >= 0 when provided")
        if self.index_price is not None:
            if not _is_finite_number(self.index_price) or float(self.index_price) < 0.0:
                errors.append("index_price must be finite and >= 0 when provided")
        return _finish_validation("DecisionQuoteSnapshot", errors, strict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "instrument_kind": self.instrument_kind.value,
            "quote_ts": float(self.quote_ts),
            "last_price": float(self.last_price),
            "best_bid": float(self.best_bid),
            "best_ask": float(self.best_ask),
            "bid_size": float(self.bid_size),
            "ask_size": float(self.ask_size),
            "mark_price": None if self.mark_price is None else float(self.mark_price),
            "index_price": None if self.index_price is None else float(self.index_price),
            "contract_id": self.contract_id,
            "venue": self.venue,
            "source_kind": self.source_kind.value,
            "contract_multiplier": float(self.contract_multiplier),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DecisionQuoteSnapshot":
        return cls(
            symbol=str(payload.get("symbol", "") or ""),
            instrument_kind=_coerce_enum(
                InstrumentKind,
                payload.get("instrument_kind", InstrumentKind.UNKNOWN.value),
                InstrumentKind.UNKNOWN,
            ),
            quote_ts=float(payload.get("quote_ts", 0.0) or 0.0),
            last_price=float(payload.get("last_price", 0.0) or 0.0),
            best_bid=float(payload.get("best_bid", 0.0) or 0.0),
            best_ask=float(payload.get("best_ask", 0.0) or 0.0),
            bid_size=float(payload.get("bid_size", 0.0) or 0.0),
            ask_size=float(payload.get("ask_size", 0.0) or 0.0),
            mark_price=None if payload.get("mark_price") in (None, "") else float(payload.get("mark_price")),
            index_price=None if payload.get("index_price") in (None, "") else float(payload.get("index_price")),
            contract_id=str(payload.get("contract_id", "") or ""),
            venue=str(payload.get("venue", "") or ""),
            source_kind=_coerce_enum(
                QuoteSourceKind,
                payload.get("source_kind", QuoteSourceKind.UNKNOWN.value),
                QuoteSourceKind.UNKNOWN,
            ),
            contract_multiplier=float(payload.get("contract_multiplier", 1.0) or 1.0),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ExecutionQuote1s:
    """Second-level quote snapshot used by execution tracking.

    字段约定：
    - `ts`: 秒级执行窗口中的事件时刻
    - 其余价格/盘口字段语义与 `DecisionQuoteSnapshot` 一致
    - `sequence_no`: 上游若有严格顺序号可放这里
    - `exchange_latency_ms`: 行情到达本地时观测到的链路延迟
    - 该对象只表达执行跟踪，不表达分钟冻结事实
    """

    # 标识与时间
    symbol: str
    instrument_kind: InstrumentKind
    ts: float

    # 秒级可观察报价
    last_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0

    # 衍生品估值锚点
    mark_price: Optional[float] = None
    index_price: Optional[float] = None

    # 来源与链路观测字段
    contract_id: str = ""
    venue: str = ""
    source_kind: QuoteSourceKind = QuoteSourceKind.UNKNOWN
    sequence_no: Optional[int] = None
    exchange_latency_ms: Optional[float] = None

    # 非关键扩展字段
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, strict: bool = False) -> List[str]:
        errors: List[str] = []
        if not self.symbol:
            errors.append("symbol is required")
        if not _is_finite_number(self.ts) or float(self.ts) <= 0.0:
            errors.append("ts must be > 0")
        for name in ("last_price", "best_bid", "best_ask", "bid_size", "ask_size"):
            value = getattr(self, name)
            if not _is_finite_number(value):
                errors.append(f"{name} must be finite")
            elif float(value) < 0.0:
                errors.append(f"{name} must be >= 0")
        if self.best_bid > 0.0 and self.best_ask > 0.0 and self.best_ask < self.best_bid:
            errors.append("best_ask must be >= best_bid")
        if self.exchange_latency_ms is not None:
            if not _is_finite_number(self.exchange_latency_ms) or float(self.exchange_latency_ms) < 0.0:
                errors.append("exchange_latency_ms must be finite and >= 0 when provided")
        return _finish_validation("ExecutionQuote1s", errors, strict)

    def to_decision_snapshot(
        self,
        source_kind: Optional[QuoteSourceKind] = None,
    ) -> DecisionQuoteSnapshot:
        return DecisionQuoteSnapshot(
            symbol=self.symbol,
            instrument_kind=self.instrument_kind,
            quote_ts=self.ts,
            last_price=self.last_price,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            bid_size=self.bid_size,
            ask_size=self.ask_size,
            mark_price=self.mark_price,
            index_price=self.index_price,
            contract_id=self.contract_id,
            venue=self.venue,
            source_kind=source_kind or self.source_kind,
            metadata=dict(self.metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "instrument_kind": self.instrument_kind.value,
            "ts": float(self.ts),
            "last_price": float(self.last_price),
            "best_bid": float(self.best_bid),
            "best_ask": float(self.best_ask),
            "bid_size": float(self.bid_size),
            "ask_size": float(self.ask_size),
            "mark_price": None if self.mark_price is None else float(self.mark_price),
            "index_price": None if self.index_price is None else float(self.index_price),
            "contract_id": self.contract_id,
            "venue": self.venue,
            "source_kind": self.source_kind.value,
            "sequence_no": self.sequence_no,
            "exchange_latency_ms": None if self.exchange_latency_ms is None else float(self.exchange_latency_ms),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExecutionQuote1s":
        return cls(
            symbol=str(payload.get("symbol", "") or ""),
            instrument_kind=_coerce_enum(
                InstrumentKind,
                payload.get("instrument_kind", InstrumentKind.UNKNOWN.value),
                InstrumentKind.UNKNOWN,
            ),
            ts=float(payload.get("ts", 0.0) or 0.0),
            last_price=float(payload.get("last_price", 0.0) or 0.0),
            best_bid=float(payload.get("best_bid", 0.0) or 0.0),
            best_ask=float(payload.get("best_ask", 0.0) or 0.0),
            bid_size=float(payload.get("bid_size", 0.0) or 0.0),
            ask_size=float(payload.get("ask_size", 0.0) or 0.0),
            mark_price=None if payload.get("mark_price") in (None, "") else float(payload.get("mark_price")),
            index_price=None if payload.get("index_price") in (None, "") else float(payload.get("index_price")),
            contract_id=str(payload.get("contract_id", "") or ""),
            venue=str(payload.get("venue", "") or ""),
            source_kind=_coerce_enum(
                QuoteSourceKind,
                payload.get("source_kind", QuoteSourceKind.UNKNOWN.value),
                QuoteSourceKind.UNKNOWN,
            ),
            sequence_no=payload.get("sequence_no"),
            exchange_latency_ms=(
                None
                if payload.get("exchange_latency_ms") in (None, "")
                else float(payload.get("exchange_latency_ms"))
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class AlphaFrameItem:
    """Minute-frozen decision facts for a single tradable symbol.

    字段约定：
    - `alpha_label_ts`: 该分钟消费的 alpha 标签时刻，通常是 `minute_ts - 60`
    - `alpha_available_ts`: 该分钟真正允许下单的时刻，通常等于 `minute_ts`
    - `frame_id`: 用于把 item 挂回所属分钟帧
    - `reference_price`: 分钟冻结时的基础价格锚点
    - `cs_alpha_z` / `vol_z` / `roc_5m` / `macd*` / `snap_roc`: 分钟冻结特征
    - `decision_quote`: 策略做决定时实际看到的可交易快照
    - `tags`: 扩展标签区，只放非关键补充信息
    """

    # 标识与资产语义
    symbol: str
    instrument_traits: InstrumentTraits

    # 核心分钟事实
    alpha: float
    alpha_label_ts: int
    alpha_available_ts: int

    # 帧内定位信息
    batch_idx: int = -1
    frame_id: str = ""

    # 分钟冻结价格与信号特征
    reference_price: float = 0.0
    cs_alpha_z: float = 0.0
    vol_z: float = 0.0
    roc_5m: float = 0.0
    macd: float = 0.0
    macd_slope: float = 0.0
    snap_roc: float = 0.0
    event_prob: float = 0.0
    is_ready: bool = False
    correction_mode: str = "NORMAL"

    # 策略做决定时所见快照，不应晚于 alpha_available_ts
    decision_quote: Optional[DecisionQuoteSnapshot] = None

    # 非关键扩展标签
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def minute_ts(self) -> int:
        return int(self.alpha_available_ts)

    def validate(self, strict: bool = False) -> List[str]:
        errors: List[str] = []
        if not self.symbol:
            errors.append("symbol is required")
        if self.symbol != self.instrument_traits.symbol:
            errors.append("instrument_traits.symbol must match item symbol")
        if self.batch_idx < -1:
            errors.append("batch_idx must be >= -1")
        if not _is_finite_number(self.alpha):
            errors.append("alpha must be finite")
        if self.alpha_label_ts <= 0:
            errors.append("alpha_label_ts must be > 0")
        if self.alpha_available_ts <= 0:
            errors.append("alpha_available_ts must be > 0")
        if self.alpha_available_ts <= self.alpha_label_ts:
            errors.append("alpha_available_ts must be > alpha_label_ts")
        if self.alpha_available_ts - self.alpha_label_ts != 60:
            errors.append("alpha_available_ts must equal alpha_label_ts + 60 for minute windows")
        if not _is_finite_number(self.reference_price) or float(self.reference_price) < 0.0:
            errors.append("reference_price must be finite and >= 0")
        if not _is_finite_number(self.event_prob) or not (0.0 <= float(self.event_prob) <= 1.0):
            errors.append("event_prob must be within [0, 1]")
        if self.decision_quote is not None:
            quote_errors = self.decision_quote.validate(strict=False)
            errors.extend([f"decision_quote.{msg}" for msg in quote_errors])
            if self.decision_quote.symbol != self.symbol:
                errors.append("decision_quote.symbol must match item symbol")
            if self.decision_quote.instrument_kind != self.instrument_traits.instrument_kind:
                errors.append("decision_quote.instrument_kind must match instrument traits")
            if self.decision_quote.quote_ts > float(self.alpha_available_ts):
                errors.append("decision_quote.quote_ts must be <= alpha_available_ts")
        return _finish_validation("AlphaFrameItem", errors, strict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "instrument_traits": self.instrument_traits.to_dict(),
            "alpha": float(self.alpha),
            "alpha_label_ts": int(self.alpha_label_ts),
            "alpha_available_ts": int(self.alpha_available_ts),
            "batch_idx": int(self.batch_idx),
            "frame_id": self.frame_id,
            "reference_price": float(self.reference_price),
            "cs_alpha_z": float(self.cs_alpha_z),
            "vol_z": float(self.vol_z),
            "roc_5m": float(self.roc_5m),
            "macd": float(self.macd),
            "macd_slope": float(self.macd_slope),
            "snap_roc": float(self.snap_roc),
            "event_prob": float(self.event_prob),
            "is_ready": bool(self.is_ready),
            "correction_mode": self.correction_mode,
            "decision_quote": None if self.decision_quote is None else self.decision_quote.to_dict(),
            "tags": dict(self.tags or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AlphaFrameItem":
        decision_quote_payload = payload.get("decision_quote")
        return cls(
            symbol=str(payload.get("symbol", "") or ""),
            instrument_traits=InstrumentTraits.from_dict(payload.get("instrument_traits") or {}),
            alpha=float(payload.get("alpha", 0.0) or 0.0),
            alpha_label_ts=int(payload.get("alpha_label_ts", 0) or 0),
            alpha_available_ts=int(payload.get("alpha_available_ts", 0) or 0),
            batch_idx=int(payload.get("batch_idx", -1) or -1),
            frame_id=str(payload.get("frame_id", "") or ""),
            reference_price=float(payload.get("reference_price", 0.0) or 0.0),
            cs_alpha_z=float(payload.get("cs_alpha_z", 0.0) or 0.0),
            vol_z=float(payload.get("vol_z", 0.0) or 0.0),
            roc_5m=float(payload.get("roc_5m", 0.0) or 0.0),
            macd=float(payload.get("macd", 0.0) or 0.0),
            macd_slope=float(payload.get("macd_slope", 0.0) or 0.0),
            snap_roc=float(payload.get("snap_roc", 0.0) or 0.0),
            event_prob=float(payload.get("event_prob", 0.0) or 0.0),
            is_ready=bool(payload.get("is_ready", False)),
            correction_mode=str(payload.get("correction_mode", "NORMAL") or "NORMAL"),
            decision_quote=(
                None if not decision_quote_payload else DecisionQuoteSnapshot.from_dict(decision_quote_payload)
            ),
            tags=dict(payload.get("tags") or {}),
        )


@dataclass(frozen=True)
class AlphaFrame:
    """Minute-frozen cross-section of tradable facts.

    字段约定：
    - `minute_ts`: 当前分钟窗口左边界，也是该帧的 canonical 时刻
    - `alpha_label_ts`: 被消费的分钟标签时刻
    - `alpha_available_ts`: 该帧进入 OMS 后允许决策的时刻
    - `items`: 所有单标的冻结事实，symbol 在帧内必须唯一
    - `index_trend` / `market_regime` / `is_zombie_market`: 帧级市场状态
    - `metadata`: 帧级扩展信息
    """

    # 帧主键与时间锚点
    frame_id: str
    minute_ts: int
    alpha_label_ts: int
    alpha_available_ts: int

    # 分钟冻结横截面
    items: List[AlphaFrameItem] = field(default_factory=list)

    # 帧级市场状态
    index_trend: int = 0
    market_regime: str = "unknown"
    is_zombie_market: bool = False

    # 帧级补充字段
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_items(
        cls,
        minute_ts: int,
        items: List[AlphaFrameItem],
        frame_id: str = "",
        **kwargs: Any,
    ) -> "AlphaFrame":
        m = int(minute_ts)
        return cls(
            frame_id=frame_id,
            minute_ts=m,
            alpha_label_ts=m - 60,
            alpha_available_ts=m,
            items=list(items or []),
            **kwargs,
        )

    def validate(self, strict: bool = False) -> List[str]:
        errors: List[str] = []
        if self.minute_ts <= 0:
            errors.append("minute_ts must be > 0")
        if self.alpha_label_ts != self.minute_ts - 60:
            errors.append("alpha_label_ts must equal minute_ts - 60")
        if self.alpha_available_ts != self.minute_ts:
            errors.append("alpha_available_ts must equal minute_ts")
        seen_symbols = set()
        for idx, item in enumerate(self.items):
            item_errors = item.validate(strict=False)
            errors.extend([f"items[{idx}].{msg}" for msg in item_errors])
            if item.symbol in seen_symbols:
                errors.append(f"items[{idx}].symbol duplicated: {item.symbol}")
            seen_symbols.add(item.symbol)
            if item.alpha_label_ts != self.alpha_label_ts:
                errors.append(f"items[{idx}].alpha_label_ts must match frame alpha_label_ts")
            if item.alpha_available_ts != self.alpha_available_ts:
                errors.append(f"items[{idx}].alpha_available_ts must match frame alpha_available_ts")
            if item.frame_id and self.frame_id and item.frame_id != self.frame_id:
                errors.append(f"items[{idx}].frame_id must match frame_id")
        return _finish_validation("AlphaFrame", errors, strict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "minute_ts": int(self.minute_ts),
            "alpha_label_ts": int(self.alpha_label_ts),
            "alpha_available_ts": int(self.alpha_available_ts),
            "items": [item.to_dict() for item in self.items],
            "index_trend": int(self.index_trend),
            "market_regime": self.market_regime,
            "is_zombie_market": bool(self.is_zombie_market),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AlphaFrame":
        return cls(
            frame_id=str(payload.get("frame_id", "") or ""),
            minute_ts=int(payload.get("minute_ts", 0) or 0),
            alpha_label_ts=int(payload.get("alpha_label_ts", 0) or 0),
            alpha_available_ts=int(payload.get("alpha_available_ts", 0) or 0),
            items=[AlphaFrameItem.from_dict(item) for item in (payload.get("items") or [])],
            index_trend=int(payload.get("index_trend", 0) or 0),
            market_regime=str(payload.get("market_regime", "unknown") or "unknown"),
            is_zombie_market=bool(payload.get("is_zombie_market", False)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ExecutionWindow:
    """Minute execution window binding one alpha frame and its second-level quotes.

    字段约定：
    - `minute_ts`: 当前执行窗口左边界
    - `alpha_label_ts`: 当前窗口消费的标签时刻
    - `alpha_available_ts`: 当前窗口开始允许执行的时刻
    - `alpha_frame`: 分钟边界冻结事实，只应评估一次
    - `quotes_1s`: 该分钟内 `[minute_ts, minute_ts + 60)` 的秒级执行行情
    """

    # 窗口时间锚点
    minute_ts: int
    alpha_label_ts: int
    alpha_available_ts: int

    # 分钟事实 + 秒级执行轨迹
    alpha_frame: AlphaFrame
    quotes_1s: List[ExecutionQuote1s] = field(default_factory=list)

    @classmethod
    def from_frame(
        cls,
        minute_ts: int,
        alpha_frame: AlphaFrame,
        quotes_1s: List[ExecutionQuote1s],
    ) -> "ExecutionWindow":
        m = int(minute_ts)
        return cls(
            minute_ts=m,
            alpha_label_ts=m - 60,
            alpha_available_ts=m,
            alpha_frame=alpha_frame,
            quotes_1s=list(quotes_1s or []),
        )

    def validate(self, strict: bool = False) -> List[str]:
        errors: List[str] = []
        if self.minute_ts <= 0:
            errors.append("minute_ts must be > 0")
        if self.alpha_label_ts != self.minute_ts - 60:
            errors.append("alpha_label_ts must equal minute_ts - 60")
        if self.alpha_available_ts != self.minute_ts:
            errors.append("alpha_available_ts must equal minute_ts")
        frame_errors = self.alpha_frame.validate(strict=False)
        errors.extend([f"alpha_frame.{msg}" for msg in frame_errors])
        if self.alpha_frame.minute_ts != self.minute_ts:
            errors.append("alpha_frame.minute_ts must match minute_ts")
        prev_ts: Optional[float] = None
        lo, hi = float(self.minute_ts), float(self.minute_ts + 60)
        for idx, quote in enumerate(self.quotes_1s):
            quote_errors = quote.validate(strict=False)
            errors.extend([f"quotes_1s[{idx}].{msg}" for msg in quote_errors])
            if not (lo <= float(quote.ts) < hi):
                errors.append(f"quotes_1s[{idx}].ts must be within [{int(lo)}, {int(hi)})")
            if prev_ts is not None and float(quote.ts) < prev_ts:
                errors.append(f"quotes_1s[{idx}].ts must be non-decreasing")
            prev_ts = float(quote.ts)
        return _finish_validation("ExecutionWindow", errors, strict)

    def summary(self) -> str:
        return (
            f"win[{self.minute_ts}] alpha={self.alpha_label_ts}->avail={self.alpha_available_ts} "
            f"items={len(self.alpha_frame.items)} quotes={len(self.quotes_1s)}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "minute_ts": int(self.minute_ts),
            "alpha_label_ts": int(self.alpha_label_ts),
            "alpha_available_ts": int(self.alpha_available_ts),
            "alpha_frame": self.alpha_frame.to_dict(),
            "quotes_1s": [quote.to_dict() for quote in self.quotes_1s],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExecutionWindow":
        return cls(
            minute_ts=int(payload.get("minute_ts", 0) or 0),
            alpha_label_ts=int(payload.get("alpha_label_ts", 0) or 0),
            alpha_available_ts=int(payload.get("alpha_available_ts", 0) or 0),
            alpha_frame=AlphaFrame.from_dict(payload.get("alpha_frame") or {}),
            quotes_1s=[ExecutionQuote1s.from_dict(it) for it in (payload.get("quotes_1s") or [])],
        )


@dataclass(frozen=True)
class PositionSnapshot:
    """Pure position semantics kept separate from quote and alpha payloads.

    字段约定：
    - `side`: 仓位方向语义，只表达持仓，不表达交易信号
    - `quantity`: 绝对持仓量，方向由 `side` 单独表达
    - `avg_entry_price` / `entry_ts`: 开仓成本与开仓时刻
    - `entry_frame_id`: 本仓位来源于哪个分钟帧
    - `entry_quote_ts`: 开仓决策时所见 quote 的时刻
    - `realized_pnl` / `unrealized_pnl`: 已实现/未实现盈亏
    - `max_favorable_excursion` / `max_adverse_excursion`: 持仓过程中的最大有利/不利偏移
    - `metadata`: 非关键扩展字段
    """

    # 标识与资产语义
    symbol: str
    instrument_traits: InstrumentTraits

    # 纯持仓语义
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    entry_ts: float = 0.0

    # 开仓来源追踪
    contract_id: str = ""
    entry_frame_id: str = ""
    entry_quote_ts: Optional[float] = None

    # 仓位结果指标
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # 非关键扩展字段
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.side != PositionSide.FLAT and self.quantity > 0.0

    @property
    def signed_quantity(self) -> float:
        if self.side == PositionSide.LONG:
            return float(self.quantity)
        if self.side == PositionSide.SHORT:
            return -float(self.quantity)
        return 0.0

    def validate(self, strict: bool = False) -> List[str]:
        errors: List[str] = []
        if not self.symbol:
            errors.append("symbol is required")
        if self.symbol != self.instrument_traits.symbol:
            errors.append("instrument_traits.symbol must match position symbol")
        if not _is_finite_number(self.quantity) or float(self.quantity) < 0.0:
            errors.append("quantity must be finite and >= 0")
        if not _is_finite_number(self.avg_entry_price) or float(self.avg_entry_price) < 0.0:
            errors.append("avg_entry_price must be finite and >= 0")
        if not _is_finite_number(self.entry_ts) or float(self.entry_ts) < 0.0:
            errors.append("entry_ts must be finite and >= 0")
        for name in ("realized_pnl", "unrealized_pnl", "max_favorable_excursion", "max_adverse_excursion"):
            value = getattr(self, name)
            if not _is_finite_number(value):
                errors.append(f"{name} must be finite")
        if self.side == PositionSide.FLAT:
            if float(self.quantity) != 0.0:
                errors.append("flat positions must have quantity == 0")
        else:
            if float(self.quantity) <= 0.0:
                errors.append("open positions must have quantity > 0")
            if float(self.avg_entry_price) <= 0.0:
                errors.append("open positions must have avg_entry_price > 0")
            if float(self.entry_ts) <= 0.0:
                errors.append("open positions must have entry_ts > 0")
        if self.side == PositionSide.SHORT and not self.instrument_traits.supports_short:
            errors.append("instrument does not support short positions")
        if self.entry_quote_ts is not None:
            if not _is_finite_number(self.entry_quote_ts) or float(self.entry_quote_ts) <= 0.0:
                errors.append("entry_quote_ts must be finite and > 0 when provided")
        return _finish_validation("PositionSnapshot", errors, strict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "instrument_traits": self.instrument_traits.to_dict(),
            "side": self.side.value,
            "quantity": float(self.quantity),
            "avg_entry_price": float(self.avg_entry_price),
            "entry_ts": float(self.entry_ts),
            "contract_id": self.contract_id,
            "entry_frame_id": self.entry_frame_id,
            "entry_quote_ts": None if self.entry_quote_ts is None else float(self.entry_quote_ts),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "max_favorable_excursion": float(self.max_favorable_excursion),
            "max_adverse_excursion": float(self.max_adverse_excursion),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PositionSnapshot":
        return cls(
            symbol=str(payload.get("symbol", "") or ""),
            instrument_traits=InstrumentTraits.from_dict(payload.get("instrument_traits") or {}),
            side=_coerce_enum(
                PositionSide,
                payload.get("side", PositionSide.FLAT.value),
                PositionSide.FLAT,
            ),
            quantity=float(payload.get("quantity", 0.0) or 0.0),
            avg_entry_price=float(payload.get("avg_entry_price", 0.0) or 0.0),
            entry_ts=float(payload.get("entry_ts", 0.0) or 0.0),
            contract_id=str(payload.get("contract_id", "") or ""),
            entry_frame_id=str(payload.get("entry_frame_id", "") or ""),
            entry_quote_ts=None if payload.get("entry_quote_ts") in (None, "") else float(payload.get("entry_quote_ts")),
            realized_pnl=float(payload.get("realized_pnl", 0.0) or 0.0),
            unrealized_pnl=float(payload.get("unrealized_pnl", 0.0) or 0.0),
            max_favorable_excursion=float(payload.get("max_favorable_excursion", 0.0) or 0.0),
            max_adverse_excursion=float(payload.get("max_adverse_excursion", 0.0) or 0.0),
            metadata=dict(payload.get("metadata") or {}),
        )


__all__ = [
    "AlphaFrame",
    "AlphaFrameItem",
    "DecisionQuoteSnapshot",
    "ExecutionQuote1s",
    "ExecutionWindow",
    "InstrumentKind",
    "InstrumentTraits",
    "PositionSide",
    "PositionSnapshot",
    "QuoteSourceKind",
]
