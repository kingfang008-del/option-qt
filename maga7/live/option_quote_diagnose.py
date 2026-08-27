"""Diagnose why a locked Mag7 option ladder has no usable NBBO."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class OptionQuoteDiagnosis:
    symbol: str
    code: str
    detail: str
    actionable: str
    exclude: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_adjusted_local_symbol(symbol: str, local_symbol: str) -> bool:
    compact = "".join(str(local_symbol or "").upper().split())
    return bool(compact) and not compact.startswith(str(symbol or "").upper())


def diagnose_missing_option_quotes(
    *,
    symbol: str,
    locks: Iterable[Any],
    option_quotes: dict[tuple[str, str], dict[str, float]] | None = None,
    subscribed_con_ids: Iterable[int] | None = None,
    allowed_dte: Iterable[int] | None = None,
    ticker_snapshots: dict[int, dict[str, Any]] | None = None,
) -> OptionQuoteDiagnosis:
    """Classify a no-quote condition into an actionable root cause.

    Expected healthy path when front DTEs are missing:
    lock nearest standard-class expiry and receive bid/ask. A missing NBBO after
    that is a separate upstream problem and must be localized, not blamed on DTE.
    """
    symbol_u = str(symbol or "").upper()
    rows = list(locks or [])
    if not rows:
        return OptionQuoteDiagnosis(
            symbol=symbol_u,
            code="empty_lock",
            detail="no contracts locked",
            actionable="check chain discovery / allowed_dte / nearest fallback",
            exclude=True,
        )

    locals_ = [str(getattr(lock, "local_symbol", "") or "") for lock in rows]
    con_ids = [int(getattr(lock, "con_id", 0) or 0) for lock in rows]
    dtes = sorted({int(getattr(lock, "front_dte", -1) or -1) for lock in rows})
    allowed = {int(value) for value in (allowed_dte or [])}
    nearest_fallback = bool(allowed) and bool(dtes) and not set(dtes).issubset(allowed)
    adjusted = any(is_adjusted_local_symbol(symbol_u, local) for local in locals_ if local)
    if adjusted:
        return OptionQuoteDiagnosis(
            symbol=symbol_u,
            code="adjusted_stub_class",
            detail=f"locked adjusted locals={locals_[:4]} dte={dtes}",
            actionable=(
                "prefer tradingClass==symbol and fall back to nearest liquid expiry; "
                "do not subscribe adjusted stubs such as 2MSFT"
            ),
            exclude=True,
        )

    quotes = option_quotes or {}
    if any((symbol_u, local) in quotes for local in locals_):
        code = "nearest_fallback_quoted" if nearest_fallback else "ok_quoted"
        return OptionQuoteDiagnosis(
            symbol=symbol_u,
            code=code,
            detail=f"nbbo present dte={dtes} locals={locals_[:2]}",
            actionable="none",
            exclude=False,
        )

    subscribed = {int(value) for value in (subscribed_con_ids or [])}
    missing_sub = [con_id for con_id in con_ids if con_id > 0 and con_id not in subscribed]
    if missing_sub:
        return OptionQuoteDiagnosis(
            symbol=symbol_u,
            code="not_subscribed",
            detail=(
                f"locked conIds missing from subscriptions={missing_sub[:6]} "
                f"nearest_fallback={nearest_fallback} dte={dtes}"
            ),
            actionable="raise max_option_subscriptions or ensure preferred DTE ladder is subscribed",
            exclude=False,
        )

    snaps = ticker_snapshots or {}
    alive_no_nbbo = []
    for con_id in con_ids:
        snap = snaps.get(int(con_id)) or {}
        bid = float(snap.get("bid") or 0.0)
        ask = float(snap.get("ask") or 0.0)
        has_model = bool(snap.get("has_model"))
        if has_model or float(snap.get("close") or 0.0) > 0 or "bid" in snap:
            if not (ask >= bid > 0):
                alive_no_nbbo.append(con_id)
    if alive_no_nbbo:
        return OptionQuoteDiagnosis(
            symbol=symbol_u,
            code="ticker_alive_no_nbbo",
            detail=(
                f"market data callbacks alive but bid/ask invalid for conIds={alive_no_nbbo[:6]} "
                f"nearest_fallback={nearest_fallback} dte={dtes} locals={locals_[:2]}"
            ),
            actionable=(
                "verify OPRA entitlement and that locked series is the liquid equity class; "
                "if only adjusted stubs exist for front DTE, nearest standard-class fallback is required"
            ),
            exclude=False,
        )

    if nearest_fallback:
        return OptionQuoteDiagnosis(
            symbol=symbol_u,
            code="nearest_fallback_awaiting_nbbo",
            detail=f"using nearest dte={dtes} locals={locals_[:2]} but NBBO not observed yet",
            actionable="wait for OPRA NBBO on nearest expiry; if persistent, inspect exchange/entitlement",
            exclude=False,
        )

    return OptionQuoteDiagnosis(
        symbol=symbol_u,
        code="awaiting_nbbo",
        detail=f"subscribed locks have no NBBO yet dte={dtes} locals={locals_[:2]}",
        actionable="confirm reqMktData and OPRA live feed; retry quote wait",
        exclude=False,
    )
