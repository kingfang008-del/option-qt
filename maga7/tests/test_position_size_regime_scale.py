from maga7.common.position_size import (
    apply_size_scale,
    coerce_size_scale,
    regime_scale_from_meta,
    resolve_size_frac,
)


def test_coerce_and_apply_size_scale():
    assert coerce_size_scale(0.5) == 0.5
    assert coerce_size_scale(1.5) == 1.0
    assert coerce_size_scale(-1) == 0.0
    assert apply_size_scale(0.2, 0.5) == 0.1


def test_regime_scale_from_meta():
    assert regime_scale_from_meta({"regime_size_scale": 0.5}) == 0.5
    assert regime_scale_from_meta({}) == 1.0


def test_resolve_then_apply_matches_offline_pattern():
    trade = {"position_frac": 0.2, "position_sizing": "concurrent", "max_concurrent_positions": 2}
    size, mode, n, allow, reason = resolve_size_frac(trade, top_k=2, open_until={}, symbol="NVDA", entry_ts=None)
    assert allow and size == 0.2
    assert apply_size_scale(size, 0.5) == 0.1
