from maga7.common.position_size import (
    block_same_dir_after_win_enabled,
    is_symbol_dir_big_win,
    post_win_cooldown_action,
    post_win_cooldown_sessions,
)


def test_post_win_off_by_default():
    assert post_win_cooldown_action({}, prev_day_ret=0.2) == ("", 1.0)


def test_post_win_skip_on_threshold():
    trade = {"post_win_cooldown_mode": "skip", "post_win_cooldown_day_ret": 0.10}
    assert post_win_cooldown_action(trade, prev_day_ret=0.12) == ("skip", 0.0)
    assert post_win_cooldown_action(trade, prev_day_ret=0.05) == ("", 1.0)


def test_post_win_scale_and_sessions():
    trade = {
        "post_win_cooldown_mode": "scale",
        "post_win_cooldown_scale": 0.5,
        "post_win_cooldown_sessions": 2,
    }
    assert post_win_cooldown_sessions(trade) == 2
    assert post_win_cooldown_action(trade, cooldown_left=1) == ("scale", 0.5)
    assert post_win_cooldown_action(trade, cooldown_left=0, prev_day_ret=0.2) == (
        "scale",
        0.5,
    )


def test_same_dir_after_win_helpers():
    assert not block_same_dir_after_win_enabled({})
    assert block_same_dir_after_win_enabled({"block_same_dir_after_win": True})
    trade = {"block_same_dir_after_win_ret": 0.50}
    assert is_symbol_dir_big_win(ret=0.60, reason="TP", trade=trade)
    assert is_symbol_dir_big_win(ret=0.55, reason="T+30", trade=trade)
    assert not is_symbol_dir_big_win(ret=0.20, reason="T+30", trade=trade)
