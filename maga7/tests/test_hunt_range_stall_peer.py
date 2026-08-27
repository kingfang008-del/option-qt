from __future__ import annotations

from maga7.common.range_stall_gate import parse_range_stall_gate


def test_parse_hunt_peer_align_default_false():
    cfg = parse_range_stall_gate({"enabled": True, "max_peer": 5})
    assert cfg.hunt_peer_align is False


def test_parse_hunt_peer_align_true():
    cfg = parse_range_stall_gate(
        {"enabled": True, "hunt_peer_align": True, "peer_pre5_max_peer": 3}
    )
    assert cfg.hunt_peer_align is True
