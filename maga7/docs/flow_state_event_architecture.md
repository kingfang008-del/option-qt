# Flow → Response → State → Event Architecture

Status: **research** (post `DEMOTE_TO_RESEARCH_FEATURE` of pure flow + T30).

## Why

Freeze OOS (2026-07-20) showed:

- ask/bid does **not** kill the May–Jul edge (~78% retained);
- weak Feb–Apr retains only ~10% of strong;
- freeze day was 100% UP concentration with large loss.

So the problem is not fill frac — it is **regime fragility** and **label/hold mismatch**.

## Pipeline

```
Rule-A flow candidate
        ↓
stock_path_confirm   (stock must respond in-direction)
        ↓
state_gate           (trend / mixed_wash / reclaim_trap — veto only)
        ↓
event exits          (delta/roi time-stops; cond ladder on wash)
```

The model / gate **never outputs Call or Put**. Direction still comes from Rule-A.

## Modules

| Stage | Code | Profile knobs |
|-------|------|----------------|
| Flow candidate | `signals.py` / scanner | existing peer3 Rule-A |
| Response confirm | `replay.stock_path_confirm_ok` | `trade.stock_path_confirm` |
| State veto | `common/state_gate.py` | `state_gate` |
| Event exit | `delta_time_stop` / `roi_time_stop` / `ladder_active` | `trade.*` |

## Ablation results (2026-07-21)

| Layer | Strong | Weak | Keep strong | Note |
|-------|--------|------|-------------|------|
| L0 baseline | +1856% | +296% | 100% | demoted pure flow+T30 |
| L1 hard path | +172% | +100% | 9% | timeout-block kills winners |
| L3 full stack | +5% | +31% | ~0% | **REJECT** — always-on time-stops |
| **S1 soft path** | **+1863%** | **+336%** | **~100%** | **research candidate** |
| S0 state_only | +1466% | +321% | 79% | better weak/strong; alt candidate |

Soft path = adverse-first veto only (`on_timeout=allow`, no fill delay).

## Profile

`CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_flow_state_event_v1.json`

Default = **S1 soft path**. `state_gate` present but **off** (enable for S0-style wash veto).

Hard path + always-on `delta_time_stop` / `roi_time_stop` are **not** in the candidate.

## Ablation

```bash
python -m maga7.tools.run_flow_state_event_ablation \
  --out /mnt/s990/data/maga7/results/flow_state_event_ablation_v1
# soft follow-up artifacts:
# /mnt/s990/data/maga7/results/flow_state_event_ablation_v1b/
```

## Live

Offline first. Live scanner still lacks `stock_path_confirm` — must mirror before any size-up.
