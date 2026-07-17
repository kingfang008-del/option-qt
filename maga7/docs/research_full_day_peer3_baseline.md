# Research baseline: full_day peer3 + streak3 + mf_idio

**Profile:** `CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json`  
**Frozen:** 2026-07-18  
**Role:** `research_baseline` (research / shadow default; not yet production freeze)

## Stack

| Layer | Setting |
|-------|---------|
| Causal core | Mag7+GOOGL open_ladder OTM5, peer_align_min=3, QQQ align, rails delay 60s |
| Exit | `hold_extend` T30→T45, MTM≥0, `hold_extend_require_mf=false` |
| Events | `event_calendar_block` + `event_calendar_live.json` |
| Day halt | `day_loss_streak_halt=3` |
| Size gate | `mf_idio_mode=pos`, `action=scale`, `scale=0.5`, `after_loss_streak=1`, ret β 5d |
| Risk | `sl_mult=0.40`, `position_frac=0.2`, concurrent max 2 |

## Snapshot metrics (offline replay)

| Window | total_ret | MaxDD | notes |
|--------|----------:|------:|-------|
| May–Jul (→07-16) | ~+875% | −13.2% | primary research window |
| Feb–Apr | ~+140% | −28.9% | weaker tape; bad3 eased vs pre-streak3 |
| Trades (both windows) | n≈123 | win≈54% | payoff≈2.2 (avg win / \|avg loss\|) |

## Ops

- Live/shadow default: `maga7/SHELL/start_maga7_live_session.sh`
- Calendar sync: `./start_maga7_live_session.sh sync-calendar`
- Replay results stay local under `maga7/results/` (gitignored)

## Non-goals in this freeze

- Do not bare-enable `mf_flip` / streak early exits (hurts May–Jul)
- Do not global-tighten `sl_mult` for premium stops
- TCN / DL probability gates are research follow-ups, not part of this freeze
