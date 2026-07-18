# P2.1 Hunt 专用退出消融

**日期：** 2026-07-18  
**工具：** `python -m maga7.tools.run_hunt_exit_ablation`  
**产物：** `results/watchdog/hunt_exit_ablation_p21/`

## 目标

在 **不拧** `wash_drop_min` / opp 的前提下，给 Hunt 单独退出（或仓位），使：

1. **2025H2 OOS：L2 ≥ L1**
2. 双窗（May–Jul / Feb–Apr）vs L0 ≥ 95%

## 选定变体：`hold20_noext`

| 字段 | 值 | 说明 |
|------|-----|------|
| `hunter.hold_minutes` | 20 | 相对基线 T30 更短 |
| `hunter.exit_mode` | `none` | **关闭** hold_extend（阴跌不再磨到 30/45） |
| `hunter.hold_extend_minutes` | 20 | 与 exit_mode=none 配套，不生效 |

未选 `hold15_size10`（OOS 略高但强窗仅 ~104% L0，且混了仓位旋钮）。`hold20_noext` 是纯退出、强窗仍 ~132% L0。

已写入：

- `…_watchdog_hunter_washout_reclaim_v1.json`
- `mu_only_0717_watchdog_hunter_washout_reclaim_v1.json`

## Scoreboard（cache_1m offline）

| variant | strong vs L0 | weak vs L0 | OOS vs L1 | promote? |
|---------|-------------:|-----------:|----------:|:--------:|
| l2_default (T30+extend) | 155% | 108% | **90%** | no |
| **hold20_noext** | **132%** | **118%** | **118%** | **yes** |
| hold15_size10 | 104% | 102% | 119% | yes（备选） |
| hold15_noext | 108% | 111% | 115% | yes |
| size10 | 136% | 101% | 107% | yes |
| mae20_fast | 95% | 94% | 77% | no |

**Verdict: `PASS_P21_CANDIDATE`** — 已落盘 L2 研究 profile。

## 升 research_baseline？

流式对拍（P2.1）已 PASS（`parity_l2_hold20_noext_20260501_0717`）。  
仍 **默认不** 把 `hunter.enabled=true` 并入 `peer3_v1`——需你显式确认升线。
