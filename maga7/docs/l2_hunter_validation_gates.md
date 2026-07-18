# L2（L1 + washout_reclaim Hunt）升线验证闸门

**日期：** 2026-07-18（P2.1 更新）  
**Profile：** `…_watchdog_hunter_washout_reclaim_v1.json`  
**默认参：** `wash_drop_min=0.015` · `allow_baseline_opposite=true` · `mutex_scope=symbol_dir`  
**Hunt 退出（基线）：** 与 L0 相同 `hold_extend` T30→T45（无 Hunt 专用覆盖）

## 闸门总表

| # | 闸门 | 结果 | 明细 |
|---|------|------|------|
| G1 | 双窗 scoreboard vs L0 ≥95% | **PASS**（P2.1 后） | May–Jul **132%** / Feb–Apr **118%**（原 155%/108%） |
| G2 | **流式对拍** offline↔stream（`stock_1s`） | **PASS**（P2.1） | 60 笔对齐 |
| G3 | **邻域** wd∈{1.2,1.5,1.8}% × opp on/off | **PASS_NEIGHBORHOOD** | 默认稳；**勿放宽到 1.2%** |
| G4 | Hold-out OOS（P0） | **观察**（非硬否决） | 2025H2 波动市 + 周五期权 vs 2026 短 DTE；旧 L2 仍 ~172% L0 |
| G5 | 升 research_baseline | **YES**（2026-07-18） | 门槛改为：**Feb–Apr vs L0 提升**（~108%）；并入 **旧 L2 T30+extend** |

**升线裁决：已并入 `peer3_v1`（旧 L2 / +1255% 退出）。** P2.1 `hold20_noext` 仅作备选研究，未进基线。

---

## G2 — 流式对拍

工程：`stream_engine` 已镜像 Hunt 注入 + `hunt_trade_overrides`（含 hold/exit）。

```bash
python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_watchdog_hunter_washout_reclaim_v1.json \
  --scheme single --stock-source stock_1s \
  --start-date 2026-05-01 --end-date 2026-07-17 \
  --tag parity_l2_hold20_noext_20260501_0717
```

| tag | period | n | ok |
|-----|--------|--:|:--:|
| `parity_l2_hold20_noext_smoke_20260507_15` | 05-07..15 | 10 | true |
| `parity_l2_hold20_noext_20260501_0717` | 05-01..07-17 | **60** | **true** |
| `parity_l2_hunter_washout_reclaim_20260501_0717` | 05-01..07-17 | 60 | true（旧退出，历史） |

`only_*=0`，`ret` / `size_frac` / `reason` 差 = 0。P2.1 全窗 `stock_1s` 权益约 **+894%**（与 cache_1m 消融 ~+840% 口径不同，对拍一致即可）。

---

## G3 — 邻域扫描

```bash
python -m maga7.tools.run_hunter_wd_neighborhood \
  --out maga7/results/watchdog/hunter_wd_neighborhood
```

| wash_drop | opp | vs L0 强 | vs L0 弱 | 双窗≥90% |
|----------:|-----|---------:|---------:|:--------:|
| 1.2% | off/on | 121–152% | **~60%** | **FAIL** |
| **1.5%** | off | 123% | 108% | PASS |
| **1.5% + opp（默认）** | on | **155%** | **108%** | **PASS**（旧退出） |
| 1.8% | off/on | 102–129% | ~104% | PASS |

**Verdict: `PASS_NEIGHBORHOOD`** — 默认点不孤立；**禁止把 `wash_drop_min` 降到 1.2%**（弱窗断崖）。  
注：邻域表仍为旧 Hunt 退出；P2.1 未改 wd/opp。

---

## G4 — OOS + P2.1

| window | L0 | L1 vs L0 | L2 default vs L0 | L2 `hold20_noext` |
|--------|-----|----------|------------------|-------------------|
| 2026-01 | +47% | 100% | 100% | （同空通过） |
| **2025 H2** | +28% | **191%** | **172%**（&lt; L1） | **~228% L0 / ~118% L1** |

消融明细：[`hunt_exit_ablation_p21.md`](hunt_exit_ablation_p21.md) · `results/watchdog/hunt_exit_ablation_p21/`。

---

## 升线依据（最终）

- Feb–Apr：L0 ~+140% → L2 ~+152%（**~108% of L0**）→ 短期足够  
- May–Jul：~**+1255%**（~155% L0）  
- 2025H2：不作硬否决；P2.1 打折退出不进基线  

## 下一步

1. Live/shadow 观测 Hunt 触发与增量  
2. 禁止把 `wash_drop_min` 降到 1.2%  
3. `hold20_noext` 仅在需要压 Hunt 阴跌时再议
