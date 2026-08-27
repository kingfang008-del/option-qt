# Peer3 单笔尾损优先（状态记录）

**Status (2026-07-22):** research / **shadow** — **未**并入 `research_baseline`（`...peer3_v1`）。  
**Goal 切换：** 主优化 **单笔尾损**（不是 May–Jul 总收益倍数）。  
**产品上层：** 尾损栈仍是 ARMED 后管理；若波段本身未确认，见 [`wave_confirm_spec.md`](wave_confirm_spec.md)（确认失败不得持有到时钟）。

## 验收条（本轮）

排序键（升序优先）：

1. `n(ret ≤ −25%)`
2. `worst`（单笔最差）
3. `left_tail_sum`（亏损笔收益和）
4. `total_ret` 仅作次要参考

产物：`/mnt/s990/data/maga7/results/peer3_tail_loss_scoreboard_v1/`  
（`scoreboard.csv` · `verdict.json` · `scoreboard_combo.json`）

## Shadow profile

| Profile | Role | 相对 peer3_v1 的 diff |
|---------|------|------------------------|
| `CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_peer3_tail_tox20_wash_m3_v1.json` | **`research_tail`** | `trade_toxic.cut_ret` **0.25→0.20** + L3 `STOCK_REV` = wash_m3（`mixed_wash_up` · stock≤−0.3% · opt MTM≤0 · ≥10m） |

基于：[`peer3_l3_wash_m3_v1`](../CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_peer3_l3_wash_m3_v1.json)（见 [`peer3_l3_causal_exit_research.md`](peer3_l3_causal_exit_research.md)）。

## 记分板摘要

### May–Jul（窗至 2026-07-21）

| 变体 | ≤−25% 笔数 | worst | total_ret |
|------|----------:|------:|----------:|
| baseline | 4 | −27% | +17.5 |
| tox_cut20 alone | 2 | −26% | +14.1 |
| l3_uw_m3_h15 | 3 | −27% | +15.8 |
| l3_wash_m3 | 4 | −27% | +16.2 |
| **tox20 + l3_wash（采用）** | **1** | −26% | **+13.3** |
| sl40 | 5 | **−43%** | +15.7 |
| srev_m3_h5_always | 4 | −38%（STOCK_REV） | +7.3 |

### Jul21（AMD 毒笔）

| 变体 | AMD 出场 |
|------|----------|
| baseline / tox_* / sl40 | T+30 **−26%**（tox 未截住时钟磨） |
| **tox20 + wash_m3** | STOCK_REV **−16%** |
| l3_uw_m3_h15 | STOCK_REV −19% |
| srev_h5_always | STOCK_REV −11%（强窗不可接受） |

### Jan–Mar

| 变体 | ≤−25% | worst |
|------|------:|------:|
| baseline | 10 | −41% |
| tox_cut20 | **5** | −41% |
| tox20 + wash_m3 | **5** | −41% |

弱窗最差笔仍是 Jan-02 AMZN T+30 −41%——本组合 **减频不减最深单笔**；更深尾损要另刀（更早路径截断 / 日态停手）。

## 结论与否决

**采用（shadow）：** `tox_cut20`（减 ≤−25% 笔数）+ `wash_m3` L3（砍 T30 路径磨，如 Jul21）。

**否决：**

- `sl40`（硬 SL 更紧）→ worst 更差  
- 单独 `srev@5m always` → 自造 STOCK_REV 尾损、强窗收益崩  

**未解决：** Jan–Mar 仍有 ~−40% 级 T30；Jul20 offline 本轮无成交（锁约未含该日，fused 证据仍见 L3 文档）。

## 与基线关系

- `...peer3_v1` **保持** research_baseline（收益导向栈不变）  
- 本 profile = **尾损优先对照 / shadow**  
- 升线门槛（另议）：双窗 `n≤−25%` 不回升 + worst 不劣于基线 + OMS 确认 `STOCK_REV` + 接受强窗收益缺口  

## Ops

```bash
# offline 单日/窗
python -m maga7.tools.run_replay_offline \
  --profile maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_peer3_tail_tox20_wash_m3_v1.json \
  --scheme single \
  --start-date 2026-07-21 --end-date 2026-07-21 \
  --tag replay_tail_tox20_wash_2026-07-21
```

Live：profile 指向本文件；OMS early-exit 须含 `STOCK_REV` 与 `TRADE_TOX`。
