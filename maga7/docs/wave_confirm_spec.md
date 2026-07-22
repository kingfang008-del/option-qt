# 波段确认（Wave Confirm）— 最小因果规格

**Status (2026-07-22):** research spec only — **无参数网格、不升线、不改 peer3_v1**。  
**动机：** peer3 生产不合格的根因不是「倍数不够」，而是 **波段从未被确认**——Rule-A 只点火，T30 在未证伪路径上硬扛。  
**产品句：** 微观优势 = **先确认一段波段，再在波段内做最好的标的表达**；行情叙事只作背景，不作主依赖。

相关：[`toxic_callback_factor_autopsy.md`](toxic_callback_factor_autopsy.md) · [`entry_quality_false_break_research.md`](entry_quality_false_break_research.md) · [`flow_state_event_architecture.md`](flow_state_event_architecture.md) · [`peer3_tail_loss_research.md`](peer3_tail_loss_research.md)

---

## 1. 问题定义

| 现状 | 规格要求 |
|------|----------|
| Rule-A（mf10 / streak / peer）= 候选点火 | 点火 **≠** 允许持有满时钟 |
| 入场横截面几乎分不开毒/赢 | **禁止**再加超买超卖 / `|fp|` 帽当主门 |
| 毒笔在入场后 1–5m 已分叉（期权阴跌、正股几乎不跟） | **确认窗**必须落在持仓早期路径 |
| T30/T45 当主裁判 | 时钟降级为 **未确认时的硬废止上限**，不是续持理由 |

一句话：**没有波段确认，就没有生产级持仓权。**

---

## 2. 状态机（因果顺序）

```
Rule-A 候选 (flow)
      │
      ▼
  FILL 探针仓          ← 允许极小/标准仓，但持仓权未授予
      │
      ▼
 ┌─ WAVE_CONFIRM 窗 ─┐     正股路径必须「同向推进且未逆向废证」
 │  pass → ARMED     │──► 才允许持有到 TP / 软出场 / 时钟上限
 │  fail → ABORT     │──► 立即平仓（原因 WAVE_ABORT），禁止拖到 T30
 │  timeout          │──► 默认 ABORT（硬确认）；研究对照才允许 soft-allow
 └───────────────────┘
      │
      ▼
  ARMED 持仓           ← 波段内管理（trail / toxic / STOCK_REV…）
      │
      ▼
  EXIT                 ← TP / 因果软出场 / 时钟安全阀
```

与旧栈差异：

- **旧：** fill → 默认可拿到 T30  
- **新：** fill → 仅探针；**确认失败不得进入「波段交易」语义**

---

## 3. 确认对象（先股票，后期权）

波段是 **标的价格路径**，不是期权 MTM。

| 证据 | 角色 |
|------|------|
| 入场后签名股票收益 `stock_signed` | **主确认**（方向推进） |
| 确认窗内最大逆向 `stock_adverse` | **废证**（假突破） |
| 期权 MTM | **副证据**（可加速 ABORT，不可单独 ARMED） |
| QQQ / VIXY / 叙事 | **日态臂**（可缩仓/停新手），不替代单笔确认 |

设计约束（来自已有否决）：

- Hard「必须涨够才给仓」会砍强窗赢家 → 规格默认用 **adverse-first 废证 + 超时默认废**，而不是「必须打到 +X 才算确认」。  
- Soft path（`on_timeout=allow`）只作研究对照，**不是生产语义**（生产要的是「确认不了就不做波段」）。

---

## 4. 最小规则（逻辑规格，非调参）

记：

- `t0` = 期权成交时刻  
- `W` = 确认窗长度（秒），量级 **60–300s**（与尸检 1–5m 分叉对齐；具体值后验）  
- `stock_signed(t)` = 相对 fill 的签名股票收益  
- `stock_min(t)` = 窗内最差签名收益（逆向极值）

### 4.1 ABORT（任一触发即废，原因 `WAVE_ABORT`）

1. **逆向废证：** 存在 `t ∈ (t0, t0+W]` 使 `stock_min(t) ≤ −A`（`A>0`，小阈值逆向）  
2. **期权加速废证（可选 AND/OR）：** `opt_mtm(t) ≤ −M` 且股票未同向推进（防止「股横盘、权已死」拖满窗）  
3. **超时未武装：** 到 `t0+W` 仍未满足 ARMED → **ABORT**（生产默认）

### 4.2 ARMED（全部满足才授予持仓权）

1. 确认窗内 **未**触发逆向废证  
2. **推进证据（弱）：** 存在时刻使 `stock_signed ≥ +P`（`P` 可为 0+ε，表示「动过」；或用「同向时间占比」）  
   - 若实证表明弱推进仍放进大量毒笔，再升为更强推进条；**先记录，后网格**  
3. （可选）同向 peer/QQQ 未全面翻脸——只作加严，不作唯一条件

### 4.3 ARMED 之后

- 才允许：TP、trail、L3 `STOCK_REV`、`trade_toxic`、时钟上限  
- **未 ARMED 禁止：** `hold_extend` 续礼、Hunter 扩仓、加仓  

### 4.4 与现有旋钮的关系

| 已有 | 在本规格中的位置 |
|------|------------------|
| `trade.stock_path_confirm`（soft / hard） | **雏形**；生产语义应对齐本文件的 ABORT-default timeout，而非 `on_timeout=allow` |
| `trade_toxic` / L3 `STOCK_REV` | **ARMED 后** 的持仓管理，不替代确认 |
| `delta_time_stop` / `roi_time_stop` always-on | 曾 REJECT（毁强窗）；不进本规格默认 |
| 入场 `|fp|` 帽 / 全周 confirm | **禁止**当波段确认 |

---

## 5. 标的选择（波段内，第二步）

仅在 **同日已有 ARMED 波段**（或候选池同步确认）时：

- 在 Mag7 池内按 **相对强度 / 同向推进质量** 选表达标的（谁的路径更干净），而不是「谁先亮 Rule-A 谁上」。  
- 本规格 **v0 不实现选股秩**；先把「单笔确认失败 → 必废」做实。  
- 锁约/短链不可交易（如 Jul21 TSLA DTE3）记为 **不可表达**，不是策略胜利。

---

## 6. 验收条（通过才谈实现升线）

相对 peer3 基线，**同一成交集合或同信号池** 上：

| 门 | 要求 |
|----|------|
| 尾损 | `n(ret≤−25%)` 明显下降；Jul21 类 T30 毒磨应变为 `WAVE_ABORT`（损失应显著浅于 −25%） |
| 强窗 | 总收益 retain **不作为主门**；但若 retain ≪ 0.5 且尾损未改善 → 规格失败 |
| 弱窗 | 最差笔 / ≤−25% 不恶化 |
| 语义 | ABORT 占比可审计；**禁止**大量「未确认却 T30 离场」 |
| 否决 | 仅靠收紧 SL / 入场横截面「假确认」 |

未过门 → 维持「研究失败、不上生产」结论，不改 peer3_v1。

---

## 7. 非目标（v0）

- 不预测 Call/Put；方向仍来自 Rule-A  
- 不把 VIX/成交量当单笔确认（可作日后态臂）  
- 不在本规格内调 `cut_ret` / wash_m3（那是 ARMED 后尾损栈）  
- 不开大网格；实现后先 **单窗尸检 + 小对照**（baseline vs WAVE_ABORT-default）

---

## 8. 最小对照结果（2026-07-22）— 成交前 hard

产物：`/mnt/s990/data/maga7/results/wave_confirm_min_contrast_v1/`  
实现代理：**成交前** `stock_path_confirm`（`thr_pos=+15bp` / `thr_neg=−30bp` / 300s）。  
`wave_hard` = `on_timeout=block`；`wave_soft` = `on_timeout=allow`；`wave_hard_delay` = hard + 确认后再定价成交。

### May–Jul（→07-21）

| 变体 | total_ret | retain | n | path_block | ≤−25% | worst |
|------|----------:|-------:|--:|-----------:|------:|------:|
| baseline | +17.54 | 1.00 | 58 | 0 | 4 | −27% |
| **wave_hard** | +6.78 | **0.42** | 37 | 27 | **2** | −26% |
| wave_soft | +17.61 | 1.00 | 55 | 7 | 3 | −27% |
| wave_hard_delay | +1.57 | 0.14 | 37 | 27 | 7 | **−58%** |

### Jan–Mar

| 变体 | total_ret | n | ≤−25% | worst |
|------|----------:|--:|------:|------:|
| baseline | +0.91 | 58 | 10 | −41% |
| **wave_hard** | **+1.04** | 20 | **0** | **−13%** |
| wave_soft | +0.98 | 53 | 7 | −41% |
| wave_hard_delay | −0.08 | 19 | 3 | −56% |

### Jul21 AMD

| 变体 | 结果 |
|------|------|
| baseline / soft / hard | 仍成交，**T+30 −26%** |
| hard | 挡掉 1 个候选（TSLA），**AMD 确认通过**后仍磨到 T+30 |

### 读数（prefill）

1. **方向有效（弱窗）：** hard 把 Jan–Mar ≤−25% 从 10→0。  
2. **强窗代价大：** retain **0.42**。  
3. **Jul21 类未被挡住：** 一次 touch 即 ARMED 不够 → 需要成交后可撤销废证。  
4. **hard_delay 否决**。

---

## 8b. 成交后 `WAVE_ABORT`（2026-07-22）

产物：`/mnt/s990/data/maga7/results/wave_abort_postfill_v1/`  
实现：`maga7/common/wave_confirm.py` → replay exit `WAVE_ABORT`；OMS 镜像。  
参数（单点，非网格）：`thr_pos=+15bp` / `thr_neg=−30bp` / `max_wait=300s` / `revoke_seconds=1800` / `on_timeout=abort`。  
工具：`python -m maga7.tools.run_wave_abort_ablation`  
Profile（research only）：`..._peer3_wave_abort_v1.json` — **不升线**。

| 窗 | 变体 | total_ret | retain | n | ≤−25% | worst | WAVE_ABORT | clock_share |
|----|------|----------:|-------:|--:|------:|------:|-----------:|------------:|
| May–Jul | baseline | +17.54 | 1.00 | 58 | 4 | −27% | 0 | 0.47 |
| May–Jul | **wave_abort** | +2.54 | **0.19** | 58 | 3 | −30% | **34** | 0.12 |
| May–Jul | hard_prefill | +6.61 | 0.41 | 33 | 2 | −26% | 0 | 0.42 |
| Jan–Mar | baseline | +0.91 | 1.00 | 58 | 10 | −41% | 0 | 0.64 |
| Jan–Mar | **wave_abort** | +0.23 | 0.65 | 58 | **5** | −30% | **38** | 0.16 |
| Jan–Mar | hard_prefill | +1.04 | 1.07 | 20 | **0** | −13% | 0 | 0.65 |
| Jul21 | baseline | −5.2%日 | — | 1 | 1 | **T+30 −26%** | 0 | 1.0 |
| Jul21 | **wave_abort** | −2.1%日 | — | 1 | **0** | **WAVE_ABORT −11%** | 1 | 0 |
| Jul21 | hard_prefill | −5.2%日 | — | 1 | 1 | T+30 −26% | 0 | 1.0 |

### 读数（postfill）

1. **机制对 Jul21 成立：** AMD 先触 +15bp 再逆向 → `revoke` → **−11%**，不再磨到 T+30 −26%。Hard prefill 仍放行。  
2. **强窗 FAIL（§6）：** May–Jul retain **0.19 ≪ 0.5**；34/58 被 `WAVE_ABORT`；其中 **13 笔 baseline TP（合计 +8.0）被 revoke 砍掉**（典型：先推进再 −30bp 回撤后仍能走到 TP）。  
3. **弱窗半改善：** Jan–Mar ≤−25% 10→5，但仍不如 hard_prefill 的 0；且保留大量误杀 TP。  
4. **尾损门未过：** May–Jul ≤−25% 仅 4→3，worst 甚至略差（AMZN/NVDA 在废证时已深亏）。

**结论：** 可撤销废证 **方向正确、默认参数过热** — 裸 `signed≤−30bp` 在 `revoke=1800s` 内当生产确认会毁掉强窗趋势回撤。  
**不升线。** → 见 §8c。

---

## 8c. Knife-2 / 2b（降温废证，2026-07-22）

产物：  
- v2（短 revoke / 非对称 / opt 同坏）：`.../wave_abort_postfill_v2/`  
- v2b（**timeout 语义**）：`.../wave_abort_postfill_v2b/`  
旋钮扩展：`thr_neg_revoke` / `revoke_opt_mtm_max` / `allow_revoke`（`wave_confirm.py`）。  
工具：`python -m maga7.tools.run_wave_abort_ablation --mode knife2|knife2b`。

### 尸检：谁在杀强窗 TP？

对 `no_revoke`（只做未 ARMED 废证）：25 笔 `WAVE_ABORT` 里 **23 笔 held≈300s** → **主杀因是 `on_timeout=abort`**，不是 revoke。  
被砍的 baseline TP 几乎全是「5 分钟内未触 +15bp、之后仍走到 TP」的慢启动赢家。

### Knife-2（只降温 revoke）— FAIL

| 变体 | May–Jul retain | n_abort | ≤−25% | Jul21 |
|------|---------------:|--------:|------:|-------|
| hot_rev1800 | 0.19 | 34 | 3 | −11% OK |
| rev600 / rev900 | 0.19 | 32 | 4 | −11% OK |
| asym_r5* / opt0* | ≤0.21 | 30–34 | 3–6 | −11%~−14% OK |
| no_revoke | 0.25 | 25 | 5 | **仍 T+30 −26%** |

短 revoke / 更深 revoke thr / opt 同坏 **几乎不动 retain**。`no_revoke` 证明 Jul21 需要 revoke，且 timeout 才是强窗毒药。

### Knife-2b（`on_timeout=allow` + 保留 revoke）— 强窗接近门

| 变体 | May–Jul ret / retain | ≤−25% | Jan–Mar ≤−25% | Jul21 |
|------|---------------------:|------:|--------------:|-------|
| baseline | +17.54 / 1.00 | 4 | 10 | T+30 −26% |
| hot_abort_timeout | +2.54 / 0.19 | 3 | 5 | WAVE_ABORT −11% |
| **allow_to_rev1800** | **+11.69 / 0.68** | **3** | 10 | **WAVE_ABORT −11%** |
| allow_to_rev600 | +11.80 / 0.69 | 4 | 10 | −11% |
| allow_to_asym_r5_rev900 | +12.64 / **0.74** | **6**↑ | 10 | −14% |
| wait600/900_abort | +4.0~5.7 / ≤0.36 | 4 | 6–11 | −11% |

### 读数

1. **生产语义修正：** 确认窗超时默认应从硬 ABORT 改为 **allow（未武装则交给后续 tox/clock）**；**逆向 / revoke 仍硬废**。这与 §3「soft 只作研究」冲突处：实证表明硬 timeout 不可用，**revoke 才是 Jul21 的必要件**。  
2. **研究 shadow 候选：** `allow_to_rev1800`（profile `..._peer3_wave_abort_v1.json`）— Jul21 过、May–Jul retain≈0.68（近 0.7）、≤−25% 略好；**Jan–Mar 尾损回到基线 10，不帮忙**。  
3. `asym_r5` 把 retain 抬过 0.7，但 May–Jul ≤−25% 恶化到 6 → **否决**。  
4. **仍不升线：** 弱窗尾损未改善；强窗仍让出 ~30% 收益；需与 tox20/wash 等 ARMED 后栈叠才谈生产。

---

## 9. 建议实现顺序

1. ~~对照：peer3 vs timeout-block~~（§8）  
2. ~~成交后探针 + 可撤销废证~~（§8b）  
3. ~~降温 revoke / 修 timeout~~（§8c；shadow=`on_timeout=allow`+revoke）  
4. **下一刀：** `allow_to_rev1800` × tail 栈（tox20 / wash_m3），看 Jan–Mar ≤−25% 与 May–Jul retain 能否同时过门  
5. 通过 §6 后再谈波段内选标的秩  

---

## 10. 一句话冻结

> **没有股票路径上的波段确认，就没有持仓到时钟的权利。**  
> 确认必须可撤销（Jul21）；**硬 timeout=abort 会误杀慢启动 TP，不能用。**  
> 研究影子：`on_timeout=allow` + `revoke@−30bp/1800s` — **仍不进 peer3_v1。**
