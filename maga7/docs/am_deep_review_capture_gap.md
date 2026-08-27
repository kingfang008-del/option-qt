# AM 深审：机会很多，为何账上很少？

> 复盘日期：2026-08-01  
> 证据：`research_am_vwap_foresight_map_may_jul` · `research_am_pocket_*` · `research_am_morning_densify_20260728` · Pulse A/B dual  
> 问题：每天 Mag7 早盘波动很大，为什么策略只抓住很少利润？

## 一句话

**边大量存在，但是「路径边」；现有臂用「稀触发 + 固定 TP」故意少做，加仓/降阈值并不能线性放大利润。**  
不是「没机会」，是 **选哪一笔 + 怎么吃路径** 没解决。

---

## 1. 机会有多大？（前视，非可交易）

协议：60s 网格 × A/B 窗 × ATM 方向；对齐 `dir == sign(from_open)`；edge = `oracle_ret@900s ≥ 15%`。

| 指标 | discover may–jul09 | blind jul10–23 |
|------|-------------------:|---------------:|
| 对齐探针 / 日（中位） | ~620 | ~ — |
| **edge 探针 / 日（中位）** | **~400** | **~265** |
| 当日有 edge 的名字数 | **≈8/8** | ≈8/8 |
| 对齐 edge_rate | 58.7% | 55.7% |
| mean **oracle**（路径最优卖） | **~+30%** | ~+28% |
| mean **clock@900s**（死拿） | **≈−0.3%** | ≈−0.2% |

读法：

- 「每天那么多机会」——前视上成立：几乎每天每只 Mag7 都有可达 ≥15% 的期权路径。  
- 同批 **固定拿 15 分钟几乎没边** → 利润在 MFE 路径里，不在「买了放着」。  
- 探针高度重叠（同一波段被 60s 网格反复点到），**≠ 每天有 400 笔独立可下单机会**。独立维度更接近 **每天 1–8 个名字级波段**。

---

## 2. 策略实际抓了多少？

| 臂 | 密度（约） | 经济结果 | 状态 |
|----|-----------:|----------|------|
| 口袋 champ TP8 | **0.7 笔/日**（35/≈50 日） | disc +44% / blind +5%，**capture≈13%** | trade-last PASS，未接线 |
| Pulse A FO0.8 DN | **~0.9 笔/日** | 旧 quote lag5 很强；**decision+60 因果 quote REJECT** | shadow，因果存疑 |
| densify `fo08_both` | ~1.4 笔/日 | add ×1.35 vs 基线 | 质量稀释 |
| densify `fo06_both_a2` | **~2.3 笔/日** | n×2.5 但 add 仅 ×1.36 | **加量不加成** |
| launch_slope / activity / session | — | 双窗 FAIL / quote REJECT | 已弃 |

口袋自身：`mean_oracle≈56%`，实现 `mean_ret≈7%`，**capture≈13%**——和「TP8 相对 oracle≈30–48% → 理论天花板 ~17%」同阶。  
**固定 TP8 不是小 bug，是主漏斗。**

---

## 3. 漏斗：钱漏在哪五层

```text
每天 ~8 名 × 多段路径 oracle
        │
        ▼ ① 时间重叠：60s 网格把 1 波算成几十个「edge」
        │
        ▼ ② 入场门控：FO / vd_soft∩cont∩mf∩volr / peer / max_alerts=1
        │     → 落到 0.7–2 笔/日
        │
        ▼ ③ 可执行性：09:30–10:00 历史 quote 中位 lag~30min
        │     → trade-last 能记账，FillSpec 不能；因果 bar_delay 再砍一刀
        │
        ▼ ④ 退出：固定 TP8/TP15 只吃 oracle 的 ~1/6–1/3
        │     path_exit / scaleout / flip 扫过 → 没抬 capture
        │
        ▼ ⑤ 账本：20%×max5 + 与 CORE 硬互斥 + 10:45 flatten
              → 即便单笔不错，日内复利上限被座位卡住
```

densify 铁证：**笔数 ×2.5 ≠ 利润 ×2.5**（retain add≈1.36）。  
说明当前阈值附近已经在挖「次优 FO」，不是「还有一大池同质 alpha 没进场」。

---

## 4. 对各臂的再裁决（2026-08）

| 臂 | 是否「机会没抓到」？ | 真正瓶颈 |
|----|---------------------|----------|
| **Pulse A/B** | 部分是——降 FO / 加 UP 能加笔，但边际差；因果 delay 后可交易性崩 | **可成交 + 因果时序**，不是 FO 网格 |
| **口袋** | 是——capture 13% 是主伤；宇宙还可略扩（vd+volr12） | **退出吃路径**；quote 离线不可验 |
| **launch / activity / morning_sleeve** | 否——双窗已否，不是「没做满」 | 边不稳 / quote 死 |
| **再扫 FO×TP×SL 全窗** | **禁止**（前视 lift≈1.0） | 方向错 |

---

## 5. 口径修正（用户裁决）

1. **离线账本只用 trade prints**（+slip）。09:30 历史 quote **不再当研究硬闸**；quote 只留给 live/IB。  
2. **退出不应写死一套 TP/SL**；应按行情分档 / 路径自适应（regime ladder）。  
   - 已试：`tools/scan_am_pocket_regime_ladder.py` → `research_am_pocket_regime_ladder`  
   - 首版 CHOP/TREND/IMPULSE 阶梯：econ dual 过，但 **capture 0.108 &lt; 固定 TP8 的 0.129**，disc 复利也更低；`path_adapt` fail-fast 直接打爆。  
   - 结论：方向对，**参数/状态机还没赢过 TP8**——继续调 ladder，而不是退回 quote 纠结。

## 6. 若要「抓住更多」，正道

1. **trade-mark + 动态阶梯（主线）** — 抬 capture≥20% 且 dual 不炸。  
2. **名字级 Top1–2/日** — 解决「400 探针 ≠ 独立单」。  
3. Live 才谈 NBBO；离线永远 trade-last。

**明确不做：** 全窗 FO thr 网格、再拿历史 quote 否决口袋、用固定 hold 替代路径退出。

---

## 6. 建议的下一实验（单刀）

**`scan_am_name_rank_capture`（新建）**

- 输入：对齐 foresight probes（或 1s FO 事件）  
- 每日因果打分 → 强制 Top1 / Top2  
- 退出：基线 TP8 vs 二段 TP（8/20）vs trail  
- 双窗：retain add、capture、blind n  
- 成功线：相对口袋 champ，`disc compound≥+44%` 且 `n≥1.5×` 且 capture≥20%

未过线则承认：**AM 期权卫星的经济上限就是「少而精 + 低 capture」**，主账仍归 CORE；AM 只做 shadow 卫星。
