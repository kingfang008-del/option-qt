# AM 策略架构重审：为什么会变差

> 复盘：2026-08-08  
> 范围：早盘（≈09:30–11:30）相对全路径机会的账上表现  
> 结论先行：**主要不是「边突然没了」，而是架构选择把边关在门外，再用互相打架的验收标准反复自我否决。**

---

## 0. 现行 AM 实际是什么

```text
生产主账   peer3 CORE（Rule-A）     信号窗从 10:30 起 —— 故意不做开盘
AM 卫星    am_pulse A/B             唯一接进 scanner，但 execute_mode=shadow
关闭/未接  qqq_open_cont、launch、pocket、morning_sleeve、densify…
```

读法：今天说的「AM 策略」，live 上几乎只是 **shadow Pulse**；研究上则是一堆 **未接线冠军 + 已否决线**。  
「变差」有两层含义，必须分开：

1. **账上少** —— 相对每天 Mag7 早盘路径机会，策略抓得极少。  
2. **叙事乱** —— 研究不断 promote，live 几乎不执行，验收标准还在漂移。

---

## 1. 架构图：钱漏在哪里

```text
前视事实：每天 ~8 名 × 多段期权路径（oracle 常 +20%～50%）
        │
        ▼ ① 主账设计：CORE 从 10:30 开火
        │     → 开盘黄金段留给「卫星」
        │
        ▼ ② 卫星故意稀：FO0.8 / vd_soft∩… / first-per-day
        │     → 落到 ~0.7–1.5 笔/日（Pulse / pocket）
        │
        ▼ ③ 退出固定 TP：TP8 / TP15
        │     → capture 只剩 oracle 的 ~10%–15%
        │
        ▼ ④ 可执行双轨：研究 trade-last PASS，live quote FillSpec
        │     → 开盘 quote lag 大 → QUOTE_REJECT / 不能升线
        │
        ▼ ⑤ 座位与互斥：卫星 10% + flatten≤10:45 + 不占 TopK
              → 即便单笔尚可，日内复利天花板很低
```

这不是调参漏了一格，是 **五层设计选择叠出来的漏斗**。

---

## 2. 六大设计问题（按严重度）

### D1. 目标函数不唯一（最致命）

同一 AM 战场上，目标在来回切：

| 阶段目标 | 典型产物 | 结果 |
|----------|----------|------|
| 少而精、复利正 | Pulse / vd_soft TP8 | ~1 笔/日，capture 低 |
| 捕获 ≥20% | vd_acc0bp | 过线但太稀 → 退回 |
| 一天几十笔 × ~5% | FO 高频 / impact | 均笔变负 |
| 密度 + 正期望 | launch multi ~22 笔/日 | trade-last PASS，未接线、未 quote 验 |

**没有单一晋升门**（例如：「可成交 quote 口径下，日均 N 笔、胜率、均笔、双窗」），研究债必然堆积。

### D2. 主账放弃开盘，卫星又养不活

- CORE baseline：**10:30 才开火**（全路径研究里为控开盘噪声）。  
- 开盘被定义为「卫星问题」，但卫星预算小、shadow、且 quote 难验。  
→ 架构上等于：**承认开盘有机会，却不给主账能力，也不给卫星生产权。**

### D3. 把前视网格当成交易引擎

`am_vwap_foresight_map`：~400 edge 探针/日 ≠ 400 笔独立单。  
探针是 60s 网格对同一波段的重复采样。  
后续 densify / high_freq 仍按「探针密度」加压 → **问题陈述错了**，答案必然错。

### D4. 固定 TP 是主漏斗，却长期当默认退出

深审已定性：oracle 均值很高，clock 死拿≈0，固定 TP8/15 只吃 ~1/6。  
path / scaleout / ladder / ride 扫过，在稀入口上最多抬到 capture~0.15。  
**入口砍稀 + 退出封顶** 双重锁死；不是「再找一个 FO 阈值」能解开。

### D5. 验收双轨：trade-last vs quote

| 口径 | 常见结果 | 用途 |
|------|----------|------|
| trade-last | pocket / launch multi PASS | 离线研究 |
| quote FillSpec | launch / impulse / pulse-delay REJECT | live OMS |

用户后裁决「离线认 prints」，但 live 仍走 quote →  
**研究冠军无法升线，升线叙事与否决叙事并存**（假晋级 / 假否决都有）。

### D6. Pulse「KEEP」与因果修复后负账未 closure

- 旧：quote lag5 champion → ACCEPT_RESEARCH / shadow KEEP。  
- 新：`decision+60` 因果口径下 activity 扫描显示 LOCK 双窗大亏。  

同一臂两套时间语义，**没有正式 demote 或重验**。  
生产相邻叙事建立在可能已死的边上。

---

## 3. 「变差」的根因排序（证据强度）

| 秩 | 根因 | 是设计问题？ | 证据 |
|----|------|:------------:|------|
| 1 | CORE 不做开盘 + 卫星稀触发 | **是** | peer3 window 10:30；Pulse/pocket ~1 笔/日 |
| 2 | 固定 TP 低 capture | **是** | deep_review；capture_levers |
| 3 | 加密度挖次优 FO → 均笔塌 | **是（目标错）** | densify；high_freq |
| 4 | 开盘 quote 不可验 / lag | **部分是（数据+门）** | quote_ready；QUOTE_REJECT 族 |
| 5 | 验收标准漂移 | **是** | trades PASS ↔ quote REJECT |
| 6 | 目标函数来回切 | **是** | capture20 / 高频 / 少而精 |
| 7 | Pulse 因果存疑仍 KEEP | **是（治理）** | activity_overlay |
| 8 | 叠层 morph / 多臂无统一晋升 | **是（架构味）** | catalog 残骸 |

**不是主因：** 「TFT 没接上」「特征不够秒级」。  
QQQ TFT 侧信道已验 **NOT_USEFUL**；秒级 launch 有边但卡在 quote/接线。

---

## 4. 架构级裁决

1. **是设计问题，不只是参数问题。**  
   当前 AM 更像「主账让出开盘 + 卫星研究牧场」，不是一条可闭合的交易系统。

2. **三条线必须先选一条当北星，其他冻结：**  
   - **A. 可执行卫星**：以 live quote/FillSpec 为唯一验收；trade-last 只作诊断。  
   - **B. 路径捕获袖**：接受稀触发，全力做退出 / 分批，目标 capture，不追求几十笔。  
   - **C. 开盘主账前移**：把 09:30–10:30 收回 CORE 族（需重做噪声控制），卫星降级。

3. **立刻该停的：**  
   - 同时推进 densify、capture20、FO overlay、launch multi、Pulse KEEP。  
   - 用 foresight edge_rate 当「该做多少笔」的 KPI。

4. **立刻该 closure 的：**  
   - Pulse A/B：在 **统一因果时序 + quote** 下重验；不过则 demote，勿继续 shadow 叙事。  
   - pocket / launch multi：要么做 quote 对拍升线方案，要么标成 offline-only。

---

## 5. 下一刀（已开工）

**北星 A 已开新路径** → [`am_v2_path.md`](am_v2_path.md)

```text
profile: am_v2_executable_path_v1
mark:    quote FillSpec only for promotion
Step1:   quote coverage baseline（进行中）
旧 AM:   叙事冻结，零件可复用
```

---

## 6. 一句话

AM 变差，是因为系统在架构上选择了 **「主账避开开盘、卫星又稀又低 capture、研究与实盘验收分裂」**；  
后续越扫越多臂，是在错误目标函数上叠层，不是信号宇宙突然变坏。
