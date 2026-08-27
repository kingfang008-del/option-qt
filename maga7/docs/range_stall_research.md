# Range-chase + pre5 stall（剩余大亏日）

> 模块：`common/range_stall_gate.py` · 工具：`tools/run_range_stall_dual.py`  
> 产物：`results/range_stall_dual_v2`（v1 在 feature_ts 测量 → 失效）  
> 基线：spine 已含 overnight_gap + peer_gap

## 1. 目标日（接线 peer_gap 后仍 ≤−3%）

02-17 / 02-26 / 03-16 / 04-06 / 02-06 / 03-18 / 03-12 / 04-23 / 02-25 / 06-11

## 2. 特征

大亏单常在 **方向极端区间位置**（UP 近日内高 / DN 近日内低，`chase≥0.9`）且 **进场前 5 分钟有利动量≈0**。  
赢家同高 chase 时 pre5 通常仍为正。

**因果注意：** 在 `feature_ts` 测量时 pre5 往往仍热；停滞出现在 confirm 之后。门必须挂在 **最终 entry 时钟**。

## 3. 规则（WIRE）

`RS90_FFO2_P5`：`chase≥0.9` ∧ `pre5≤2bp` ∧ `peer≤5` ∧ `fav_from_open≥2%` → **block**

## 4. 双窗（`range_stall_dual_v2`）

| arm | weak | strong | 裁决 |
|-----|------|--------|------|
| **RS90_FFO2_P5** | **1.128** | **1.031** | **`DUAL_PASS_WIRE`** |
| RS90 | 0.917 | 1.031 | RESEARCH（弱窗不够） |
| RS90_P3PRE | 1.015 | 1.003 | WIRE 但误杀 04-20 META TP → 不取 |
| RS90_UNI | 1.244 | 0.951 | 强窗不够 |

Focus 日（相对 OFF）：
| 日 | OFF | FFO2_P5 |
|----|-----|---------|
| 04-06 | −5.25% | **0** |
| 04-23 | −3.70% | **0** |
| 02-26 | −8.28% | **−1.77%**（GOOGL 跳过，AMD 仍在） |
| 02-06 | −4.71% | −5.27%（拦 NVDA 后 TSLA 独仓放大） |
| 02-17 / 02-25 / 03-12 / 03-16 / 03-18 / 06-11 | 未覆盖 | 同 |

## 5. 接线

**2026-07-26：已 WIRE research baseline** `trade.range_stall_gate` = RS90_FFO2_P5。  
**Live：** 同日 `Mag7Scanner._entry_morph_range_stall` 在 final entry 时钟对齐 offline（与 `fo_lod` / gap stalls 一并进 live）。

### 5.1 peer_pre5 臂（清 03-16，接受强窗小回撤）

规则加臂：`peer_align ≤ 3` ∧ `pre5 ≤ 2bp` → block（与 chase 臂共用 `max_pre5`）。

相对当时 spine（已含 RS90_FFO2_P5）：

| arm | weak vsCUR | strong vsCUR | 03-16 |
|-----|------------|--------------|-------|
| **P3PRE_KEEPFLAT** | **1.102** | **0.973** | **跳过** |

**2026-07-26：用户接受强窗 keep≥0.97（存活优先）→ 已并入 spine**（`peer_pre5_max_peer=3`）。

仍未覆盖（接线 C7_FFO012_P25 前）：02-17 GOOGL SL、02-06 独仓放大、03-18（ffo 1.4%<2%）、06-11。

```bash
PYTHONPATH=. python -m maga7.tools.run_range_stall_dual \
  --out maga7/results/range_stall_dual_v2
```

## 6. 残留 02-06 / 03-18（2026-07-26）

### 尸检

| 日 | 标的 | 缺口 |
|----|------|------|
| **02-06** | TSLA UP | `peer_align=7` → chase 臂 `max_peer=5` 跳过；`pre5≈9.6bp` > 2bp → peer_pre5 也不触发 |
| **03-18** | AMZN DN | `fav_fo≈1.41%` < 2%；且门控 asof=`12:46:00` 时 `pre5≈2.15bp` 刚过 2bp（fill `12:46:03` 窗口滑掉 12:41 bar → pre5≈0） |

### 新臂

- **CROWD7：** `peer≥7` ∧ `chase≥0.9` ∧ `pre5≤10bp` ∧ `ffo≥2%` → block（清 02-06 / 02-25）
- **FFO012_P25：** chase 臂 `min_fav_from_open=1.2%` + `max_pre5=2.5bp`（清 03-18，不伤 06-10）

### 双窗（`range_stall_residual_dual_v2`，相对当时 spine OFF）

| arm | weak vs | strong vs | MaxDD weak | 02-06 | 02-25 MSFT | 03-18 |
|-----|--------:|----------:|-----------:|-------|------------|-------|
| CROWD7 | **1.134** | 1.000 | −11.5% | 清 | 清 | 仍 TOX |
| FFO012_P25 | 1.055 | 1.000 | **−7.4%** | 仍 | 仍 | **清** |
| **C7_FFO012_P25** | **1.047** | **1.000** | **−7.4%** | **清** | **清** | **清** |

**升线：`C7_FFO012_P25`。** 工具：`tools/run_range_stall_residual_dual.py`。

## 6. 优先残差日 02-18 / 03-12 / 06-11（`priority3_dual_v2`）

尸检（相对已接线 spine）：

| 日 | 交易 | 缺口 |
|----|------|------|
| **02-18** | AMZN UP −27% TOX | crowd 已 peer=7/chase/fo≥2%，但 `pre5_e≈11.9bp` > 当时 10bp |
| **03-12** | AAPL/NVDA DN | 同上 + NVDA 门控整分钟 `pre5≈24.9bp`（fill+1s 窗口滑到 4.9bp） |
| **06-11** | TSLA UP −18% | gap≈1.75%、**feature** `|fo|≈0`、chase≥0.9、sess≈32m；entry 时钟 fo/chase 已漂移 |

### 新臂

- **CROWD25：** crowd `pre5≤25bp` + 独立 `crowd_min_fav_from_open=1%`（Arm A 仍 ffo≥1.2%）
- **UGS15：** `up_gap_stall_gate` @ **feature_ts** — UP ∧ gap≥1.5% ∧ `|fo|≤0.1%` ∧ chase≥0.9 ∧ sess≤40m

### 双窗（相对当时 spine OFF）

| arm | weak | strong | 02-18 | 03-12 | 06-11 |
|-----|-----:|-------:|-------|-------|-------|
| CROWD25 | **1.111** | **1.053** | 清 | 清 | 仍 |
| UGS15 | 1.000 | **1.038** | 仍 | 仍 | **清** |
| **BOTH** | **1.111** | **1.093** | **清** | **清** | **清** |

**升线：`BOTH` = CROWD25 + UGS15。** `DUAL_PASS_WIRE`。工具：`tools/run_priority3_dual.py` · 模块：`common/up_gap_stall_gate.py`。

## 7. CORE「大 fo + LOD 追尾」BLOCK（`fo_lod_chase_dual_v1`）

07-24 TSLA：Rule-A 10:38 时已 fo≈−3.5%、chase≈1.0、贴 LOD；买 0DTE put 是脉冲尾部追涨，T+30 −17.7%。

### 规则（DN30）

feature 时钟：`DN` ∧ `fav_fo≥3%` ∧ `chase≥0.9` ∧ `dist_LOD=(px-lo)/open ≤30bp` → block。

| arm | weak | strong(→07-24) | 07-24 | 备注 |
|-----|-----:|---------------:|-------|------|
| **DN30** | **1.000** | **1.044** | **清** | 升线 |
| DN25 (fo≥2.5%) | 0.934 | 0.841 | 清 | 砍 DN 赢家 |
| BOTH_DIRS | 0.828 | 1.145 | 清 | 砍 04-15/04-16 UP |

**升线：`DN30`。** 模块：`common/fo_lod_chase_gate.py` · 工具：`tools/run_fo_lod_chase_dual.py`。  
开盘脉冲本身仍属 AM sleeve（`launch_slope`）；本门只拦 CORE 追尾。

## 8. AM Pulse Scout（侦察-only · 不接线）

09:30–10:30 独立侦察：`AM_SCOUT_ALERT`（FO≥1% / LB 2m≥0.8%）。**不进 OMS**。见 [`am_pulse_scout.md`](am_pulse_scout.md) · `tools/scan_am_pulse_scout.py`。07-24 TSLA：LB@09:36 + FO@09:44。
