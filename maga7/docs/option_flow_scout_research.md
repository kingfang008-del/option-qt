# Option 1s put-flow 侦查兵

> 独立袖；用现有 `/mnt/s990/new_option_data_s3_trades`（1s 聚合），**不重下**。  
> 无 aggressor；代理 = put/call volume share + put volume z。  
> 工具：`common/option_flow.py` · `tools/scan_option_flow_scout_tpsl.py`

## 形态（DN）

会话内首次同时满足：

- `put_share = Σput_v / (put+call)` ≥ τ  
- `put_vol_z`（窗内 put 速率 / 更长基线）≥ z  
- `put_v` ≥ min_v  
- 可选：正股 `ret_lb ≤ −0.3%`（`stk_m3`）

入场：lock-map ATM put；定价 trades last±slip。

## 双窗（2026-07-25）

| 账本 | 产物 | 裁决 |
|------|------|------|
| trades | `research_option_flow_scout_dn_tpsl_dual` | **REJECT**（dual_pass_n=0） |

近优细胞（`put_f120_sh0.55_z2.0_*_stk_m3`）：MJ mean 可到 ≈+7–8%，但 **day_win≈0.38–0.54** 不过 0.55 闸；Jul n 常 &lt;6。  
`flow_only` 更吵，未形成稳定日胜率。

## Tick 验证（2026-07-25）

数据：`/mnt/s990/new_option_data_s3_tick`，**10 个交易日** `2026-07-10`…`07-23`（休市日无文件）。  
工具：`tools/scan_option_flow_tick_validate.py`  
产物：`results/research_option_flow_tick_validate_jul10_23`

| 项 | 结果 |
|----|------|
| 单窗闸（n≥6, day_win≥0.55, mean>0） | **VALIDATE_PASS**（61 cells） |
| 全过闸细胞 | **全部** `stk_m3`（股价跌确认）；`flow_only` **0** |
| 较稳口袋（n≥10） | `put_f120_sh0.55_z1.5_*_stk_m3_tp0.25_sl0.15`：n=22 mean≈+3.2% day_win≈71%（7 个有成交日） |
| add 冠军（n=6） | `put_f60_…_z3.0_…_tp0.25_sl0.15` mean≈+13% — **样本太薄，勿当冠军升格** |

说明：tick 仍无 aggressor；只是把 size 按秒汇总成 put/call 量。相对 1s 聚合，Jul 口袋能过**单窗**软闸，但 **May–Jul9 未覆盖 → 不是 dual PASS，禁止接线**。

### Multi-fire / 上升沿 / 新脉冲（2026-07-25）

`iter_put_flow_dn_in_window(fire_mode=...)`：`hold` | `rising` | `pulse` | `first`。  
验证默认 `rising`。产物：

| tag | mode |
|-----|------|
| `research_option_flow_tick_validate_jul10_23` | first（会话首枪） |
| `research_option_flow_tick_multifire_jul10_23` | hold（闸亮连开） |
| `research_option_flow_tick_rising_jul10_23` | rising（关→开） |
| `research_option_flow_tick_pulse_jul10_23` | pulse（边 + z/v 新冲） |

同形态 `put_f120_sh0.55_z1.5_v200_stk_m3_tp25/sl15`：

| mode | n | mean | 账本 add |
|------|---|------|----------|
| first | 22 | +3.2% | **+7.8%** |
| hold | 35 | −3.5% | −11.8% |
| rising | 34 | −3.0% | −9.6% |

rising/pulse 另有弱 PASS（n≈33，mean≈0，账本≈+0.3%）——笔数够了，**几乎不赚钱**。

**结论：** 加笔三种做法里，只有 **first-fire 稀疏口袋** 在这 10 天有经济意义；上升沿/新脉冲并未保住那笔边（多半是后段 episode 更差）。Mag7 秒级要对齐的可能是 **别的形态**（Rule-A / Hunt），不是把 put-flow 闸加密。

### 前视大波段 × tick（2026-07-25）

工具：`tools/scan_option_flow_tick_foresight_waves.py`  
产物：`research_option_flow_tick_foresight_waves_jul10_23`

方法：用正股前视标出大跌波（H=5/10/15min，深度 0.5/0.8/1.2%），在波起点只看**因果** tick put_share/z，并算 ATM put oracle。

| 发现 | 数据 |
|------|------|
| 大波本身 | oracle put 很肥（深波 mean 可到 +60%～+80%）→ 行情可做 |
| 波前 put 流 | 平均 put_share 仅 ≈0.35–0.38，**多数波前并没有 put 主导** |
| vs 对照 | `share≥0.55` 的 lift 多数 ≈0.7–1.1；深波 lift 可到 ~2× 但 n=10 且覆盖率仅 20% |
| 裁决 | **`FORESIGHT_NO_DISTILL`** — 前视也提炼不出稳定「tick put 流 → 大波」规则 |

## 结论

1. 1s 双窗 **REJECT**；tick Jul **first-fire** 单窗弱 PASS（~+7.8% / 22 笔）。  
2. hold / rising / pulse 加笔后边消失或趋零 → **禁止**为冲笔数接线。  
3. **前视大波也抽不出 put-flow 签名** → 瓶颈不是「没扫够」，而是当前 tick 代理特征不对版大波段。  
4. 下一刀若仍做 OF：要 side/quotes 或换特征（ATM 单约 tape、IV 冲击），勿再拧 put_share 阈值。

## 正股流 → 买期权（2026-07-25）

换假设：**信号看股价 1s（跌 + down-tick vol share），车用 ATM put（option tick 定价）**。  
工具：`tools/scan_stock_flow_opt_foresight.py`  
产物：`research_stock_flow_opt_jul10_23`

| 闸 | 结果 |
|----|------|
| 前视 distill（严格 lift） | 仍 `FORESIGHT_NO_DISTILL`（波前 dn_share≈0.50–0.54，比期权 put_share≈0.36 更贴，但覆盖率仍不够严格闸） |
| 因果 TP/SL（Jul 单窗） | **`VALIDATE_PASS`** · 38 cells |

Champion（rising）：`stk_d0.003_f120_sh0.6_tp0.25_sl0.2`  
n=164 · mean≈+2.3% · day_win=70% · 账本 add≈**+33%**（10 天，`position_frac=10%`）。

相对期权 put 流：同窗笔数与账本都明显更好 → **「看股价买期权」方向对了**；仍须 May–Jul9 + quote 双窗才谈升格。
