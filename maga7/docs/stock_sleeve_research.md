# 独立股票袖仓（MF first top2）— 研究线

**日期：** 2026-07-19  
**角色：** `research_only` · **Verdict: `REJECT_BASELINE`**  
**状态：** 只做研究归档；**不进**期权 `research_baseline` / peer3，不分配实盘仓位。  

**结论（2026-07-19）：** 能抓住 META/NVDA 故事日的**正股方向**，但收益相对期权基线**低到离谱**（强窗袖仓约 +9%，折全账户 ~+2%），不值得用 25% 权益置换期权 alpha。保留代码与 scoreboard 供对照，不再升线。

## 规则（当前默认）

| 项 | 设定 |
|----|------|
| 资金 | 袖仓权益归一 100（语义上 = 组合里专做股票的 ~25% 切片） |
| 入场 | 多因子 **首次进入 top2** → UP 做多 / DN 做空（1m close） |
| 仓位 | 每笔 `position_frac=0.25`（相对袖仓权益） |
| 席位 | UP≤2 · DN≤2 · 同日同票一侧；满员可按 score **displace**；反向 top2 **后讯翻转** |
| 成本 | 双边各 1bp |
| 出场消融 | `eod` / `window_end` / `hold_60` / `hold_120` × `stable_bars∈{1,5}` |

代码：`maga7/common/stock_sleeve.py` · `maga7/common/multifactor_rank.py`  
工具：`python -m maga7.tools.run_stock_sleeve_replay`  
Profile 标注：`CONFIG/strategy_profiles/stock_sleeve_mf_top2_v1.json`  
产物：`maga7/results/stock_sleeve_mf_top2/`

## Focus（07-07..10，`eod_s1`）

| 故事 | 结果 |
|------|------|
| **07-08 NVDA UP** | **抓住** · 10:53 入 · 持有到收盘 **股票 +3.06%**（袖仓贡献 ≈ 0.25×） |
| **07-09 META UP** | **抓住** · 11:41 入（曾短暂 DN，后讯翻转）· 至收盘 **股票 +4.14%** |

期权基线在这两天难兑现；股票袖仓用同一多因子时钟可以落到正股路径上。

## 双窗 scoreboard（袖仓权益，非合并组合）

推荐对照档：**`eod_s1`**（stable=1，收到收盘）

| window | total_ret | maxdd | n_trades | win_rate |
|--------|----------:|------:|---------:|---------:|
| May–Jul (strong) | **+9.3%** | −2.9% | 198 | 54% |
| Feb–Apr (weak) | **+3.7%** | −5.2% | 213 | 47% |
| focus 07-07..10 | **+0.74%** | −0.3% | 15 | 47% |

其它出场：`window_end_s1` 弱窗更好（~+5.9%）但强窗略低；`hold_60_s1` 弱窗为负。  
**当前研究默认出场：`eod`。**

## 与期权基线的关系

- 本线与 peer3 **账本隔离**；不改 QQQ / Rule-A / Hunt。  
- 正股能验证「信号找得到人」，**不能**用低 beta 收益去占期权预算。  
- 离线对 displace 掉的早笔不计中间段损益 → 数字还偏乐观，更不宜升线。

## 工程备注

1. 早版「全日只取最早 2 笔」会把席位给 10:30 噪音，**META/NVDA 进不来** → 已改为分向席位 + displace + 同票翻转。  
2. `stable_bars=5` 在强窗多数更差；默认 1。  
3. **不要**为故事日去松期权门；大波动兑现仍应在期权表达层研究（时机 / 合约），而非正股替身。

## 非目标 / 冻结

- **不进基线、不 live on、不合并 75/25 组合账**  
- 不替代 L0/L1/L2 期权栈  
- 不把袖仓 PnL 写进期权 `summary.json`
