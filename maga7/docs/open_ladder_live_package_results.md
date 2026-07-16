# Mag7 开盘阶梯 + 实盘仓位/出场：当前结论（2026-07-16）

> 窗口：`2026-01-02` ~ `2026-07-13`  
> 成交：ATM，`fill_frac=0.8`，账户复利 MaxDD  
> 方案：`m5_circuit` + QQQ align + **only_win**  
> 选约：`open_ladder` OTM5（因果开盘阶梯）vs `day_lock`（前视对照）  
> 详表与消融链：[`jan_jul_replay_versions.md`](jan_jul_replay_versions.md)

---

## 1. 推荐研究包（可控回撤）

| 项 | 值 |
|---|---|
| Profile | [`m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p15_v1`](../CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p15_v1.json) |
| 复入 | `reentry_mode=only_win` |
| 仓位 | `position_sizing=concurrent`，`position_frac=0.15`（独处 15%；并发再开一笔 7.5%；最多 2 腿） |
| 出场 | `exit_mode=mf_flip`（mf10 翻向提前平，60s grace；仍保留 TP1.6 / SL0.4 / T+30） |
| 选约 | `contract_mode=open_ladder`，`ladder_otm_rungs=5` |

```bash
python -m maga7.tools.run_open_ladder_ab \
  --ladder-profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p15_v1.json \
  --quote-source 1s --scheme m5_circuit \
  --reentry-mode only_win --position-sizing concurrent --position-frac 0.15 --exit-mode mf_flip \
  --tag open_ladder_ab_1s_otm5_ow_conc_p15_mf_flip_jan_jul
```

产物：`results/open_ladder_ab_1s_otm5_ow_conc_p15_mf_flip_jan_jul/`

---

## 2. 1s 主结果表（day_lock vs open_ladder）

| 配置 | day 收益 | day MaxDD | lad 收益 | lad MaxDD | 结果目录 |
|---|---:|---:|---:|---:|---|
| only_win + topk p25（旧生产均分 12.5%） | +1471% | -22% | +718% | -26% | `open_ladder_ab_1s_otm5_jan_jul` |
| only_win + conc p25 + rails | +6849% | -33% | +1427% | **-58%** | `..._onlywin_conc25_...` |
| only_win + conc p15 + rails | +1468% | -23% | +714% | -33% | `..._ow_conc_p15_rails_...` |
| **only_win + conc p15 + mf_flip** | **+2007%** | **-18%** | **+1445%** | **-17%** | `..._ow_conc_p15_mf_flip_...` |
| **only_win + conc p20 + mf_flip** | **+5373%** | **-23%** | **+3375%** | **-22%** | `..._ow_conc_p20_mf_flip_...` |
| cooldown + conc p25（不可上线） | +2143% | **-52%** | +2732% | **-61%** | `..._cooldown_conc25_...` |

要点：

- **-58% / -61% 不可上线**；根因是满仓 25% + 允许亏后再开（或无窗口提前平）。
- **mf_flip**（同款滑动窗口：mf10 翻向）把 ladder 回撤压到约 **-17%～-22%**，且收益高于同仓位 rails。
- **p20** = `position_frac=0.20`（独处 20% / 并发 10%）。相对 p15 是仓位放大；相对同 p20 无 mf_flip（约 +994% / -42%）则是窗口出场带来的主增益。

出场构成（p15 mf_flip ladder 约）：`MF_FLIP` ~80%，`TP` ~16%，几乎无硬 SL。

---

## 3. 机制结论（短）

| 模块 | 结论 |
|---|---|
| 开盘阶梯 OTM5 | 因果可实盘；day_iv 已贴近日锁；1s 总收益差主要来自路径/复利，非缺 quote |
| `cooldown_only` | 集合对齐好，适合对拍；**不要做生产默认** |
| `concurrent` 仓位 | 实盘不知后面几笔：独处满袖套；最多再开 1 腿；禁止 25%+25%=50% |
| 完整反向 Rule-A 出场 | 30m 持仓内几乎触发不到 |
| **mf_flip 出场** | 有效：砍硬亏、降 MaxDD、抬总收益 |

---

## 4. 相关代码

| 路径 | 作用 |
|---|---|
| `maga7/common/open_lock.py` / `entry_contract.py` | 开盘锁 / 阶梯选约 |
| `maga7/common/reentry.py` | `reentry_mode` |
| `maga7/common/position_size.py` | concurrent / max 2 腿 |
| `maga7/common/replay.py` | `exit_mode=mf_flip` |
| `maga7/tools/run_open_ladder_ab.py` | A/B CLI（`--exit-mode` / `--position-frac`） |
| `maga7/tools/prepare_open_lock_quotes.py` | 开盘锁 1s 下载流水线 |

---

## 5. 暂不宣称生产冻结

当前推荐包为 **研究默认**。旧生产稳定口径仍是 `m5c_qqq_onlywin_stable_v1`（day_lock + topk 均分 + only_win，无 mf_flip）。上线因果锁约前需再做 stream/live 对拍与纸面验证。
