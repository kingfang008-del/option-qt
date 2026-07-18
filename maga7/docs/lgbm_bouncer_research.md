# LightGBM Smart Bouncer（研究开关）

在 **Rule-A 候选不变** 的前提下，用表格特征做「是否放行 / 缩仓」的可插拔否决层。默认 **OFF**，不改变 freeze。

与 TCN gate 同钩子风格；优先学 **期权路径三元标签**（+1 优质 / 0 耗损 / −1 恶性反转），缺 quote 时回退正股三元并打标。

## Config

```json
"lgbm_bouncer": {
  "enabled": false,
  "action": "scale",
  "p_min": 0.55,
  "scale_when_low": 0.5,
  "model_path": "maga7/results/lgbm_bouncer/lgbm_bouncer_v1.txt",
  "block_on_missing": false
}
```

| action | 行为 |
|--------|------|
| `off` / `enabled:false` | no-op |
| `scale` | `P(allow) < p_min` 时仓位 × `scale_when_low` |
| `block` | `P(allow) < p_min` 拒入 |

先试 **scale**，慎 hard-block（水床效应）。

## 模块

| Path | Role |
|------|------|
| `maga7/common/lgbm_bouncer.py` | 特征、标签、推理、`load_lgbm_bouncer` |
| `maga7/common/replay.py` | entry block + size scale；summary `n_lgbm_*` |
| `maga7/tools/build_lgbm_bouncer_dataset.py` | Rule-A → 表格特征 + 三元标签 |
| `maga7/tools/train_lgbm_bouncer.py` | walk-forward 训练 |
| `maga7/tools/run_lgbm_bouncer_scoreboard.py` | May–Jul + Feb–Apr 双窗 |

## 实验命令

```bash
# 1) 数据集（期权优先）
python -m maga7.tools.build_lgbm_bouncer_dataset \
  --start-date 2026-02-01 --end-date 2026-07-17 \
  --out maga7/results/lgbm_bouncer/dataset_rule_a.parquet

# 2) 训练（train≤Apr / valid=May–Jul）
python -m maga7.tools.train_lgbm_bouncer \
  --dataset maga7/results/lgbm_bouncer/dataset_rule_a.parquet \
  --train-end 2026-04-30 --valid-start 2026-05-01 \
  --out maga7/results/lgbm_bouncer/lgbm_bouncer_v1.txt

# 可选：仅期权标签
python -m maga7.tools.train_lgbm_bouncer --option-only ...

# 3) 双窗 scoreboard
python -m maga7.tools.run_lgbm_bouncer_scoreboard \
  --model maga7/results/lgbm_bouncer/lgbm_bouncer_v1.txt \
  --out maga7/results/lgbm_bouncer/scoreboard_dual_window
```

## 验收（对齐 TCN 教训）

- May–Jul（强窗）总收益 ≥ **95%** 基线，且 MaxDD 不明显恶化
- Feb–Apr（弱窗）有改善
- 两者同时满足才考虑升格；否则保持研究开关

## 特征（因果）

`dir_sign, mf10, mf_fast, streak, from_prev, vol_z, tod_min, gap_open, bounce_lod, above_open, above_vwap, qqq_*`

## 期权标签阈值（v1）

- toxic (−1): `MAE ≥ 30%`（持有窗内期权卖价回撤）
- quality (+1): `MFE ≥ 40%` 且 `MAE ≤ 15%`
- 其余 = 0（耗损）；训练目标 `y_allow = 1[y ≠ -1]`（模型可训 `P(toxic)`，推理转 `p_allow`）

## 2026-07-18 探针结果 → **研究 REJECT（不升 freeze）**

数据集（Feb–Jul Rule-A first fires）：

| 项 | 值 |
|----|----|
| n | 406（期权标签 370 / 正股回退 36） |
| ternary | −1:133 · 0:181 · +1:92 |
| allow rate | ~67% |

Walk-forward（train≤Apr / valid≥May）：

| 模型 | valid AUC (allow) | 备注 |
|------|-------------------|------|
| all labels + early stop | ~0.52 | 预测几乎常数（std≈0.014） |
| option-only · 80 round | ~0.45 | train AUC 0.94 → 过拟合，**劣于随机** |

双窗 scoreboard（`lgbm_bouncer_opt_v1` · action=scale）：

| window | variant | total_ret | MaxDD | vs baseline |
|--------|---------|-----------|-------|-------------|
| May–Jul | baseline | **+810%** | −13.2% | — |
| May–Jul | scale p55 | +410% | −11.5% | **50.6%** |
| May–Jul | scale p60 | +361% | −11.0% | 44.6% |
| Feb–Apr | baseline | +140% | −28.9% | — |
| Feb–Apr | scale p55 | +149% | −21.3% | 106% |
| Feb–Apr | scale p60 | +161% | −19.6% | 115% |

结论与 TCN / 结构硬门一致：**弱窗略好、强窗腰斩** → 水床效应；表格特征在 Rule-A 触发点对「期权毒性」跨 regime 无稳定可学信号。保持 `lgbm_bouncer.enabled=false`。

## 续：DN-only 子集（2026-07-18）

配置：`only_directions: ["DN"]`，模型 `lgbm_bouncer_dn_v1`（option-only · toxic MAE≥25% · train≤Apr）。

| 项 | 值 |
|----|----|
| n_train / n_valid | 89 / 91 |
| valid AUC | **0.50**（仍不可学） |
| 07-17 NVDA/TSLA `p_allow` | 0.66 / 0.63（p55 缩不到） |

Scoreboard（`scoreboard_dn_only/`）：

| window | variant | total_ret | vs baseline |
|--------|---------|-----------|-------------|
| May–Jul | baseline | +810% | — |
| May–Jul | DN scale p55 | +622% | **77%** |
| May–Jul | DN scale p65 | +618% | 76% |
| Feb–Apr | baseline | +140% | — |
| Feb–Apr | DN scale p55 | +120% | **85%**（弱窗也变差） |

→ 仍 **REJECT**。DN 收窄减轻强窗伤害（50%→77%），但弱窗不再受益，未过 95% 线。

结构硬子集（DN∧above_open / VWAP / bounce≥2%）样本 <15，不够训模型；且 `QQQ>开 ∧ bounce≥2%` 在 valid 上 **0 条 toxic**，挡的是好单。

旁路更轻的规则见 `dn_structure_gate_ablation.md`（`scale_dn_if_qqq_above_open`）：强窗约 86%、07-17 半损，仍不升 freeze。

后续更值得试：**持仓内早切**（入场后期权 MAE 触阈平仓），而不是再加 entry 否决。