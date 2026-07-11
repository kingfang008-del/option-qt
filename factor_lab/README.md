# factor_lab

因子 / tradeprint / state-gate / routing 研究线归档目录。

生产主路径仍是 `qqq_btc` 的 V4（+ 可选 weekly finetune）。本目录只保留此前「因子模式」相关脚本、结果与设计文档，避免与主模型混在一起。

## 布局

```
factor_lab/
  tools/      # 原 qqq_btc/tools 下 0dte / tradeprint / state_gate / routing 等
  results/    # 原 qqq_btc/results/0dte_* 实验结果
  docs/       # 架构与策略设计文档
  CONFIG/     # state_gate_profiles.json 等
```

## 依赖

脚本仍复用 `qqq_btc.common` / `qqq_btc.qqq` 的数据与 replay 基础设施；包内互引用使用 `factor_lab.tools.*`。

```bash
export PYTHONPATH=/path/to/option-qt
python -m factor_lab.tools.run_0dte_factor_score_loop --help
```

## 不在此目录

- V4 训练 / 推理 / weekly finetune：`qqq_btc/`
- 更早的 baseline 实验：`New_Pro/baseline_qqq/`（未迁入）
