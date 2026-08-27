# TFT Deploy Bundle + July 全月 Frozen Replay

**日期：** 2026-07-22  
**Bundle：** `qqq_btc/CONFIG/deploy_bundle_v4_frozen_v1.json`  
**Profile：** `qqq_btc/CONFIG/strategy_profiles/v4_vx_jul_full_frozen_v1.json`

## 控制原则

1. TFT 只提案；VX / q10 / quarantine / 仓位 / 熔断在外壳。  
2. **rolling_norm = RESEARCH_ONLY**；离线 KPI / 流式 / 实盘共用 `frozen_norm_qqq_daily.npz`。  
3. Gate1 raw → Gate2 frozen norm → Gate3 成交；任一失败按 bundle `degrade_on_gate_fail` 降到 `NO_NEW` / `HALT`。

## July 全月重跑

```bash
bash qqq_btc/tools/run_july_full_frozen_replay.sh
```

覆盖（以本机数据为准）：

| 日 | 状态 |
|----|------|
| 07-01..02, 06..10, 13..17, 20 | 重跑目标 |
| 07-03 | NYSE 休市，跳过 |
| 07-21 | 需有效 `MASSIVE_API_KEY`/`POLYGON_API_KEY` 后再补下载 |

补 07-21：

```bash
export MASSIVE_API_KEY=... POLYGON_API_KEY=$MASSIVE_API_KEY
python preprocess/download/run_backfill_open_lock_pipeline.py \
  --start-date 2026-07-21 --end-date 2026-07-21 --symbols QQQ --force
END=2026-07-21 bash qqq_btc/tools/run_july_full_frozen_replay.sh
```

产物：

- Infer：`qqq_btc/results/v4_jul_full_frozen_infer/`
- Replay：`qqq_btc/results/offline_live_aligned/v4_vx_jul_full_frozen_v1_offline/summary.json`

## 与旧 670% 口径

May/Jun `+175.84%` / `+68.41%` 是 **分月 rolling 研究 KPI**。  
本 profile 是 **frozen + 全月（可得日）**，数字不可与 670% 直接横比；用途是 deploy 对齐基线。

## 第一次全月结果（2026-07-22）

范围：13 个交易日（01–02,06–10,13–17,20）；缺 03（休市）、21（API key 无效未下）。

| 口径 | acct25 | 笔数 | MaxDD | 备注 |
|------|-------:|-----:|------:|------|
| **VX regime（deploy）** | **+36.40%** | 17 | −8.25% | CHOP 2 日；相对 baseline −5.25pp |
| baseline TREND_PUT_OK | +41.65% | 24 | — | 无 VX 日切 |
| eval_test_set 内置 LIVE（无 VX） | +17.41% | 34 | −12.2% | frozen infer 自带 replay |

日路径（regime）：07-07 +22.0% 为最大日；07-08/09/15/16 无成交；07-13 −5.2%。

产物：`qqq_btc/results/offline_live_aligned/v4_vx_jul_full_frozen_v1_offline/summary.json`
