# 离线 Replay / 流式对拍统一 Strategy Profile

## 目标

离线 replay 与流式 FCS → SE → OMS 使用同一份策略配方，避免以下参数散落在
`config.py`、Python inline `replace()` 和 shell env 后发生静默分叉：

- `ReplayConfig` 入场门控和跨日 quarantine / defense
- VX / VIXY rule profile selector
- 实时 `put_gate` 模式
- tick exit、标签偏移、fill spread
- checkpoint、feature config、frozen normalizer
- 指定月份的 infer / raw 依赖

## Profile 文件

当前提供四份版本化配方：

- `qqq_btc/CONFIG/strategy_profiles/v4_vx_live_aligned_v1.json`
  - V4 4–7 月标准离线 live-aligned replay。
- `qqq_btc/CONFIG/strategy_profiles/v4_honest_v0_parity_v1.json`
  - V4 V0 honest 无 VX：`open30 + bounce_cut + lock45`。
  - `tick_exits=off`，用于 `+65.09%` 的正式三闸门流式闭环。
- `qqq_btc/CONFIG/strategy_profiles/ft56_honest_vx_parity_v1.json`
  - FT56 Jul W1 honest 分钟逻辑对拍，`tick_exits=off`。
  - 指定 FT56 infer、honest raw1、VX selector、因果 `vixy_z` put gate。
- `qqq_btc/CONFIG/strategy_profiles/ft56_honest_vx_production_v1.json`
  - 与上一份 ReplayConfig/输入相同，但 `tick_exits=disaster_only`。
  - 用于 FCS+OMS+tick 的生产执行对拍。

每次改策略语义应新建 profile ID（例如 `_v2`），不要原地覆盖历史 profile。
结果通过完整 SHA-256 与 profile 快照绑定，而不是只依赖文件名。

## 成对运行

FT56 Jul W1 离线：

```bash
python qqq_btc/tools/replay_offline_live_aligned.py \
  --months 2026-07 \
  --strategy-profile \
  qqq_btc/CONFIG/strategy_profiles/ft56_honest_vx_parity_v1.json \
  --out-name ft56_honest_vx_parity_v1_offline
```

同 profile 流式逻辑对拍（tick off）：

```bash
QQQ_BTC_STRATEGY_PROFILE="$PWD/qqq_btc/CONFIG/strategy_profiles/ft56_honest_vx_parity_v1.json" \
  bash qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh
```

生产执行对拍（脚本默认 profile）：

```bash
python qqq_btc/tools/replay_offline_live_aligned.py \
  --months 2026-07 \
  --strategy-profile \
  qqq_btc/CONFIG/strategy_profiles/ft56_honest_vx_production_v1.json \
  --out-name ft56_honest_vx_production_v1_offline

QQQ_BTC_STRATEGY_PROFILE="$PWD/qqq_btc/CONFIG/strategy_profiles/ft56_honest_vx_production_v1.json" \
  bash qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh
```

检查两边 `manifest.json` 的以下字段完全一致：

- `strategy_profile_id`
- `strategy_profile_sha256`
- `selector_source` / `rule_profile_selector`
- infer、raw1、checkpoint 路径

离线 manifest 保存完整 `resolved_replay_config`；流式目录保存
`strategy_profile.resolved.json`，并记录 git commit / branch / dirty。

## 覆盖优先级

流式配置优先级：

1. 显式传给 governor 的 `ReplayConfig`
2. `QQQ_BTC_STRATEGY_PROFILE`
3. `qqq_btc/qqq/config.py` 默认值
4. 最后应用 `QQQ_BTC_*` env 紧急覆盖

env 覆盖适合临时诊断，但会导致实际运行值偏离 profile。正式 KPI 运行应避免额外
env；若确需覆盖，必须在结果命名和说明中标为变体。

离线 CLI 的 selector / quarantine 参数同样高于 profile；默认不传即可严格使用
profile。

## 已知边界

统一 profile 解决的是“策略配方一致”，不等于分钟 replay 与流式成交必然逐笔相同：

- 离线分钟 replay 没有真实 tick 序列；精确对拍 profile 默认 `tick_exits=off`。
- 流式仍有 FCS 特征生成、V0 前置检查、订单状态和成交时序。
- `put_gate` 离线来自 raw1 `+1min merge_asof(backward)`，流式来自实时 `vixy_z`；
  必须继续通过 Gate-1 / Gate-2 验证特征因果与数值一致。

因此验收顺序仍为：profile/hash 一致 → Gate-1 raw → Gate-2 normalized →
Gate-3 signals/trades/PnL。
