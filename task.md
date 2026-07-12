# 高保真模拟器 Guard 开关对齐

## Phase 1: 添加开关 → 全关 → 验证与 plan_a 一致
- [x] 在 [strategy_config.py](file:///Users/fangshuai/Documents/GitHub/option-qt/production/baseline/strategy_config.py) 添加 13 个 Guard 开关
- [x] 在 [strategy_core_v1.py](file:///Users/fangshuai/Documents/GitHub/option-qt/production/baseline/strategy_core_v1.py) 使用开关控制每个 Guard
- [x] 在 [strategy_core_v1.py](file:///Users/fangshuai/Documents/GitHub/option-qt/production/baseline/strategy_core_v1.py) 添加 plan_a 独有的 EARLY_STOP / NO_MOMENTUM 规则
- [x] 在 [s4_run_historical_replay.py](file:///Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/s4_run_historical_replay.py) 添加 PARITY_MODE=PLAN_A 配置注入
- [/] 全部关闭 s4 独有 Guard，运行 parity 验证
  - ✅ 信号生成: 141 (之前为 0)
  - ✅ 实际交易: 4 笔 (之前为 0)
  - ❌ 目标: 21 笔 (与 plan_a 一致)
  - 🔍 根因: 架构层差异 — 见下文

## 发现: s4 vs plan_a 架构级差异

### 差异 1: 持仓价格追踪
- **plan_a**: 每个 tick 更新所有标的的 `lpm[(sym,act)]`，持仓总能获到最新价
- **s4**: inner-join 后只含有 alpha 信号的标的，已持仓标的可能不在下一批次中

### 差异 2: 截面排序与开仓节流
- **plan_a**: 无截面排序，满足条件即逐个开仓
- **s4**: `entry_candidates[:3]` 每批最多开 3 笔，且有 59 秒保护期

### 差异 3: 退出后再入场
- **plan_a**: EARLY/NO_MOM 清仓后立即释放仓位，下一分钟可重新开仓
- **s4**: 59 秒保护期 + cooldown 机制阻止快速再入场

## Phase 2: 逐个开启 Guard 验证正/负向
- [ ] 待 Phase 1 完成后执行
