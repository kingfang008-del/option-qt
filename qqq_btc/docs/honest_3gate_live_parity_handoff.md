# FT56 July W1 — Honest 3-Gate Live Parity 交接文档

> 用途：Cursor 闪退后，新 agent 可直接按本文继续。  
> 日期：2026-07-13  
> 对话 transcript：`9992cc1b-0d37-4d5b-8cd3-84bfd916ff73`

---

## 1. 目标（不可妥协）

对 **QQQ FT56 July W1** 做 **诚实模拟实盘** 特征/交易对拍，硬顺序三闸门：

| Gate | 比什么 | 在线侧 | 离线金标 |
|------|--------|--------|----------|
| **1** | 归一化前 raw 特征 | FCS `debug_raw_*` / 进程内等价 raw | `quote_features_raw` |
| **2** | 仅 Gate1 PASS 后：norm 特征 | FCS `debug_slow_*`（frozen/rolling） | `quote_features_test` |
| **3** | 仅 Gate1+2 PASS 后：交易/PnL | OMS fill_audit / signals | （parity 汇总，非开卷） |

### 诚实约束（禁止开卷）

- **禁止** `--greek-parity` / day_iv **IV/Greeks 注入**（FCS 自算 BSM）
- **允许** day_iv **仅补 volume**（openwin cbbo raw_1s 无 trade volume）
- `put_gate = vixy_z`（因果 VIXY），**禁止** feature5m 金标
- regime gold = **off**
- 归一化 = deploy 同款 `qqq_btc/CONFIG/frozen_norm_qqq_daily.npz`
- `FCS_DEBUG_RAW=1`、`RECALC_GREEKS=1`、`FCS_FORCE_RECALC_GREEKS=1`

---

## 2. 总体架构

```
raw_1s (stock + option cbbo)
        │
        ▼
┌───────────────────┐
│  Pitcher 1s       │  redis_fused_pitcher_1s.py
│  聚合/ffill 盘口   │  volume ← day_iv（非 greek-parity）
│  组装 fused 包     │
└─────────┬─────────┘
          │  (慢路径: Redis stream)
          │  (快路径目标: 进程内直喂 FCS)
          ▼
┌───────────────────┐
│  FCS v8           │  FeatureComputeService
│  1s→1m finalize   │  + RealtimeFeatureEngine
│  自算 Greeks/IV   │  debug_raw = pre-norm
│  frozen/rolling   │  debug_slow = normed
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
 Gate1/2     SE → OMS → fills
 (特征对拍)   (Gate3 交易)
```

### 两条执行路径

| 路径 | 入口 | 用途 | 现状 |
|------|------|------|------|
| **慢路径（Redis 全栈）** | `restart_ft56_july_w1_honest_live_parity.sh` → `run_qqq_btc_redis_sim.py` | Gate1+2+3 端到端 | Jul1 smoke PASS；全周 Gate1 未全过；week_v2 中断 |
| **快路径（目标）** | 新建：进程内 `1s → FCS only` | **只做 Gate1/2**，秒级完成 | **未落地**（闪退前正在做） |

快路径应借鉴：

- `preprocess/backtest/second/s2_run_realtime_replay_sqlite_1s.py` 的 **turbo**：`process_market_data` → `run_compute_cycle`（分钟边界才算特征）
- `qqq_btc/tools/honest_parity_fast.py`：无 Redis、只算价格层（缺期权全量）
- `qqq_btc/tools/option_chain_parity_sim.py`：期权链离线仿真（非完整 FCS）

**用户明确要求**：不要为 Gate1/2 再跑 pitcher+SE+OMS 全 Redis；**只调用 FCS 算分钟特征比对一次**。

---

## 3. 关键路径与配置

### 脚本 / 工具

| 文件 | 作用 |
|------|------|
| `qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh` | 诚实三闸门编排（慢路径） |
| `qqq_btc/tools/run_qqq_btc_redis_sim.py` | Redis 多进程仿真 |
| `qqq_btc/tools/redis_fused_pitcher_1s.py` | 1s fused 发球机；`greek_parity=False` 时只注 volume |
| `qqq_btc/tools/compare_debug_raw_offline.py` | **Gate1** |
| `qqq_btc/tools/compare_debug_slow_offline.py` | **Gate2**（判据/SKIP 列也在这里） |
| `qqq_btc/tools/honest_parity_fast.py` | 价格层快对拍（参考） |
| `qqq_btc/CONFIG/slow_feature_qqq_v4.json` | 慢特征配置（见下） |
| `qqq_btc/CONFIG/frozen_norm_qqq_daily.npz` | deploy 归一化 |

### 数据根

```bash
OPT_ROOT=/mnt/s990/data/v4_original_jul5/databento_july_w1_openwin/raw_1s
HONEST_FEAT_ROOT=~/train_data/july_w1_v4_honest_openwin
GREEK_ROOT=$HONEST_FEAT_ROOT/quote_options_day_iv          # 仅 volume
OFFLINE_RAW=$HONEST_FEAT_ROOT/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet
OFFLINE_NORM=$HONEST_FEAT_ROOT/quote_features_test/QQQ/regular/09:30-16:00/1min/2026-07.parquet
MAP=~/train_data/locked_targets_map_1dte_jul2026_openwin.parquet
```

重建离线特征（已做过，一般不用重跑）：

```bash
bash qqq_btc/tools/rebuild_july_w1_honest_openwin_features.sh
```

### 环境变量（诚实栈）

```bash
FCS_DEBUG_RAW=1
RECALC_GREEKS=1
FCS_FORCE_RECALC_GREEKS=1
FCS_OPTION_T_LABEL=end
FCS_IV_PRICE_MODE=close
FCS_TA_MONTH_ISOLATED=1
FCS_FROZEN_NORM_PATH=qqq_btc/CONFIG/frozen_norm_qqq_daily.npz
QQQ_BTC_PUT_GATE_MODE=vixy_z
QQQ_BTC_REGIME_GOLD_1M=0
# 禁止: GREEK_PARITY_MODE / FCS_MINUTE_PARITY_INJECT / put_gate=feature5m
```

### Gate 判据默认

- `med_tol=0.001`，`corr_min=0.90`，`ts_shift_sec=60`（live 分钟起点 +60s = 离线 end-label）
- SKIP 列见 `compare_debug_slow_offline.py` 的 `SKIP_FEATURES`（time/vix_level/volume_ratio/trend_* 等）

---

## 4. 已落地的关键修复（务必保留）

1. **FCS `debug_raw`**：`FCS_DEBUG_RAW=1` 写 pre-norm（`feature_compute_service_v8.py` + support handler）
2. **分辨率**：`slow_feature_qqq_v4.json` 中 `adx_smooth_10` / `bb_width` / `garman_klass_vol` / `options_struc_atm_iv` / `options_struc_skew` 改为 **`1min`**（原先误标 5min → FCS 台阶广播）
3. **时间特征**：`qqq_btc/live/fcs_adapter.py` 用 end-label（start+1min）对齐离线
4. **期权 T / IV 价**：`FCS_OPTION_T_LABEL=end`，`FCS_IV_PRICE_MODE=close`
5. **Volume**：pitcher 在非 greek-parity 时从 day_iv 只取 volume；`load_minute_option_ref` 优先显式 `greek_root`
6. **IV 冻结**：非 parity 时每新分钟强制 recalc（避免 sticky IV）
7. **ATM IV 半腿 bug**（week_v2 已含）：`realtime_feature_engine.py` 中 `options_vw_iv` / `struc_atm_iv` 只对 **有效腿(IV>0.01)** 平均，避免单腿 BSM 失败把 IV 腰斩（Jul2 下午典型）
8. **Compare soft-pass**：`compare_debug_slow_offline.py` 对尖峰敏感 corr 有 robust 路径

---

## 5. 当前结果快照

### Jul1 smoke（`july_w1_ft56_honest_3gate_smoke7`）

- Gate1 **PASS 100%**，Gate2 **PASS 100%**
- Gate3 解锁：约 2 笔 PUT，`@25%` ≈ **−5.3%**（诊断级）

### 全周 v1（`qqq_btc/results/july_w1_ft56_honest_3gate_week/`）

- Redis 全栈跑完 7 天
- **Gate1 overall FAIL**（`feat_parity_gate1_raw.json`）：

| 日期 | pass_rate | 主要失败列 |
|------|-----------|------------|
| 07-01 | 100% | — |
| 07-02 | 83.9% | iv_div, iv_mom, struc_atm_iv, struc_skew, vw_iv |
| 07-06 | 93.5% | iv_div, iv_mom |
| 07-07 | 83.9% | iv_div, iv_mom, struc_*, vw_iv |
| 07-08 | 90.3% | iv_div, iv_mom, struc_skew |
| 07-09 | 87.1% | iv_div, iv_mom, struc_atm_iv, vw_iv |
| 07-10 | 90.3% | flow_skew, iv_div, iv_mom |

- Gate2/3 被挡；有 diagnostic fills，**不能宣称交易 parity**

### 全周 v2（`july_w1_ft56_honest_3gate_week_v2/`）— **未完成**

- 含 ATM-IV 有效腿修复
- 已完成流式：07-01..08；**07-09 中途被 kill**（用户嫌慢 + 应用闪退）
- **无 Gate JSON**；只有 fill_audit / signals / stream logs
- Redis 相关进程已停（`pkill`）

---

## 6. 下一步（新 agent 优先执行）

### A. 实现快速 Gate1/2（最高优先）

新建例如：`qqq_btc/tools/honest_gate1_fcs_fast.py`（名称可改）

**建议实现要点：**

1. 设好诚实 env（§3），`SKIP_DEEP_WARMUP=1`（或按日可控 warmup）
2. 用 `redis_fused_pitcher_1s.FusedPitcherRaw` / `_load_day_maps` 加载：
   - `OPT_ROOT` openwin raw_1s
   - stock 1s / fallback
   - `greek_root=day_iv`，**`greek_parity=False`**
3. 进程内实例化 `FeatureComputeService`（同 s2 turbo）：
   ```python
   await feat_svc.process_market_data(batch_payloads)   # 每秒
   if minute_boundary:
       feat_payload = await feat_svc.run_compute_cycle(..., return_payload=True)
       # 从 cached_batch_raw / _build_raw_data_list_from_cache 收集 Gate1 向量
   ```
4. **不要**起 SE/OMS/多进程 Redis 消费（可连 Redis 做 BAR 状态，但不要全栈）
5. 输出 DataFrame → 复用 `compare_day`（`compare_debug_slow_offline.py`）对 `OFFLINE_RAW`
6. 可选：同一轮收 norm 向量做 Gate2 vs `OFFLINE_NORM`

**验收：**

```bash
# 期望：全周 7 天 Gate1 报告，分钟级完成（远快于 Redis 全栈）
python qqq_btc/tools/honest_gate1_fcs_fast.py \
  --dates 2026-07-01,2026-07-02,2026-07-06,2026-07-07,2026-07-08,2026-07-09,2026-07-10 \
  --option-root /mnt/s990/data/v4_original_jul5/databento_july_w1_openwin/raw_1s \
  --greek-root ~/train_data/july_w1_v4_honest_openwin/quote_options_day_iv \
  --offline-raw ~/train_data/july_w1_v4_honest_openwin/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet \
  --out qqq_btc/results/july_w1_ft56_honest_gate1_fast/
```

### B. Gate1 全绿后再做

1. Gate2（frozen norm）
2. 必要时才用慢路径跑 Gate3 交易对拍（或更薄的 SE/OMS）

### C. 不要做

- 不要用 greek-parity / day_iv IV 注入“刷绿”Gate1
- 不要用 feature5m / regime gold 开卷
- 不要把 `debug_slow`（已 norm）当成 Gate1
- 不要为 Gate1 再开完整 Redis pitcher+SE+OMS 周跑（除非快路径验证后专门做 Gate3）

---

## 7. 已知坑（排障）

| 现象 | 原因 | 方向 |
|------|------|------|
| FCS 特征台阶/错分辨率 | config 标 5min，离线按 1min 算 | 保持 v4 json 里相关列为 `1min` |
| IV 全天粘住 | snap 已有 IV 跳过 recalc | `FCS_FORCE_RECALC_GREEKS` |
| Jul2 下午 IV≈一半 | ATM 单腿 IV=0 仍 `(p+c)/2` | `realtime_feature_engine` 有效腿平均（已修） |
| volume 全 0 / 错 | cbbo 无成交量；或 ref 指错 monthly | day_iv volume + 显式 greek_root |
| Gate1 用错表 | 比了 debug_slow | 必须 raw |
| corr 过严尖峰 | iv_momentum 等 | compare 已有 robust；仍 fail 则查 IV 路径 |
| 时间错位 1min | label 起点 vs 终点 | `ts_shift_sec=60`；T 用 end-label |

---

## 8. 给新 agent 的一句话任务

> 停用 Redis 全栈周跑；实现 **s2-turbo 风格进程内 FCS**：用 openwin `raw_1s` + day_iv volume 喂 FCS，只算分钟 raw（及可选 norm）特征，对拍 `july_w1_v4_honest_openwin` 的 `quote_features_{raw,test}`，先把 July W1 Gate1（再 Gate2）跑绿；Gate3 最后再说。

---

## 9. 相关结果目录

```
qqq_btc/results/july_w1_ft56_honest_3gate_smoke7/   # Jul1 三闸门 PASS 参考
qqq_btc/results/july_w1_ft56_honest_3gate_week/     # 全周 v1，Gate1 FAIL JSON 在此
qqq_btc/results/july_w1_ft56_honest_3gate_week_v2/  # 含 ATM 修复，流式未完成，无 Gate JSON
```
