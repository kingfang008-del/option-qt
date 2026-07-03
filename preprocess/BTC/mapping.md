# BTC Derivatives Feature Engineering: Feature → Data → Implementation Mapping

## 📌 目标

将 BTC `deriv_` 特征体系：

👉 **从概念 → 数据字段 → 代码实现**

做到：

* 可直接接入你的 data pipeline
* 对齐 TFT slow/fast channel
* 保证训练 / 推理一致

---

# 🧱 总体结构

```text
Raw Data (CoinAPI / Exchange)
        ↓
Normalized Base Fields
        ↓
Feature Calculation (calc)
        ↓
deriv_* Features
        ↓
TFT Input
```

---

# 🟢 一、基础数据字段（必须具备）

| 数据类型    | 字段                         | 说明     |
| ------- | -------------------------- | ------ |
| OHLCV   | open/high/low/close/volume | 基础价格   |
| Trades  | price, size, side          | 成交流    |
| Funding | funding_rate               | 永续资金费率 |
| OI      | open_interest              | 持仓量    |
| Mark    | mark_price                 | 标记价格   |
| Index   | index_price                | 指数价格   |
| Futures | future_price               | 交割合约价格 |

---

# 🔴 二、Deriv Feature Mapping（核心）

---

## 1️⃣ Funding（Theta proxy）

### Feature

* `deriv_funding_rate`
* `deriv_funding_change`
* `deriv_funding_zscore`

### Data Source

```text
funding_rate
```

### Implementation

```python
df["deriv_funding_rate"] = df["funding_rate"]

df["deriv_funding_change"] = df["funding_rate"].diff()

df["deriv_funding_zscore"] = (
    (df["funding_rate"] - df["funding_rate"].rolling(288).mean())
    / df["funding_rate"].rolling(288).std()
)
```

---

## 2️⃣ Open Interest（Delta flow）

### Feature

* `deriv_open_interest`
* `deriv_oi_change`
* `deriv_oi_momentum`
* `deriv_price_oi_divergence`

### Data Source

```text
open_interest
close
```

### Implementation

```python
df["deriv_open_interest"] = df["open_interest"]

df["deriv_oi_change"] = df["open_interest"].diff()

df["deriv_oi_momentum"] = df["open_interest"].pct_change()

df["deriv_price_oi_divergence"] = (
    df["close"].pct_change() - df["open_interest"].pct_change()
)
```

---

## 3️⃣ Basis / Curve（Term Structure）

### Feature

* `deriv_basis_perp_vs_index`
* `deriv_basis_momentum`
* `deriv_current_future_basis`
* `deriv_next_future_basis`
* `deriv_basis_slope`

### Data Source

```text
perp_price
index_price
future_price_front
future_price_next
```

### Implementation

```python
df["deriv_basis_perp_vs_index"] = (
    df["perp_price"] - df["index_price"]
) / df["index_price"]

df["deriv_basis_momentum"] = df["deriv_basis_perp_vs_index"].diff()

df["deriv_current_future_basis"] = (
    df["future_price_front"] - df["index_price"]
) / df["index_price"]

df["deriv_next_future_basis"] = (
    df["future_price_next"] - df["index_price"]
) / df["index_price"]

df["deriv_basis_slope"] = (
    df["deriv_next_future_basis"] - df["deriv_current_future_basis"]
)
```

---

## 4️⃣ Mark / Index Spread（Execution Layer）

### Feature

* `deriv_mark_price_diff`
* `deriv_index_price_diff`

### Data Source

```text
mark_price
index_price
close
```

### Implementation

```python
df["deriv_mark_price_diff"] = (
    df["mark_price"] - df["close"]
) / df["close"]

df["deriv_index_price_diff"] = (
    df["index_price"] - df["close"]
) / df["close"]
```

---

## 5️⃣ Trade Flow（Microstructure）

### Feature

* `deriv_aggressive_buy_volume`
* `deriv_aggressive_sell_volume`
* `deriv_trade_imbalance`

### Data Source

```text
trade_price
trade_size
trade_side (buy/sell)
```

### Implementation

```python
buy = df[df["side"] == "buy"]["size"].groupby("timestamp").sum()
sell = df[df["side"] == "sell"]["size"].groupby("timestamp").sum()

df["deriv_aggressive_buy_volume"] = buy
df["deriv_aggressive_sell_volume"] = sell

df["deriv_trade_imbalance"] = (
    (buy - sell) / (buy + sell + 1e-6)
)
```

---

## 6️⃣ Liquidation（Gamma proxy）

### Feature

* `deriv_liquidation_long`
* `deriv_liquidation_short`
* `deriv_liquidation_imbalance`

### Data Source

```text
liquidation_size
liquidation_side
```

### Implementation

```python
long_liq = df[df["liq_side"] == "long"]["liq_size"].groupby("timestamp").sum()
short_liq = df[df["liq_side"] == "short"]["liq_size"].groupby("timestamp").sum()

df["deriv_liquidation_long"] = long_liq
df["deriv_liquidation_short"] = short_liq

df["deriv_liquidation_imbalance"] = (
    (long_liq - short_liq) / (long_liq + short_liq + 1e-6)
)
```

---

# 🟡 三、Resolution 对齐规则

| 特征类型                 | 推荐分辨率 |
| -------------------- | ----- |
| funding / OI / basis | 5min  |
| volatility / regime  | 5min  |
| trade flow           | 1min  |
| price return         | 1min  |

---

# 🔵 四、Slow vs Fast Channel Mapping

## Slow（结构）

* deriv_funding_*
* deriv_oi_*
* deriv_basis_*
* deriv_liquidation_*
* realized_vol

---

## Fast（执行）

* return_1m
* deriv_trade_imbalance
* deriv_mark_price_diff
* volume_log

---

# ⚠️ 五、关键工程注意事项

## 1. 时间对齐

* funding / OI 延迟
* mark/index 不同步

👉 使用：

```python
pd.merge_asof(...)
```

---

## 2. 滚动归一化（必须）

```python
z = (x - rolling_mean) / rolling_std
```

避免未来信息泄露

---

## 3. 缺失值处理

```python
df.fillna(method="ffill")
df.fillna(0)
```

---

## 4. 特征一致性（训练 vs 推理）

必须保证：

* 顺序一致
* 名称一致
* scaler一致

---

# 🚀 六、最小可运行版本（推荐）

先实现：

```text
deriv_funding_rate
deriv_oi_change
deriv_basis_perp_vs_index
deriv_basis_slope
deriv_trade_imbalance
realized_vol_5m
```

---

# 🎯 总结

## 一句话核心

> **BTC deriv 特征工程 = 用 funding / OI / basis / liquidation 替代期权 Greeks，并通过结构化 bucket + 聚合构建稳定输入。**

---

## 最重要的三类 alpha

* Flow（OI + trade）
* Cost（funding）
* Structure（basis curve）

---

👉 这是你从 **期权体系 → BTC体系** 的核心迁移完成标志
