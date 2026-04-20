# BTC Derivatives Feature Engineering: From Options Surface to Flow Structure

## 📌 背景

在股票期权系统中，我们采用了如下处理方式：

* 按 **ATM / OTM / 次月** 锁定合约
* 对同类合约进行 **VW（或 Vega）加权平均**
* 构建稳定的结构化特征输入模型

这套方法的核心目的不是“选合约”，而是：

> **将离散、噪音大的期权市场压缩为稳定的结构表示**

---

## ❗问题

在 BTC 永续 / 期货（derivatives）中：

* 没有 ATM / OTM
* 合约数量远少于股票期权
* 结构核心不在 strike，而在 **期限结构 + 杠杆行为**

那么：

👉 是否需要沿用“ATM / OTM / 次月 + VW 加权”这套逻辑？

---

# ✅ 核心结论

## ✔ 保留思想，不照搬实现

> **要保持“结构化聚合”的思想一致，但必须重构 bucket 设计方式**

---

# 🧠 本质对比

| 维度    | 股票期权               | BTC Derivatives            |
| ----- | ------------------ | -------------------------- |
| 核心结构轴 | Moneyness（ATM/OTM） | Contract Role（perp/future） |
| 第二结构轴 | Tenor（到期时间）        | Curve（期限结构）                |
| 合约数量  | 极多（分散）             | 较少（集中）                     |
| 聚合必要性 | 极高                 | 中等                         |
| 驱动因素  | IV / Greeks        | Funding / OI / Liquidation |

---

# 🔄 结构映射（最关键）

## 股票期权

```text
spot → ATM / OTM / next-month → VW 聚合 → 特征
```

---

## BTC Derivatives（推荐）

```text
市场状态 → perp / front future / next future → 桶内计算 → 桶间关系 → 特征
```

---

# 🧱 BTC Bucket 设计

## 1️⃣ 主交易桶（核心）

建议固定三个角色：

* `perp`（永续）
* `front_future`（当季）
* `next_future`（次季）

---

## 2️⃣ 桶内特征（单合约）

示例：

* `deriv_perp_return`

* `deriv_perp_oi_change`

* `deriv_perp_funding_rate`

* `deriv_front_basis`

* `deriv_next_basis`

---

## 3️⃣ 桶间特征（结构核心）

这是 BTC 的“期权 surface 等价物”：

* `deriv_basis_slope = next - front`
* `deriv_curve_spread`
* `deriv_perp_vs_future_spread`

👉 **这是最重要的 alpha 来源之一**

---

## 4️⃣ 聚合特征（替代 VW）

不是简单平均，而是结构化聚合：

* `deriv_curve_basis_oi_weighted_mean`
* `deriv_curve_basis_oi_weighted_std`

---

# ⚖️ 加权方式选择

## 推荐优先级：

### 1️⃣ OI-weighted（最重要）

* 代表真实杠杆暴露
* 适合：

  * basis
  * funding
  * curve

---

### 2️⃣ Volume-weighted

* 代表短期交易活跃度
* 适合：

  * flow
  * burst

---

### 3️⃣ 固定权重（第一阶段推荐）

```text
perp: 0.5
front future: 0.3
next future: 0.2
```

👉 更稳定，避免流动性噪音

---

# ❌ 不推荐做法

## ❌ 所有合约直接 VW 平均

原因：

* 破坏结构信息
* 混淆期限差异
* 降低 alpha 密度

---

## ❌ 套用 ATM / OTM 思路

BTC 没有：

* strike
* moneyness

👉 会导致错误建模

---

# 🔥 为什么这套方法更强

## 股票期权成功的真正原因是：

> 不是 ATM，而是 **结构压缩**

---

## BTC 的结构压缩方式是：

* perp（资金成本）
* future（期限结构）
* curve（预期）

---

👉 所以：

## **BTC = Flow + Leverage + Curve**

---

# 🚀 第一阶段推荐最小实现

只用：

## 单桶

* `deriv_perp_return`
* `deriv_perp_oi_change`
* `deriv_perp_funding_rate`

## 曲线

* `deriv_front_basis`
* `deriv_next_basis`
* `deriv_basis_slope`

## Flow

* `deriv_trade_imbalance`

---

👉 已足够支撑 TFT 第一版训练

---

# 🔮 第二阶段扩展

当你引入：

* 多交易所
* 更多 futures tenor
* options surface

再扩展：

* 多 bucket
* OI-weighted 聚合
* curve decomposition

---

# 🧠 最终总结（一句话）

> **BTC deriv 特征工程的本质，是将“ATM/OTM/tenor”这套期权结构，替换为“perp/future/curve”这套结构，并通过 OI/volume 加权完成稳定表达。**

---

# 🎯 关键 takeaway

* 不要复制期权结构
* 要迁移结构思想
* BTC 的核心是：

  * funding
  * OI
  * basis
  * curve

---

👉 **这是从“期权 surface alpha” → “flow alpha”的本质跃迁**
