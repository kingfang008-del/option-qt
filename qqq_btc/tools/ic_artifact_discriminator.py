#!/usr/bin/env python3
"""判别实验: bak 冻住特征的 IC 优势是否集中在与重建特征分歧大的日子。

三臂 (2025H2, 同时间戳同标签):
  A = 旧 V4 ckpt + bak 冻住特征        (H2 IC ~0.224)
  B = prefer_primary 重训 ckpt + 重建特征 (IC ~0.036)
  C = 旧 V4 ckpt + 重建特征            (IC ~0.065; 与 A 同权重, 只换特征)

逐日计算:
  - ic_A / ic_B / ic_C: 当日 Spearman(net_edge, label_return_fwd_net)
  - feat_div: 当日 bak vs 重建 options_* 列的 1 - median Spearman
  - sec_share: prefer_primary 重建里当日 secondary 补洞行占比

判别逻辑:
  若 (ic_A - ic_C) 集中在 feat_div 高 / sec_share 高的日子 → 0.2 IC 依赖数据态伪影;
  若特征几乎一致的日子 A 仍显著强于 C → 重建管线仍缺一环, alpha 可能是真的。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("/home/kingfang007/文档/GitHub/option-qt/qqq_btc/results")
OUT = BASE / "ic_artifact_discriminator"
OUT.mkdir(exist_ok=True)

ARMS_H2 = {
    "A_bak": BASE / "v4_replay_2025h2_bak/test_infer.parquet",
    "B_pp_retrain": BASE / "v4_prefer_primary_h2_after_train/test_infer.parquet",
    "C_v4_on_pp": BASE / "v4_replay_2025h2_prefer_primary_rebuild/test_infer.parquet",
}
ASSEMBLE = Path.home() / "train_data/builds/0dte_prefer_primary/assemble_per_day.json"


def load(path):
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
    return df


def daily_ic(df):
    out = {}
    for d, g in df.groupby("date"):
        m = g["label_return_fwd_net"].notna() & g["net_edge"].notna()
        if m.sum() < 30:
            continue
        ic, _ = stats.spearmanr(g.loc[m, "net_edge"], g.loc[m, "label_return_fwd_net"])
        out[d] = ic
    return pd.Series(out, name="ic")


def main():
    dfs = {k: load(p) for k, p in ARMS_H2.items()}
    a, b, c = dfs["A_bak"], dfs["B_pp_retrain"], dfs["C_v4_on_pp"]
    assert (a["timestamp"].values == c["timestamp"].values).all()
    assert (a["timestamp"].values == b["timestamp"].values).all()

    ic = pd.DataFrame({
        "ic_A": daily_ic(a),
        "ic_B": daily_ic(b),
        "ic_C": daily_ic(c),
    })

    # 逐日特征分歧: bak vs rebuild 的 options_* 列 (同时间戳截面 Spearman)
    opt_cols = [x for x in a.columns if x.startswith("options_")]
    div_rows = {}
    for d in ic.index:
        ga = a[a["date"] == d]
        gc = c[c["date"] == d]
        corrs = []
        for col in opt_cols:
            va, vc = ga[col].values, gc[col].values
            m = np.isfinite(va) & np.isfinite(vc)
            if m.sum() < 30 or np.std(va[m]) == 0 or np.std(vc[m]) == 0:
                continue
            r, _ = stats.spearmanr(va[m], vc[m])
            corrs.append(r)
        if corrs:
            div_rows[d] = 1.0 - float(np.median(corrs))
    ic["feat_div"] = pd.Series(div_rows)

    # 逐日 secondary 补洞占比 (重建侧的覆盖修补强度, 代理当年数据态的"混合度")
    per_day = json.load(open(ASSEMBLE))
    sec = {r["day"]: (r.get("n_secondary_rows", 0) / max(r.get("rows", 1), 1)) for r in per_day}
    ic["sec_share"] = ic.index.map(lambda d: sec.get(d, np.nan))

    ic["gap_AC"] = ic["ic_A"] - ic["ic_C"]
    ic["gap_AB"] = ic["ic_A"] - ic["ic_B"]
    ic.to_csv(OUT / "daily_ic_h2.csv")

    rep = {"n_days": len(ic)}
    rep["mean_ic"] = ic[["ic_A", "ic_B", "ic_C"]].mean().round(4).to_dict()
    rep["median_ic"] = ic[["ic_A", "ic_B", "ic_C"]].median().round(4).to_dict()
    # A 的日 IC 为正的天数占比
    rep["pos_day_share"] = {k: float((ic[k] > 0).mean().round(3)) for k in ["ic_A", "ic_B", "ic_C"]}

    # 相关性: 优势 vs 分歧/补洞
    for x in ["feat_div", "sec_share"]:
        m = ic[[x, "gap_AC", "gap_AB", "ic_A"]].dropna()
        if len(m) > 10:
            rep[f"spearman_{x}"] = {
                "vs_gap_AC": round(float(stats.spearmanr(m[x], m["gap_AC"])[0]), 3),
                "vs_gap_AB": round(float(stats.spearmanr(m[x], m["gap_AB"])[0]), 3),
                "vs_ic_A": round(float(stats.spearmanr(m[x], m["ic_A"])[0]), 3),
                "n": len(m),
            }

    # 分桶: 特征分歧低(最一致的1/3)天 vs 高(最分歧的1/3)天
    m = ic.dropna(subset=["feat_div"])
    q1, q2 = m["feat_div"].quantile([1 / 3, 2 / 3])
    low, high = m[m["feat_div"] <= q1], m[m["feat_div"] >= q2]
    rep["bucket_low_div"] = {
        "n": len(low), "feat_div_med": round(float(low["feat_div"].median()), 4),
        "ic_A": round(float(low["ic_A"].mean()), 4), "ic_B": round(float(low["ic_B"].mean()), 4),
        "ic_C": round(float(low["ic_C"].mean()), 4), "gap_AC": round(float(low["gap_AC"].mean()), 4),
    }
    rep["bucket_high_div"] = {
        "n": len(high), "feat_div_med": round(float(high["feat_div"].median()), 4),
        "ic_A": round(float(high["ic_A"].mean()), 4), "ic_B": round(float(high["ic_B"].mean()), 4),
        "ic_C": round(float(high["ic_C"].mean()), 4), "gap_AC": round(float(high["gap_AC"].mean()), 4),
    }

    # 分桶: secondary 补洞 = 0 的天 vs > 0 的天
    m2 = ic.dropna(subset=["sec_share"])
    zero, nz = m2[m2["sec_share"] == 0], m2[m2["sec_share"] > 0]
    rep["bucket_sec_zero"] = {
        "n": len(zero), "ic_A": round(float(zero["ic_A"].mean()), 4),
        "ic_C": round(float(zero["ic_C"].mean()), 4), "gap_AC": round(float(zero["gap_AC"].mean()), 4),
    }
    rep["bucket_sec_pos"] = {
        "n": len(nz), "sec_share_med": round(float(nz["sec_share"].median()), 4),
        "ic_A": round(float(nz["ic_A"].mean()), 4),
        "ic_C": round(float(nz["ic_C"].mean()), 4), "gap_AC": round(float(nz["gap_AC"].mean()), 4),
    }

    # 逐月均值, 看优势是否全时段均匀
    ic["month"] = pd.to_datetime(pd.Series(ic.index, index=ic.index)).dt.strftime("%Y-%m")
    rep["monthly"] = {
        mth: {k: round(float(g[k].mean()), 4) for k in ["ic_A", "ic_B", "ic_C", "feat_div"]}
        for mth, g in ic.groupby("month")
    }

    json.dump(rep, open(OUT / "verdict_h2.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
