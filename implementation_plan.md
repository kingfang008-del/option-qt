# Implementation Plan - Market Regime Comparison (Mar 3rd vs Others)

The user wants to understand why **2026-03-03** performed so poorly compared to other days. We suspect a significant market regime shift (VIX spike, geopolitical impact) occurred on this day.

## Proposed Changes

### [NEW] [compare_regimes.py](file:///Users/fangshuai/Documents/GitHub/option-qt/production/baseline/compare_regimes.py)
A diagnostic script to extract and compare key metrics across multiple SQLite databases.

#### Logic:
1. **Target Dates**: 2026-03-02, 03-03, 03-04, 03-05, 03-10.
2. **Metrics**:
    - **Volatility (VIXY)**: 
        - Max price change (%) from open.
        - Average 1m absolute return (Speed of move).
        - Max 5m "Volatility Spike" (Highest cluster of returns).
    - **Alpha Score Divergence**:
        - Correlation between `alpha_score` and forward 15m price returns.
        - Distribution of `alpha_score` (Is it extremely skewed or oscillating?).
    - **Market "Choppiness"**:
        - Number of "reversals" (Price goes up 0.1% then down 0.1% within 5 mins).
3. **Output**: A console summary and a data frame for report generation.

### [NEW] [comparison_report_0303.md](file:///Users/fangshuai/.gemini/antigravity/brain/b236241a-7bc1-4b23-833f-79e8d01e0884/comparison_report_0303.md)
Final artifact summarizing the statistical differences.

---

## User Review Required

> [!IMPORTANT]
> I will be using **VIXY** as a proxy for the VIX index, as it is standard in this dataset. I will analyze the **alpha_score** correlation to see if the model was "broken" by the regime shift or simply hitting stops due to pure volatility.

## Open Questions
- Is there a specific symbol besides VIXY that you believe represents the "Middle East issue" impact (e.g., Oil ETFs like USO)?

## Verification Plan

### Automated Tests
- Run `compare_regimes.py` across all available databases.
- Verify that VIXY price action matches the "spike" hypothesis.

### Manual Verification
- Review the generated comparison table to confirm the anomaly on March 3rd.
