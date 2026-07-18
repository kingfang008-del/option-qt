# TCN probability gate (pluggable)

Optional **Rule-A + TCN** hybrid: keep the frozen `full_day_peer3` trigger, add a
switchable neural gate that scores “smooth continuation” probability on a short
minute tensor.

**Baseline default: OFF.** Enabling it never changes the freeze unless you set
`tcn_gate.enabled=true` and provide a checkpoint.

## Config

Under `signal.tcn_gate` or top-level `tcn_gate`:

```json
"tcn_gate": {
  "enabled": false,
  "action": "scale",
  "p_min": 0.55,
  "scale_when_low": 0.5,
  "scale_mode": "floor",
  "window": 15,
  "model_path": "maga7/results/tcn_gate/tcn_gate_v1.pt",
  "block_on_missing": false,
  "channels": ["net$", "ret1", "range_pct", "qqq_ret1", "vol_z", "tod"],
  "only_when": {"vixy_z_max": null, "abs_qqq_fp_min": null, "abs_qqq_fp_max": null}
}
```

| action | behavior |
|--------|----------|
| `off` / `enabled:false` | no-op (`NullTcnGate`) |
| `scale` | keep entry; if `p < p_min`, multiply size by `scale_when_low` (or linear in `p`) |
| `block` | reject entry when `p < p_min` |

Prefer **`scale` first** in ablations (same lesson as mf_idio hard-block).

## Modules

| Path | Role |
|------|------|
| `maga7/common/tcn_gate.py` | config, features, TinyTCN, `load_tcn_gate` |
| `maga7/common/replay.py` | hooks (entry block + size scale); summary `n_tcn_*` |
| `maga7/tools/build_tcn_gate_dataset.py` | Rule-A fires → tensor + underlying path label |
| `maga7/tools/train_tcn_gate.py` | train checkpoint |

## Minimal experiment

```bash
# 1) dataset — prefer MFE labels (smooth/strict labels were nearly unlearnable)
python -m maga7.tools.build_tcn_gate_dataset \
  --start-date 2025-07-01 --end-date 2026-07-16 \
  --label-mode mfe --breakout-pct 0.003 \
  --out maga7/results/tcn_gate/dataset_rule_a_mfe.parquet

# 2) train (hold out 2026 May–Jul); pos_weight=auto; early-stop on valid AUC
python -m maga7.tools.train_tcn_gate \
  --dataset maga7/results/tcn_gate/dataset_rule_a_mfe.parquet \
  --train-end 2026-04-30 --valid-start 2026-05-01 \
  --epochs 40 --hidden 48 --pos-weight auto \
  --out maga7/results/tcn_gate/tcn_gate_v1.pt

# 3) dual-window scoreboard lives under results/tcn_gate/scoreboard_dual_window/
```

Label notes (v1 probe):

| label | valid logreg/TCN AUC | note |
|-------|----------------------|------|
| `smooth` / `strict` (MFE+MAE) | ~0.45–0.53 | collapsed / no edge |
| `mfe` (MFE≥0.3% only) | ~0.67–0.70 | usable; channels +`vol_z`,`tod` |

## Dual-window scoreboard (2026-07-17 checkpoint)

Frozen baseline = `full_day_peer3` (+streak3 + mf_idio). Checkpoint: MFE label, 6ch, valid AUC≈0.70.

| window | variant | total_ret | MaxDD | vs baseline ret |
|--------|---------|-----------|-------|-----------------|
| May–Jul | baseline | **+874.7%** | −13.2% | — |
| May–Jul | tcn_scale p50 | +768.7% | −13.2% | **87.9%** of base |
| May–Jul | tcn_scale p55 | +672.2% | −13.2% | 76.8% |
| May–Jul | tcn_block p55 | +490.2% | −16.7% | 56.0% |
| Feb–Apr | baseline | +140.0% | −28.9% | — |
| Feb–Apr | tcn_scale p55 | +169.8% | **−16.6%** | +30pp / DD better |
| Feb–Apr | tcn_block p55 | **+197.5%** | **−14.9%** | best weak-regime |

## Accept / reject

Adopt into the research baseline only if:

1. May–Jul total_ret ≥ **95%** of frozen baseline  
2. 2025H2 or Feb–Apr MaxDD / bad clusters improve  
3. Live/shadow can load the same checkpoint (parity later)

Milder floors (`scale_when_low=0.75`): May–Jul `p50` → +820.9% (**93.8%** of base); Feb–Apr still only modest (+144.9%, DD −25.9%). Still fails the 95% bar.

## Option-ret labels (v1b)

Built `baseline_trades_2025h2_2026jul.csv` (n=140, only 2026-01…07 — no 2025H2 fills under this freeze).  
Logreg on option `ret>0`: train AUC≈0.82 / **valid AUC≈0.28** (overfit). MFE-TCN vs option win valid AUC≈0.43.  
**Do not train on option ret yet** — sample too small / regime shift.

## Weak-regime conditional gate (`only_when`)

Config (nested under `tcn_gate.only_when` or flat keys):

| key | meaning |
|-----|---------|
| `vixy_z_max` / `vixy_z_min` | apply gate only when VIXY z in range |
| `abs_qqq_fp_max` / `abs_qqq_fp_min` | apply when \|QQQ from_prev\| in range |

Otherwise passthrough (`tcn_skip_regime`). Scoreboard: `results/tcn_gate/scoreboard_weak_regime/`.

Notable Pareto points:

| variant | May–Jul vs base | Feb–Apr | note |
|---------|-----------------|---------|------|
| always scale p55 | 76.8% | +170% / DD−16.6% | helps weak, kills strong |
| abs_qfp&lt;0.8% × p50×0.75 | **100.8%** | 94.9% | preserves May–Jul; Feb–Apr ~flat |
| abs_qfp≥1% × p55 | 80.6% | **+162% / DD−24.6%** | helps weak; fails 95% |
| calm vixy&lt;0 block p55 | 63% | **+221%** | too harsh on May–Jul |

**v1/v1b verdict: REJECT for freeze.** No setting jointly clears May–Jul ≥95% **and** a clear Feb–Apr MaxDD/ret upgrade. Keep gate **pluggable / OFF**; `only_when` stays available for research.

## Non-goals (v1)

- Block-trade / L2 OIB channels (need finer data)  
- Predicting option premium directly  
- Replacing Rule-A  
