# Random forests with pipeline calibration for RBNS and IBNR reserving

Companion code for the paper *Random forests with pipeline calibration for RBNS and IBNR reserving* by Jan
Janoušek and Michal Pešta (Charles University, Department of Probability and Mathematical Statistics; submitted,
2026). An earlier version of the code
accompanied the manuscript *Random Forest with Pipeline Calibration for Individual Claims Reserving*.

The code reproduces the numerical pipeline from the paper on motor-third-party-liability claims data. An
**illustrative dataset provided by an insurance company** is shipped in `data/` so the pipeline runs end-to-end on
a clean clone (see [Data](#data) below).

## What this version adds (release v1.0, 2026)

- `tools/v1_decomposition.py`, `tools/v1_calibration_factors.py`, `tools/v1_ibnr_restate.py`, `tools/v1_figures.py` —
  the scripts that produce the paper's decomposition table, realised-factor table, IBNR/combined tables and figures;
- `tools/cohort_cl.py`, `tools/cohort_cl_by_h.py`, `tools/cohort_cl_by_type.py` — the standard and the report-cohort-consistent
  chain ladders, in total, by horizon and stratified by claim type (Schnieper-style separation of RBNS from IBNR);
- `tools/v2_arms_table.py` — the correction-arm table of the reproduction run (Section S5) from `results/correction_comparison.csv`
  and `results/cvz/correction_comparison.csv`; `tools/rebuild_summaries.py` — rebuilds the summary CSVs of a run from its per-configuration partials;
- `tools/ablation_tuned.py` — the gate-threshold, interaction-feature and depth sensitivities of Section 7 and Section S5,
  refitted with the tuned hyperparameters persisted by the reproduction run (`results/ablation_flagship_tuned.csv`);
- `src/gabrielli_*.py`, `run_gabrielli_nn.py` — the neural-network reference point (subnet architecture) re-implemented
  from the literature;
- `results/` — the deposited summaries cited in the paper. **These were computed on the insurer's confidential full
  data; running the scripts on the illustrative subset shipped in `data/` reproduces the procedure, not the numbers.**

## What the method does

Claims reserving estimates how much an insurer still owes on claims that
have been reported but not yet fully paid (RBNS) and on claims that have
occurred but have not yet been reported (IBNR). The traditional approach,
**Chain Ladder**, aggregates historical payments into a development
triangle and projects the lower-right by volume-weighted age-to-age
factors. It is simple and unbiased in expectation but blind to
claim-level information.

This project trains a **Random Forest** on claim-by-development-year
panel data to predict per-claim incremental payments, then sums the
per-claim predictions into aggregate reserves. The RF replaces Chain
Ladder on the RBNS part; a separate CL-frequency + RF-severity model
handles IBNR.

The contribution is the **pipeline calibration layer**, which restores
the *balance property* (aggregate predictions equal aggregate actuals;
the defining property of Chain Ladder, see Wüthrich, 2020) on the
cross-fitted record, before shrinkage and clipping. Random Forest plus a `log1p` target transform breaks this
property: `expm1(mean(log1p(y)))` under-estimates `E[Y|X]` by
Jensen's inequality, and a two-stage (classify-then-regress) pipeline
adds variance that the regressor alone cannot see. The calibration
layer fixes both in one step:

1. Run the full two-stage prediction pipeline out-of-sample via 3-fold CV
   grouped by accident year (GroupKFold; the folds are cohort groups, not a
   forward-in-time split).
2. Compute the ratio `sum(y_true) / sum(y_pred)` overall and per claim
   type.
3. Shrink the raw ratio toward 1.0 with the heuristic linear weight
   `Z = min(n_eff / K_hat, 1)`, where `K_hat = max(n_eff * s2, 50)` uses the
   between-fold variance `s2` of the per-fold ratios, and clip the result to
   [0.5, 5]. The unshrunk ratio restores the balance property exactly on the
   cross-fitted record; the shrunk per-type factors leave the record's
   aggregate about 6.5% short of it on our data (exact where Z is 1 and no clip
   bound binds, far off
   at the late development years where Z is small; `results/full25/record_diagnostics.csv`).
   When folds agree and n_eff >= 50, Z = 1 (no shrinkage); when folds
   disagree, shrink hard. The floor of 50 is an ad hoc numerical floor; below
   it Z is capped at n_eff/50.
4. Clip to `[0.5, 5.0]` as a safety rail.
5. Multiply the final-model predictions by the shrunken factor.

Detailed references — Wüthrich (2020), Manning & Mullahy (2001),
Bühlmann (1967), Duan (1983) — are in `src/models.py`.

## Code layout

```
main.py             Entry point for the RBNS pipeline
  ├─ prepare_data                 (src/data_prep.py)
  ├─ engineer_features            (src/feature_engineering.py)
  ├─ run_chain_ladder             (src/chain_ladder.py)   — baseline
  ├─ run_all_experiments          (src/train.py)          — RF tuning + calibration
  │    └─ tune_model / train_final_model / predict       (src/models.py)
  └─ build_comparison_table / plot_results / print_summary (src/evaluate.py)

ibnr_model.py       Entry point for the IBNR model: chain ladder on the claim-count triangle, random-forest
                    severities trained on the mature accident years (paid-to-date target), Monte Carlo draws

plot_violin.py      Monte-Carlo violin plots of per-tree RF predictions
                    (ultimate reserve, next-year reserve, aggregate and
                    per-AY panels) using bootstrap draws over trees.

src/config.py       Paths, experiment matrix, hyperparameter search spaces
src/data_prep.py    Raw CSV → claim × development-year panel
src/feature_engineering.py   34 engineered features per claim × DY (30 for the per-DY models)
src/chain_ladder.py          Volume-weighted CL baseline
src/models.py                RF training, Optuna tuning, pipeline calibration
src/train.py                 Experiment driver (iterates the 2×2×2 grid)
src/evaluate.py              Metrics, comparison tables, plots
```

## Experiment matrix

The paper evaluates 8 RF configurations — the Cartesian product of:

| Dimension | Values |
|---|---|
| Model structure | `single_model` (one RF with `dev_lag` as feature) · `per_dy` (one RF per development year) |
| Target transform | `raw` (incremental payment directly) · `log1p` (log-transformed target) |
| Zero handling | `include_zeros` (regression on all rows) · `two_stage` (classifier → regressor on positives) |

Pipeline calibration is applied to the `log1p` configurations only
(where the retransformation bias exists). The eight configurations are
`uv run python main.py --methods random_forest --objectives mse_raw mse_log1p`;
the default matrix of `src/config.py` also runs the Poisson forest and the
gradient-boosting objectives of the reproduction run (Section S5).

## Provenance of the deposited summaries

The CSVs in `results/` are the summaries the paper's tables were produced from, computed on the insurer's
confidential full data with this code. One property of the runs is worth stating, because it is a deliberate
choice rather than a default:

- The main run of the paper kept the observed cells of a claim that precede its reporting year in the training
  panel (5.6% of observed cells; Section 2.4 of the paper). The reproduction run of Section S5 excludes them, so
  that the risk set at development year *k* holds only the claims reported by *k*; `RESERVING_KEEP_PREREPORT=1`
  restores the main run's panel and reproduces its total of +3.9% exactly. The panel is worth about 0.4 points
  on the total here.

## Deposited files

| File | Used for |
|---|---|
| `results/model_comparison.csv`, `results/chain_ladder_results.csv` | Table 5 (RF configurations, standard chain ladder) |
| `results/chain_ladder_cohort*.csv`, `results/chain_ladder_by_h.csv` | Tables 5–6: cohort-consistent chain ladder in total, by horizon, by development year, stratified by type (`_min500`, `_min100`) |
| `results/v1_decomposition_by_dy.csv`, `results/v1_decomposition_by_h.csv` | Table 6 (decomposition of the leading forest) |
| `results/v1_calibration_factors.csv` | Table 4 (realised calibration factors) |
| `results/ibnr_reserves_by_ay.csv`, `results/combined_rbns_ibnr_reserves.csv`, `results/ibnr_severity_distribution.csv`, `results/v1_ibnr_restated.csv` | Tables 8–9 and Section S2 (IBNR; `_joint`, `_settled`, `_eventual` are the Section S5 sensitivity runs) |
| `results/violin_aggregate_summary.csv` | Figure 3 (ensemble-dispersion ranges) |
| `results/correction_comparison.csv`, `results/cvz/correction_comparison.csv` | Table S1 (correction arms and objectives of the reproduction run) |
| `results/full25/record_diagnostics.csv` | Section 4.3 (balance of the shrunk factors on the cross-fitted record) |
| `results/ablation_flagship_tuned.csv` | Section 7 and Section S5 (gate-threshold, interaction and depth sensitivities on the tuned hyperparameters; `tools/ablation_tuned.py`) |

## Data

The pipeline reads from `data/converted/` (annual 10×10 triangle).
Expected contents:

| File | Content |
|---|---|
| `claims_individual.csv` | one row per claim: `claim_no`, `accident_year`, `occurrence_period`, `notidel`, `report_dy`, `ibnr_flag`, `claim_type`, `claim_size`, `cum_paid_observed`, `cum_paid_full`, `reserve_true` |
| `transactions_individual.csv` | one row per payment: `claim_no`, `payment_period`, `payment_size`, `observed_flag`, `dev_lag` |
| `triangle_observed.csv`, `triangle_full.csv`, `triangle_incremental.csv` | paid-loss triangles (observed / true / incremental), indexed by accident year, columns `DY1..DY10` |
| `triangle_observed_rbns.csv`, `triangle_full_rbns.csv`, `triangle_incremental_rbns.csv` | same but restricted to RBNS claims |

`data/convert_raw_data.py` produces these files from the raw
`raw_claims.csv` export.

### Illustrative dataset

The data shipped in `data/` is an illustrative dataset provided by
an insurance company, suitable for running the pipeline end-to-end
and for demonstrating the method.

Ground-truth columns (`cum_paid_full`, `reserve_true`,
`true_incremental_payment`) are kept in memory for evaluation but are
defensively excluded from the feature set by a frozenset guard in
`src/data_prep.py` and `src/feature_engineering.py`.

## Install

Python 3.12+, [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                  # core; add --extra boosting (LightGBM rows of S5) and --extra nn (neural reference)
```

## Run

If you swap in your own raw `data/raw_claims.csv` and want to rebuild
the converted triangles from it, run the converter:

```bash
uv run python data/convert_raw_data.py
```

Otherwise the shipped files in `data/converted/` are ready to use
directly.

RBNS pipeline:

```bash
uv run python main.py                    # full tuning
uv run python main.py --quick            # 5 Optuna trials, single_model only
uv run python main.py --no-plots         # skip figure generation
```

IBNR model:

```bash
uv run python ibnr_model.py                       # defaults: --severity-train mature --min-dev 8
uv run python ibnr_model.py --severity-train settled   # sensitivity variant of Section S5
```

Monte-Carlo violin plots (requires `results/chain_ladder_results.csv`
and `results/model_comparison.csv` from `main.py`, so run `main.py`
first):

```bash
uv run python plot_violin.py
uv run python plot_violin.py --quick
```

Outputs land in `results/`.

## License

[MIT](LICENSE).
