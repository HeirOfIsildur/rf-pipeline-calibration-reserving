#!/usr/bin/env python3
"""
Violin plots of per-AY reserve predictions using individual RF tree estimates.

For the best config (per_dy__log1p__two_stage), this script:
  1. Loads data and engineers features (same as main pipeline)
  2. For each DY, tunes + trains the two-stage RF model
  3. Extracts predictions from each individual tree in the RF ensemble
  4. Aggregates per-tree predictions to AY-level reserves using bootstrap
     draws across DY-specific forests:
       - next_year: predicted incremental payment for the first unobserved DY
       - ultimate: sum of predicted incremental payments for ALL unobserved DYs
  5. Plots violin distributions with CL and true-value reference lines

Each DY has its own RF with potentially different n_estimators. To construct
a proper distribution over "full ensemble" reserve estimates, we take N_DRAWS
bootstrap samples: in each draw, we sample one tree from each DY's forest,
compute its AY-level contribution, and sum across DYs.

Usage:
    python plot_violin.py                 # full tuning (25 Optuna trials/DY)
    python plot_violin.py --quick         # fast test (5 trials)
    python plot_violin.py --n-trials 10   # custom trial count
    python plot_violin.py --plot-only     # re-render PNGs from full_data_ cache
"""

import argparse
import pickle
import sys
import os
import time
import warnings
import gc

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    MAX_DEV_LAG,
    MAX_ACCIDENT_YEAR,
    RANDOM_STATE,
    RESULTS_DIR,
    GLOBAL_CALIB_KEY,
    get_test_ays_for_dy,
)
from src.data_prep import prepare_data, get_split_masks
from src.feature_engineering import engineer_features
from src.models import (
    tune_model,
    train_final_model,
    DEFAULT_PARAMS,
    DEFAULT_CLF_PARAMS,
)

N_DRAWS = 500  # bootstrap draws for the violin distribution
BASE_YEAR = 2000


def _cache_path():
    return os.path.join(RESULTS_DIR, "full_data_violin_cache.pkl")


def _save_violin_cache(cache):
    path = _cache_path()
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Saved: {path}")


def _load_violin_cache():
    path = _cache_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python plot_violin.py` (without --quick) once "
            f"to produce the full_data_violin_cache.pkl."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------
# Per-tree prediction for the two-stage model
# -----------------------------------------------------------------------


def predict_per_tree(model, X, target_type, zero_handling, calibration_factor=1.0):
    """
    Generate predictions from each individual tree in the RF ensemble.

    For two_stage: clf uses predict_proba (soft gating, matching the
    production pipeline in models.py:492-518) with a 0.01 threshold,
    then each regressor tree predicts independently on the positive rows.
    Results are multiplied by the pipeline calibration factor.

    Returns (n_trees, n_samples) array in the original payment scale.
    """
    if zero_handling == "two_stage":
        clf, reg = model

        # Soft gating via predict_proba — matches production pipeline
        # (models.py:497-505). Guard against single-class classifiers.
        proba = clf.predict_proba(X)
        if proba.shape[1] == 1:
            if clf.classes_[0] == 0:
                prob_positive = np.zeros(len(X))
            else:
                prob_positive = np.ones(len(X))
        else:
            prob_positive = proba[:, 1]

        likely_positive = prob_positive > 0.01

        n_trees = len(reg.estimators_)
        n_samples = X.shape[0]
        tree_preds = np.zeros((n_trees, n_samples))

        if likely_positive.sum() > 0:
            X_pos = X[likely_positive]
            for i, tree in enumerate(reg.estimators_):
                raw = tree.predict(X_pos)
                if target_type == "log1p":
                    raw = np.expm1(raw)
                raw = np.clip(raw, 0, None)
                # Soft gating: multiply by probability (not hard 0/1)
                tree_preds[i, likely_positive] = (
                    prob_positive[likely_positive] * raw * calibration_factor
                )

        return tree_preds

    elif zero_handling == "include_zeros":
        reg = model
        n_trees = len(reg.estimators_)
        n_samples = X.shape[0]
        tree_preds = np.zeros((n_trees, n_samples))

        for i, tree in enumerate(reg.estimators_):
            raw = tree.predict(X)
            if target_type == "log1p":
                raw = np.expm1(raw)
            raw = np.clip(raw, 0, None)
            tree_preds[i] = raw * calibration_factor

        return tree_preds
    else:
        raise ValueError(f"Unknown zero_handling: {zero_handling}")


def aggregate_tree_preds_to_ay(tree_preds, test_data_info):
    """
    Aggregate per-tree claim-level predictions to AY-level totals.

    Parameters
    ----------
    tree_preds : np.ndarray, shape (n_trees, n_samples)
    test_data_info : pd.DataFrame with columns [accident_year, dev_lag]

    Returns
    -------
    ay_tree_totals : dict[int, np.ndarray]
        For each AY, a 1-D array of length n_trees with the summed prediction.
    """
    ays = test_data_info["accident_year"].values
    unique_ays = np.unique(ays)
    result = {}
    for ay in unique_ays:
        mask = ays == ay
        # Sum across claims for each tree
        result[int(ay)] = tree_preds[:, mask].sum(axis=1)
    return result


# -----------------------------------------------------------------------
# Compute: Steps 1-3 (feature prep, per-DY training, bootstrap)
# -----------------------------------------------------------------------


def _compute_draws(args):
    """
    Run Steps 1-3 and return a dict with all inputs the plotting code needs.
    """
    method = "random_forest"
    target_type = "log1p"
    zero_handling = "two_stage"
    per_dy_trials = max(args.n_trials // 2, 5)

    # ------------------------------------------------------------------
    # Step 1: Load data and engineer features
    # ------------------------------------------------------------------
    print("=" * 60)
    print("VIOLIN PLOT: Per-tree RF predictions by Accident Year")
    print("=" * 60)

    print("\n[1/4] Loading data and engineering features...")
    t0 = time.time()
    claim_dy, claims_rbns, transactions_obs, triangles = prepare_data()
    claim_dy, feature_cols, feature_cols_per_dy = engineer_features(
        claim_dy, transactions_obs
    )
    feat_cols = feature_cols_per_dy
    print(f"  Done in {time.time() - t0:.1f}s")

    # Try to load per-DY models persisted by a previous main.py full run.
    # If present, skip Optuna tuning and use those models so the bootstrap
    # matches the predictions reported in the tables.
    persisted_path = os.path.join(
        RESULTS_DIR,
        "persisted_models__random_forest__per_dy__log1p__two_stage.pkl",
    )
    persisted_models = None
    if os.path.exists(persisted_path):
        with open(persisted_path, "rb") as f:
            persisted_models = pickle.load(f)
        print(
            f"  Loaded persisted per-DY models from {persisted_path} "
            f"({len(persisted_models)} DYs)"
        )

    # ------------------------------------------------------------------
    # Step 2: Train per-DY models and extract per-tree AY-level aggregates
    # ------------------------------------------------------------------
    if persisted_models is None:
        print(
            f"\n[2/4] Training per-DY models (log1p + two_stage), "
            f"{per_dy_trials} Optuna trials/DY..."
        )
    else:
        print(
            "\n[2/4] Using persisted per-DY models (skipping Optuna)..."
        )

    # For each DY, we store:
    #   dy_ay_tree_totals[dy] = {ay: np.array of shape (n_trees_for_dy,)}
    # This is the sum of per-tree predictions across claims, by AY.
    dy_ay_tree_totals = {}  # ultimate contributions per DY
    dy_ay_tree_next = {}  # next-year contributions (only the relevant DY)
    dy_n_trees = {}  # number of trees per DY

    # Also collect true AY-level reserves per DY for ground truth
    dy_ay_true = {}
    dy_ay_true_next = {}

    for dy in range(1, MAX_DEV_LAG + 1):
        t_dy = time.time()
        train_mask, val_mask, test_mask = get_split_masks(
            claim_dy, dy, structure="per_dy"
        )

        train_data = claim_dy[train_mask]
        val_data = claim_dy[val_mask]
        test_data = claim_dy[test_mask]

        if len(train_data) == 0 or len(test_data) == 0:
            print(f"  DY {dy:2d}: skip (no train or test data)")
            continue

        X_train = train_data[feat_cols].values
        y_train = train_data["incremental_payment"].values
        X_val = val_data[feat_cols].values if len(val_data) > 0 else X_train[:0]
        y_val = (
            val_data["incremental_payment"].values if len(val_data) > 0 else y_train[:0]
        )

        if persisted_models is not None and dy in persisted_models:
            final_model = persisted_models[dy]["model"]
        else:
            # Tune
            if len(val_data) > 10:
                try:
                    best_params, best_clf_params = tune_model(
                        method,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        target_type,
                        zero_handling,
                        per_dy_trials,
                    )
                except Exception as e:
                    print(f"  DY {dy:2d}: tuning failed ({e}), using defaults")
                    best_params = DEFAULT_PARAMS[method].copy()
                    best_clf_params = DEFAULT_CLF_PARAMS.copy()
            else:
                best_params = DEFAULT_PARAMS[method].copy()
                best_clf_params = DEFAULT_CLF_PARAMS.copy()

            # Train on full train+val (with claim_types for calibration)
            X_full = np.vstack([X_train, X_val]) if len(X_val) > 0 else X_train
            y_full = np.concatenate([y_train, y_val]) if len(y_val) > 0 else y_train

            has_claim_type = "claim_type" in claim_dy.columns
            ct_train = train_data["claim_type"].values if has_claim_type else None
            ct_val = (
                val_data["claim_type"].values
                if has_claim_type and len(val_data) > 0
                else None
            )
            ct_full = (
                np.concatenate([ct_train, ct_val])
                if ct_train is not None and ct_val is not None
                else ct_train
            )

            final_model = train_final_model(
                method,
                X_full,
                y_full,
                best_params,
                best_clf_params,
                target_type,
                zero_handling,
                claim_types=ct_full,
            )

        # Extract calibration factor from the regressor
        if zero_handling == "two_stage":
            _, reg = final_model
        else:
            reg = final_model
        calib_factors = getattr(reg, "pipeline_calibration_factors_", {})
        global_calib = calib_factors.get(GLOBAL_CALIB_KEY, 1.0)

        # Extract per-tree predictions on test set (with calibration)
        X_test = test_data[feat_cols].values
        tree_preds = predict_per_tree(
            final_model,
            X_test,
            target_type,
            zero_handling,
            calibration_factor=global_calib,
        )
        n_trees = tree_preds.shape[0]
        dy_n_trees[dy] = n_trees

        test_info = test_data[["accident_year", "dev_lag"]].reset_index(drop=True)

        # Aggregate to AY-level totals per tree
        ay_totals = aggregate_tree_preds_to_ay(tree_preds, test_info)
        dy_ay_tree_totals[dy] = ay_totals

        # True values aggregated to AY level
        true_vals = test_data.groupby("accident_year")["true_incremental_payment"].sum()
        dy_ay_true[dy] = true_vals.to_dict()

        # Mark which AYs have this DY as their "next year" (first unobserved)
        # For AY a, first unobserved DY = MAX_AY - a + 2
        # So for DY d, the AY that has d as its next year is: a = MAX_AY - d + 2
        next_year_ay = MAX_ACCIDENT_YEAR - dy + 2
        if next_year_ay in ay_totals:
            dy_ay_tree_next[dy] = {next_year_ay: ay_totals[next_year_ay]}
            dy_ay_true_next[dy] = {next_year_ay: dy_ay_true[dy].get(next_year_ay, 0.0)}

        elapsed = time.time() - t_dy
        test_ays = get_test_ays_for_dy(dy)
        print(
            f"  DY {dy:2d}: {n_trees} trees, calib={global_calib:.3f}, "
            f"test AYs={test_ays}, {elapsed:.1f}s"
        )

        del final_model, tree_preds
        gc.collect()

    # ------------------------------------------------------------------
    # Step 3: Bootstrap draws to build AY-level reserve distributions
    # ------------------------------------------------------------------
    print(f"\n[3/4] Building distributions via {N_DRAWS} bootstrap draws...")

    rng = np.random.RandomState(RANDOM_STATE)

    # Determine which AYs need predictions
    all_test_ays = set()
    for dy_dict in dy_ay_tree_totals.values():
        all_test_ays.update(dy_dict.keys())
    all_test_ays = sorted(all_test_ays)

    # For each AY, which DYs contribute?
    # AY a has unobserved DYs from (MAX_AY - a + 2) to MAX_DEV_LAG
    ay_dys = {}
    for ay in all_test_ays:
        first_unobs = MAX_ACCIDENT_YEAR - ay + 2
        ay_dys[ay] = [
            d for d in range(first_unobs, MAX_DEV_LAG + 1) if d in dy_ay_tree_totals
        ]

    # Bootstrap: for each draw, sample one tree index from each DY's forest
    # and sum per-AY contributions
    ultimate_draws = {ay: np.zeros(N_DRAWS) for ay in all_test_ays}
    next_year_draws = {}  # only for AYs that have a "next year" DY

    # Identify next-year AY for each AY
    for ay in all_test_ays:
        first_unobs = MAX_ACCIDENT_YEAR - ay + 2
        if first_unobs in dy_ay_tree_next:
            next_year_draws[ay] = np.zeros(N_DRAWS)

    for draw_i in range(N_DRAWS):
        # Sample one tree per DY
        sampled_tree_idx = {}
        for dy, n_t in dy_n_trees.items():
            sampled_tree_idx[dy] = rng.randint(0, n_t)

        for ay in all_test_ays:
            total = 0.0
            for dy in ay_dys[ay]:
                t_idx = sampled_tree_idx[dy]
                ay_trees = dy_ay_tree_totals[dy].get(ay, None)
                if ay_trees is not None:
                    total += ay_trees[t_idx]
            ultimate_draws[ay][draw_i] = total

            # Next-year draw
            first_unobs = MAX_ACCIDENT_YEAR - ay + 2
            if ay in next_year_draws and first_unobs in dy_ay_tree_totals:
                t_idx = sampled_tree_idx[first_unobs]
                ay_trees = dy_ay_tree_totals[first_unobs].get(ay, None)
                if ay_trees is not None:
                    next_year_draws[ay][draw_i] = ay_trees[t_idx]

    # True values
    true_ultimate = {}
    for ay in all_test_ays:
        true_ultimate[ay] = sum(
            dy_ay_true.get(dy, {}).get(ay, 0.0) for dy in ay_dys[ay]
        )

    true_next_year = {}
    for ay in next_year_draws:
        first_unobs = MAX_ACCIDENT_YEAR - ay + 2
        true_next_year[ay] = dy_ay_true.get(first_unobs, {}).get(ay, 0.0)

    # Chain Ladder from saved results
    cl_path = os.path.join(RESULTS_DIR, "chain_ladder_results.csv")
    cl_df = pd.read_csv(cl_path)
    cl_reserves = dict(zip(cl_df["accident_year"].astype(int), cl_df["CL_reserve"]))

    # CL next-year estimates
    cl_next_year = {}
    for ay in next_year_draws:
        last_obs_dy = MAX_ACCIDENT_YEAR - ay + 1
        first_unobs_dy = last_obs_dy + 1
        if first_unobs_dy <= MAX_DEV_LAG:
            cl_row = cl_df[cl_df["accident_year"] == ay]
            if len(cl_row) > 0:
                cum_first = cl_row[f"DY{first_unobs_dy}"].values[0]
                cum_last = cl_row[f"DY{last_obs_dy}"].values[0]
                cl_next_year[ay] = cum_first - cum_last
            else:
                cl_next_year[ay] = np.nan
        else:
            cl_next_year[ay] = 0.0

    # Calibrated RF prediction from saved pipeline results
    mc_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    mc_df = pd.read_csv(mc_path)
    best_row = mc_df[mc_df["name"].str.contains("per_dy__log1p__two_stage")]
    if len(best_row) > 0:
        total_rf_calib_ult = float(best_row["total_predicted_reserve"].values[0])
    else:
        total_rf_calib_ult = None

    total_rf_calib_ny = _calibrated_next_year_total(MAX_ACCIDENT_YEAR)

    return {
        "next_year_draws": next_year_draws,
        "ultimate_draws": ultimate_draws,
        "true_next_year": true_next_year,
        "true_ultimate": true_ultimate,
        "cl_next_year": cl_next_year,
        "cl_reserves": cl_reserves,
        "all_test_ays": list(all_test_ays),
        "total_rf_calib_ult": total_rf_calib_ult,
        "total_rf_calib_ny": total_rf_calib_ny,
        "n_draws": N_DRAWS,
    }


def _calibrated_next_year_total(max_ay):
    """
    Sum the best-config per-DY predictions over each AY's first unobserved DY.

    Returns None if the predictions CSV is missing.
    """
    pred_path = os.path.join(
        RESULTS_DIR,
        "full_data_predictions__random_forest__per_dy__log1p__two_stage.csv",
    )
    if not os.path.exists(pred_path):
        pred_path = os.path.join(
            RESULTS_DIR,
            "predictions__random_forest__per_dy__log1p__two_stage.csv",
        )
    if not os.path.exists(pred_path):
        return None
    df = pd.read_csv(pred_path)
    total = 0.0
    for ay in sorted(df["accident_year"].dropna().unique()):
        next_dy = max_ay - int(ay) + 2
        mask = (df["accident_year"] == ay) & (df["dev_lag"] == next_dy)
        total += float(df.loc[mask, "predicted_payment"].sum())
    return total


# -----------------------------------------------------------------------
# Render: Step 4 (all violin figures)
# -----------------------------------------------------------------------


def _render_plots(cache):
    """Render all violin figures and save summary CSV from a cache dict."""
    next_year_draws = cache["next_year_draws"]
    ultimate_draws = cache["ultimate_draws"]
    true_next_year = cache["true_next_year"]
    true_ultimate = cache["true_ultimate"]
    cl_next_year = cache["cl_next_year"]
    cl_reserves = cache["cl_reserves"]
    all_test_ays = cache["all_test_ays"]
    total_rf_calib_ult = cache["total_rf_calib_ult"]
    total_rf_calib_ny = cache.get("total_rf_calib_ny")
    if total_rf_calib_ny is None:
        total_rf_calib_ny = _calibrated_next_year_total(MAX_ACCIDENT_YEAR)
    n_draws = cache["n_draws"]

    # Re-center the per-tree bootstrap distribution on the production
    # RF-calibrated prediction.  Per-tree expm1 is convex-biased upward
    # vs. the ensemble's expm1(mean(tree_preds)), so the raw draws
    # overshoot the calibrated point estimate by the Jensen gap, and the
    # gap varies by AY because payment-scale variance differs per row.
    # We therefore rescale each AY's draws independently against the
    # production per-AY prediction loaded from the predictions CSV.
    pred_path = os.path.join(
        RESULTS_DIR,
        "full_data_predictions__random_forest__per_dy__log1p__two_stage.csv",
    )
    if not os.path.exists(pred_path):
        pred_path = os.path.join(
            RESULTS_DIR,
            "predictions__random_forest__per_dy__log1p__two_stage.csv",
        )
    rf_calib_by_ay_ult = {}
    rf_calib_by_ay_ny = {}
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        rf_calib_by_ay_ult = (
            pred_df.groupby("accident_year")["predicted_payment"].sum().to_dict()
        )
        for ay in sorted(pred_df["accident_year"].dropna().unique()):
            next_dy = MAX_ACCIDENT_YEAR - int(ay) + 2
            m = (pred_df["accident_year"] == ay) & (pred_df["dev_lag"] == next_dy)
            rf_calib_by_ay_ny[int(ay)] = float(pred_df.loc[m, "predicted_payment"].sum())

    for ay in ultimate_draws:
        target = rf_calib_by_ay_ult.get(float(ay), rf_calib_by_ay_ult.get(int(ay)))
        if target is None:
            continue
        bm = float(np.mean(ultimate_draws[ay]))
        if bm > 0:
            ultimate_draws[ay] = ultimate_draws[ay] * (target / bm)

    for ay in next_year_draws:
        target = rf_calib_by_ay_ny.get(int(ay))
        if target is None or target <= 0:
            continue
        bm = float(np.mean(next_year_draws[ay]))
        if bm > 0:
            next_year_draws[ay] = next_year_draws[ay] * (target / bm)

    print("\n[4/4] Creating violin plots...")

    def _plot_violin_panel(ax, draws_dict, true_dict, cl_dict, title_str):
        """Plot a single violin panel with log-scale y-axis."""
        plot_ays = sorted(draws_dict.keys())

        positions = []
        violin_data = []
        ay_labels = []
        for i, ay in enumerate(plot_ays):
            vals = draws_dict[ay]
            # Clamp to small positive for log scale (avoid log(0))
            vals = np.clip(vals, 1.0, None)
            violin_data.append(vals)
            positions.append(i)
            ay_labels.append(f"AY {ay}")

        parts = ax.violinplot(
            violin_data,
            positions=positions,
            showmeans=True,
            showmedians=False,
            showextrema=False,
            widths=0.7,
        )

        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0")
            pc.set_alpha(0.55)
            pc.set_edgecolor("#2a4a7f")
            pc.set_linewidth(0.8)
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(3.0)
        parts["cmeans"].set_linestyle("--")

        marker_size = 9
        for i, ay in enumerate(plot_ays):
            true_val = true_dict.get(ay, np.nan)
            cl_val = cl_dict.get(ay, np.nan)

            if true_val is not None and not np.isnan(true_val) and true_val > 0:
                ax.plot(
                    i,
                    true_val,
                    "D",
                    color="#2ca02c",
                    markersize=marker_size,
                    zorder=5,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label="True" if i == 0 else "",
                )

            if cl_val is not None and not np.isnan(cl_val) and cl_val > 0:
                ax.plot(
                    i,
                    cl_val,
                    "s",
                    color="#d62728",
                    markersize=marker_size,
                    zorder=5,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label="Chain Ladder" if i == 0 else "",
                )

        ax.plot([], [], "k--", linewidth=1.5, label="RF Mean")

        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels(ay_labels, fontsize=25)
        ax.set_ylabel("Reserve (log scale)", fontsize=28)
        ax.tick_params(axis="y", labelsize=24)
        ax.grid(axis="y", alpha=0.3, linestyle=":", which="both")
        ax.legend(fontsize=25, loc="upper left")

    # --- Figure 1a: Next-year predictions (log scale) ---
    fig_ny, ax_ny = plt.subplots(1, 1, figsize=(16, 7))
    _plot_violin_panel(
        ax_ny,
        next_year_draws,
        true_next_year,
        cl_next_year,
        "Next-Year Prediction (First Unobserved DY)",
    )
    fig_ny.tight_layout()
    out_ny = os.path.join(RESULTS_DIR, "violin_per_tree_next_year.pdf")
    fig_ny.savefig(out_ny, dpi=150, bbox_inches="tight")
    plt.close(fig_ny)
    print(f"\n  Saved: {out_ny}")

    # --- Figure 1b: Ultimate reserve predictions (log scale) ---
    fig_ult, ax_ult = plt.subplots(1, 1, figsize=(16, 7))
    _plot_violin_panel(
        ax_ult,
        ultimate_draws,
        true_ultimate,
        cl_reserves,
        "Ultimate Reserve (Sum of All Unobserved DYs)",
    )
    fig_ult.tight_layout()
    out_ult = os.path.join(RESULTS_DIR, "violin_per_tree_ultimate.pdf")
    fig_ult.savefig(out_ult, dpi=150, bbox_inches="tight")
    plt.close(fig_ult)
    print(f"\n  Saved: {out_ult}")

    # --- Figure 2: Aggregate violin plot (total reserves across all AYs) ---
    print("  Creating aggregate violin plot...")

    # Sum bootstrap draws across AYs for aggregate distributions
    total_ult_draws = np.zeros(n_draws)
    for ay in all_test_ays:
        total_ult_draws += ultimate_draws[ay]

    total_ny_draws = np.zeros(n_draws)
    for ay in next_year_draws:
        total_ny_draws += next_year_draws[ay]

    # Reference values: sum across AYs
    total_true_ult = sum(true_ultimate.get(ay, 0.0) for ay in all_test_ays)
    total_true_ny = sum(true_next_year.get(ay, 0.0) for ay in next_year_draws)
    total_cl_ult = sum(cl_reserves.get(ay, 0.0) for ay in all_test_ays)
    total_cl_ny = sum(cl_next_year.get(ay, 0.0) for ay in next_year_draws)

    # Paper Figure 3 uses standalone per-panel PDFs
    # (violin_aggregate_next_year.pdf, violin_aggregate_ultimate.pdf).
    single_specs = [
        (total_ny_draws, total_true_ny, total_cl_ny, total_rf_calib_ny,
         "Total Next-Year Reserve", "violin_aggregate_next_year.pdf", True),
        (total_ult_draws, total_true_ult, total_cl_ult, total_rf_calib_ult,
         "Total Ultimate Reserve", "violin_aggregate_ultimate.pdf", False),
    ]
    for draws, true_val, cl_val, rf_calib_val, title, fname, show_var in single_specs:
        fig_s, ax_s = plt.subplots(figsize=(6, 5))
        parts = ax_s.violinplot(
            [draws], positions=[0],
            showmeans=False, showmedians=False, showextrema=False, widths=0.7,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0")
            pc.set_alpha(0.55)
            pc.set_edgecolor("#2a4a7f")
            pc.set_linewidth(0.8)
        line_xmin, line_xmax = -0.45, 0.45
        ci_lo, ci_hi = np.percentile(draws, [2.5, 97.5])
        ax_s.hlines([ci_lo, ci_hi], line_xmin, line_xmax, colors="#555555",
                    linewidths=1.2, linestyles=":", zorder=4,
                    label=f"95% CI [{ci_lo/1e6:,.0f}M, {ci_hi/1e6:,.0f}M]")
        if show_var:
            var_995 = np.percentile(draws, 99.5)
            ax_s.hlines(var_995, line_xmin, line_xmax, colors="#8c564b",
                        linewidths=1.8, linestyles="-.", zorder=4,
                        label=f"99.5% VaR ({var_995/1e6:,.0f}M)")
        ax_s.hlines(true_val, line_xmin, line_xmax, colors="#2ca02c",
                    linewidths=2.5, linestyles="-", zorder=5,
                    label=f"True ({true_val / 1e6:,.0f}M)")
        if rf_calib_val is not None:
            ax_s.hlines(rf_calib_val, line_xmin, line_xmax, colors="#ff7f0e",
                        linewidths=2.5, linestyles="-", zorder=5,
                        label=f"RF calibrated ({rf_calib_val / 1e6:,.0f}M)")
        ax_s.hlines(cl_val, line_xmin, line_xmax, colors="#d62728",
                    linewidths=2.5, linestyles="--", zorder=5,
                    label=f"CL ({cl_val / 1e6:,.0f}M)")
        ax_s.set_ylabel("Reserve (millions)", fontsize=14)
        ax_s.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, p: f"{x / 1e6:,.0f}")
        )
        ax_s.set_xticks([])
        ax_s.grid(axis="y", alpha=0.3, linestyle=":")
        ax_s.tick_params(axis="y", labelsize=15)
        ax_s.legend(fontsize=11, loc="upper right", bbox_to_anchor=(1, 0.85))
        fig_s.tight_layout()
        out_single = os.path.join(RESULTS_DIR, fname)
        fig_s.savefig(out_single, dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        print(f"  Saved: {out_single}")


    # Save summary statistics
    rows = []
    for ay in all_test_ays:
        ult = ultimate_draws[ay]
        row = {
            "accident_year": ay,
            "year": BASE_YEAR + ay - 1,
            "ult_mean": np.mean(ult),
            "ult_median": np.median(ult),
            "ult_std": np.std(ult),
            "ult_p5": np.percentile(ult, 5),
            "ult_p95": np.percentile(ult, 95),
            "ult_true": true_ultimate.get(ay, np.nan),
            "ult_cl": cl_reserves.get(ay, np.nan),
        }
        if ay in next_year_draws:
            ny = next_year_draws[ay]
            row.update(
                {
                    "ny_mean": np.mean(ny),
                    "ny_median": np.median(ny),
                    "ny_std": np.std(ny),
                    "ny_p5": np.percentile(ny, 5),
                    "ny_p95": np.percentile(ny, 95),
                    "ny_true": true_next_year.get(ay, np.nan),
                    "ny_cl": cl_next_year.get(ay, np.nan),
                }
            )
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "violin_summary_stats.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Violin plots of per-tree RF predictions"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="Optuna trials per DY (default: 25)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: 5 trials, cap n_estimators at 100",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip all computation; re-render PNGs from the full_data_violin_cache.pkl "
        "produced by a previous full (non-quick) run.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    total_start = time.time()

    if args.plot_only:
        print("Loading cached violin draws for re-rendering...")
        cache = _load_violin_cache()
        print(
            f"  Loaded {len(cache['all_test_ays'])} AYs, "
            f"{cache['n_draws']} bootstrap draws"
        )
        _render_plots(cache)
    else:
        if args.quick:
            args.n_trials = 5
            import src.config as cfg

            def hp_rf_quick(trial):
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 100, step=50),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "min_samples_split": trial.suggest_int(
                        "min_samples_split", 2, 60, log=True
                    ),
                    "min_samples_leaf": trial.suggest_int(
                        "min_samples_leaf", 1, 40, log=True
                    ),
                    "max_features": trial.suggest_categorical(
                        "max_features", ["sqrt", "log2"]
                    ),
                }

            cfg.HP_SPACES["random_forest"] = hp_rf_quick
            DEFAULT_PARAMS["random_forest"]["n_estimators"] = 100
            DEFAULT_CLF_PARAMS["n_estimators"] = 100
            import src.models as mod

            mod.QUICK_MODE = True

        cache = _compute_draws(args)
        if not args.quick:
            _save_violin_cache(cache)
        _render_plots(cache)

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"DONE — Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
