#!/usr/bin/env python3
"""
Post-run analysis script.

Reads saved results from the RF pipeline run and produces:
1. Model comparison summary (all RF configs + CL)
2. Per-AY reserve breakdowns for best RF config vs CL
3. Feature importance ranking
4. RBNS predictions analysis at claim × DY granularity
5. Configuration pattern analysis

Usage:
    uv run python analysis.py      # or: python analysis.py inside the project environment
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = "results"
DATA_DIR = "data/converted"


def load_results():
    """Load all result files from the pipeline run."""
    comparison = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    cl_results = pd.read_csv(os.path.join(RESULTS_DIR, "chain_ladder_results.csv"))
    ay_reserves = pd.read_csv(
        os.path.join(RESULTS_DIR, "all_models_reserves_by_ay.csv")
    )

    # Load feature importances if available
    fi_path = os.path.join(RESULTS_DIR, "feature_importances.csv")
    fi = pd.read_csv(fi_path) if os.path.exists(fi_path) else None

    # Load all prediction files
    predictions = {}
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith("predictions__") and f.endswith(".csv"):
            name = f.replace("predictions__", "").replace(".csv", "")
            predictions[name] = pd.read_csv(os.path.join(RESULTS_DIR, f))

    return comparison, cl_results, ay_reserves, fi, predictions


def load_data():
    """Load the underlying data for context."""
    claims = pd.read_csv(os.path.join(DATA_DIR, "claims_individual.csv"))
    return claims


def section_divider(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


# -----------------------------------------------------------------------
# 1. Model Comparison Summary
# -----------------------------------------------------------------------
def analyze_model_comparison(comparison):
    section_divider("1. MODEL COMPARISON SUMMARY (8 RF configs + Chain Ladder)")

    print(
        f"{'Rank':<5} {'Config':<55} {'Error%':>10} {'RMSE':>14} {'MAE':>14} {'R²':>8}"
    )
    print("-" * 110)

    for i, (_, row) in enumerate(comparison.iterrows()):
        name = (
            row["name"].replace("__", " | ")
            if isinstance(row["name"], str)
            else str(row["name"])
        )
        err_pct = row.get("total_reserve_error_pct", np.nan)
        rmse = row.get("rmse", np.nan)
        mae = row.get("mae", np.nan)
        r2 = row.get("r2", np.nan)

        rmse_str = f"{rmse:>14,.0f}" if not np.isnan(rmse) else f"{'N/A':>14}"
        mae_str = f"{mae:>14,.0f}" if not np.isnan(mae) else f"{'N/A':>14}"
        r2_str = f"{r2:>8.4f}" if not np.isnan(r2) else f"{'N/A':>8}"

        marker = " <-- BEST" if i == 0 and row.get("method") != "chain_ladder" else ""
        if row.get("method") == "chain_ladder":
            marker = " <-- CL BASELINE"
        print(
            f"{i + 1:<5} {name:<55} {err_pct:>+9.2f}% {rmse_str} {mae_str} {r2_str}{marker}"
        )

    # Key comparisons
    ml_rows = comparison[comparison["method"] != "chain_ladder"]
    cl_row = comparison[comparison["method"] == "chain_ladder"]

    if not ml_rows.empty and not cl_row.empty:
        best_ml = ml_rows.iloc[0]
        cl = cl_row.iloc[0]
        print(f"\n--- Key Comparison ---")
        print(f"  Best RF config: {best_ml['name']}")
        print(f"  Best RF error:  {best_ml['total_reserve_error_pct']:+.2f}%")
        print(f"  CL error:       {cl['total_reserve_error_pct']:+.2f}%")
        print(
            f"  RF advantage:   {abs(cl['total_reserve_error_pct']) - abs(best_ml['total_reserve_error_pct']):.2f} pp closer to zero"
        )

        # Count how many RF configs beat CL
        cl_abs = abs(cl["total_reserve_error_pct"])
        beats_cl = (ml_rows["total_reserve_error_pct"].abs() < cl_abs).sum()
        print(f"  RF configs beating CL: {beats_cl}/{len(ml_rows)}")


# -----------------------------------------------------------------------
# 2. Per-AY Reserve Breakdowns
# -----------------------------------------------------------------------
def analyze_per_ay(ay_reserves, comparison, cl_results):
    section_divider("2. PER-ACCIDENT-YEAR RESERVE BREAKDOWN")

    # Best ML model
    ml_rows = comparison[comparison["method"] != "chain_ladder"]
    if ml_rows.empty:
        print("No ML results to analyze.")
        return

    best_name = ml_rows.iloc[0]["name"]
    best_ay = ay_reserves[ay_reserves["model_name"] == best_name].copy()
    best_ay = best_ay.sort_values("accident_year")

    # CL per-AY
    cl_ay = cl_results.copy()

    print(f"Best RF config: {best_name}")
    print(
        f"\n{'AY':<5} {'True Reserve':>15} {'RF Predicted':>15} {'RF Err%':>10} {'CL Predicted':>15} {'CL Err%':>10}"
    )
    print("-" * 75)

    total_true = 0
    total_rf = 0
    total_cl = 0

    for _, row in best_ay.iterrows():
        ay = int(row["accident_year"])
        true_r = row["true_reserve"]
        rf_pred = row["predicted_reserve"]
        rf_err = row["reserve_error_pct"]

        # Find CL for this AY
        cl_match = cl_ay[cl_ay["accident_year"] == ay]
        if not cl_match.empty and cl_match.iloc[0].get("true_reserve", 0) > 0:
            cl_pred = cl_match.iloc[0].get(
                "CL_reserve", cl_match.iloc[0].get("predicted_reserve", np.nan)
            )
            cl_true = cl_match.iloc[0].get("true_reserve", true_r)
            cl_err = (cl_pred - cl_true) / cl_true * 100 if cl_true > 0 else 0
        else:
            cl_pred = np.nan
            cl_err = np.nan

        total_true += true_r
        total_rf += rf_pred
        if not np.isnan(cl_pred):
            total_cl += cl_pred

        cl_pred_str = f"{cl_pred:>15,.0f}" if not np.isnan(cl_pred) else f"{'N/A':>15}"
        cl_err_str = f"{cl_err:>+9.2f}%" if not np.isnan(cl_err) else f"{'N/A':>10}"

        print(
            f"{ay:<5} {true_r:>15,.0f} {rf_pred:>15,.0f} {rf_err:>+9.2f}% {cl_pred_str} {cl_err_str}"
        )

    print("-" * 75)
    total_rf_err = (total_rf - total_true) / total_true * 100 if total_true > 0 else 0
    total_cl_err = (total_cl - total_true) / total_true * 100 if total_true > 0 else 0
    print(
        f"{'TOTAL':<5} {total_true:>15,.0f} {total_rf:>15,.0f} {total_rf_err:>+9.2f}% {total_cl:>15,.0f} {total_cl_err:>+9.2f}%"
    )

    # Identify where RF wins/loses vs CL
    print("\n--- Per-AY Winner (lower absolute error) ---")
    rf_wins = 0
    cl_wins = 0
    for _, row in best_ay.iterrows():
        ay = int(row["accident_year"])
        rf_abs = abs(row["reserve_error_pct"])
        cl_match = cl_ay[cl_ay["accident_year"] == ay]
        if not cl_match.empty and cl_match.iloc[0].get("true_reserve", 0) > 0:
            cl_true = cl_match.iloc[0]["true_reserve"]
            cl_pred = cl_match.iloc[0].get(
                "CL_reserve", cl_match.iloc[0].get("predicted_reserve", np.nan)
            )
            cl_abs = abs((cl_pred - cl_true) / cl_true * 100) if cl_true > 0 else 999
            winner = "RF" if rf_abs < cl_abs else "CL"
            if winner == "RF":
                rf_wins += 1
            else:
                cl_wins += 1
            print(f"  AY {ay}: RF {rf_abs:.1f}% vs CL {cl_abs:.1f}% -> {winner}")
    print(f"\n  RF wins {rf_wins} AYs, CL wins {cl_wins} AYs")


# -----------------------------------------------------------------------
# 3. Feature Importance
# -----------------------------------------------------------------------
def analyze_feature_importance(fi, comparison):
    section_divider("3. FEATURE IMPORTANCE (Best RF Config)")

    if fi is None:
        print("No feature importances saved.")
        return

    # Best RF config
    ml_rows = comparison[comparison["method"] != "chain_ladder"]
    if ml_rows.empty:
        return

    best_name = ml_rows.iloc[0]["name"]
    best_fi = fi[fi["model_name"] == best_name].copy()

    if best_fi.empty:
        print(f"No feature importances for {best_name}. Showing first available.")
        best_fi = fi[fi["model_name"] == fi["model_name"].iloc[0]].copy()

    best_fi = best_fi.sort_values("importance", ascending=False)

    print(f"Model: {best_fi['model_name'].iloc[0]}")
    print(f"\n{'Rank':<5} {'Feature':<40} {'Importance':>12} {'Cumulative':>12}")
    print("-" * 72)

    cum = 0
    for i, (_, row) in enumerate(best_fi.iterrows()):
        cum += row["importance"]
        print(
            f"{i + 1:<5} {row['feature']:<40} {row['importance']:>12.4f} {cum:>11.4f}"
        )
        if i >= 19:  # Top 20
            break

    # Summarize importance by feature category
    print(f"\n--- Importance by Category ---")
    categories = {
        "Claim type": [f for f in best_fi["feature"].values if f.startswith("type_")],
        "Development pattern": ["dev_lag", "elapsed_pct", "remaining_dys"],
        "Accident year": ["accident_year", "ay_normalized"],
        "Payment history": [
            f
            for f in best_fi["feature"].values
            if any(x in f for x in ["prev_", "cum_", "paid_", "payment_", "pmt_"])
        ],
        "Claim status": [
            f
            for f in best_fi["feature"].values
            if any(x in f for x in ["open_", "is_open", "settled"])
        ],
    }

    for cat_name, cat_features in categories.items():
        cat_imp = best_fi[best_fi["feature"].isin(cat_features)]["importance"].sum()
        if cat_imp > 0:
            print(f"  {cat_name:<30}: {cat_imp:.4f} ({cat_imp * 100:.1f}%)")


# -----------------------------------------------------------------------
# 4. RBNS Predictions Analysis
# -----------------------------------------------------------------------
def analyze_rbns_predictions(predictions, comparison, claims):
    section_divider("4. RBNS PREDICTIONS ANALYSIS (Claim × DY Granularity)")

    if not predictions:
        print("No prediction files found.")
        return

    # Best config
    ml_rows = comparison[comparison["method"] != "chain_ladder"]
    if ml_rows.empty:
        return

    best_name = ml_rows.iloc[0]["name"]

    # Summary for all 8 RF configs
    print(
        f"{'Config':<55} {'N rows':>8} {'N claims':>10} {'N nonzero':>10} {'Mean pred':>12} {'Sum pred':>15}"
    )
    print("-" * 115)

    for name, preds in sorted(predictions.items()):
        n_rows = len(preds)
        n_claims = preds["claim_no"].nunique()
        n_nonzero = (preds["predicted_payment"] > 0).sum()
        mean_pred = preds["predicted_payment"].mean()
        sum_pred = preds["predicted_payment"].sum()
        marker = " *" if name == best_name else ""
        print(
            f"{name:<55} {n_rows:>8} {n_claims:>10} {n_nonzero:>10} {mean_pred:>12,.0f} {sum_pred:>15,.0f}{marker}"
        )

    # Detailed analysis of best config
    if best_name in predictions:
        preds = predictions[best_name]
        print(f"\n--- Detailed: {best_name} ---")

        # By DY
        print(f"\nPer Development Year:")
        print(
            f"{'DY':<5} {'N rows':>8} {'N nonzero true':>15} {'N nonzero pred':>15} {'True sum':>15} {'Pred sum':>15} {'Error%':>10}"
        )
        print("-" * 88)

        for dy in sorted(preds["dev_lag"].unique()):
            dy_data = preds[preds["dev_lag"] == dy]
            n = len(dy_data)
            nz_true = (dy_data["true_incremental_payment"] > 0).sum()
            nz_pred = (dy_data["predicted_payment"] > 0).sum()
            true_sum = dy_data["true_incremental_payment"].sum()
            pred_sum = dy_data["predicted_payment"].sum()
            err = (pred_sum - true_sum) / true_sum * 100 if true_sum > 0 else 0
            print(
                f"{dy:<5} {n:>8} {nz_true:>15} {nz_pred:>15} {true_sum:>15,.0f} {pred_sum:>15,.0f} {err:>+9.2f}%"
            )

        # By AY
        print(f"\nPer Accident Year:")
        print(
            f"{'AY':<5} {'N rows':>8} {'True sum':>15} {'Pred sum':>15} {'Error%':>10}"
        )
        print("-" * 58)

        for ay in sorted(preds["accident_year"].unique()):
            ay_data = preds[preds["accident_year"] == ay]
            n = len(ay_data)
            true_sum = ay_data["true_incremental_payment"].sum()
            pred_sum = ay_data["predicted_payment"].sum()
            err = (pred_sum - true_sum) / true_sum * 100 if true_sum > 0 else 0
            print(f"{ay:<5} {n:>8} {true_sum:>15,.0f} {pred_sum:>15,.0f} {err:>+9.2f}%")


# -----------------------------------------------------------------------
# 5. Config comparison analysis
# -----------------------------------------------------------------------
def analyze_config_patterns(comparison):
    section_divider("5. CONFIGURATION PATTERN ANALYSIS")

    ml_rows = comparison[comparison["method"] != "chain_ladder"].copy()
    if ml_rows.empty:
        return

    # Structure comparison
    print("--- By Model Structure ---")
    for struct in ["single_model", "per_dy"]:
        subset = ml_rows[ml_rows["structure"] == struct]
        if not subset.empty:
            mean_err = subset["total_reserve_error_pct"].abs().mean()
            best_err = subset["total_reserve_error_pct"].abs().min()
            print(
                f"  {struct:<15}: mean |error| = {mean_err:.2f}%, best = {best_err:.2f}%"
            )

    # Target comparison
    print("\n--- By Target Transform ---")
    for target in ["raw", "log1p"]:
        subset = ml_rows[ml_rows["target"] == target]
        if not subset.empty:
            mean_err = subset["total_reserve_error_pct"].abs().mean()
            best_err = subset["total_reserve_error_pct"].abs().min()
            print(
                f"  {target:<15}: mean |error| = {mean_err:.2f}%, best = {best_err:.2f}%"
            )

    # Zero handling comparison
    print("\n--- By Zero Handling ---")
    for zh in ["include_zeros", "two_stage"]:
        subset = ml_rows[ml_rows["zero_handling"] == zh]
        if not subset.empty:
            mean_err = subset["total_reserve_error_pct"].abs().mean()
            best_err = subset["total_reserve_error_pct"].abs().min()
            print(f"  {zh:<15}: mean |error| = {mean_err:.2f}%, best = {best_err:.2f}%")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  ML CLAIMS RESERVING — POST-RUN ANALYSIS")
    print("=" * 80)

    # Load everything
    comparison, cl_results, ay_reserves, fi, predictions = load_results()
    claims = load_data()

    # Run all analyses
    analyze_model_comparison(comparison)
    analyze_per_ay(ay_reserves, comparison, cl_results)
    analyze_feature_importance(fi, comparison)
    analyze_rbns_predictions(predictions, comparison, claims)
    analyze_config_patterns(comparison)

    section_divider("ANALYSIS COMPLETE")
    print(f"Prediction files: {len(predictions)}")
    print(f"Total prediction rows: {sum(len(p) for p in predictions.values()):,}")


if __name__ == "__main__":
    main()
