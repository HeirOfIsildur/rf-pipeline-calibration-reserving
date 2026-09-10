"""
Evaluation module.

Computes claim-level and aggregate-level metrics, builds comparison tables,
and generates plots for all experiment results.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import src.config as cfg

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from src.chain_ladder import load_triangle


# -----------------------------------------------------------------------
# Claim-level metrics
# -----------------------------------------------------------------------


def compute_claim_level_metrics(y_true, y_pred):
    """
    Compute RMSE, MAE, R² on individual claim × DY predictions.

    Returns dict with rmse, mae, r2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    if n == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"rmse": rmse, "mae": mae, "r2": r2}


# -----------------------------------------------------------------------
# Aggregate reserve metrics
# -----------------------------------------------------------------------


def compute_aggregate_reserves(predictions_df, pred_col="predicted_payment"):
    """
    Aggregate individual claim × DY predictions to accident-year level reserves.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must have columns: accident_year, dev_lag, predicted_payment,
                          true_incremental_payment

    Returns
    -------
    reserves_df : pd.DataFrame
        One row per AY with predicted_reserve, true_reserve, error, error_pct.
    """
    if predictions_df.empty:
        return pd.DataFrame(
            columns=[
                "accident_year",
                "predicted_reserve",
                "true_reserve",
                "reserve_error",
                "reserve_error_pct",
            ]
        )

    # Warn if any predictions are NaN — nansum will silently underestimate
    # the predicted reserve for affected AYs.
    n_nan = predictions_df[pred_col].isna().sum()
    if n_nan > 0:
        nan_by_ay = (
            predictions_df[predictions_df[pred_col].isna()]
            .groupby("accident_year")
            .size()
        )
        warnings.warn(
            f"{n_nan} NaN predictions found — predicted reserves may be "
            f"underestimated. Affected AYs: {dict(nan_by_ay)}",
            stacklevel=2,
        )

    # Sum predictions and ground truth by AY.
    # Use nansum for predicted_payment so that DYs where prediction failed
    # (NaN) don't contaminate the total, while true_reserve always reflects
    # the full ground truth regardless of prediction success.
    agg = (
        predictions_df.groupby("accident_year")
        .agg(
            predicted_reserve=(pred_col, lambda x: np.nansum(x)),
            true_reserve=("true_incremental_payment", "sum"),
        )
        .reset_index()
    )

    agg["reserve_error"] = agg["predicted_reserve"] - agg["true_reserve"]
    # Edge case: when true_reserve is 0, error % is undefined.
    # If predicted is also ~0, report 0%; otherwise NaN to flag the discrepancy.
    agg["reserve_error_pct"] = np.where(
        agg["true_reserve"] != 0,
        agg["reserve_error"] / agg["true_reserve"] * 100,
        np.where(np.isclose(agg["predicted_reserve"], 0, atol=1e-8), 0.0, np.nan),
    )

    return agg


# -----------------------------------------------------------------------
# Build comparison table across all experiments
# -----------------------------------------------------------------------


def build_comparison_table(all_results, chain_ladder_results=None):
    """
    Build a comprehensive comparison table of all model results.

    Parameters
    ----------
    all_results : list of dicts
        Each from run_single_experiment.
    chain_ladder_results : dict or None
        Results from chain_ladder.run_chain_ladder().

    Returns
    -------
    comparison_df : pd.DataFrame
        Sorted by total_reserve_error_pct (ascending absolute value).
    """
    rows = []

    for result in all_results:
        preds = result["predictions"]
        name = result["name"]

        if preds.empty:
            rows.append(
                {
                    "name": name,
                    "method": result["method"],
                    "structure": result["structure"],
                    "target": result["target"],
                    "zero_handling": result["zero_handling"],
                    "rmse": np.nan,
                    "mae": np.nan,
                    "r2": np.nan,
                    "total_predicted_reserve": np.nan,
                    "total_true_reserve": np.nan,
                    "total_reserve_error": np.nan,
                    "total_reserve_error_pct": np.nan,
                    "max_ay_error_pct": np.nan,
                    "runtime_seconds": result.get("runtime_seconds", 0),
                }
            )
            continue

        # Claim-level metrics (only on rows with actual predictions)
        valid_mask = preds["predicted_payment"].notna()
        preds_valid = preds[valid_mask]
        claim_metrics = compute_claim_level_metrics(
            preds_valid["true_incremental_payment"].values,
            preds_valid["predicted_payment"].values,
        )

        # Aggregate reserves
        reserves = compute_aggregate_reserves(preds)
        total_pred = reserves["predicted_reserve"].sum()
        total_true = reserves["true_reserve"].sum()
        total_error = total_pred - total_true
        total_error_pct = (total_error / total_true * 100) if total_true > 0 else 0.0
        max_ay_error_pct = (
            reserves["reserve_error_pct"].abs().max() if len(reserves) > 0 else np.nan
        )

        rows.append(
            {
                "name": name,
                "objective": result.get("objective"),
                "method": result["method"],
                "structure": result["structure"],
                "target": result["target"],
                "zero_handling": result["zero_handling"],
                "rmse": claim_metrics["rmse"],
                "mae": claim_metrics["mae"],
                "r2": claim_metrics["r2"],
                "total_predicted_reserve": total_pred,
                "total_true_reserve": total_true,
                "total_reserve_error": total_error,
                "total_reserve_error_pct": total_error_pct,
                "max_ay_error_pct": max_ay_error_pct,
                "runtime_seconds": result.get("runtime_seconds", 0),
            }
        )

    # Chain-ladder rows, each labelled with the truth it targets.  The ML
    # models reserve the RBNS cohort, so the like-for-like baseline is the
    # cohort-consistent RBNS chain ladder; the standard CL (factors contain
    # late reporters' payments) targets the TOTAL reserve and is evaluated
    # against the total truth.  Falls back to the legacy single row when the
    # individual data needed for the split is unavailable.
    if chain_ladder_results is not None:
        cl = chain_ladder_results
        cl_df = cl["projected_df"]
        cl_ay_errors = cl_df[cl_df["true_reserve"] > 0]["CL_error_pct"].abs()
        max_cl_ay_error = cl_ay_errors.max() if len(cl_ay_errors) > 0 else np.nan

        def _cl_row(name, target, pred_by_ay, true_by_ay, max_ay=np.nan):
            pred, true = pred_by_ay.sum(), true_by_ay.sum()
            return {
                "name": name,
                "method": "chain_ladder",
                "structure": "aggregate",
                "target": target,
                "zero_handling": "N/A",
                "rmse": np.nan,  # no claim-level metrics for CL
                "mae": np.nan,
                "r2": np.nan,
                "total_predicted_reserve": pred,
                "total_true_reserve": true,
                "total_reserve_error": pred - true,
                "total_reserve_error_pct": (pred - true) / true * 100 if true > 0 else 0.0,
                "max_ay_error_pct": max_ay,
                "runtime_seconds": 0,
            }

        rbns_cl = cl.get("rbns_cohort_reserves_by_ay")
        total_truth = cl.get("total_true_reserves_by_ay")
        if rbns_cl is not None:
            rows.append(_cl_row("chain_ladder_rbns_cohort", "RBNS cohort",
                                rbns_cl, cl["true_reserves_by_ay"]))
        if total_truth is not None:
            rows.append(_cl_row("chain_ladder_total", "total (RBNS+IBNR)",
                                cl["reserves_by_ay"], total_truth))
        if rbns_cl is None and total_truth is None:
            rows.append(_cl_row("chain_ladder", "cumulative",
                                cl["reserves_by_ay"], cl["true_reserves_by_ay"],
                                max_ay=max_cl_ay_error))

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(
        "total_reserve_error_pct",
        key=lambda x: x.abs(),
        ascending=True,
    ).reset_index(drop=True)

    return comparison_df


# -----------------------------------------------------------------------
# Correction arms (revision v2): one row per configuration x arm
# -----------------------------------------------------------------------


def build_correction_table(all_results):
    """
    Total-reserve accuracy of every post-hoc correction arm for every fitted
    configuration.  Arms are the `pred__<arm>` columns produced by
    src/corrections.py; `paper` is the first-submission method.
    """
    rows = []
    for result in all_results:
        preds = result["predictions"]
        if preds.empty:
            continue
        arm_cols = [c for c in preds.columns if c.startswith("pred__")]
        for col in arm_cols:
            if preds[col].notna().sum() == 0:
                continue
            reserves = compute_aggregate_reserves(preds, pred_col=col)
            total_pred = reserves["predicted_reserve"].sum()
            total_true = reserves["true_reserve"].sum()
            err = total_pred - total_true
            valid = preds[col].notna()
            resid = preds.loc[valid, col].values - preds.loc[valid, "true_incremental_payment"].values
            rows.append(
                {
                    "name": result["name"],
                    "objective": result.get("objective"),
                    "method": result["method"],
                    "structure": result["structure"],
                    "target": result["target"],
                    "zero_handling": result["zero_handling"],
                    "arm": col.replace("pred__", "", 1),
                    "total_predicted_reserve": total_pred,
                    "total_true_reserve": total_true,
                    "total_reserve_error": err,
                    "total_reserve_error_pct": (err / total_true * 100) if total_true > 0 else np.nan,
                    "max_ay_error_pct": reserves["reserve_error_pct"].abs().max() if len(reserves) else np.nan,
                    "rmse": float(np.sqrt(np.mean(resid ** 2))) if len(resid) else np.nan,
                    "mae": float(np.mean(np.abs(resid))) if len(resid) else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["name", "arm"]).reset_index(drop=True)
    return df


def collect_calibration_diagnostics(all_results):
    """Concatenate the per-configuration calibration diagnostics (alpha, s2, v, Z, factors)."""
    frames = [
        r["calibration_diagnostics"]
        for r in all_results
        if isinstance(r.get("calibration_diagnostics"), pd.DataFrame)
        and not r["calibration_diagnostics"].empty
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------


def plot_results(all_results, chain_ladder_results, comparison_df):
    """Generate all evaluation plots and save to RESULTS_DIR."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    _plot_reserve_comparison(all_results, chain_ladder_results, comparison_df)
    _plot_error_distribution(all_results, comparison_df)

    print(f"\nPlots saved to {cfg.RESULTS_DIR}/")


def _plot_reserve_comparison(all_results, chain_ladder_results, comparison_df):
    """Bar chart: total predicted vs actual reserve for top models + CL."""
    top_n = min(10, len(comparison_df))
    top = comparison_df.head(top_n)

    # Figure sized to match \textwidth in the paper (A4, 2.5cm margins)
    # so that 12pt fonts in matplotlib render as 12pt in the PDF.
    fig, ax = plt.subplots(figsize=(6.3, 4.5))

    x = np.arange(top_n)
    width = 0.35

    scale = 1e9
    pred_b = top["total_predicted_reserve"] / scale
    true_b = top["total_true_reserve"] / scale

    ax.bar(
        x - width / 2,
        pred_b,
        width,
        label="Predicted Reserve",
        color="steelblue",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        true_b,
        width,
        label="True Reserve",
        color="coral",
        alpha=0.8,
    )

    ax.set_xlabel("Model", fontsize=9)
    ax.set_ylabel("Total Reserve (billions)", fontsize=9)
    ax.set_title("")
    ax.set_xticks(x)
    labels = [_short_name(n) for n in top["name"]]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Add error % labels
    for i, (_, row) in enumerate(top.iterrows()):
        pct = row["total_reserve_error_pct"]
        ax.annotate(
            f"{pct:+.1f}%",
            xy=(i, max(row["total_predicted_reserve"], row["total_true_reserve"]) / scale),
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "reserve_comparison_top10.pdf"), dpi=300)
    plt.close()


def _plot_method_heatmap(comparison_df):
    """Heatmap: total_reserve_error_pct by method × (structure + target + zero_handling)."""
    ml_only = comparison_df[comparison_df["method"] != "chain_ladder"].copy()
    if ml_only.empty:
        return

    ml_only["config"] = (
        ml_only["structure"]
        + "\n"
        + ml_only["target"]
        + "\n"
        + ml_only["zero_handling"]
    )

    pivot = ml_only.pivot_table(
        index="method",
        columns="config",
        values="total_reserve_error_pct",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Reserve Error %"},
    )
    ax.set_title("Total Reserve Error (%) by Method and Configuration", fontsize=16)
    ax.set_xlabel("Configuration (structure / target / zero_handling)", fontsize=14)
    ax.set_ylabel("Method", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "method_heatmap.pdf"), dpi=150)
    plt.close()


def _plot_error_distribution(all_results, comparison_df):
    """Histogram of claim-level prediction errors for top 3 models."""
    top3 = comparison_df[comparison_df["method"] != "chain_ladder"].head(3)

    # Figure sized to match \textwidth in the paper (A4, 2.5cm margins)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5))

    for idx, (_, row) in enumerate(top3.iterrows()):
        name = row["name"]
        result = next((r for r in all_results if r["name"] == name), None)
        if result is None or result["predictions"].empty:
            continue

        preds = result["predictions"]
        errors = (
            preds["predicted_payment"].values - preds["true_incremental_payment"].values
        )
        # Drop NaN errors (from DYs where prediction failed)
        errors = errors[~np.isnan(errors)]
        if len(errors) == 0:
            continue

        ax = axes[idx]
        # Focus on non-zero errors to reveal distribution shape
        errors_nonzero = errors[errors != 0]
        if len(errors_nonzero) == 0:
            errors_nonzero = errors
        # Clip extreme outliers for better visualization
        clip_lo, clip_hi = np.percentile(errors_nonzero, [1, 99])
        errors_clipped = errors_nonzero[
            (errors_nonzero >= clip_lo) & (errors_nonzero <= clip_hi)
        ]

        ax.hist(
            errors_clipped, bins=80, color="steelblue", alpha=1.0, edgecolor="steelblue"
        )
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(_short_name(name), fontsize=8)
        ax.set_xlabel("Prediction Error", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "error_distribution_top3.pdf"), dpi=300)
    plt.close()


def _plot_per_ay_reserves(all_results, chain_ladder_results, comparison_df):
    """Per-AY reserve comparison for the best ML model vs Chain Ladder vs truth."""
    # Find best ML model
    ml_models = comparison_df[comparison_df["method"] != "chain_ladder"]
    if ml_models.empty:
        return

    best_name = ml_models.iloc[0]["name"]
    best_result = next((r for r in all_results if r["name"] == best_name), None)
    if best_result is None or best_result["predictions"].empty:
        return

    # Compute per-AY reserves for best model
    ml_reserves = compute_aggregate_reserves(best_result["predictions"])

    # True reserves and CL reserves per AY
    full_tri = load_triangle("triangle_full_rbns.csv")

    fig, ax = plt.subplots(figsize=(12, 6))

    ays = ml_reserves["accident_year"].values
    ax.bar(
        ays - 0.25,
        ml_reserves["true_reserve"],
        0.25,
        label="True Reserve",
        color="coral",
        alpha=0.8,
    )
    ax.bar(
        ays,
        ml_reserves["predicted_reserve"],
        0.25,
        label=f"Best ML ({_short_name(best_name)})",
        color="steelblue",
        alpha=0.8,
    )

    if chain_ladder_results is not None:
        cl_reserves = chain_ladder_results["reserves_by_ay"]
        # Only plot AYs that have reserves (matching ML AYs)
        cl_ays = np.arange(1, cfg.MAX_ACCIDENT_YEAR + 1)
        cl_mask = np.isin(cl_ays, ays)
        if cl_mask.any():
            ax.bar(
                ays + 0.25,
                cl_reserves[cl_mask],
                0.25,
                label="Chain Ladder",
                color="green",
                alpha=0.8,
            )

    ax.set_xlabel("Accident Year", fontsize=14)
    ax.set_ylabel("Reserve Amount", fontsize=14)
    ax.set_title("Predicted vs True Reserves by Accident Year", fontsize=16)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(ays)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "per_ay_reserves.pdf"), dpi=150)
    plt.close()


# -----------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------


def save_results(comparison_df, all_results, chain_ladder_results, is_full_run=False):
    """Save comparison table and detailed predictions to CSV.

    When is_full_run is True, additionally write full_data_* copies so a later
    --quick run does not overwrite data used by --plot-only.
    """
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    def _emit(df, fname):
        df.to_csv(os.path.join(cfg.RESULTS_DIR, fname), index=False)
        if is_full_run:
            df.to_csv(os.path.join(cfg.RESULTS_DIR, "full_data_" + fname), index=False)

    # Save comparison table
    _emit(comparison_df, "model_comparison.csv")
    print(f"Saved comparison table to {cfg.RESULTS_DIR}/model_comparison.csv")

    # Correction arms and calibration diagnostics (revision v2)
    corr_df = build_correction_table(all_results)
    if not corr_df.empty:
        _emit(corr_df, "correction_comparison.csv")
        print(f"Saved correction-arm table to {cfg.RESULTS_DIR}/correction_comparison.csv")
    diag_df = collect_calibration_diagnostics(all_results)
    if not diag_df.empty:
        _emit(diag_df, "calibration_factors.csv")
        print(f"Saved calibration diagnostics to {cfg.RESULTS_DIR}/calibration_factors.csv")

    # Save predictions from ALL models (claim × DY granularity)
    ml_models = comparison_df[comparison_df["method"] != "chain_ladder"]
    for _, row in ml_models.iterrows():
        name = row["name"]
        result = next((r for r in all_results if r["name"] == name), None)
        if result is not None and not result["predictions"].empty:
            _emit(result["predictions"], f"predictions__{name}.csv")

    # Save chain ladder results
    if chain_ladder_results is not None:
        _emit(chain_ladder_results["projected_df"], "chain_ladder_results.csv")

    # Save per-AY reserve comparison for all models
    ay_reserve_rows = []
    for result in all_results:
        if result["predictions"].empty:
            continue
        reserves = compute_aggregate_reserves(result["predictions"])
        reserves["model_name"] = result["name"]
        ay_reserve_rows.append(reserves)

    if ay_reserve_rows:
        ay_reserves_df = pd.concat(ay_reserve_rows, ignore_index=True)
        _emit(ay_reserves_df, "all_models_reserves_by_ay.csv")

    print(f"All results saved to {cfg.RESULTS_DIR}/")


def save_feature_importances(all_results, is_full_run=False):
    """Save feature importances from all models that have them."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    fi_rows = []
    for result in all_results:
        fi = result.get("feature_importances")
        if fi is not None:
            for feat_name, importance in fi.items():
                fi_rows.append(
                    {
                        "model_name": result["name"],
                        "method": result["method"],
                        "structure": result["structure"],
                        "target": result["target"],
                        "zero_handling": result["zero_handling"],
                        "feature": feat_name,
                        "importance": importance,
                    }
                )

    if fi_rows:
        fi_df = pd.DataFrame(fi_rows)
        fi_df.to_csv(
            os.path.join(cfg.RESULTS_DIR, "feature_importances.csv"), index=False
        )
        if is_full_run:
            fi_df.to_csv(
                os.path.join(cfg.RESULTS_DIR, "full_data_feature_importances.csv"),
                index=False,
            )
        print(f"Saved feature importances to {cfg.RESULTS_DIR}/feature_importances.csv")


def plot_from_full_data():
    """Re-render the RBNS pipeline plots from full_data_* CSVs.

    Reconstructs the comparison_df, chain_ladder_results dict and all_results
    list that plot_results() expects, then calls plot_results. Used by
    `python main.py --plot-only`.
    """
    import glob

    def _require(fname):
        path = os.path.join(cfg.RESULTS_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run `python main.py` (without --quick) once "
                f"to produce the full_data_* cache."
            )
        return path

    comparison_df = pd.read_csv(_require("full_data_model_comparison.csv"))
    cl_projected = pd.read_csv(_require("full_data_chain_ladder_results.csv"))
    chain_ladder_results = {
        "projected_df": cl_projected,
        "reserves_by_ay": cl_projected["CL_reserve"].values,
        "true_reserves_by_ay": cl_projected["true_reserve"].values,
    }

    pred_paths = sorted(
        glob.glob(os.path.join(cfg.RESULTS_DIR, "full_data_predictions__*.csv"))
    )
    if not pred_paths:
        raise FileNotFoundError(
            f"No full_data_predictions__*.csv files in {cfg.RESULTS_DIR}. "
            f"Run `python main.py` (without --quick) once to produce them."
        )

    all_results = []
    name_to_meta = {
        row["name"]: row for _, row in comparison_df.iterrows() if pd.notna(row.get("name"))
    }
    for path in pred_paths:
        base = os.path.basename(path)
        name = base[len("full_data_predictions__"):-len(".csv")]
        meta = name_to_meta.get(name, {})
        all_results.append(
            {
                "name": name,
                "method": meta.get("method"),
                "structure": meta.get("structure"),
                "target": meta.get("target"),
                "zero_handling": meta.get("zero_handling"),
                "predictions": pd.read_csv(path),
                "feature_importances": None,
            }
        )
    print(f"  Loaded {len(all_results)} model prediction sets from full_data_*")

    plot_results(all_results, chain_ladder_results, comparison_df)


# -----------------------------------------------------------------------
# Print summary
# -----------------------------------------------------------------------


def print_summary(comparison_df, chain_ladder_results):
    """Print a formatted summary to console."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Chain Ladder baseline
    if chain_ladder_results is not None:
        cl_row = comparison_df[comparison_df["method"] == "chain_ladder"]
        if not cl_row.empty:
            cl = cl_row.iloc[0]
            print(f"\n--- Chain Ladder Baseline ---")
            print(
                f"  Total reserve error: {cl['total_reserve_error']:,.0f} "
                f"({cl['total_reserve_error_pct']:+.2f}%)"
            )

    # Top 10 ML models
    ml_models = comparison_df[comparison_df["method"] != "chain_ladder"]
    print(f"\n--- Top 10 ML Models (by total reserve error %) ---")
    print(
        f"{'Rank':<5} {'Name':<50} {'RMSE':>12} {'MAE':>12} {'R²':>8} "
        f"{'Reserve Err%':>14} {'Max AY Err%':>13}"
    )
    print("-" * 114)

    for i, (_, row) in enumerate(ml_models.head(10).iterrows()):
        print(
            f"{i + 1:<5} {_short_name(row['name']):<50} "
            f"{row['rmse']:>12,.0f} {row['mae']:>12,.0f} {row['r2']:>8.4f} "
            f"{row['total_reserve_error_pct']:>+13.2f}% "
            f"{row['max_ay_error_pct']:>12.1f}%"
        )

    # Best model details
    if not ml_models.empty:
        best = ml_models.iloc[0]
        print(f"\n--- Best Model ---")
        print(f"  Name:           {best['name']}")
        print(f"  Method:         {best['method']}")
        print(f"  Structure:      {best['structure']}")
        print(f"  Target:         {best['target']}")
        print(f"  Zero handling:  {best['zero_handling']}")
        print(f"  RMSE:           {best['rmse']:,.0f}")
        print(f"  MAE:            {best['mae']:,.0f}")
        print(f"  R²:             {best['r2']:.4f}")
        print(
            f"  Total reserve error: {best['total_reserve_error']:,.0f} "
            f"({best['total_reserve_error_pct']:+.2f}%)"
        )
        print(f"  Runtime:        {best['runtime_seconds']:.1f}s")

    print("\n" + "=" * 80)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _short_name(name):
    """Shorten a config name for display using abbreviations."""
    abbrevs = {
        "random_forest": "",
        "per_dy": "PD",
        "single_model": "SM",
        "log1p": "L",
        "raw": "R",
        "two_stage": "2S",
        "include_zeros": "IZ",
        "chain_ladder": "CL",
    }
    parts = name.split("__")
    short = [abbrevs.get(p, p) for p in parts]
    return " / ".join(p for p in short if p)
