#!/usr/bin/env python3
"""
IBNR (Incurred But Not Reported) reserve model.

Combines:
  1. Chain Ladder on a 10x10 claim-count triangle (frequency component)
  2. Random Forest for individual claim severity

The frequency model predicts how many more claims will be reported per AY.
The severity model predicts the expected cost of each late-reported claim,
using claim_type mix estimated from observed late reporters.

The IBNR reserve is then combined with RBNS reserves from the main pipeline
to produce total reserve estimates.

No future information is used in training or prediction:
  - Frequency: observed claim count triangle only
  - Severity target: paid-to-date of the reported claims of mature accident years (>= 8 development
    years observed at valuation; prospective). --severity-train settled|all for the two alternatives.
  - Claim-type mix: RBNS late reporters (report_dy >= 2)
  - IBNR ground truth: used for evaluation only

Usage:
    python ibnr_model.py                   # full tuning (25 Optuna trials)
    python ibnr_model.py --quick           # fast test (5 trials)
    python ibnr_model.py --n-trials 10     # custom trial count
"""

import argparse
import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GroupKFold

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    DATA_DIR,
    RESULTS_DIR,
    MAX_ACCIDENT_YEAR,
    MAX_DEV_LAG,
    RANDOM_STATE,
    DEFAULT_BEST_MODEL,
    GLOBAL_CALIB_KEY,
)

BASE_YEAR = 2000  # AY 1 = calendar year 2000


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------


def load_claims():
    """Load claims_individual.csv and split into RBNS / IBNR."""
    path = os.path.join(DATA_DIR, "claims_individual.csv")
    df = pd.read_csv(path)

    # Reporting development year: which DY the claim first appears in.
    # This is computed during data conversion using calendar years
    # (report_dy = reporting_cal_year - accident_cal_year + 1) so that
    # it aligns with the observability rule AY + DY - 1 <= MAX_AY.
    # A notidel-based formula (ceil(notidel)) would misclassify claims
    # reported in the next calendar year with notidel < 1 as DY 1,
    # contaminating the observed triangle with IBNR claims.
    if "report_dy" not in df.columns:
        # Fallback for older data files without pre-computed report_dy.
        df["report_dy"] = np.ceil(df["notidel"]).astype(int).clip(lower=1)

    rbns = df[df["ibnr_flag"] == 0].copy()
    ibnr = df[df["ibnr_flag"] == 1].copy()

    print(f"  Total claims: {len(df):,}")
    print(f"  RBNS claims:  {len(rbns):,}")
    print(f"  IBNR claims:  {len(ibnr):,}")
    print(f"  IBNR reserve (true): {ibnr['claim_size'].sum():,.0f}")

    return df, rbns, ibnr


# ---------------------------------------------------------------------------
# 2. Chain Ladder frequency model (claim count triangle)
# ---------------------------------------------------------------------------


def build_count_triangle(claims_all):
    """
    Build a 10x10 claim count triangle: rows = AY (1..10), cols = report DY (1..10).

    Uses the same dimensions as the paid triangle (MAX_ACCIDENT_YEAR x MAX_DEV_LAG).
    Report DYs beyond ~5 will have zero counts (max reporting delay < 5 years).

    Returns:
        full_inc: incremental count triangle (full, including IBNR)
        full_cum: cumulative count triangle (full)
        obs_cum:  cumulative count triangle (observed only, NaN for unobservable)
    """
    n_ay = MAX_ACCIDENT_YEAR
    n_dy = MAX_DEV_LAG

    # Full incremental count triangle (includes IBNR — ground truth)
    full_inc = np.zeros((n_ay, n_dy))
    for _, row in claims_all.iterrows():
        ay = int(row["accident_year"]) - 1  # 0-indexed
        rdy = min(int(row["report_dy"]), n_dy) - 1  # 0-indexed, capped
        if 0 <= ay < n_ay and 0 <= rdy < n_dy:
            full_inc[ay, rdy] += 1

    # Cumulative count triangle
    full_cum = np.cumsum(full_inc, axis=1)

    # Observed cumulative: mask out cells where AY + DY - 1 > MAX_ACCIDENT_YEAR
    obs_cum = full_cum.copy().astype(float)
    for i in range(n_ay):
        for j in range(n_dy):
            ay_1based = i + 1
            dy_1based = j + 1
            if ay_1based + dy_1based - 1 > MAX_ACCIDENT_YEAR:
                obs_cum[i, j] = np.nan

    return full_inc, full_cum, obs_cum


def cl_frequency(obs_cum, full_cum, window: "int | None" = 5):
    """
    Apply Chain Ladder to the cumulative claim count triangle.

    When ``window`` is set, only the most recent ``window`` accident years
    (that have observable data for each transition) are used to compute the
    volume-weighted development factors.  This addresses the common problem
    of a trend in reporting speed across accident years — older AYs with
    slower reporting inflate the DY1→DY2 link ratio, causing systematic
    over-prediction of IBNR counts for recent AYs.

    Parameters:
        obs_cum:  observed cumulative count triangle (NaN for unobservable)
        full_cum: full cumulative count triangle (ground truth)
        window:   number of most-recent AYs to use for factor estimation.
                  None = all available AYs (original CL behavior).

    Returns:
        predicted_counts: array of predicted total claim counts per AY
        ibnr_counts: array of predicted IBNR claim counts per AY
        dev_factors: development factors (length MAX_DEV_LAG - 1)
        true_counts: array of true total counts per AY
        observed_counts: latest observed cumulative count per AY
    """
    n_ay, n_dy = obs_cum.shape

    # Volume-weighted development factors (optionally windowed)
    factors = np.zeros(n_dy - 1)
    for j in range(n_dy - 1):
        # AYs that have observable data in both columns j and j+1
        mask = ~np.isnan(obs_cum[:, j]) & ~np.isnan(obs_cum[:, j + 1])
        eligible_ays = np.where(mask)[0]

        if len(eligible_ays) == 0:
            factors[j] = 1.0
            continue

        # Apply window: keep only the last `window` eligible AYs
        if window is not None and len(eligible_ays) > window:
            eligible_ays = eligible_ays[-window:]

        use_mask = np.zeros(n_ay, dtype=bool)
        use_mask[eligible_ays] = True

        num = obs_cum[use_mask, j + 1].sum()
        den = obs_cum[use_mask, j].sum()
        factors[j] = num / den if den > 0 else 1.0

    # Project
    projected = obs_cum.copy()
    for i in range(n_ay):
        for j in range(n_dy - 1):
            if np.isnan(projected[i, j + 1]):
                projected[i, j + 1] = projected[i, j] * factors[j]

    predicted_counts = projected[:, -1]
    true_counts = full_cum[:, -1]

    # Observed counts = latest non-NaN cumulative per AY
    observed_counts = np.zeros(n_ay)
    for i in range(n_ay):
        for j in range(n_dy - 1, -1, -1):
            if not np.isnan(obs_cum[i, j]):
                observed_counts[i] = obs_cum[i, j]
                break

    ibnr_counts = predicted_counts - observed_counts

    return predicted_counts, ibnr_counts, factors, true_counts, observed_counts


# ---------------------------------------------------------------------------
# 3. Late-reporter claim-type mix
# ---------------------------------------------------------------------------


def estimate_late_reporter_mix(rbns):
    """
    Estimate claim_type proportions among late reporters (report_dy >= 2).

    These proportions represent the expected composition of IBNR claims,
    since IBNR claims are by definition late reporters.
    """
    late = rbns[rbns["report_dy"] >= 2]

    if len(late) == 0:
        # Fallback: use overall mix
        late = rbns

    type_counts = late["claim_type"].value_counts()
    type_props = type_counts / type_counts.sum()

    print(f"\n  Late-reporter claim-type mix (n={len(late)}):")
    for ct, prop in type_props.items():
        print(f"    {ct:12s}: {prop:.3f} ({type_counts[ct]} claims)")

    return type_props


# ---------------------------------------------------------------------------
# 4. RF severity model
# ---------------------------------------------------------------------------


def prepare_severity_features(df):
    """
    Prepare features for the severity RF model.

    Features: accident_year, notidel, log_notidel, claim_type (one-hot).
    """
    feature_df = pd.DataFrame()
    feature_df["accident_year"] = df["accident_year"].values
    feature_df["notidel"] = df["notidel"].values
    feature_df["log_notidel"] = np.log1p(df["notidel"].values)

    # One-hot encode claim_type
    claim_types = df["claim_type"].values
    unique_types = sorted(set(claim_types))
    for ct in unique_types:
        feature_df[f"type_{ct}"] = (claim_types == ct).astype(float)

    feature_cols = list(feature_df.columns)
    return feature_df, feature_cols, unique_types


def tune_severity_model(X_train, y_train, n_trials, quick=False):
    """Tune RF hyperparameters with Optuna using 3-fold CV."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100)
            if not quick
            else trial.suggest_int("n_estimators", 50, 100, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 60, log=True
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 40, log=True),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5, 0.75]
            ),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=3, scoring="neg_mean_absolute_error", n_jobs=1
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  Best Optuna params: {best}")
    print(f"  Best CV MAE (log-scale): {-study.best_value:.4f}")

    return best


def train_severity_model(X_train, y_train, params):
    """Train the final severity RF model on all training data."""
    model_params = {
        "n_estimators": params.get("n_estimators", 200),
        "max_depth": params.get("max_depth", 15),
        "min_samples_split": params.get("min_samples_split", 5),
        "min_samples_leaf": params.get("min_samples_leaf", 3),
        "max_features": params.get("max_features", "sqrt"),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)
    return model


def compute_calibration_factors(
    X_train, y_train, params, n_folds=3, claim_types=None, accident_years=None
):
    """
    Compute pipeline-level ratio calibration for IBNR severity model.

    For the IBNR model (single-stage regressor on log1p(claim_size)),
    this computes calibration factors via cross-validation:

        factor = sum(y_true) / sum(expm1(y_pred))

    where y_pred is the out-of-sample log1p prediction.

    This corrects the systematic downward bias from Jensen's inequality
    when retransforming log-scale predictions (Manning & Mullahy, 2001).
    The factors are shrunk toward 1.0 using Bühlmann credibility
    (Bühlmann, 1967), with the reliability parameter K estimated from
    the between-fold variance of per-fold ratios.

    Parameters
    ----------
    accident_years : array-like or None
        If provided, used as groups for GroupKFold to prevent claims from
        the same AY from appearing in both train and val within a fold.
        This avoids optimistic calibration from AY-level effects.

    Returns
    -------
    calibration_factors : dict
        Mapping of claim_type -> calibration factor. Always includes GLOBAL_CALIB_KEY.
    """
    n = len(y_train)
    if n < n_folds * 2:
        return {GLOBAL_CALIB_KEY: 1.0}

    model_params = {
        "n_estimators": min(params.get("n_estimators", 200), 100),  # cap for CV speed
        "max_depth": params.get("max_depth", 15),
        "min_samples_split": params.get("min_samples_split", 5),
        "min_samples_leaf": params.get("min_samples_leaf", 3),
        "max_features": params.get("max_features", "sqrt"),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    # Compute original-scale true values from log1p targets
    y_orig = np.expm1(y_train)  # y_train is in log1p-space

    oos_preds = np.zeros(n)
    fold_ratios = []  # per-fold sum(actual)/sum(pred) for Bühlmann K estimation

    # Use GroupKFold on accident_year when available (consistent with main
    # pipeline in models.py:706-707), fall back to regular KFold otherwise.
    if accident_years is not None:
        unique_ays = np.unique(accident_years)
        actual_folds = min(n_folds, len(unique_ays))
        if actual_folds >= 2:
            kf = GroupKFold(n_splits=actual_folds)
            split_iter = kf.split(X_train, groups=accident_years)
        else:
            kf = KFold(n_splits=n_folds, shuffle=False)
            split_iter = kf.split(X_train)
    else:
        kf = KFold(n_splits=n_folds, shuffle=False)
        split_iter = kf.split(X_train)

    for train_idx, val_idx in split_iter:
        fold_model = RandomForestRegressor(**model_params)
        fold_model.fit(X_train[train_idx], y_train[train_idx])
        fold_log_preds = fold_model.predict(X_train[val_idx])
        oos_preds[val_idx] = np.clip(np.expm1(fold_log_preds), 0, None)
        fold_ratios.append(
            float(y_orig[val_idx].sum() / max(oos_preds[val_idx].sum(), 1e-12))
        )
        del fold_model

    # Credibility-inspired shrinkage (see models.py for full derivation).
    #
    # NOTE ON NON-STANDARD FORMULATION:
    # K_hat = max(n_eff * s²_alpha, 50),  Z = min(n_eff / K_hat, 1).
    # When n_eff * s²_alpha > 50, this simplifies to Z = 1 / s²_alpha,
    # independent of sample size.  Credibility is governed by fold-ratio
    # stability: when folds agree (s²_alpha < 1), Z = 1 (full trust);
    # when they disagree, Z < 1 (shrink toward 1.0).  The floor of 50
    # guards against overconfident calibration for small training sets.
    # Note: s²_alpha is estimated from only J=3 fold ratios (2 d.f.).
    n_eff = n  # all samples are non-zero for IBNR severity
    fold_ratios_arr = np.array(fold_ratios)
    fold_var = float(np.var(fold_ratios_arr, ddof=1)) if len(fold_ratios) > 1 else 0.0
    n_reliable = max(int(n_eff * fold_var), 50)
    shrinkage = min(n_eff / n_reliable, 1.0)

    def _ratio(actual, predicted):
        s_pred = predicted.sum()
        if s_pred <= 0:
            return 1.0
        raw = actual.sum() / s_pred
        factor = 1.0 + shrinkage * (raw - 1.0)
        return float(np.clip(factor, 0.5, 5.0))

    global_factor = _ratio(y_orig, oos_preds)
    calibration_factors = {GLOBAL_CALIB_KEY: global_factor}

    # Per-type calibration if claim_types provided
    if claim_types is not None:
        unique_types = np.unique(claim_types)
        if len(unique_types) > 1:
            for ct in unique_types:
                mask = claim_types == ct
                if mask.sum() >= 10:
                    ct_factor = _ratio(y_orig[mask], oos_preds[mask])
                    calibration_factors[ct] = ct_factor
                else:
                    calibration_factors[ct] = global_factor

    return calibration_factors


# ---------------------------------------------------------------------------
# 5. IBNR prediction: generate synthetic claims and predict severity
# ---------------------------------------------------------------------------


JOINT_TYPE_DELAY = False
OUT_SUFFIX = ""  # appended to output file names by --out-suffix (sensitivity runs must not overwrite the main outputs)


def _out(name):
    base, ext = os.path.splitext(name)
    return os.path.join(RESULTS_DIR, f"{base}{OUT_SUFFIX}{ext}")


def generate_ibnr_claims(ibnr_counts, type_props, rbns, rng, joint=False):
    """
    Generate synthetic IBNR claim records for each AY.

    For each AY with predicted IBNR > 0:
      - Draw claim count from Poisson(lambda=predicted_count) to capture
        frequency uncertainty across bootstrap draws
      - Assign claim_type based on late-reporter proportions
      - Sample notidel from the empirical distribution of late reporters
        in the same AY (or overall if too few)

    The Poisson draw (rather than deterministic rounding) ensures that
    bootstrap confidence intervals reflect both frequency and severity
    uncertainty.  For large expected counts the Poisson is tightly
    concentrated around the mean, so point estimates are barely affected.

    Returns a DataFrame with columns: accident_year, notidel, claim_type.
    """
    late_rbns = rbns[rbns["report_dy"] >= 2]
    type_list = list(type_props.index)
    type_probs = np.array([type_props[t] for t in type_list])

    rows = []
    for ay_idx in range(len(ibnr_counts)):
        ay = ay_idx + 1  # 1-based
        lam = max(float(ibnr_counts[ay_idx]), 0.0)
        n_ibnr = int(rng.poisson(lam)) if lam > 0 else 0
        if n_ibnr <= 0:
            continue

        # Sample claim types (independently of the delay unless --joint-type-delay)
        types = rng.choice(type_list, size=n_ibnr, p=type_probs)

        # Sample notidel from reported claims whose delay would have made them IBNR
        # in accident year `ay`: reported after the valuation date, i.e. report_dy
        # >= MAX_ACCIDENT_YEAR - ay + 2 (9 Sep 2026; previously the same-AY late
        # reporters were used, which for older accident years admits delays that
        # would already have been observed and for the newest one is empty).
        min_dy = MAX_ACCIDENT_YEAR - ay + 2
        eligible = rbns[rbns["report_dy"] >= min_dy]
        if len(eligible) >= 5:
            notidels = eligible["notidel"].values
        elif len(late_rbns) > 0:
            notidels = late_rbns["notidel"].values
        else:
            notidels = np.array([1.5])  # fallback

        if joint:
            # draw whole late-reporter records, so that type and delay keep their joint distribution
            pool = eligible if len(eligible) >= 5 else late_rbns
            if len(pool) > 0:
                idx = rng.randint(0, len(pool), size=n_ibnr)
                sampled_notidels = pool["notidel"].values[idx]
                types = pool["claim_type"].values[idx]
            else:
                sampled_notidels = rng.choice(notidels, size=n_ibnr, replace=True)
        else:
            sampled_notidels = rng.choice(notidels, size=n_ibnr, replace=True)

        for i in range(n_ibnr):
            rows.append(
                {
                    "accident_year": ay,
                    "notidel": sampled_notidels[i],
                    "claim_type": types[i],
                }
            )

    if len(rows) == 0:
        return pd.DataFrame(columns=["accident_year", "notidel", "claim_type"])

    return pd.DataFrame(rows)


def predict_ibnr_severity(
    model, synthetic_claims, feature_cols, unique_types, calibration_factors=None
):
    """
    Predict claim_size for synthetic IBNR claims using the trained RF model.

    Applies pipeline-level ratio calibration for log-retransformation
    bias correction: expm1(model.predict(X)) * factor[claim_type].

    Parameters
    ----------
    calibration_factors : dict or None
        Mapping of claim_type -> calibration factor, with GLOBAL_CALIB_KEY as fallback.

    Returns the synthetic_claims DataFrame with 'predicted_claim_size' column.
    """
    if len(synthetic_claims) == 0:
        synthetic_claims = synthetic_claims.copy()
        synthetic_claims["predicted_claim_size"] = []
        return synthetic_claims

    # Build feature matrix matching training format
    X = pd.DataFrame()
    X["accident_year"] = synthetic_claims["accident_year"].values
    X["notidel"] = synthetic_claims["notidel"].values
    X["log_notidel"] = np.log1p(synthetic_claims["notidel"].values)

    for ct in unique_types:
        X[f"type_{ct}"] = (synthetic_claims["claim_type"].values == ct).astype(float)

    # Ensure columns match
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols]

    # Standard expm1 retransformation — pipeline calibration (below) handles
    # the full Jensen's inequality correction.  Per-tree averaging is NOT used
    # because it partially corrects the same bias, causing double-correction.
    pred = np.expm1(model.predict(X.values))

    # Per-type calibration: correct residual-variance bias
    if calibration_factors is not None:
        global_factor = calibration_factors.get(GLOBAL_CALIB_KEY, 1.0)
        claim_type_arr = synthetic_claims["claim_type"].values
        multipliers = np.array(
            [calibration_factors.get(ct, global_factor) for ct in claim_type_arr]
        )
        pred = pred * multipliers

    pred = np.clip(pred, 0, None)

    synthetic_claims = synthetic_claims.copy()
    synthetic_claims["predicted_claim_size"] = pred

    return synthetic_claims


# ---------------------------------------------------------------------------
# 6. Evaluation and reporting
# ---------------------------------------------------------------------------


def evaluate_ibnr(
    ibnr_pred_by_ay,
    ibnr_true_by_ay,
    ibnr_counts_pred,
    ibnr_counts_true,
    observed_counts,
):
    """Build a per-AY comparison table for IBNR reserves."""
    rows = []
    for ay in range(1, MAX_ACCIDENT_YEAR + 1):
        pred_reserve = ibnr_pred_by_ay.get(ay, 0.0)
        true_reserve = ibnr_true_by_ay.get(ay, 0.0)
        error = pred_reserve - true_reserve
        error_pct = (error / true_reserve * 100) if true_reserve > 0 else 0.0
        pred_count = ibnr_counts_pred[ay - 1]
        true_count = ibnr_counts_true.get(ay, 0)
        obs_count = observed_counts[ay - 1]

        rows.append(
            {
                "accident_year": ay,
                "calendar_year": BASE_YEAR + ay - 1,
                "observed_claims": int(obs_count),
                "pred_ibnr_claims": pred_count,  # unrounded; tables round at display time
                "true_ibnr_claims": int(true_count),
                "pred_ibnr_reserve": pred_reserve,
                "true_ibnr_reserve": true_reserve,
                "ibnr_error": error,
                "ibnr_error_pct": error_pct,
            }
        )

    return pd.DataFrame(rows)


def combine_with_rbns(ibnr_results_df):
    """
    Load RBNS results from main pipeline and combine with IBNR predictions.
    """
    rbns_path = os.path.join(RESULTS_DIR, "all_models_reserves_by_ay.csv")
    if not os.path.exists(rbns_path):
        print("  WARNING: RBNS results not found, skipping combination")
        return None

    all_reserves = pd.read_csv(rbns_path)
    best_model = DEFAULT_BEST_MODEL
    rbns = all_reserves[all_reserves["model_name"] == best_model].copy()

    if len(rbns) == 0:
        print(f"  WARNING: {best_model} not found in RBNS results")
        return None

    # Load Chain Ladder RBNS results
    cl_path = os.path.join(RESULTS_DIR, "chain_ladder_results.csv")
    cl_df = pd.read_csv(cl_path) if os.path.exists(cl_path) else None

    combined_rows = []
    for ay in range(1, MAX_ACCIDENT_YEAR + 1):
        rbns_row = rbns[rbns["accident_year"] == ay]
        ibnr_row = ibnr_results_df[ibnr_results_df["accident_year"] == ay]

        rbns_pred = (
            rbns_row["predicted_reserve"].values[0] if len(rbns_row) > 0 else 0.0
        )
        rbns_true = rbns_row["true_reserve"].values[0] if len(rbns_row) > 0 else 0.0
        ibnr_pred = (
            ibnr_row["pred_ibnr_reserve"].values[0] if len(ibnr_row) > 0 else 0.0
        )
        ibnr_true = (
            ibnr_row["true_ibnr_reserve"].values[0] if len(ibnr_row) > 0 else 0.0
        )

        cl_rbns = 0.0
        if cl_df is not None:
            cl_row = cl_df[cl_df["accident_year"] == ay]
            cl_rbns = cl_row["CL_reserve"].values[0] if len(cl_row) > 0 else 0.0

        total_pred = rbns_pred + ibnr_pred
        total_true = rbns_true + ibnr_true
        total_error = total_pred - total_true
        total_error_pct = (total_error / total_true * 100) if total_true > 0 else 0.0

        # CL has RBNS component only — compare against total true
        cl_total_error = cl_rbns - total_true
        cl_error_pct = (cl_total_error / total_true * 100) if total_true > 0 else 0.0

        combined_rows.append(
            {
                "accident_year": ay,
                "calendar_year": BASE_YEAR + ay - 1,
                "rbns_pred": rbns_pred,
                "rbns_true": rbns_true,
                "ibnr_pred": ibnr_pred,
                "ibnr_true": ibnr_true,
                "total_pred": total_pred,
                "total_true": total_true,
                "total_error": total_error,
                "total_error_pct": total_error_pct,
                "cl_rbns_only": cl_rbns,
                "cl_vs_total_true_error_pct": cl_error_pct,
            }
        )

    return pd.DataFrame(combined_rows)


# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------


def plot_ibnr_results(ibnr_df, combined_df):
    """Create individual subplot images for IBNR and combined reserves."""
    # Figure width matched to \textwidth (A4, 2.5cm margins) for subfigure use
    half_w = 3.15  # half of 6.3in textwidth

    ays_with_ibnr = ibnr_df[
        (ibnr_df["true_ibnr_claims"] > 0) | (ibnr_df["pred_ibnr_claims"] > 0.5)
    ]

    paths = []

    # --- Panel 1: IBNR claim counts ---
    if len(ays_with_ibnr) > 0:
        fig, ax = plt.subplots(figsize=(half_w, 3))
        x = np.arange(len(ays_with_ibnr))
        width = 0.35
        ax.bar(x - width / 2, ays_with_ibnr["pred_ibnr_claims"], width,
               label="CL Predicted", color="#4C72B0", alpha=0.8)
        ax.bar(x + width / 2, ays_with_ibnr["true_ibnr_claims"], width,
               label="True", color="#2ca02c", alpha=0.8)
        labels = [f"AY {int(r['accident_year'])}"
                  for _, r in ays_with_ibnr.iterrows()]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Number of Claims", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(RESULTS_DIR, "ibnr_counts.pdf")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # --- Panel 2: IBNR reserves ---
    if len(ays_with_ibnr) > 0:
        fig, ax = plt.subplots(figsize=(half_w, 3))
        x = np.arange(len(ays_with_ibnr))
        width = 0.35
        ax.bar(x - width / 2, ays_with_ibnr["pred_ibnr_reserve"] / 1e6, width,
               label="RF Predicted", color="#4C72B0", alpha=0.8)
        ax.bar(x + width / 2, ays_with_ibnr["true_ibnr_reserve"] / 1e6, width,
               label="True", color="#2ca02c", alpha=0.8)
        labels = [f"AY {int(r['accident_year'])}"
                  for _, r in ays_with_ibnr.iterrows()]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Reserve (millions)", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(RESULTS_DIR, "ibnr_reserves_only.pdf")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # --- Panel 3: Combined reserves (RBNS + IBNR) ---
    if combined_df is not None:
        ays_with_reserve = combined_df[combined_df["total_true"] > 0]
        if len(ays_with_reserve) > 0:
            fig, ax = plt.subplots(figsize=(half_w, 3))
            x = np.arange(len(ays_with_reserve))
            width = 0.25
            ax.bar(x - width, ays_with_reserve["rbns_pred"] / 1e6, width,
                   label="RF RBNS", color="#4C72B0", alpha=0.8)
            ax.bar(x - width, ays_with_reserve["ibnr_pred"] / 1e6, width,
                   bottom=ays_with_reserve["rbns_pred"] / 1e6,
                   label="RF IBNR", color="#ff7f0e", alpha=0.8)
            ax.bar(x, ays_with_reserve["total_true"] / 1e6, width,
                   label="True Total", color="#2ca02c", alpha=0.8)
            ax.bar(x + width, ays_with_reserve["cl_rbns_only"] / 1e6, width,
                   label="CL (RBNS only)", color="#d62728", alpha=0.8)
            labels = [f"AY {int(r['accident_year'])}"
                      for _, r in ays_with_reserve.iterrows()]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
            ax.set_ylabel("Reserve (millions)", fontsize=10)
            ax.tick_params(axis="y", labelsize=9)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            p = os.path.join(RESULTS_DIR, "ibnr_combined.pdf")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)

    # --- Panel 4: Reserve error comparison ---
    if combined_df is not None:
        ays_with_reserve = combined_df[combined_df["total_true"] > 0]
        if len(ays_with_reserve) > 0:
            fig, ax = plt.subplots(figsize=(half_w, 3))
            x = np.arange(len(ays_with_reserve))
            width = 0.35
            ax.bar(x - width / 2, ays_with_reserve["total_error_pct"], width,
                   label="RF (RBNS+IBNR)", color="#4C72B0", alpha=0.8)
            ax.bar(x + width / 2, ays_with_reserve["cl_vs_total_true_error_pct"], width,
                   label="CL (RBNS only)", color="#d62728", alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.8)
            labels = [f"AY {int(r['accident_year'])}"
                      for _, r in ays_with_reserve.iterrows()]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
            ax.set_ylabel("Reserve Error (%)", fontsize=10)
            ax.tick_params(axis="y", labelsize=9)
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            p = os.path.join(RESULTS_DIR, "ibnr_error.pdf")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)

    for p in paths:
        print(f"  Saved: {p}")

    return paths[0] if paths else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def replot_from_full_data():
    ibnr_full = os.path.join(RESULTS_DIR, "full_data_ibnr_reserves_by_ay.csv")
    combined_full = os.path.join(
        RESULTS_DIR, "full_data_combined_rbns_ibnr_reserves.csv"
    )
    if not os.path.exists(ibnr_full):
        raise FileNotFoundError(
            f"{ibnr_full} not found. Run `python ibnr_model.py` (without --quick) "
            f"once to produce the full_data_* cache."
        )
    ibnr_df = pd.read_csv(ibnr_full)
    combined_df = pd.read_csv(combined_full) if os.path.exists(combined_full) else None
    print(f"  Loaded: {ibnr_full}")
    if combined_df is not None:
        print(f"  Loaded: {combined_full}")
    if not OUT_SUFFIX:
        plot_ibnr_results(ibnr_df, combined_df)


def main():
    parser = argparse.ArgumentParser(
        description="IBNR reserve model: CL frequency + RF severity"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="Optuna trials for severity model (default: 25)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: 5 trials, capped n_estimators",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=200,
        help="Bootstrap draws for IBNR severity estimation (default: 200)",
    )
    parser.add_argument(
        "--cl-window",
        type=int,
        default=5,
        help="Window size for CL frequency factors (most recent N AYs). "
        "0 = all AYs (original CL). Default: 5.",
    )
    parser.add_argument(
        "--severity-train",
        choices=["settled", "mature", "all"],
        default="mature",
        help="Severity training sample: mature = reported claims of accident years with >= --min-dev development years observed, paid-to-date target (default; the paper's run); settled = claims settled by the valuation date, paid-to-date target; all = every RBNS claim with the eventual claim_size (not prospective).",
    )
    parser.add_argument("--joint-type-delay", action="store_true",
                        help="Draw claim type and notification delay of a synthetic IBNR claim from the same late-reporter record (joint distribution) instead of independently.")
    parser.add_argument("--out-suffix", default="", help="Suffix for the output CSV names (sensitivity runs); plots are skipped when set.")
    parser.add_argument("--min-dev", type=int, default=8, help="For --severity-train mature: minimum development years observed (default 8).")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip all computation; regenerate PNGs from full_data_* CSVs "
        "saved by a previous full (non-quick) run.",
    )
    args = parser.parse_args()
    global OUT_SUFFIX, JOINT_TYPE_DELAY
    OUT_SUFFIX = args.out_suffix
    JOINT_TYPE_DELAY = args.joint_type_delay

    if args.plot_only:
        replot_from_full_data()
        return

    if args.quick:
        args.n_trials = 5

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    total_start = time.time()

    print("=" * 65)
    print("IBNR RESERVE MODEL: CL Frequency + RF Severity")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading claims data...")
    claims_all, rbns, ibnr = load_claims()

    # ------------------------------------------------------------------
    # Step 2: Chain Ladder frequency model (windowed)
    # ------------------------------------------------------------------
    cl_window = args.cl_window if args.cl_window > 0 else None
    window_label = f"last {cl_window} AYs" if cl_window else "all AYs"
    print(
        f"\n[2/6] Building 10x10 claim count triangle and applying CL ({window_label})..."
    )
    full_inc, full_cum, obs_cum = build_count_triangle(claims_all)
    pred_counts, ibnr_counts, dev_factors, true_counts, observed_counts = cl_frequency(
        obs_cum, full_cum, window=cl_window
    )

    print(f"\n  Count triangle development factors (non-trivial only):")
    for j in range(len(dev_factors)):
        if abs(dev_factors[j] - 1.0) > 1e-6:
            print(f"    DY{j + 1} -> DY{j + 2}: {dev_factors[j]:.6f}")

    print(f"\n  IBNR claim count predictions:")
    true_ibnr_counts = {}
    for ay in range(1, MAX_ACCIDENT_YEAR + 1):
        true_n = len(ibnr[ibnr["accident_year"] == ay])
        true_ibnr_counts[ay] = true_n
        if ibnr_counts[ay - 1] > 0.5 or true_n > 0:
            print(
                f"    AY {ay:2d} ({BASE_YEAR + ay - 1}): "
                f"predicted={ibnr_counts[ay - 1]:6.1f}, true={true_n}"
            )

    total_pred_n = sum(ibnr_counts)
    total_true_n = len(ibnr)
    print(f"    TOTAL: predicted={total_pred_n:.1f}, true={total_true_n}")

    # ------------------------------------------------------------------
    # Step 3: Estimate late-reporter claim-type mix
    # ------------------------------------------------------------------
    print("\n[3/6] Estimating late-reporter claim-type mix...")
    type_props = estimate_late_reporter_mix(rbns)

    # ------------------------------------------------------------------
    # Step 4: Train RF severity model on RBNS claims
    # ------------------------------------------------------------------
    print(f"\n[4/6] Training RF severity model ({args.n_trials} Optuna trials)...")
    t0 = time.time()

    # Prospective severity sample (9 Sep 2026): only claims whose development is
    # complete at the valuation date -- settled by calendar year MAX_ACCIDENT_YEAR
    # (accident_year + setldel <= MAX_ACCIDENT_YEAR; on this data every such claim
    # has cum_paid_observed == cum_paid_full) -- with the paid-to-date amount as the
    # target.  The earlier target, claim_size, is the eventual claim size and
    # includes payments after the valuation date; --severity-train all restores it
    # for comparison only.
    if args.severity_train == "settled":
        settled = (rbns["accident_year"] + rbns["setldel"]) <= MAX_ACCIDENT_YEAR
        sev_train = rbns[settled].copy()
        y_train = np.log1p(sev_train["cum_paid_observed"].values)
        print(f"  Severity training sample: {len(sev_train):,} claims settled by the valuation date "
              f"(of {len(rbns):,} RBNS); target = paid to date")
    elif args.severity_train == "mature":
        # accident years with at least --min-dev development years observed at valuation
        mature = rbns["accident_year"] <= MAX_ACCIDENT_YEAR - args.min_dev + 1
        sev_train = rbns[mature].copy()
        y_train = np.log1p(sev_train["cum_paid_observed"].values)
        print(f"  Severity training sample: {len(sev_train):,} claims of accident years with >= {args.min_dev} "
              f"development years observed (AY <= {MAX_ACCIDENT_YEAR - args.min_dev + 1}); target = paid to date")
    else:
        sev_train = rbns.copy()
        y_train = np.log1p(sev_train["claim_size"].values)  # eventual size (not prospective)
        print(f"  Severity training sample: all {len(sev_train):,} RBNS claims; target = eventual claim_size")
    feat_df, feature_cols, unique_types = prepare_severity_features(sev_train)
    X_train = feat_df[feature_cols].values

    print(f"  Training samples: {len(X_train):,}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    best_params = tune_severity_model(X_train, y_train, args.n_trials, quick=args.quick)
    severity_model = train_severity_model(X_train, y_train, best_params)

    print(f"  Severity model trained in {time.time() - t0:.1f}s")
    print(
        f"  n_estimators={severity_model.n_estimators}, "
        f"n_trees={len(severity_model.estimators_)}"
    )

    # Compute per-type calibration factors via 3-fold CV (GroupKFold on AY)
    claim_types_train = sev_train["claim_type"].values
    accident_years_train = sev_train["accident_year"].values
    calibration_factors = compute_calibration_factors(
        X_train,
        y_train,
        best_params,
        claim_types=claim_types_train,
        accident_years=accident_years_train,
    )
    print(f"  Calibration factors (3-fold CV):")
    for ct, factor in sorted(calibration_factors.items()):
        print(f"    {ct:15s}: {factor:.4f}")

    # Feature importance
    importances = severity_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n  Feature importances:")
    for i in sorted_idx:
        print(f"    {feature_cols[i]:20s}: {importances[i]:.4f}")

    # ------------------------------------------------------------------
    # Step 5: Generate synthetic IBNR claims and predict severity
    # ------------------------------------------------------------------
    print(f"\n[5/6] Predicting IBNR severity ({args.n_draws} bootstrap draws)...")

    rng = np.random.RandomState(RANDOM_STATE)

    all_ibnr_by_ay = {ay: [] for ay in range(1, MAX_ACCIDENT_YEAR + 1)}

    for draw in range(args.n_draws):
        draw_rng = np.random.RandomState(RANDOM_STATE + draw)
        synthetic = generate_ibnr_claims(ibnr_counts, type_props, rbns, draw_rng, joint=JOINT_TYPE_DELAY)

        if len(synthetic) == 0:
            continue

        synthetic = predict_ibnr_severity(
            severity_model,
            synthetic,
            feature_cols,
            unique_types,
            calibration_factors=calibration_factors,
        )

        for ay in range(1, MAX_ACCIDENT_YEAR + 1):
            ay_claims = synthetic[synthetic["accident_year"] == ay]
            total = (
                ay_claims["predicted_claim_size"].sum() if len(ay_claims) > 0 else 0.0
            )
            all_ibnr_by_ay[ay].append(total)

    # Point estimate = mean across draws
    ibnr_pred_by_ay = {}
    for ay in range(1, MAX_ACCIDENT_YEAR + 1):
        if len(all_ibnr_by_ay[ay]) > 0:
            ibnr_pred_by_ay[ay] = np.mean(all_ibnr_by_ay[ay])
        else:
            ibnr_pred_by_ay[ay] = 0.0

    # True IBNR reserves
    ibnr_true_by_ay = {}
    for ay in range(1, MAX_ACCIDENT_YEAR + 1):
        ay_ibnr = ibnr[ibnr["accident_year"] == ay]
        ibnr_true_by_ay[ay] = ay_ibnr["claim_size"].sum()

    print(f"\n  IBNR reserve predictions (mean of {args.n_draws} draws):")
    for ay in range(1, MAX_ACCIDENT_YEAR + 1):
        pred = ibnr_pred_by_ay[ay]
        true = ibnr_true_by_ay[ay]
        if pred > 0 or true > 0:
            draws = all_ibnr_by_ay[ay]
            p5 = np.percentile(draws, 5) if draws else 0
            p95 = np.percentile(draws, 95) if draws else 0
            err_pct = ((pred - true) / true * 100) if true > 0 else 0
            print(
                f"    AY {ay:2d}: pred={pred:>12,.0f}  true={true:>12,.0f}  "
                f"err={err_pct:+.1f}%  [P5={p5:,.0f}, P95={p95:,.0f}]"
            )

    total_ibnr_pred = sum(ibnr_pred_by_ay.values())
    total_ibnr_true = sum(ibnr_true_by_ay.values())
    total_err = total_ibnr_pred - total_ibnr_true
    total_err_pct = (total_err / total_ibnr_true * 100) if total_ibnr_true > 0 else 0
    print(
        f"\n  IBNR Total: pred={total_ibnr_pred:,.0f}  "
        f"true={total_ibnr_true:,.0f}  err={total_err_pct:+.1f}%"
    )

    # ------------------------------------------------------------------
    # Step 6: Evaluate and combine with RBNS
    # ------------------------------------------------------------------
    print("\n[6/6] Evaluating and combining with RBNS results...")

    ibnr_df = evaluate_ibnr(
        ibnr_pred_by_ay,
        ibnr_true_by_ay,
        ibnr_counts,
        true_ibnr_counts,
        observed_counts,
    )

    # Save IBNR results
    ibnr_csv = _out("ibnr_reserves_by_ay.csv")
    ibnr_df.to_csv(ibnr_csv, index=False)
    print(f"  Saved: {ibnr_csv}")
    if not args.quick:
        ibnr_full = _out("full_data_ibnr_reserves_by_ay.csv")
        ibnr_df.to_csv(ibnr_full, index=False)
        print(f"  Saved: {ibnr_full}")

    # Combine with RBNS
    combined_df = combine_with_rbns(ibnr_df)
    if combined_df is not None:
        combined_csv = _out("combined_rbns_ibnr_reserves.csv")
        combined_df.to_csv(combined_csv, index=False)
        print(f"  Saved: {combined_csv}")
        if not args.quick:
            combined_full = _out("full_data_combined_rbns_ibnr_reserves.csv")
            combined_df.to_csv(combined_full, index=False)
            print(f"  Saved: {combined_full}")

        # Print summary
        print("\n" + "=" * 65)
        print("COMBINED RESERVE SUMMARY (RBNS + IBNR)")
        print("=" * 65)

        total_rbns_pred = combined_df["rbns_pred"].sum()
        total_rbns_true = combined_df["rbns_true"].sum()
        total_ibnr_pred_c = combined_df["ibnr_pred"].sum()
        total_ibnr_true_c = combined_df["ibnr_true"].sum()
        total_pred = combined_df["total_pred"].sum()
        total_true = combined_df["total_true"].sum()
        total_cl = combined_df["cl_rbns_only"].sum()

        rbns_err = (total_rbns_pred - total_rbns_true) / total_rbns_true * 100
        print(
            f"\n  RF RBNS reserve:   pred={total_rbns_pred:>14,.0f}  "
            f"true={total_rbns_true:>14,.0f}  err={rbns_err:+.1f}%"
        )

        if total_ibnr_true_c > 0:
            ibnr_err = (total_ibnr_pred_c - total_ibnr_true_c) / total_ibnr_true_c * 100
            print(
                f"  RF IBNR reserve:   pred={total_ibnr_pred_c:>14,.0f}  "
                f"true={total_ibnr_true_c:>14,.0f}  err={ibnr_err:+.1f}%"
            )

        total_err_c = (total_pred - total_true) / total_true * 100
        print(
            f"  RF TOTAL reserve:  pred={total_pred:>14,.0f}  "
            f"true={total_true:>14,.0f}  err={total_err_c:+.1f}%"
        )

        cl_err = (total_cl - total_true) / total_true * 100
        print(
            f"  CL (RBNS only):    pred={total_cl:>14,.0f}  "
            f"true={total_true:>14,.0f}  err={cl_err:+.1f}%"
        )

    # Save bootstrap distribution stats
    dist_rows = []
    for ay in range(1, MAX_ACCIDENT_YEAR + 1):
        draws = all_ibnr_by_ay[ay]
        if draws:
            dist_rows.append(
                {
                    "accident_year": ay,
                    "calendar_year": BASE_YEAR + ay - 1,
                    "ibnr_mean": np.mean(draws),
                    "ibnr_median": np.median(draws),
                    "ibnr_std": np.std(draws),
                    "ibnr_p5": np.percentile(draws, 5),
                    "ibnr_p25": np.percentile(draws, 25),
                    "ibnr_p75": np.percentile(draws, 75),
                    "ibnr_p95": np.percentile(draws, 95),
                    "ibnr_true": ibnr_true_by_ay.get(ay, 0),
                }
            )
    if dist_rows:
        dist_df = pd.DataFrame(dist_rows)
        dist_csv = _out("ibnr_severity_distribution.csv")
        dist_df.to_csv(dist_csv, index=False)
        print(f"  Saved: {dist_csv}")

    # Plot (sensitivity runs with --out-suffix must not overwrite the main figures)
    if not OUT_SUFFIX:
        plot_ibnr_results(ibnr_df, combined_df)

    total_time = time.time() - total_start
    print(f"\n{'=' * 65}")
    print(f"DONE — Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
