"""
Comprehensive feature engineering for the claim × DY dataset.

All features are computed using ONLY observed (historical) information —
no leakage from unobservable rows or ground-truth columns.

-----------------------------------------------------------------------
COLUMN SAFETY REFERENCE — which columns are safe to use as features?
-----------------------------------------------------------------------

SAFE (observable at prediction time):
  accident_year, occurrence_period, occurrence_month, notidel,
  dev_lag, incremental_payment (observed DYs only),
  cum_paid_prior, cum_paid_through (both built from observed txns),
  claimant_age, injury_severity, vehicle_class, legal_rep,
  claim_count, claim_type, cc, AQ, age, inj_part
  ... and any feature engineered from the above.

FORBIDDEN — these contain future / ground-truth information and must
NEVER appear in feature_cols:
  claim_size       — total of ALL payments including future unobserved ones
  setldel          — settlement delay derived from last payment date (future)
  cum_paid_full    — cumulative paid including future payments
  cum_paid_full_prior — same, shifted by one DY
  cum_paid_observed — safe in principle, but easily confused with cum_paid_full;
                      use cum_paid_prior / cum_paid_through instead
  reserve_true     — the ground-truth reserve we are trying to predict
  true_incremental_payment — ground-truth incremental (includes unobserved txns)
-----------------------------------------------------------------------
"""

import pandas as pd
import os
import numpy as np
import src.config as cfg

# Columns that must NEVER be used as model features (future leakage / ground truth).
_FORBIDDEN_FEATURES = frozenset(
    {
        "claim_size",
        "setldel",
        "cum_paid_full",
        "cum_paid_full_prior",
        "cum_paid_observed",
        "reserve_true",
        "true_incremental_payment",
    }
)


def engineer_features(claim_dy, transactions_obs):
    """
    Add engineered features to the claim × DY dataframe.

    Parameters
    ----------
    claim_dy : pd.DataFrame
        The claim × development-year dataset from data_prep.
    transactions_obs : pd.DataFrame
        Observed transactions (observed_flag == 1, RBNS only).

    Returns
    -------
    claim_dy : pd.DataFrame
        Same dataframe with additional feature columns.
    feature_cols : list[str]
        Names of all feature columns (for model training).
    """
    df = claim_dy.copy()

    # -------------------------------------------------------------------
    # Pre-compute transaction-level aggregates per claim × DY
    # (some of these are already in df from data_prep, but we need
    #  additional detail from the raw transactions)
    # -------------------------------------------------------------------
    txn_claim_dy = (
        transactions_obs.groupby(["claim_no", "dev_lag"])
        .agg(
            pmt_count=("payment_size", "count"),
            pmt_sum=("payment_size", "sum"),
            pmt_mean=("payment_size", "mean"),
            pmt_std=("payment_size", "std"),
            pmt_max=("payment_size", "max"),
            pmt_min=("payment_size", "min"),
            first_pay_period=("payment_period", "min"),
            last_pay_period=("payment_period", "max"),
        )
        .reset_index()
    )
    txn_claim_dy["pmt_std"] = txn_claim_dy["pmt_std"].fillna(0.0)

    # -------------------------------------------------------------------
    # 1. CLAIM-LEVEL STATIC FEATURES
    # -------------------------------------------------------------------
    # These are already in df: accident_year, occurrence_period, notidel,
    #                          occurrence_month
    df["log_notidel"] = np.log1p(df["notidel"])

    # -------------------------------------------------------------------
    # 1b. CLAIM-LEVEL FEATURES (from CL-hostile data, if present)
    # -------------------------------------------------------------------
    has_claim_features = "claimant_age" in df.columns
    claim_feature_cols = []

    if has_claim_features:
        # One-hot encode injury_severity (baseline = minor)
        for sev in ["moderate", "serious", "severe"]:
            col = f"injury_{sev}"
            df[col] = (df["injury_severity"] == sev).astype(int)
            claim_feature_cols.append(col)

        # One-hot encode vehicle_class (baseline = passenger_car)
        for vc in ["SUV_truck", "motorcycle"]:
            col = f"vehicle_{vc}"
            df[col] = (df["vehicle_class"] == vc).astype(int)
            claim_feature_cols.append(col)

        # Numeric features: use directly
        claim_feature_cols.extend(["claimant_age", "legal_rep", "claim_count"])

        # Interaction features with claim-level covariates
        df["age_x_severe"] = df["claimant_age"] * (
            df["injury_severity"] == "severe"
        ).astype(int)
        df["motorcycle_x_severe"] = df["vehicle_motorcycle"] * df["injury_severe"]

        claim_feature_cols.extend(
            [
                "age_x_severe",
                "motorcycle_x_severe",
            ]
        )
        # Note: legal_x_cum_paid deferred until after cum_paid_prior is computed

    # -------------------------------------------------------------------
    # 1c. MTPL DATA: claim_type feature (if present)
    # -------------------------------------------------------------------
    has_claim_type = "claim_type" in df.columns
    claim_type_cols = []

    if has_claim_type:
        # One-hot encode claim_type (baseline = Material, the most common)
        for ct in ["Bodily", "Technical", "Annuity"]:
            col = f"type_{ct}"
            df[col] = (df["claim_type"] == ct).astype(int)
            claim_type_cols.append(col)

    # -------------------------------------------------------------------
    # 1d. GABRIELLI DATA: cc, AQ, age, inj_part features (if present)
    # -------------------------------------------------------------------
    has_gabrielli_features = "cc" in df.columns and "inj_part" in df.columns
    gabrielli_feature_cols = []

    if has_gabrielli_features:
        # Age: use directly as numeric (already integer)
        df["gabrielli_age"] = df["age"].astype(float)
        gabrielli_feature_cols.append("gabrielli_age")

        # AQ (accident quarter): one-hot (baseline = Q1)
        for q in [2, 3, 4]:
            col = f"quarter_{q}"
            df[col] = (df["AQ"] == q).astype(int)
            gabrielli_feature_cols.append(col)

        # cc (claim code): too many levels (51) for one-hot — use as numeric
        # The code is already integer in the data
        df["claim_code"] = df["cc"].astype(float)
        gabrielli_feature_cols.append("claim_code")

        # inj_part (injured body part): too many levels (46) — use as numeric
        df["injured_part"] = df["inj_part"].astype(float)
        gabrielli_feature_cols.append("injured_part")

        print(f"  Gabrielli features: {gabrielli_feature_cols}")

    # -------------------------------------------------------------------
    # 2. CURRENT DY CONTEXT
    # -------------------------------------------------------------------
    df["is_first_dy"] = (df["dev_lag"] == 1).astype(int)
    late_dy_threshold = max(2, int(np.ceil(0.7 * cfg.MAX_DEV_LAG)))
    df["is_late_dy"] = (df["dev_lag"] >= late_dy_threshold).astype(int)
    df["dev_lag_sq"] = df["dev_lag"] ** 2

    # -------------------------------------------------------------------
    # 3. CUMULATIVE PAYMENT HISTORY (through prior DYs)
    #    cum_paid_prior is already computed in data_prep
    # -------------------------------------------------------------------
    df["log_cum_paid_prior"] = np.log1p(df["cum_paid_prior"])
    df["has_prior_payments"] = (df["cum_paid_prior"] > 0).astype(int)

    # Deferred claim-feature interaction (needs cum_paid_prior)
    if has_claim_features:
        df["legal_x_cum_paid"] = df["legal_rep"] * df["cum_paid_prior"]
        claim_feature_cols.append("legal_x_cum_paid")

    # Number of observed payments through prior DYs
    # We need to build this from the transaction-level aggregates
    _prior_pmt_counts = _compute_cumulative_prior(
        df, txn_claim_dy, value_col="pmt_count", new_col="num_payments_prior"
    )
    df = df.merge(_prior_pmt_counts, on=["claim_no", "dev_lag"], how="left")
    df["num_payments_prior"] = df["num_payments_prior"].fillna(0)

    # Number of active DYs prior (DYs with at least one payment)
    _active_dys = _compute_active_dys_prior(df, txn_claim_dy)
    df = df.merge(_active_dys, on=["claim_no", "dev_lag"], how="left")
    df["num_active_dys_prior"] = df["num_active_dys_prior"].fillna(0)

    # -------------------------------------------------------------------
    # 4. PAYMENT PATTERNS
    # -------------------------------------------------------------------
    # Average payment in prior DYs
    df["avg_payment_prior"] = np.where(
        df["num_payments_prior"] > 0,
        df["cum_paid_prior"] / df["num_payments_prior"],
        0.0,
    )

    # Max single payment in prior DYs
    _prior_max = _compute_cumulative_prior(
        df,
        txn_claim_dy,
        value_col="pmt_max",
        new_col="max_payment_prior",
        agg_func="max",
    )
    df = df.merge(_prior_max, on=["claim_no", "dev_lag"], how="left")
    df["max_payment_prior"] = df["max_payment_prior"].fillna(0)

    # Payment in previous DY (DY k-1) and DY k-2
    _prev_payments = _compute_lagged_payments(df)
    df = df.merge(_prev_payments, on=["claim_no", "dev_lag"], how="left")
    df["paid_in_prev_dy"] = df["paid_in_prev_dy"].fillna(0)
    df["paid_in_prev2_dy"] = df["paid_in_prev2_dy"].fillna(0)

    # Payment velocity: payments per active DY
    df["payment_velocity"] = np.where(
        df["num_active_dys_prior"] > 0,
        df["num_payments_prior"] / df["num_active_dys_prior"],
        0.0,
    )

    # Cumulative paid per active DY
    df["cum_paid_per_active_dy"] = np.where(
        df["num_active_dys_prior"] > 0,
        df["cum_paid_prior"] / df["num_active_dys_prior"],
        0.0,
    )

    # -------------------------------------------------------------------
    # 5. PAYMENT TREND (slope of incremental payments across prior DYs)
    # -------------------------------------------------------------------
    _trends = _compute_payment_trend(df)
    df = df.merge(_trends, on=["claim_no", "dev_lag"], how="left")
    df["payment_trend"] = df["payment_trend"].fillna(0)

    # -------------------------------------------------------------------
    # 6. EWMA (exponentially weighted moving average of DY payments)
    # -------------------------------------------------------------------
    _ewma = _compute_ewma(df)
    df = df.merge(_ewma, on=["claim_no", "dev_lag"], how="left")
    df["ewma_payment"] = df["ewma_payment"].fillna(0)

    # -------------------------------------------------------------------
    # 7. PAYMENT DECAY RATE
    # -------------------------------------------------------------------
    _decay = _compute_decay_rate(df)
    df = df.merge(_decay, on=["claim_no", "dev_lag"], how="left")
    df["payment_decay_rate"] = df["payment_decay_rate"].fillna(0)

    # -------------------------------------------------------------------
    # 8. PAYMENT ACCELERATION (change in velocity)
    # -------------------------------------------------------------------
    df["payment_acceleration"] = df["paid_in_prev_dy"] - df["paid_in_prev2_dy"]

    # -------------------------------------------------------------------
    # 9. INTERACTION FEATURES
    # -------------------------------------------------------------------
    df["cum_paid_x_notidel"] = df["cum_paid_prior"] * df["notidel"]
    df["dev_lag_x_cum_paid"] = df["dev_lag"] * df["cum_paid_prior"]
    df["notidel_x_dev_lag"] = df["notidel"] * df["dev_lag"]
    df["log_cum_paid_x_dev_lag"] = df["log_cum_paid_prior"] * df["dev_lag"]
    df["velocity_x_dev_lag"] = df["payment_velocity"] * df["dev_lag"]

    # -------------------------------------------------------------------
    # 10. FRACTION OF EXPECTED ULTIMATE PAID
    #     We compute average ultimate per AY from FULLY DEVELOPED AYs (1-11)
    #     to avoid leakage. This is a train-time aggregate feature.
    # -------------------------------------------------------------------
    _frac = _compute_fraction_paid(df)
    df = df.merge(_frac, on=["claim_no", "dev_lag"], how="left")
    df["fraction_paid_vs_avg_ultimate"] = df["fraction_paid_vs_avg_ultimate"].fillna(0)

    # -------------------------------------------------------------------
    # 11. DY-LEVEL STATISTICS FROM PRIOR DYs (std of DY payments)
    # -------------------------------------------------------------------
    _dy_std = _compute_dy_payment_std(df)
    df = df.merge(_dy_std, on=["claim_no", "dev_lag"], how="left")
    df["std_dy_payments_prior"] = df["std_dy_payments_prior"].fillna(0)

    # -------------------------------------------------------------------
    # Collect feature column names
    # -------------------------------------------------------------------
    feature_cols = [
        # Static claim features
        "accident_year",
        "occurrence_period",
        "occurrence_month",
        "notidel",
        "log_notidel",
        # DY context
        "dev_lag",
        "is_first_dy",
        "is_late_dy",
        "dev_lag_sq",
        # Cumulative history
        "cum_paid_prior",
        "log_cum_paid_prior",
        "has_prior_payments",
        "num_payments_prior",
        "num_active_dys_prior",
        # Payment patterns
        "avg_payment_prior",
        "max_payment_prior",
        "paid_in_prev_dy",
        "paid_in_prev2_dy",
        "payment_velocity",
        "cum_paid_per_active_dy",
        # Trend & dynamics
        "payment_trend",
        "ewma_payment",
        "payment_decay_rate",
        "payment_acceleration",
        # Interactions
        "cum_paid_x_notidel",
        "dev_lag_x_cum_paid",
        "notidel_x_dev_lag",
        "log_cum_paid_x_dev_lag",
        "velocity_x_dev_lag",
        # Context
        "fraction_paid_vs_avg_ultimate",
        "std_dy_payments_prior",
    ]

    # Add claim-level features if present in this dataset
    if has_claim_features:
        feature_cols.extend(claim_feature_cols)

    # Add MTPL claim_type features if present
    if has_claim_type:
        feature_cols.extend(claim_type_cols)

    # Add Gabrielli features if present
    if has_gabrielli_features:
        feature_cols.extend(gabrielli_feature_cols)

    # --- Guard rail: ensure no forbidden columns leaked into features ---
    leaked = _FORBIDDEN_FEATURES.intersection(feature_cols)
    assert not leaked, (
        f"DATA LEAKAGE: forbidden columns found in feature_cols: {leaked}. "
        "These contain future information and must not be used as features."
    )

    # For per_dy models, we drop dev_lag-related features since they're constant
    feature_cols_per_dy = [
        c
        for c in feature_cols
        if c not in ("dev_lag", "is_first_dy", "is_late_dy", "dev_lag_sq")
    ]

    # Reduced-covariate check (Paper B, Assumption A3): drop the two accident-year
    # coordinates when RESERVING_EXCLUDE_AY_FEATURES=1, so that the fitted
    # predictor is a function of the reduced feature vector by construction.
    if os.environ.get("RESERVING_EXCLUDE_AY_FEATURES") == "1":
        # occurrence_month is derived from occurrence_period and equals the accident year in annual data
        # (found in review, 9 Sep 2026): it must go too, or the reduced vector still encodes the accident year.
        ay_cols = ("accident_year", "occurrence_period", "occurrence_month")
        feature_cols = [c for c in feature_cols if c not in ay_cols]
        feature_cols_per_dy = [c for c in feature_cols_per_dy if c not in ay_cols]
        print("RESERVING_EXCLUDE_AY_FEATURES=1: accident_year, occurrence_period and occurrence_month removed from the features")

    print(
        f"Engineered {len(feature_cols)} features (single_model), "
        f"{len(feature_cols_per_dy)} features (per_dy)"
    )

    return df, feature_cols, feature_cols_per_dy


# -------------------------------------------------------------------
# Direct multi-step (horizon-frozen) features
# -------------------------------------------------------------------
# History features for target DY k with the history frozen h years before
# the target are exactly the history features of row (claim, k - h + 1):
# every helper above aggregates DYs 1..(dev_lag - 1), so shifting the source
# row shifts the freeze point.  Only the interactions that mix the target
# dev_lag with history need recomputation; the pure DY-context columns
# (dev_lag, dev_lag_sq, ...) are functions of the target k and stay.
# h = 1 is the identity (the standard construction).

HISTORY_FEATURES = [
    "cum_paid_prior", "log_cum_paid_prior", "has_prior_payments",
    "num_payments_prior", "num_active_dys_prior", "avg_payment_prior",
    "max_payment_prior", "paid_in_prev_dy", "paid_in_prev2_dy",
    "payment_velocity", "cum_paid_per_active_dy", "payment_trend",
    "ewma_payment", "payment_decay_rate", "payment_acceleration",
    "cum_paid_x_notidel", "fraction_paid_vs_avg_ultimate",
    "std_dy_payments_prior", "legal_x_cum_paid",  # last one: hostile data only
]


def horizon_shifted_features(df, h):
    """
    Return a copy of `df` whose history features are frozen h years before the
    target development year (Remark "Two remedies", variant (ii)): the history
    block of row (claim, k) is replaced by the one of row (claim, k - h + 1),
    and the dev_lag x history interactions are recomputed at the target k.
    Rows with k - h + 1 < 1 get zero history (they are never used: a cell at
    horizon h always has k >= h + 1).
    """
    if h == 1:
        return df
    hist = [c for c in HISTORY_FEATURES if c in df.columns]
    src = df[["claim_no", "dev_lag"] + hist].copy()
    src["dev_lag"] = src["dev_lag"] + (h - 1)  # source s -> target k = s + h - 1
    out = df.drop(columns=hist).merge(src, on=["claim_no", "dev_lag"], how="left")
    for c in hist:
        out[c] = out[c].fillna(0.0)
    # recompute the target-DY interactions from the shifted history
    out["dev_lag_x_cum_paid"] = out["dev_lag"] * out["cum_paid_prior"]
    out["log_cum_paid_x_dev_lag"] = out["log_cum_paid_prior"] * out["dev_lag"]
    out["velocity_x_dev_lag"] = out["payment_velocity"] * out["dev_lag"]
    # (notidel_x_dev_lag has no history component and is already correct)
    return out


# ===================================================================
# Helper functions for feature computation
# ===================================================================


def _compute_cumulative_prior(df, txn_claim_dy, value_col, new_col, agg_func="sum"):
    """
    For each (claim_no, dev_lag), compute the cumulative aggregate of
    `value_col` from DYs 1..(dev_lag - 1).
    """
    # Get the value per claim × DY
    vals = txn_claim_dy[["claim_no", "dev_lag", value_col]].copy()

    # For each target DY, sum/max the values from prior DYs
    claims = df[["claim_no", "dev_lag"]].drop_duplicates()
    result = []

    # Pivot for efficiency
    pivot = vals.pivot_table(
        index="claim_no",
        columns="dev_lag",
        values=value_col,
        aggfunc="first",
        fill_value=0,
    )

    for dy in range(1, cfg.MAX_DEV_LAG + 1):
        prior_cols = [c for c in pivot.columns if c < dy]
        if len(prior_cols) == 0:
            # DY=1 has no prior
            tmp = pd.DataFrame(
                {
                    "claim_no": pivot.index,
                    "dev_lag": dy,
                    new_col: 0.0,
                }
            )
        else:
            if agg_func == "sum":
                vals_agg = pivot[prior_cols].sum(axis=1)
            elif agg_func == "max":
                vals_agg = pivot[prior_cols].max(axis=1)
            else:
                raise ValueError(f"Unknown agg_func: {agg_func}")
            tmp = pd.DataFrame(
                {
                    "claim_no": pivot.index,
                    "dev_lag": dy,
                    new_col: vals_agg.values,
                }
            )
        result.append(tmp)

    return pd.concat(result, ignore_index=True)


def _compute_active_dys_prior(df, txn_claim_dy):
    """Count of DYs prior to current that had at least one payment."""
    # Binary indicator: did this claim have a payment in this DY?
    has_payment = txn_claim_dy[txn_claim_dy["pmt_count"] > 0][
        ["claim_no", "dev_lag"]
    ].copy()
    has_payment["has_pmt"] = 1

    pivot = has_payment.pivot_table(
        index="claim_no",
        columns="dev_lag",
        values="has_pmt",
        aggfunc="first",
        fill_value=0,
    )

    result = []
    for dy in range(1, cfg.MAX_DEV_LAG + 1):
        prior_cols = [c for c in pivot.columns if c < dy]
        if len(prior_cols) == 0:
            tmp = pd.DataFrame(
                {
                    "claim_no": pivot.index,
                    "dev_lag": dy,
                    "num_active_dys_prior": 0,
                }
            )
        else:
            count = pivot[prior_cols].sum(axis=1)
            tmp = pd.DataFrame(
                {
                    "claim_no": pivot.index,
                    "dev_lag": dy,
                    "num_active_dys_prior": count.values,
                }
            )
        result.append(tmp)

    return pd.concat(result, ignore_index=True)


def _compute_lagged_payments(df):
    """Get incremental payment in DY k-1 and DY k-2."""
    # Pivot incremental_payment by dev_lag
    pivot = df.pivot_table(
        index="claim_no",
        columns="dev_lag",
        values="incremental_payment",
        aggfunc="first",
        fill_value=0,
    )

    result = []
    for dy in range(1, cfg.MAX_DEV_LAG + 1):
        prev1 = (
            pivot[dy - 1].values if (dy - 1) in pivot.columns else np.zeros(len(pivot))
        )
        prev2 = (
            pivot[dy - 2].values if (dy - 2) in pivot.columns else np.zeros(len(pivot))
        )
        tmp = pd.DataFrame(
            {
                "claim_no": pivot.index,
                "dev_lag": dy,
                "paid_in_prev_dy": prev1,
                "paid_in_prev2_dy": prev2,
            }
        )
        result.append(tmp)

    return pd.concat(result, ignore_index=True)


def _compute_payment_trend(df):
    """
    Compute the slope of incremental DY payments across DYs 1..(k-1).
    Uses simple linear regression: slope of payment vs DY index.
    """
    pivot = df.pivot_table(
        index="claim_no",
        columns="dev_lag",
        values="incremental_payment",
        aggfunc="first",
        fill_value=0,
    )

    result = []
    for dy in range(1, cfg.MAX_DEV_LAG + 1):
        prior_cols = sorted([c for c in pivot.columns if c < dy])
        if len(prior_cols) < 2:
            # Need at least 2 points for a slope
            tmp = pd.DataFrame(
                {
                    "claim_no": pivot.index,
                    "dev_lag": dy,
                    "payment_trend": 0.0,
                }
            )
        else:
            # Vectorized slope: cov(x, y) / var(x)
            x = np.array(prior_cols, dtype=float)
            y = pivot[prior_cols].values  # (n_claims, n_prior_dys)
            x_mean = x.mean()
            y_mean = y.mean(axis=1, keepdims=True)
            cov_xy = ((y - y_mean) * (x - x_mean)).mean(axis=1)
            var_x = ((x - x_mean) ** 2).mean()
            slope = np.where(var_x > 0, cov_xy / var_x, 0.0)
            tmp = pd.DataFrame(
                {
                    "claim_no": pivot.index,
                    "dev_lag": dy,
                    "payment_trend": slope,
                }
            )
        result.append(tmp)

    return pd.concat(result, ignore_index=True)


def _compute_ewma(df, alpha=0.5):
    """
    Exponentially weighted moving average of DY-level incremental payments
    through DYs 1..(k-1).
    """
    pivot = df.pivot_table(
        index="claim_no",
        columns="dev_lag",
        values="incremental_payment",
        aggfunc="first",
        fill_value=0,
    )

    result = []
    for dy in range(1, cfg.MAX_DEV_LAG + 1):
        prior_cols = sorted([c for c in pivot.columns if c < dy])
        if len(prior_cols) == 0:
            tmp = pd.DataFrame(
                {
                    "claim_no": pivot.index,
                    "dev_lag": dy,
                    "ewma_payment": 0.0,
                }
            )
        else:
            # Compute EWMA: most recent DY gets highest weight
            weights = np.array([(1 - alpha) ** i for i in range(len(prior_cols))])[::-1]
            weights = weights / weights.sum()
            ewma = (pivot[prior_cols].values * weights).sum(axis=1)
            tmp = pd.DataFrame(
                {
                    "claim_no": pivot.index,
                    "dev_lag": dy,
                    "ewma_payment": ewma,
                }
            )
        result.append(tmp)

    return pd.concat(result, ignore_index=True)


def _compute_decay_rate(df):
    """
    Estimated decay rate of payments across DYs.
    If payments are [p1, p2, p3, ...], decay = mean(p_{i+1}/p_i) for p_i > 0.
    """
    pivot = df.pivot_table(
        index="claim_no",
        columns="dev_lag",
        values="incremental_payment",
        aggfunc="first",
        fill_value=0,
    )

    result = []
    for dy in range(1, cfg.MAX_DEV_LAG + 1):
        prior_cols = sorted([c for c in pivot.columns if c < dy])
        if len(prior_cols) < 2:
            tmp = pd.DataFrame(
                {
                    "claim_no": pivot.index,
                    "dev_lag": dy,
                    "payment_decay_rate": 0.0,
                }
            )
        else:
            # Compute ratios p_{i+1}/p_i for consecutive DYs with payments
            y = pivot[prior_cols].values  # (n_claims, n_prior_dys)
            ratios = np.zeros(len(pivot))
            counts = np.zeros(len(pivot))
            for i in range(len(prior_cols) - 1):
                p_curr = y[:, i]
                p_next = y[:, i + 1]
                mask = p_curr > 0
                ratio = np.where(mask, p_next / np.maximum(p_curr, 1e-10), 0.0)
                ratios += ratio
                counts += mask.astype(float)
            decay = np.where(counts > 0, ratios / counts, 0.0)
            tmp = pd.DataFrame(
                {
                    "claim_no": pivot.index,
                    "dev_lag": dy,
                    "payment_decay_rate": decay,
                }
            )
        result.append(tmp)

    return pd.concat(result, ignore_index=True)


LAST_AVG_ULTIMATE = None       # reference ultimate of the most recent feature run (training portfolio)
AVG_ULTIMATE_OVERRIDE = None   # set by a caller that scores a fresh portfolio with the training reference


def _compute_fraction_paid(df):
    """
    Fraction of expected ultimate paid so far.

    The reference ultimate is a credibility-weighted mean across all AYs.
    A naive "fully-developed mean" works only when at least one AY has
    reached MAX_DEV_LAG, which with the default MAX_AY=MAX_DY=10 means
    AY 1 alone — a noisy reference. Instead, each AY contributes its
    own mean cum_paid at its latest observable DY, with a credibility
    weight equal to the fraction of the development window observed
    (k / MAX_DEV_LAG).  Fully-developed AYs get weight 1.0; recent AYs
    with few observed DYs contribute little.  This produces a smoother
    estimate while keeping the feature's intent: scale cum_paid_prior
    against a typical long-run payment amount.
    """
    per_ay = []
    for ay in range(1, cfg.MAX_ACCIDENT_YEAR + 1):
        latest_dy = min(cfg.MAX_ACCIDENT_YEAR - ay + 1, cfg.MAX_DEV_LAG)
        if latest_dy < 1:
            continue
        rows = df[(df["accident_year"] == ay) & (df["dev_lag"] == latest_dy)]
        if len(rows) == 0:
            continue
        mean_cum = float(rows["cum_paid_through"].mean())
        if not np.isfinite(mean_cum) or mean_cum <= 0:
            continue
        credibility = latest_dy / cfg.MAX_DEV_LAG
        per_ay.append((credibility, mean_cum))

    if per_ay:
        w = np.array([c for c, _ in per_ay])
        v = np.array([m for _, m in per_ay])
        avg_ultimate = float((w * v).sum() / w.sum())
    else:
        avg_ultimate = 0.0

    if avg_ultimate == 0 or np.isnan(avg_ultimate):
        avg_ultimate = 1.0  # safety
    global LAST_AVG_ULTIMATE
    if AVG_ULTIMATE_OVERRIDE is not None:
        avg_ultimate = float(AVG_ULTIMATE_OVERRIDE)
    LAST_AVG_ULTIMATE = avg_ultimate

    result = df[["claim_no", "dev_lag", "cum_paid_prior"]].copy()
    result["fraction_paid_vs_avg_ultimate"] = result["cum_paid_prior"] / avg_ultimate
    result = result[["claim_no", "dev_lag", "fraction_paid_vs_avg_ultimate"]]

    return result


def _compute_dy_payment_std(df):
    """
    Standard deviation of DY-level incremental payments for DYs 1..(k-1).
    """
    pivot = df.pivot_table(
        index="claim_no",
        columns="dev_lag",
        values="incremental_payment",
        aggfunc="first",
        fill_value=0,
    )

    result = []
    for dy in range(1, cfg.MAX_DEV_LAG + 1):
        prior_cols = sorted([c for c in pivot.columns if c < dy])
        if len(prior_cols) < 2:
            tmp = pd.DataFrame(
                {
                    "claim_no": pivot.index,
                    "dev_lag": dy,
                    "std_dy_payments_prior": 0.0,
                }
            )
        else:
            std_vals = pivot[prior_cols].std(axis=1, ddof=1).fillna(0)
            tmp = pd.DataFrame(
                {
                    "claim_no": pivot.index,
                    "dev_lag": dy,
                    "std_dy_payments_prior": std_vals.values,
                }
            )
        result.append(tmp)

    return pd.concat(result, ignore_index=True)
