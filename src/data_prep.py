"""
Data preparation module.

Loads raw CSVs, filters to RBNS claims and observed transactions,
builds the claim × development-year dataset used by all downstream modules.
"""

import pandas as pd
import numpy as np
import src.config as cfg

from src.config import get_val_ay_for_dy, get_train_ays_for_dy, get_test_ays_for_dy

import os

# Columns that must NEVER be used as model features (future leakage / ground truth).
# Duplicated from feature_engineering.py so the guard rail works even if
# feature engineering is bypassed.
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

# Columns that are kept in the DataFrame for evaluation / ground-truth purposes
# but must never be passed to a model as features.
_GROUND_TRUTH_COLS = [
    "cum_paid_full",  # cumulative paid including future (from claims CSV)
    "cum_paid_observed",  # cumulative paid at valuation date (from claims CSV)
    "reserve_true",  # true outstanding reserve (from claims CSV)
    "true_incremental_payment",  # added by add_ground_truth()
]


# -----------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------


def load_raw_data():
    """Load the two main CSV files and return (claims, transactions)."""
    claims = pd.read_csv(os.path.join(cfg.DATA_DIR, "claims_individual.csv"))
    transactions = pd.read_csv(
        os.path.join(cfg.DATA_DIR, "transactions_individual.csv")
    )
    print(f"Loaded claims: {claims.shape[0]:,} rows, {claims.shape[1]} cols")
    print(
        f"Loaded transactions: {transactions.shape[0]:,} rows, {transactions.shape[1]} cols"
    )
    return claims, transactions


def load_triangles():
    """Load the three triangle files."""
    tri_obs = pd.read_csv(os.path.join(cfg.DATA_DIR, "triangle_observed.csv"))
    tri_full = pd.read_csv(os.path.join(cfg.DATA_DIR, "triangle_full.csv"))
    tri_inc = pd.read_csv(os.path.join(cfg.DATA_DIR, "triangle_incremental.csv"))
    return tri_obs, tri_full, tri_inc


# -----------------------------------------------------------------------
# Filtering
# -----------------------------------------------------------------------


def filter_rbns_observed(claims, transactions):
    """
    Filter to RBNS claims (ibnr_flag == 0) and observed transactions
    (observed_flag == 1) belonging to those claims.

    Returns (claims_rbns, transactions_obs).
    """
    claims_rbns = claims[claims["ibnr_flag"] == 0].copy()
    rbns_claim_nos = set(claims_rbns["claim_no"])

    transactions_obs = transactions[
        (transactions["observed_flag"] == 1)
        & (transactions["claim_no"].isin(rbns_claim_nos))
    ].copy()

    print(f"RBNS claims: {claims_rbns.shape[0]:,}")
    print(f"Observed RBNS transactions: {transactions_obs.shape[0]:,}")
    return claims_rbns, transactions_obs


# -----------------------------------------------------------------------
# Build claim × DY dataset
# -----------------------------------------------------------------------


def _aggregate_transactions_to_claim_dy(transactions_obs):
    """
    Group observed transactions by (claim_no, dev_lag).

    Returns a DataFrame with columns:
        claim_no, dev_lag, incremental_payment, num_payments_in_dy,
        max_payment_in_dy, min_payment_in_dy, first_pay_period, last_pay_period
    """
    agg = (
        transactions_obs.groupby(["claim_no", "dev_lag"])
        .agg(
            incremental_payment=("payment_size", "sum"),
            num_payments_in_dy=("payment_size", "count"),
            max_payment_in_dy=("payment_size", "max"),
            min_payment_in_dy=("payment_size", "min"),
            first_pay_period=("payment_period", "min"),
            last_pay_period=("payment_period", "max"),
        )
        .reset_index()
    )
    return agg


def build_claim_dy_dataset(claims_rbns, transactions_obs):
    """
    Build the full claim × development-year dataset.

    For each RBNS claim, creates rows for DY 1..10.
    Marks each row as observable or not (based on AY + DY - 1 <= MAX_ACCIDENT_YEAR).
    Merges aggregated transaction data (payment sums) for observable rows.
    Computes cumulative paid through prior DYs.

    Returns a DataFrame with ~1.45M rows.
    """
    # Step 1: aggregate transactions to claim × DY
    txn_agg = _aggregate_transactions_to_claim_dy(transactions_obs)

    # Step 2: create the full claim × DY grid
    # Core columns always present
    core_cols = [
        "claim_no",
        "accident_year",
        "occurrence_period",
        "notidel",
        "cum_paid_observed",
        "cum_paid_full",
        "reserve_true",
    ]
    if "report_dy" in claims_rbns.columns:  # calendar reporting year; not a feature (see RESERVING_EXCLUDE_PREREPORT)
        core_cols.append("report_dy")
    # Optional claim-level feature columns (present in CL-hostile data)
    optional_feature_cols = [
        "claimant_age",
        "injury_severity",
        "vehicle_class",
        "legal_rep",
        "claim_count",
        "claim_type",  # MTPL data: Material/Bodily/Technical/etc.
    ]
    extra_cols = [c for c in optional_feature_cols if c in claims_rbns.columns]
    if extra_cols:
        print(f"  Found claim-level features: {extra_cols}")

    claim_info = claims_rbns[core_cols + extra_cols].copy()

    dy_range = pd.DataFrame({"dev_lag": range(1, cfg.MAX_DEV_LAG + 1)})
    claim_dy = claim_info.merge(dy_range, how="cross")

    # Step 3: mark observable rows
    claim_dy["is_observable"] = (
        claim_dy["accident_year"] + claim_dy["dev_lag"] - 1
    ) <= cfg.MAX_ACCIDENT_YEAR

    # Step 4: merge aggregated transaction data
    claim_dy = claim_dy.merge(
        txn_agg,
        on=["claim_no", "dev_lag"],
        how="left",
    )

    # Fill NaN payment columns for DYs with no transactions
    fill_cols = [
        "incremental_payment",
        "num_payments_in_dy",
        "max_payment_in_dy",
        "min_payment_in_dy",
    ]
    for col in fill_cols:
        claim_dy[col] = claim_dy[col].fillna(0.0)

    # first/last pay period: leave as NaN for DYs with no payments
    # (feature engineering will handle these)

    # Step 5: compute cumulative paid through prior DYs
    # Sort by claim_no, dev_lag to ensure correct cumsum
    claim_dy = claim_dy.sort_values(["claim_no", "dev_lag"]).reset_index(drop=True)

    # Cumulative payment PRIOR to this DY (within-group shift by 1)
    claim_dy["cum_paid_prior"] = claim_dy.groupby("claim_no")[
        "incremental_payment"
    ].transform(lambda s: s.cumsum().shift(1, fill_value=0.0))

    # Cumulative payment INCLUDING this DY (for ground-truth / evaluation)
    claim_dy["cum_paid_through"] = claim_dy.groupby("claim_no")[
        "incremental_payment"
    ].cumsum()

    # Step 6: For unobservable rows, we need the TRUE incremental payment
    # as ground truth for evaluation.  We get this from the full transactions.
    # But during training we never use unobservable rows.
    # The incremental_payment for unobservable rows is currently 0
    # (since we only merged observed transactions).
    # We'll fill in ground truth separately for evaluation.

    # Step 7: compute additional basic columns
    if cfg.is_quarterly():
        claim_dy["occurrence_month"] = ((claim_dy["occurrence_period"] - 1) % 4) + 1
    else:
        claim_dy["occurrence_month"] = ((claim_dy["occurrence_period"] - 1) % 12) + 1

    print(f"Claim × DY dataset: {claim_dy.shape[0]:,} rows")
    print(f"  Observable rows: {claim_dy['is_observable'].sum():,}")
    print(
        f"  Unobservable (prediction targets): {(~claim_dy['is_observable']).sum():,}"
    )

    return claim_dy


# -----------------------------------------------------------------------
# Fill in ground-truth for unobservable rows (for evaluation only)
# -----------------------------------------------------------------------


def add_ground_truth(claim_dy, transactions_full):
    """
    Add the true incremental payments for unobservable rows using the full
    transaction dataset (including observed_flag == 0).

    This is ONLY used for evaluation — never for training or features.
    """
    # Filter to RBNS claims that are in our dataset
    rbns_claims = set(claim_dy["claim_no"].unique())

    # Aggregate ALL transactions (observed + unobserved) for RBNS claims
    txn_all = transactions_full[transactions_full["claim_no"].isin(rbns_claims)]
    txn_agg_all = (
        txn_all.groupby(["claim_no", "dev_lag"])["payment_size"]
        .sum()
        .reset_index()
        .rename(columns={"payment_size": "true_incremental_payment"})
    )

    # Merge
    claim_dy = claim_dy.merge(txn_agg_all, on=["claim_no", "dev_lag"], how="left")
    claim_dy["true_incremental_payment"] = claim_dy["true_incremental_payment"].fillna(
        0.0
    )

    # For observable rows, true_incremental_payment should match incremental_payment
    # For unobservable rows, true_incremental_payment is the ground truth

    print(
        f"Ground truth added. Unobservable rows with non-zero true payment: "
        f"{((~claim_dy['is_observable']) & (claim_dy['true_incremental_payment'] > 0)).sum():,}"
    )

    return claim_dy


# -----------------------------------------------------------------------
# Train / Validation / Test splitting
# -----------------------------------------------------------------------


def get_split_masks(claim_dy, dy, structure="per_dy"):
    """
    Return boolean masks for train, validation, and test rows.

    For a given development year `dy`:
      - Train: AYs 1..(MAX_AY - dy), all available AYs except validation
      - Validation: AY MAX_AY - dy + 1, the latest AY with observed DY
      - Test: AYs (MAX_AY - dy + 2)..MAX_AY, unobservable (lower triangle)

    If structure == "per_dy": filters to rows with that specific dev_lag.
    If structure == "single_model": returns masks across ALL dev_lags,
        with the appropriate train/val/test per DY.
    """
    if structure == "per_dy":
        mask_dy = claim_dy["dev_lag"] == dy
        val_ay = get_val_ay_for_dy(dy)
        train_ays = get_train_ays_for_dy(dy)
        test_ays = get_test_ays_for_dy(dy)

        train_mask = mask_dy & claim_dy["accident_year"].isin(train_ays)
        val_mask = mask_dy & (claim_dy["accident_year"] == val_ay)
        test_mask = mask_dy & claim_dy["accident_year"].isin(test_ays)

        return train_mask, val_mask, test_mask

    elif structure == "single_model":
        # For single_model, we combine all DYs.
        # For each DY, the val AY is the latest available, and test is beyond.
        # We build combined masks across all DYs.
        train_mask = pd.Series(False, index=claim_dy.index)
        val_mask = pd.Series(False, index=claim_dy.index)
        test_mask = pd.Series(False, index=claim_dy.index)

        for d in range(1, cfg.MAX_DEV_LAG + 1):
            mask_d = claim_dy["dev_lag"] == d
            val_ay = get_val_ay_for_dy(d)
            train_ays = get_train_ays_for_dy(d)
            test_ays = get_test_ays_for_dy(d)

            train_mask |= mask_d & claim_dy["accident_year"].isin(train_ays)
            val_mask |= mask_d & (claim_dy["accident_year"] == val_ay)
            test_mask |= mask_d & claim_dy["accident_year"].isin(test_ays)

        return train_mask, val_mask, test_mask

    else:
        raise ValueError(f"Unknown structure: {structure}")


# -----------------------------------------------------------------------
# Convenience: prepare everything
# -----------------------------------------------------------------------


def prepare_data():
    """
    Full data preparation pipeline. Returns:
      - claim_dy: the claim × DY dataframe with features, splits, and ground truth
      - claims_rbns: the filtered claims dataframe
      - transactions_obs: the filtered observed transactions
      - triangles: tuple of (observed, full, incremental) triangles
    """
    claims, transactions = load_raw_data()
    triangles = load_triangles()

    claims_rbns, transactions_obs = filter_rbns_observed(claims, transactions)
    claim_dy = build_claim_dy_dataset(claims_rbns, transactions_obs)
    claim_dy = add_ground_truth(claim_dy, transactions)

    # --- Guard rail: verify forbidden columns are present ONLY for evaluation,
    # not accidentally used downstream as features.  The actual feature-list
    # check lives in feature_engineering.engineer_features(), but we also
    # tag the DataFrame here so it's easy to audit.
    gt_present = [c for c in _GROUND_TRUTH_COLS if c in claim_dy.columns]
    if gt_present:
        print(
            f"  Ground-truth / evaluation-only columns in claim_dy "
            f"(NOT for features): {gt_present}"
        )

    return claim_dy, claims_rbns, transactions_obs, triangles
