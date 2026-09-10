"""
Data preparation for the Gabrielli (2021) individual RBNS claims reserving model.

Converts MTPL individual claims data (long format) into the wide format
expected by the Gabrielli NN: one row per claim with columns Pay01..Pay10
(incremental payments per development year).

Produces:
  - Categorical features: claim_type, AY, occurrence_quarter, report_dy
  - Time_Known indicators: which DYs are observed for each claim
  - PayInfo: discretised past-payment magnitudes (6 categories)
  - PayInd: binary payment indicators per DY
  - LogPay: log(payment) where payment > 0
  - Time_Predict masks for training and prediction
"""

import numpy as np
import pandas as pd
import os

import src.config as cfg


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

N_DY = cfg.MAX_DEV_LAG  # 10 for annual MTPL data

CLAIM_TYPE_MAP = {
    "Material": 0,
    "Technical": 1,
    "Bodily": 2,
    "Annuity": 3,
    "Remaining": 4,
    "Lost": 5,
}

# Payment magnitude bucket thresholds.
# Gabrielli used (-Inf,-1,0,5000,20000,100000,Inf) for Swiss personal-injury.
# MTPL data has no negative payments and higher amounts.  We keep the same
# structure but shift to MTPL scale (median single payment ~20K).
PAY_BUCKET_EDGES = [-np.inf, -1, 0, 10_000, 50_000, 500_000, np.inf]
# Category mapping:  1=recovery, 0=zero, 2=small, 3=medium, 4=large, 5=very-large


def load_and_prepare(verbose=True):
    """
    Build the Gabrielli-format dataset from MTPL data.

    Returns
    -------
    data : pd.DataFrame
        One row per RBNS claim with columns:
          claim_no, accident_year, claim_type_code, AY_code, AQ_code,
          report_dy_code, Pay01..Pay{N_DY}, Time_Known01..Time_Known{N_DY},
          Pay01Info..Pay{N_DY-1}Info, PayInd01..PayInd{N_DY},
          LogPay01..LogPay{N_DY},
          Time_Predict_Indicator01..{N_DY},
          Time_Predict_Payment01..{N_DY}
    """
    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    claims = pd.read_csv(os.path.join(cfg.DATA_DIR, "claims_individual.csv"))
    transactions = pd.read_csv(
        os.path.join(cfg.DATA_DIR, "transactions_individual.csv")
    )
    if verbose:
        print(
            f"Loaded {claims.shape[0]:,} claims, {transactions.shape[0]:,} transactions"
        )

    # Filter to RBNS claims
    claims_rbns = claims[claims["ibnr_flag"] == 0].copy()
    rbns_claim_nos = set(claims_rbns["claim_no"])
    if verbose:
        print(f"RBNS claims: {len(claims_rbns):,}")

    # ALL transactions for RBNS claims (both observed and unobserved — we need
    # the full payments for ground truth; Time_Known masks prevent leakage).
    txn_rbns = transactions[transactions["claim_no"].isin(rbns_claim_nos)].copy()

    # ------------------------------------------------------------------
    # 2. Pivot payments to wide format: Pay01..Pay{N_DY}
    # ------------------------------------------------------------------
    pay_agg = (
        txn_rbns.groupby(["claim_no", "dev_lag"])["payment_size"].sum().reset_index()
    )
    # Filter to DY 1..N_DY only
    pay_agg = pay_agg[(pay_agg["dev_lag"] >= 1) & (pay_agg["dev_lag"] <= N_DY)]
    pay_wide = pay_agg.pivot(
        index="claim_no", columns="dev_lag", values="payment_size"
    ).fillna(0.0)
    pay_wide.columns = [f"Pay{int(d):02d}" for d in pay_wide.columns]

    # Ensure all DY columns exist (some claims may have zero for certain DYs)
    for dy in range(1, N_DY + 1):
        col = f"Pay{dy:02d}"
        if col not in pay_wide.columns:
            pay_wide[col] = 0.0

    pay_wide = pay_wide[[f"Pay{dy:02d}" for dy in range(1, N_DY + 1)]]
    pay_wide = pay_wide.reset_index()

    # ------------------------------------------------------------------
    # 3. Build claim-level features
    # ------------------------------------------------------------------
    data = claims_rbns[
        [
            "claim_no",
            "accident_year",
            "occurrence_period",
            "claim_type",
            "notidel",
            "report_dy",
            "no_payments",
            "cum_paid_observed",
        ]
    ].copy()

    # claim_type -> integer code
    data["claim_type_code"] = (
        data["claim_type"].map(CLAIM_TYPE_MAP).fillna(0).astype(int)
    )

    # AY -> 0-based code
    data["AY_code"] = data["accident_year"] - 1  # 0..9

    # (A) Reporting delay: UNCAPPED, 0-based.
    # MTPL report_dy is 1-based (1 = reported in same year as accident).
    # Gabrielli RepDel is 0-based (0 = reported in same year).
    # MTPL report_dy ranges 1-6, so RepDel = 0-5, report_dy_code = 0-5.
    data["RepDel"] = data["report_dy"] - 1
    data["report_dy_code"] = data["RepDel"].astype(int)

    # (C) Repurpose AQ embedding for binned notidel (4 categories).
    # notidel is continuous (0-4.5 years).  Bin into 4 buckets:
    #   0: [0, 0.5)   fast report
    #   1: [0.5, 1.0)  within a year
    #   2: [1.0, 2.0)  1-2 years
    #   3: [2.0, inf)   slow report
    notidel_bins = pd.cut(
        data["notidel"],
        bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
        labels=[0, 1, 2, 3],
        right=False,
    )
    data["AQ_code"] = notidel_bins.astype(int)

    # (B) notidel as continuous numeric feature (standardised later in model)
    data["notidel_raw"] = data["notidel"].astype(np.float32)

    # (D) cum_paid_observed and no_payments as continuous numeric features
    # Log-transform cum_paid_observed to reduce scale (add 1 to handle zeros)
    data["log_cum_paid_obs"] = np.log1p(
        data["cum_paid_observed"].fillna(0).clip(lower=0)
    ).astype(np.float32)
    data["no_payments_raw"] = data["no_payments"].fillna(0).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Merge payments
    # ------------------------------------------------------------------
    data = data.merge(pay_wide, on="claim_no", how="left")
    for dy in range(1, N_DY + 1):
        data[f"Pay{dy:02d}"] = data[f"Pay{dy:02d}"].fillna(0.0)

    # ------------------------------------------------------------------
    # 5. Time_Known indicators
    # ------------------------------------------------------------------
    # DY t is known if accident_year + t - 1 <= MAX_ACCIDENT_YEAR (= N_DY = 10)
    # Equivalently: t <= 10 - accident_year + 1  = 11 - AY
    for t in range(1, N_DY + 1):
        data[f"Time_Known{t:02d}"] = (
            (data["accident_year"] + t - 1) <= cfg.MAX_ACCIDENT_YEAR
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # 6. PayInfo: discretised payment magnitudes (only for known DYs)
    # ------------------------------------------------------------------
    for t in range(1, N_DY):  # Pay01Info .. Pay{N_DY-1}Info (last DY never a feature)
        col_pay = f"Pay{t:02d}"
        col_known = f"Time_Known{t:02d}"
        col_info = f"Pay{t:02d}Info"

        # Default: 0 (= "zero payment" / "no info")
        data[col_info] = 0

        known_mask = data[col_known] == 1.0
        if known_mask.any():
            # Categorise: labels map to [1, 0, 2, 3, 4, 5] matching Gabrielli
            # (recovery=1, zero=0, small=2, med=3, large=4, vlarge=5)
            cats = pd.cut(
                data.loc[known_mask, col_pay],
                bins=PAY_BUCKET_EDGES,
                labels=[1, 0, 2, 3, 4, 5],
                right=True,
            )
            data.loc[known_mask, col_info] = cats.astype(int).values

    # ------------------------------------------------------------------
    # 7. Payment indicators and log-payments
    # ------------------------------------------------------------------
    for t in range(1, N_DY + 1):
        col_pay = f"Pay{t:02d}"
        col_known = f"Time_Known{t:02d}"
        col_ind = f"PayInd{t:02d}"
        col_log = f"LogPay{t:02d}"

        data[col_ind] = 0
        data[col_log] = 0.0

        known_mask = data[col_known] == 1.0
        if known_mask.any():
            data.loc[known_mask, col_ind] = (
                (data.loc[known_mask, col_pay] > 0).astype(int).values
            )
            pos_mask = known_mask & (data[col_pay] > 0)
            if pos_mask.any():
                data.loc[pos_mask, col_log] = np.log(data.loc[pos_mask, col_pay].values)

    # ------------------------------------------------------------------
    # 8. Time_Predict masks (for training: predict only known DYs)
    # ------------------------------------------------------------------
    for t in range(1, N_DY + 1):
        data[f"Time_Predict_Indicator{t:02d}"] = data[f"Time_Known{t:02d}"]
        data[f"Time_Predict_Payment{t:02d}"] = data[f"PayInd{t:02d}"].astype(np.float32)

    # ------------------------------------------------------------------
    # 9. Ground truth: true reserve per claim
    # ------------------------------------------------------------------
    # True outstanding = sum of Pay columns for unobserved DYs
    data["true_reserve"] = 0.0
    for t in range(1, N_DY + 1):
        unobserved = data[f"Time_Known{t:02d}"] == 0.0
        data.loc[unobserved, "true_reserve"] += data.loc[unobserved, f"Pay{t:02d}"]

    if verbose:
        print(f"Gabrielli-format dataset: {len(data):,} claims x {data.shape[1]} cols")
        print(f"  Total true RBNS reserve: {data['true_reserve'].sum():,.0f}")

    return data


def build_prediction_data(data):
    """
    Build the prediction-set version of the dataset.

    For prediction, we flip the Time_Predict masks:
    - Time_Predict_Indicator[t] = 1 - Time_Known[t]  (predict unknown DYs)
    - Time_Predict_Payment[t] = Time_Predict_Indicator[t]
    - For claims with RepDel > 0, zero out the last RepDel time indices.

    Only claims with accident_year > 1 are included (AY 1 is fully observed).

    Returns a copy of the prediction-set data with flipped masks.
    """
    # Only claims that have at least one unobserved DY
    pred = data[data["accident_year"] > 1].copy()

    for t in range(1, N_DY + 1):
        pred[f"Time_Predict_Indicator{t:02d}"] = 1.0 - pred[f"Time_Known{t:02d}"]
        # Zero out indices beyond the claim's development horizon
        # A claim with RepDel=r has at most N_DY - r meaningful periods
        # (periods N_DY-r+1 .. N_DY are beyond its horizon)
        if t > N_DY:
            pred[f"Time_Predict_Indicator{t:02d}"] = 0.0

    # For claims with RepDel > 0, zero out the last RepDel periods
    for r in range(1, N_DY):
        rep_mask = pred["RepDel"] == r
        if rep_mask.any():
            for t in range(N_DY - r + 1, N_DY + 1):
                pred.loc[rep_mask, f"Time_Predict_Indicator{t:02d}"] = 0.0

    # Payment prediction = same as indicator prediction (for prediction set)
    for t in range(1, N_DY + 1):
        pred[f"Time_Predict_Payment{t:02d}"] = pred[f"Time_Predict_Indicator{t:02d}"]

    return pred


def compute_starting_values(data):
    """
    Compute starting values for the NN: mean P(pay>0) and mean log(pay|pay>0)
    for each DY, using only observed data.

    Returns
    -------
    starting_values : np.ndarray  shape (N_DY, 2)
        Col 0: mean payment probability at each DY
        Col 1: mean log-payment at each DY (given payment > 0)
    """
    sv = np.zeros((N_DY, 2))
    for t in range(1, N_DY + 1):
        known = data[f"Time_Known{t:02d}"] == 1.0
        if known.sum() > 0:
            pays = data.loc[known, f"Pay{t:02d}"]
            sv[t - 1, 0] = (pays > 0).mean()
            pos = pays[pays > 0]
            sv[t - 1, 1] = np.log(pos).mean() if len(pos) > 0 else 0.0
    return sv


def compute_loss_weights(data):
    """
    Compute per-DY loss weights: inverse of average loss for each output.

    Matches Gabrielli's R code: the average is taken over ALL n claims
    (padding non-eligible claims with 0), not just eligible ones. This
    means late DYs (few eligible claims) get lower raw loss values and
    thus higher weights, which is how the paper normalizes the multi-task
    loss to equal contribution per DY.

    Returns
    -------
    weights : np.ndarray  shape (N_DY, 2)
        Col 0: weight for indicator loss
        Col 1: weight for mean loss
    """
    n = len(data)
    w = np.ones((N_DY, 2))

    for t in range(1, N_DY + 1):
        known = data[f"Time_Known{t:02d}"] == 1.0
        if known.sum() > 0:
            # Indicator weight: R code computes mean(c(bce_eligible, rep(0, n-n_eligible)))
            # which equals sum(bce_eligible) / n
            p = data.loc[known, f"PayInd{t:02d}"].values.astype(float)
            p_mean = np.clip(p.mean(), 1e-8, 1 - 1e-8)
            bce = -(p * np.log(p_mean) + (1 - p) * np.log(1 - p_mean))
            avg_bce = bce.sum() / n  # divide by total n, not eligible count
            w[t - 1, 0] = 1.0 / max(avg_bce, 1e-8)

            # Mean weight: R code pads MSE with zeros for non-positive/non-eligible
            pos = data.loc[known & (data[f"Pay{t:02d}"] > 0), f"LogPay{t:02d}"].values
            if len(pos) > 0:
                mse = (pos - pos.mean()) ** 2
                avg_mse = mse.sum() / n  # divide by total n
                w[t - 1, 1] = 1.0 / max(avg_mse, 1e-8)

    return w
