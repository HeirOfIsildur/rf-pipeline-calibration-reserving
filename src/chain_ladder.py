"""
Aggregate Chain Ladder baseline.

Implements the traditional volume-weighted Chain Ladder method on the
observed loss triangle, projects the lower-right, and computes reserves
per accident year.
"""

import pandas as pd
import numpy as np
import os
import src.config as cfg


def load_triangle(filename="triangle_observed.csv"):
    """Load a triangle CSV and return as a numpy 2D array (AY x DY)."""
    df = pd.read_csv(os.path.join(cfg.DATA_DIR, filename))
    prefix = "DQ" if cfg.is_quarterly() else "DY"
    dy_cols = [f"{prefix}{i}" for i in range(1, cfg.MAX_DEV_LAG + 1)]
    triangle = df[dy_cols].values.astype(float)
    return triangle


def compute_development_factors(triangle):
    """
    Compute volume-weighted age-to-age development factors.

    factor[j] = sum(C_{i, j+1}) / sum(C_{i, j})
    where the sum is over all AYs that have observed values at both DY j and j+1.

    Returns array of length MAX_DEV_LAG - 1 (factors for DY1→DY2, ..., DY9→DY10).
    """
    n_ay, n_dy = triangle.shape
    factors = np.zeros(n_dy - 1)

    for j in range(n_dy - 1):
        # Find AYs where both DY j and DY j+1 are observed (not NaN)
        mask = ~np.isnan(triangle[:, j]) & ~np.isnan(triangle[:, j + 1])
        if mask.sum() == 0:
            factors[j] = 1.0  # no data — assume no development
        else:
            numerator = triangle[mask, j + 1].sum()
            denominator = triangle[mask, j].sum()
            factors[j] = numerator / denominator if denominator > 0 else 1.0

    return factors


def project_triangle(triangle, factors):
    """
    Project the lower-right of the triangle using the development factors.

    Returns a completed triangle (no NaNs).
    """
    projected = triangle.copy()
    n_ay, n_dy = projected.shape

    for i in range(n_ay):
        for j in range(n_dy - 1):
            if np.isnan(projected[i, j + 1]):
                projected[i, j + 1] = projected[i, j] * factors[j]

    return projected


def compute_reserves(projected, observed):
    """
    Compute reserves per AY.

    reserve[i] = projected ultimate (DY10) - latest observed cumulative.

    Returns:
      reserves : array of shape (n_ay,)
      latest_observed : array of shape (n_ay,) — latest non-NaN cumulative per AY
    """
    n_ay, n_dy = observed.shape
    reserves = np.zeros(n_ay)
    latest_observed = np.zeros(n_ay)

    for i in range(n_ay):
        # Find latest observed DY
        last_dy = -1
        for j in range(n_dy):
            if not np.isnan(observed[i, j]):
                last_dy = j
        if last_dy >= 0:
            latest_observed[i] = observed[i, last_dy]
        reserves[i] = projected[i, -1] - latest_observed[i]

    return reserves, latest_observed


def _individual_data():
    """claims + transactions with report delay merged; None if files are absent."""
    import os as _os
    cpath = _os.path.join(cfg.DATA_DIR, "claims_individual.csv")
    tpath = _os.path.join(cfg.DATA_DIR, "transactions_individual.csv")
    if not (_os.path.exists(cpath) and _os.path.exists(tpath)):
        return None
    claims = pd.read_csv(cpath)
    rep_col = "report_dq" if "report_dq" in claims.columns else "report_dy"
    if rep_col not in claims.columns:
        return None
    lag_col = "dev_lag"
    txn = pd.read_csv(tpath, usecols=["claim_no", "accident_year", lag_col, "payment_size"])
    txn = txn.merge(claims[["claim_no", rep_col]].rename(columns={rep_col: "report_lag"}),
                    on="claim_no", how="left")
    return txn


def _observed_cumulative_from_txn(txn, mask=None):
    """Cumulative triangle (NaN below the diagonal) from transactions."""
    I, K = cfg.MAX_ACCIDENT_YEAR, cfg.MAX_DEV_LAG
    t = txn if mask is None else txn[mask]
    inc = np.zeros((I, K))
    g = t.groupby(["accident_year", "dev_lag"])["payment_size"].sum()
    for (a, k), v in g.items():
        if 1 <= a <= I and 1 <= k <= K:
            inc[int(a) - 1, int(k) - 1] = v
    cum = np.cumsum(inc, axis=1).astype(float)
    for a in range(1, I + 1):
        for k in range(1, K + 1):
            if a + k - 1 > I:
                cum[a - 1, k - 1] = np.nan
    return cum


def cohort_consistent_reserves(observed):
    """
    Report-cohort-consistent RBNS chain ladder (Schnieper 1991 with individual
    data): accident year a is projected with development factors estimated from
    the triangle of claims with report delay <= I - a + 1 — the factor
    population matches the RBNS cohort being reserved.  The standard CL, whose
    factors contain the payments of late reporters, projects the TOTAL
    (RBNS + IBNR) reserve instead; comparing it against an RBNS-only truth is
    apples to oranges.

    Returns reserves per AY, or None when individual data is unavailable.
    """
    txn = _individual_data()
    if txn is None:
        return None
    I = cfg.MAX_ACCIDENT_YEAR
    reserves = np.zeros(I)
    for a in range(2, I + 1):
        c = I - a + 1
        cum_c = _observed_cumulative_from_txn(txn, txn["report_lag"] <= c)
        f = compute_development_factors(cum_c)
        last_k = I - a + 1
        val = cum_c[a - 1, last_k - 1]
        if np.isnan(val):
            val = observed[a - 1, last_k - 1]
        ult = val
        for j in range(last_k - 1, I - 1):
            ult *= f[j]
        reserves[a - 1] = ult - val
    return reserves


def run_chain_ladder():
    """
    Run the full Chain Ladder pipeline.

    Returns:
      results : dict with keys:
        - 'factors': development factors
        - 'projected': completed triangle
        - 'reserves_by_ay': reserves per AY (array of shape MAX_ACCIDENT_YEAR)
        - 'total_reserve': sum of all reserves
        - 'projected_df': DataFrame with AY and projected values
    """
    observed = load_triangle("triangle_observed_rbns.csv")
    full = load_triangle("triangle_full_rbns.csv")

    factors = compute_development_factors(observed)
    projected = project_triangle(observed, factors)
    reserves, latest_obs = compute_reserves(projected, observed)

    # True reserves from the full triangle (RBNS cohort)
    true_reserves = full[:, -1] - latest_obs

    # Total truth (RBNS + IBNR) and the cohort-consistent RBNS chain ladder,
    # so the comparison table can label each CL row with its actual target.
    try:
        full_all = load_triangle("triangle_full.csv")
        true_total = full_all[:, -1] - latest_obs
    except Exception:
        true_total = None
    rbns_cohort_reserves = cohort_consistent_reserves(observed)

    # Build results DataFrame
    prefix = "DQ" if cfg.is_quarterly() else "DY"
    dy_cols = [f"{prefix}{i}" for i in range(1, cfg.MAX_DEV_LAG + 1)]
    projected_df = pd.DataFrame(projected, columns=dy_cols)
    projected_df.insert(0, "accident_year", range(1, cfg.MAX_ACCIDENT_YEAR + 1))
    projected_df["CL_reserve"] = reserves
    projected_df["true_reserve"] = true_reserves
    projected_df["CL_error"] = reserves - true_reserves
    # Match evaluate.py's logic: when true_reserve is 0, report 0% if
    # predicted is also ~0, otherwise NaN to flag the discrepancy.
    projected_df["CL_error_pct"] = np.where(
        true_reserves != 0,
        (reserves - true_reserves) / true_reserves * 100,
        np.where(np.isclose(reserves, 0, atol=1e-8), 0.0, np.nan),
    )

    print("\n=== Chain Ladder Results ===")
    print(f"Development factors: {np.round(factors, 4)}")
    print(f"Total CL reserve: {reserves.sum():,.0f}")
    print(f"Total true reserve: {true_reserves.sum():,.0f}")
    if true_reserves.sum() != 0:
        print(
            f"Total CL error: {reserves.sum() - true_reserves.sum():,.0f} "
            f"({(reserves.sum() - true_reserves.sum()) / true_reserves.sum() * 100:.2f}%)"
        )
    else:
        print(
            f"Total CL error: {reserves.sum() - true_reserves.sum():,.0f} (no true reserves)"
        )

    if rbns_cohort_reserves is not None:
        projected_df["CL_rbns_cohort_reserve"] = rbns_cohort_reserves
        print(f"Cohort-consistent RBNS CL reserve: {rbns_cohort_reserves.sum():,.0f} "
              f"({(rbns_cohort_reserves.sum() / true_reserves.sum() - 1) * 100:+.2f}% vs RBNS truth)")
    if true_total is not None:
        projected_df["true_total_reserve"] = true_total
        print(f"Standard CL vs TOTAL (RBNS+IBNR) truth {true_total.sum():,.0f}: "
              f"{(reserves.sum() / true_total.sum() - 1) * 100:+.2f}%")

    return {
        "factors": factors,
        "projected": projected,
        "reserves_by_ay": reserves,
        "true_reserves_by_ay": true_reserves,
        "total_reserve": reserves.sum(),
        "total_true_reserve": true_reserves.sum(),
        "projected_df": projected_df,
        "rbns_cohort_reserves_by_ay": rbns_cohort_reserves,
        "total_true_reserves_by_ay": true_total,
    }
