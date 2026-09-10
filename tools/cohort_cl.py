"""
Distinguish RBNS from IBNR in the chain ladder.

The standard chain ladder on the observed payment triangle projects the TOTAL
future payments of each accident year: its development factors contain the
historical payments of late reporters, so the projection includes claims not
yet reported at the valuation date (IBNR).  The pipeline's ML models, by
contrast, reserve only the RBNS cohort (claims reported by CY 10).  Comparing
the standard CL against the RBNS-only truth is therefore apples to oranges.

This tool computes three things from the individual data:

1. **Standard CL**, evaluated against BOTH truths: the RBNS-only lower
   triangle (what the pipeline reports) and the total lower triangle
   (the fair target of the standard CL).
2. **Report-cohort-consistent RBNS chain ladder** (Schnieper-style): for each
   accident year a, the RBNS cohort at valuation is the claims with
   report_dy <= 11 - a.  Its development is projected with factors estimated
   from a triangle restricted to claims with report_dy <= 11 - a, so the
   factor and the cohort match.  This is the apples-to-apples CL baseline for
   the RBNS reserving task.
3. **CL-implied IBNR** = (1) - (2) per accident year, comparable with the
   frequency-severity IBNR model of the first submission.

Reference: Schnieper (1991), "Separating true IBNR and IBNER claims", ASTIN
Bulletin 21(1); the report-cohort restriction implements the same separation
with individual data.

Usage (repo root, project interpreter):
    python tools/cohort_cl.py [--out results_v2/chain_ladder_cohort.csv]
"""

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "converted")
I = 10  # accident years = development years = valuation calendar year


def load():
    claims = pd.read_csv(
        os.path.join(DATA, "claims_individual.csv"),
        usecols=["claim_no", "accident_year", "report_dy", "ibnr_flag"],
    )
    tpath = os.path.join(DATA, "transactions_individual.csv")
    have = set(pd.read_csv(tpath, nrows=0).columns)
    cols = [c for c in ("claim_no", "accident_year", "dev_lag", "payment_size", "payment_inflated") if c in have]
    txn = pd.read_csv(tpath, usecols=cols)   # payment_inflated is absent from the public illustrative dataset
    txn = txn.merge(claims[["claim_no", "report_dy", "ibnr_flag"]], on="claim_no", how="left")
    return claims, txn


def square(txn, pay_col, mask=None):
    """Incremental payment square (I x I) from transactions, optionally masked."""
    t = txn if mask is None else txn[mask]
    g = t.groupby(["accident_year", "dev_lag"])[pay_col].sum()
    sq = np.zeros((I, I))
    for (a, k), v in g.items():
        if 1 <= a <= I and 1 <= k <= I:
            sq[int(a) - 1, int(k) - 1] = v
    return sq


def pick_payment_column(txn):
    """Choose the payment column that reproduces the converter's triangles."""
    ref = pd.read_csv(os.path.join(DATA, "triangle_incremental.csv"))
    ref_total = ref[[f"DY{i}" for i in range(1, I + 1)]].values.astype(float)
    for col in ("payment_size", "payment_inflated"):
        if col not in txn.columns:
            continue
        if np.allclose(np.nan_to_num(ref_total), square(txn, col), rtol=1e-6, atol=1.0):
            return col
    raise SystemExit("neither payment column reproduces triangle_incremental.csv")


def cl_factors(cum_obs):
    """Volume-weighted age-to-age factors from an observed cumulative triangle."""
    f = np.ones(I - 1)
    for j in range(I - 1):
        mask = ~np.isnan(cum_obs[:, j]) & ~np.isnan(cum_obs[:, j + 1])
        den = cum_obs[mask, j].sum()
        f[j] = cum_obs[mask, j + 1].sum() / den if mask.any() and den > 0 else 1.0
    return f


def observed_cumulative(inc_square):
    """Cumulative triangle with NaN below the CY-10 diagonal."""
    cum = np.cumsum(inc_square, axis=1).astype(float)
    for a in range(1, I + 1):
        for k in range(1, I + 1):
            if a + k - 1 > I:
                cum[a - 1, k - 1] = np.nan
    return cum


def cl_reserve_row(cum_obs, factors, a):
    """Project accident year a from its latest observed cumulative."""
    last_k = I - a + 1
    c = cum_obs[a - 1, last_k - 1]
    for j in range(last_k - 1, I - 1):
        c *= factors[j]
    return c - cum_obs[a - 1, last_k - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "results_v2", "chain_ladder_cohort.csv"))
    args = ap.parse_args()

    claims, txn = load()
    pay = pick_payment_column(txn)
    print(f"payment column: {pay}  |  claims: {len(claims):,} ({int(claims.ibnr_flag.sum()):,} IBNR)")

    total_sq = square(txn, pay)                                   # all claims
    rbns_sq = square(txn, pay, txn.ibnr_flag == 0)                # RBNS cohort

    def lower_by_ay(sq):
        return np.array([sum(sq[a - 1, k - 1] for k in range(1, I + 1) if a + k - 1 > I) for a in range(1, I + 1)])

    true_total = lower_by_ay(total_sq)
    true_rbns = lower_by_ay(rbns_sq)

    # 1. standard CL (observed triangle; identical for total and RBNS data)
    cum_obs = observed_cumulative(total_sq)
    assert np.allclose(np.nan_to_num(cum_obs), np.nan_to_num(observed_cumulative(rbns_sq))), \
        "observed triangles differ between total and RBNS — check report/payment consistency"
    f_std = cl_factors(cum_obs)
    cl_std = np.array([cl_reserve_row(cum_obs, f_std, a) for a in range(1, I + 1)])

    # 2. report-cohort-consistent RBNS CL: AY a projected with factors from the
    #    triangle of claims with report_dy <= 11 - a (its own RBNS cohort).
    cl_rbns = np.zeros(I)
    for a in range(2, I + 1):
        c = I - a + 1  # cohort truncation: claims reported by CY 10
        sq_c = square(txn, pay, txn.report_dy <= c)
        cum_c = observed_cumulative(sq_c)
        # AY a's own observed row within the c-cohort equals its full observed
        # row (all its observed payments come from claims with report_dy <= c).
        assert np.allclose(np.nan_to_num(cum_c[a - 1]), np.nan_to_num(cum_obs[a - 1])), a
        cl_rbns[a - 1] = cl_reserve_row(cum_c, cl_factors(cum_c), a)

    ibnr_implied = cl_std - cl_rbns

    df = pd.DataFrame({
        "accident_year": range(1, I + 1),
        "cl_standard": cl_std,
        "cl_rbns_cohort": cl_rbns,
        "cl_ibnr_implied": ibnr_implied,
        "true_rbns": true_rbns,
        "true_total": true_total,
        "true_ibnr": true_total - true_rbns,
    })
    df["err_std_vs_total_pct"] = np.where(df.true_total > 0, (df.cl_standard / df.true_total - 1) * 100, np.nan)
    df["err_rbnscl_vs_rbns_pct"] = np.where(df.true_rbns > 0, (df.cl_rbns_cohort / df.true_rbns - 1) * 100, np.nan)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:,.1f}"):
        show = df.copy()
        for c in show.columns:
            if c.startswith(("cl_", "true_")):
                show[c] = show[c] / 1e6
        print(show.round(1).to_string(index=False))
    T = df.sum(numeric_only=True)
    print(f"\nTOTALS (M): standard CL {T.cl_standard/1e6:,.1f} | cohort RBNS CL {T.cl_rbns_cohort/1e6:,.1f} | implied IBNR {T.cl_ibnr_implied/1e6:,.1f}")
    print(f"truth  (M): RBNS {T.true_rbns/1e6:,.1f} | total {T.true_total/1e6:,.1f} | IBNR {T.true_ibnr/1e6:,.1f}")
    print(f"standard CL vs TOTAL truth  : {(T.cl_standard/T.true_total-1)*100:+.1f}%")
    print(f"standard CL vs RBNS truth   : {(T.cl_standard/T.true_rbns-1)*100:+.1f}%   (the mismatched comparison)")
    print(f"cohort RBNS CL vs RBNS truth: {(T.cl_rbns_cohort/T.true_rbns-1)*100:+.1f}%")
    print(f"implied IBNR vs true IBNR   : {(T.cl_ibnr_implied/T.true_ibnr-1)*100:+.1f}%  ({T.cl_ibnr_implied/1e6:,.1f} vs {T.true_ibnr/1e6:,.1f} M)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
