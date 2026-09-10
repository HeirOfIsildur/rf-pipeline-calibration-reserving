#!/usr/bin/env python3
"""Claim-type-stratified report-cohort-consistent chain ladder (referee request, 9 Sep 2026).

For every claim type with at least MIN_CLAIMS reported claims, accident year a is projected from the type's own
report-cohort triangle (claims of that type with report_dy <= I - a + 1) with factors estimated on that triangle;
types below the threshold (Annuity, Remaining, Lost on the MTPL data) are projected with the all-type cohort factors
applied to the type's own observed cumulative (the sparse-segment fallback).  Sums over types give the stratified
projection per accident year and per horizon.  Reuses tools/cohort_cl.py and tools/cohort_cl_by_h.py.

Outputs chain_ladder_cohort_by_type_min<MIN>.csv, _by_h.csv and _by_dy.csv (per type: reserve, truth, error) into results_v2/ if it exists, else results/, and prints the totals against
the unstratified cohort chain ladder.      Usage: python tools/cohort_cl_by_type.py [MIN_CLAIMS]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cohort_cl as cc
from cohort_cl_by_h import projected_increments

MIN_CLAIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 500


def main():
    claims_all = pd.read_csv(os.path.join(cc.DATA, "claims_individual.csv"), usecols=["claim_no", "claim_type", "ibnr_flag"])
    claims, txn = cc.load(); pay = cc.pick_payment_column(txn); I = cc.I
    txn = txn.merge(claims_all[["claim_no", "claim_type"]], on="claim_no", how="left")
    rbns = claims_all[claims_all.ibnr_flag == 0]
    counts = rbns.claim_type.value_counts()
    own = [t for t, n in counts.items() if n >= MIN_CLAIMS]
    pooled = [t for t in counts.index if t not in own]
    rbns_sq_all = cc.square(txn, pay, txn.ibnr_flag == 0)
    rows, byh, byk = [], {}, {}
    for t in counts.index:
        m_t = (txn.claim_type == t) & (txn.ibnr_flag == 0)
        sq_t = cc.square(txn, pay, m_t)                      # the type's own RBNS square (truth in the lower triangle)
        for a in range(2, I + 1):
            c = I - a + 1
            cum_t = cc.observed_cumulative(cc.square(txn, pay, (txn.claim_type == t) & (txn.report_dy <= c)))
            if t in own:
                f = cc.cl_factors(cum_t)                       # own cohort factors
            else:
                f = cc.cl_factors(cc.observed_cumulative(cc.square(txn, pay, txn.report_dy <= c)))  # all-type fallback
            inc = projected_increments(cum_t, f, a)
            for k, v in inc.items():
                h = a + k - I - 1
                d = byh.setdefault(h, dict(pred=0.0, true=0.0)); d["pred"] += v; d["true"] += sq_t[a - 1, k - 1]
                dk = byk.setdefault(k, dict(pred=0.0, true=0.0)); dk["pred"] += v; dk["true"] += sq_t[a - 1, k - 1]
            pred = sum(inc.values()); true = sum(sq_t[a - 1, k - 1] for k in inc)
            rows.append(dict(claim_type=t, own_factors=t in own, accident_year=a, cl_cohort_type=pred, true_rbns=true))
    df = pd.DataFrame(rows)
    by_type = df.groupby(["claim_type", "own_factors"], as_index=False)[["cl_cohort_type", "true_rbns"]].sum()
    by_type["err_pct"] = (by_type.cl_cohort_type / by_type.true_rbns - 1) * 100
    OUT_DIR = os.path.join(cc.ROOT, "results_v2") if os.path.isdir(os.path.join(cc.ROOT, "results_v2")) else os.path.join(cc.ROOT, "results")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"chain_ladder_cohort_by_type_min{MIN_CLAIMS}.csv"); by_type.to_csv(out, index=False)
    tot_pred, tot_true = df.cl_cohort_type.sum(), df.true_rbns.sum()
    # unstratified cohort CL for reference
    ref = 0.0
    for a in range(2, I + 1):
        c = I - a + 1; cum_c = cc.observed_cumulative(cc.square(txn, pay, txn.report_dy <= c))
        ref += cc.cl_reserve_row(cum_c, cc.cl_factors(cum_c), a)
    print(f"own factors for {own} (>= {MIN_CLAIMS} claims); fallback for {pooled}")
    with pd.option_context("display.float_format", lambda v: f"{v:,.1f}"):
        show = by_type.copy(); show[["cl_cohort_type", "true_rbns"]] /= 1e6; print(show.to_string(index=False))
    print(f"stratified cohort CL: {tot_pred/1e6:.1f} M vs RBNS truth {tot_true/1e6:.1f} M -> {100*(tot_pred/tot_true-1):+.1f}%   "
          f"(unstratified cohort CL {ref/1e6:.1f} M, {100*(ref/tot_true-1):+.1f}%)")
    hh = pd.DataFrame([dict(h=h, pred_M=d['pred']/1e6, true_M=d['true']/1e6, err_pct=(d['pred']/d['true']-1)*100) for h, d in sorted(byh.items())])
    hh.to_csv(os.path.join(OUT_DIR, f"chain_ladder_cohort_by_type_min{MIN_CLAIMS}_by_h.csv"), index=False)
    print(hh.round(1).to_string(index=False))
    kk = pd.DataFrame([dict(dev_lag=k, pred_M=d['pred']/1e6, true_M=d['true']/1e6, err_M=(d['pred']-d['true'])/1e6) for k, d in sorted(byk.items())])
    kk.to_csv(os.path.join(OUT_DIR, f"chain_ladder_cohort_by_type_min{MIN_CLAIMS}_by_dy.csv"), index=False)
    print(f"DY 2-6 net {kk[kk.dev_lag<=6].err_M.sum():+.1f} M, DY 7-10 net {kk[kk.dev_lag>=7].err_M.sum():+.1f} M")


if __name__ == "__main__":
    main()
