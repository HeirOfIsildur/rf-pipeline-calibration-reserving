#!/usr/bin/env python3
"""Per-horizon decomposition of the two chain ladders (standard and report-cohort-consistent).

Reuses tools/cohort_cl.py. For each accident year a the projected cumulative path is expanded into incremental
payments per development year k; horizon h = a + k - 11. Outputs results_v2/chain_ladder_by_h.csv with, per h:
standard-CL projection vs the TOTAL truth, cohort-CL projection vs the RBNS truth (M and %).
Usage: python tools/cohort_cl_by_h.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cohort_cl as cc

def projected_increments(cum_obs, factors, a):
    """Projected incremental payments of accident year a for k = I-a+2 .. I (lower triangle)."""
    I = cc.I; last_k = I - a + 1
    c = cum_obs[a - 1, last_k - 1]; inc = {}
    for j in range(last_k - 1, I - 1):
        nxt = c * factors[j]; inc[j + 2] = nxt - c; c = nxt   # k = j+2 is the DY index (1-based) reached
    return inc

def main():
    claims, txn = cc.load(); pay = cc.pick_payment_column(txn); I = cc.I
    total_sq = cc.square(txn, pay); rbns_sq = cc.square(txn, pay, txn.ibnr_flag == 0)
    cum_obs = cc.observed_cumulative(total_sq); f_std = cc.cl_factors(cum_obs)
    rows = []
    for a in range(2, I + 1):
        std = projected_increments(cum_obs, f_std, a)
        c = I - a + 1; sq_c = cc.square(txn, pay, txn.report_dy <= c); cum_c = cc.observed_cumulative(sq_c)
        coh = projected_increments(cum_c, cc.cl_factors(cum_c), a)
        for k in std:
            rows.append(dict(accident_year=a, dev_lag=k, h=a + k - I - 1, cl_standard=std[k], cl_cohort=coh[k],
                             true_total=total_sq[a - 1, k - 1], true_rbns=rbns_sq[a - 1, k - 1]))
    df = pd.DataFrame(rows)
    byh = df.groupby("h")[["cl_standard", "cl_cohort", "true_total", "true_rbns"]].sum()
    byh["err_std_vs_total_pct"] = (byh.cl_standard / byh.true_total - 1) * 100
    byh["err_cohort_vs_rbns_pct"] = (byh.cl_cohort / byh.true_rbns - 1) * 100
    byh["err_std_vs_rbns_pct"] = (byh.cl_standard / byh.true_rbns - 1) * 100
    out = os.path.join(cc.ROOT, "results_v2" if os.path.isdir(os.path.join(cc.ROOT, "results_v2")) else "results", "chain_ladder_by_h.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True); byh.to_csv(out)
    T = byh.sum(numeric_only=True)
    print(f"check totals (M): std {T.cl_standard/1e6:.1f} (675.0) cohort {T.cl_cohort/1e6:.1f} (515.5) truth total {T.true_total/1e6:.1f} rbns {T.true_rbns/1e6:.1f}")
    with pd.option_context("display.float_format", lambda v: f"{v:,.1f}", "display.width", 160):
        show = byh.copy()
        for col in ["cl_standard", "cl_cohort", "true_total", "true_rbns"]: show[col] /= 1e6
        print(show)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
