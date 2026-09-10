"""Restate the first-submission IBNR and combined-reserve tables on the reconciled truth.

Why.  The first submission (ibnr_model.py) measured the IBNR truth as the claim-level ultimate
`claim_size` of the late-reported claims, which includes payments falling outside the 10 x 10
evaluation square, while the RBNS truth is the sum of payments inside the square.  The reconciled
definition used by both papers is: truth = payments inside the square, for RBNS and IBNR alike
(tools/cohort_cl.py; results_v2/chain_ladder_cohort.csv).  The differences are small (AY 9: 13.41 M vs
12.98 M; AY 10: 60.81 M vs 60.74 M) but they must not leave two papers quoting two totals.

Inputs
  results/ibnr_reserves_by_ay.csv            v1 IBNR frequency-severity model: observed_claims,
                                             pred_ibnr_claims, true_ibnr_claims, pred_ibnr_reserve
  results/combined_rbns_ibnr_reserves.csv    v1 combined table: rbns_pred, ibnr_pred (the same
                                             predictions), rbns_true
  results_v2/chain_ladder_cohort.csv         cl_standard (total-reserve projection), cl_rbns_cohort,
                                             cl_ibnr_implied, true_rbns, true_total, true_ibnr
Outputs
  results/v1_ibnr_restated.csv
  paper/eaj/tables/table_ibnr.tex            label tab:ibnr
  paper/eaj/tables/table_combined.tex        label tab:combined
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(ROOT, "paper", "eaj", "tables")


def load() -> pd.DataFrame:
    ibnr = pd.read_csv(os.path.join(RESULTS, "ibnr_reserves_by_ay.csv"))
    comb = pd.read_csv(os.path.join(RESULTS, "combined_rbns_ibnr_reserves.csv"))
    cl_path = os.path.join(ROOT, "results_v2", "chain_ladder_cohort.csv")
    if not os.path.exists(cl_path):  # deposited layout of the public repository
        cl_path = os.path.join(RESULTS, "chain_ladder_cohort.csv")
    cl = pd.read_csv(cl_path)
    df = ibnr[["accident_year", "observed_claims", "pred_ibnr_claims", "true_ibnr_claims", "pred_ibnr_reserve"]]
    df = df.merge(comb[["accident_year", "rbns_pred", "rbns_true", "ibnr_pred"]], on="accident_year")
    df = df.merge(cl[["accident_year", "cl_standard", "cl_rbns_cohort", "cl_ibnr_implied",
                      "true_rbns", "true_total", "true_ibnr"]], on="accident_year")
    assert np.allclose(df["ibnr_pred"], df["pred_ibnr_reserve"])
    assert np.allclose(df["rbns_true"], df["true_rbns"], atol=1.0), "RBNS truth differs between v1 and cohort files"
    df["total_pred"] = df["rbns_pred"] + df["ibnr_pred"]
    df["ibnr_err_pct"] = np.where(df["true_ibnr"] > 0, 100 * (df["pred_ibnr_reserve"] / df["true_ibnr"] - 1), np.nan)
    df["total_err_pct"] = np.where(df["true_total"] > 0, 100 * (df["total_pred"] / df["true_total"] - 1), np.nan)
    df["cl_err_pct"] = np.where(df["true_total"] > 0, 100 * (df["cl_standard"] / df["true_total"] - 1), np.nan)
    dist_path = os.path.join(RESULTS, "ibnr_severity_distribution.csv")
    if os.path.exists(dist_path):  # Monte Carlo percentiles of the IBNR draws (5th / 95th)
        dist = pd.read_csv(dist_path)
        df = df.merge(dist[["accident_year", "ibnr_p5", "ibnr_p95"]], on="accident_year", how="left")
    return df


def fmt_pct(x: float) -> str:
    return "---" if pd.isna(x) else f"${x:+.1f}$"


def fmt_m(x: float) -> str:
    return f"{x / 1e6:.1f}"


def fmt_iv(r) -> str:
    """[p5, p95] in millions, or --- when the percentiles are not deposited."""
    if "ibnr_p5" not in r or pd.isna(r.ibnr_p5):
        return "---"
    return f"[{r.ibnr_p5 / 1e6:.1f}, {r.ibnr_p95 / 1e6:.1f}]"


def fmt_int(x: float) -> str:
    """Integer with LaTeX thin thousands separators (2801 -> 2{,}801)."""
    return f"{int(round(x)):,}".replace(",", "{,}")


def table_ibnr(df: pd.DataFrame) -> str:
    d = df[df["accident_year"] >= 7]
    lines = [
        r"\begin{table}[tb]",
        r"\centering",
        r"\caption{IBNR reserve estimates by accident year: chain-ladder claim counts, random-forest",
        r"severities and Monte Carlo point estimates (mean of the draws) with the 90\% simulation interval",
        r"(5th and 95th percentiles of the $M=200$ draws; Section~\ref{sec:ibnr:bootstrap}), against the true IBNR payments",
        r"inside the evaluation square.  The last column is the IBNR implied by the standard chain",
        r"ladder, i.e.\ its total-reserve projection less its report-cohort-consistent RBNS projection.",
        r"Accident years 1--6 have no late-reported claims.  Amounts in millions.}",
        r"\label{tab:ibnr}",
        r"\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{crrrrrrrr}",
        r"\toprule",
        r"\textbf{AY} & \textbf{Observed} & \multicolumn{2}{c}{\textbf{IBNR claims}} & \multicolumn{4}{c}{\textbf{IBNR reserve (M)}} & \textbf{CL-implied} \\",
        r" & \textbf{claims} & pred. & true & pred. & 90\% interval & true & error\,\% & \textbf{IBNR (M)} \\",
        r"\midrule",
    ]
    for _, r in d.iterrows():
        lines.append(
            f"{int(r.accident_year)} & {fmt_int(r.observed_claims)} & {r.pred_ibnr_claims:.1f} & {int(r.true_ibnr_claims)}"
            f" & {fmt_m(r.pred_ibnr_reserve)} & {fmt_iv(r)} & {fmt_m(r.true_ibnr)} & {fmt_pct(r.ibnr_err_pct)} & {fmt_m(r.cl_ibnr_implied)} \\\\"
        )
    tot_pred, tot_true = df["pred_ibnr_reserve"].sum(), df["true_ibnr"].sum()
    lines += [
        r"\midrule",
        f"Total & {fmt_int(df.observed_claims.sum())} & {df.pred_ibnr_claims.sum():.1f} & {int(df.true_ibnr_claims.sum())}"
        f" & {fmt_m(tot_pred)} & --- & {fmt_m(tot_true)} & {fmt_pct(100 * (tot_pred / tot_true - 1))} & {fmt_m(df.cl_ibnr_implied.sum())} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    return "\n".join(lines)


def table_combined(df: pd.DataFrame) -> str:
    d = df[df["true_total"] > 0]
    lines = [
        r"\begin{table}[tb]",
        r"\centering",
        r"\caption{Combined RBNS + IBNR reserves by accident year (leading random-forest configuration),",
        r"set against the true total payments inside the evaluation square and against the standard chain ladder,",
        r"whose projection is a total (RBNS + IBNR) reserve and is therefore compared with the total truth.",
        r"Amounts in millions.}",
        r"\label{tab:combined}",
        r"\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{crrrrrrr}",
        r"\toprule",
        r"\textbf{AY} & \textbf{RF RBNS} & \textbf{RF IBNR} & \textbf{RF total} & \textbf{True total} & \textbf{Error\,\%} & \textbf{CL (total)} & \textbf{CL error\,\%} \\",
        r"\midrule",
    ]
    for _, r in d.iterrows():
        lines.append(
            f"{int(r.accident_year)} & {fmt_m(r.rbns_pred)} & {fmt_m(r.ibnr_pred)} & {fmt_m(r.total_pred)} & {fmt_m(r.true_total)}"
            f" & {fmt_pct(r.total_err_pct)} & {fmt_m(r.cl_standard)} & {fmt_pct(r.cl_err_pct)} \\\\"
        )
    T = df.sum(numeric_only=True)
    lines += [
        r"\midrule",
        f"Total & {fmt_m(T.rbns_pred)} & {fmt_m(T.ibnr_pred)} & {fmt_m(T.total_pred)} & {fmt_m(T.true_total)}"
        f" & {fmt_pct(100 * (T.total_pred / T.true_total - 1))} & {fmt_m(T.cl_standard)} & {fmt_pct(100 * (T.cl_standard / T.true_total - 1))} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    return "\n".join(lines)


def main() -> None:
    df = load()
    df.to_csv(os.path.join(RESULTS, "v1_ibnr_restated.csv"), index=False)
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "table_ibnr.tex"), "w") as f:
        f.write(table_ibnr(df))
    with open(os.path.join(TABLES, "table_combined.tex"), "w") as f:
        f.write(table_combined(df))

    T = df.sum(numeric_only=True)
    print(f"RBNS truth {T.true_rbns/1e6:.3f} M | IBNR truth {T.true_ibnr/1e6:.3f} M | total truth {T.true_total/1e6:.3f} M")
    print(f"IBNR share of total truth      : {100*T.true_ibnr/T.true_total:.1f} %")
    print(f"RF IBNR vs true IBNR           : {T.pred_ibnr_reserve/1e6:.1f} vs {T.true_ibnr/1e6:.1f} M  ({100*(T.pred_ibnr_reserve/T.true_ibnr-1):+.1f} %)")
    print(f"RF RBNS vs RBNS truth          : {T.rbns_pred/1e6:.1f} vs {T.true_rbns/1e6:.1f} M  ({100*(T.rbns_pred/T.true_rbns-1):+.1f} %)")
    print(f"RF total vs total truth        : {T.total_pred/1e6:.1f} vs {T.true_total/1e6:.1f} M  ({100*(T.total_pred/T.true_total-1):+.1f} %)")
    print(f"standard CL vs total truth     : {T.cl_standard/1e6:.1f} M  ({100*(T.cl_standard/T.true_total-1):+.1f} %)")
    print(f"standard CL vs RBNS truth      : ({100*(T.cl_standard/T.true_rbns-1):+.1f} %)   [the first submission's mismatched comparison]")
    print(f"cohort RBNS CL vs RBNS truth   : {T.cl_rbns_cohort/1e6:.1f} M  ({100*(T.cl_rbns_cohort/T.true_rbns-1):+.1f} %)")
    print(f"CL-implied IBNR vs true IBNR   : {T.cl_ibnr_implied/1e6:.1f} vs {T.true_ibnr/1e6:.1f} M  ({100*(T.cl_ibnr_implied/T.true_ibnr-1):+.1f} %)")
    print("per-AY IBNR (AY 7-10): pred / true (M):",
          "; ".join(f"AY{int(r.accident_year)} {r.pred_ibnr_reserve/1e6:.2f}/{r.true_ibnr/1e6:.2f} ({r.ibnr_err_pct:+.1f}%)"
                    for _, r in df[df.accident_year >= 7].iterrows()))


if __name__ == "__main__":
    sys.exit(main())
