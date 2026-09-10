"""Decompose the first-submission (v1, ASTIN) total reserve error by development year and by horizon.

The +3.9 % total of the flagship configuration is a net of opposite errors: over-prediction at
development years 2-6 and under-prediction at development years 7-10 (referee R1-A18).  This script
quantifies that for every v1 configuration and for the chain ladder, from the persisted v1 artefacts
only (no model is re-run).

Inputs (all in results/, the frozen v1 run)
  predictions__random_forest__<structure>__<target>__<zeros>.csv
      lower-triangle claim x DY rows: claim_no, accident_year, dev_lag,
      true_incremental_payment, predicted_payment   (AY 2..10, DY 2..10, AY + DY - 1 > 10)
  chain_ladder_results.csv
      cumulative chain-ladder projection per accident year: accident_year, DY1..DY10, CL_reserve, ...
      (upper-triangle cells are observed cumulatives, lower-triangle cells the projection)
  model_comparison.csv
      totals used as a consistency check and for the Table-5 ordering

Outputs
  results/v1_decomposition_by_dy.csv     net error (M) per method x development year, plus DY bands
  results/v1_decomposition_by_h.csv      predicted, true, error (M, %) per method x horizon h
  paper/eaj/tables/table_decomposition.tex   booktabs table for Paper A (label tab:decomposition)

Horizon h = accident_year + dev_lag - (MAX_AY + 1) is the number of calendar years after the valuation
date at which the cell is paid (h = 1 is the next calendar year).  The chain-ladder row compares the
standard chain ladder (a total-reserve projection) with the RBNS truth, exactly as the first
submission's Table 5 did; its like-for-like comparison is the cohort-consistent chain ladder reported
separately (tools/cohort_cl.py).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABLE_OUT = os.path.join(ROOT, "paper", "eaj", "tables", "table_decomposition.tex")
MAX_AY = 10

# Table-5 order of the first submission (rank by |total error|) and its labels
CONFIGS = [
    ("random_forest__per_dy__log1p__two_stage", r"RF per\_dy + log1p + two\_stage"),
    ("random_forest__single_model__log1p__include_zeros", r"RF single + log1p + incl.\ zeros"),
    ("random_forest__per_dy__raw__two_stage", r"RF per\_dy + raw + two\_stage"),
    ("random_forest__per_dy__raw__include_zeros", r"RF per\_dy + raw + incl.\ zeros"),
    ("random_forest__per_dy__log1p__include_zeros", r"RF per\_dy + log1p + incl.\ zeros"),
    ("chain_ladder", r"Chain Ladder (standard)"),
    ("random_forest__single_model__log1p__two_stage", r"RF single + log1p + two\_stage"),
    ("random_forest__single_model__raw__two_stage", r"RF single + raw + two\_stage"),
    ("random_forest__single_model__raw__include_zeros", r"RF single + raw + incl.\ zeros"),
]


def _cells_rf(name: str) -> pd.DataFrame:
    """Predicted and true payments per lower-triangle (AY, DY) cell for an RF configuration."""
    df = pd.read_csv(os.path.join(RESULTS, f"predictions__{name}.csv"))
    cy = df["accident_year"] + df["dev_lag"] - 1
    assert (cy > MAX_AY).all(), f"{name}: rows outside the lower triangle"
    g = (
        df.groupby(["accident_year", "dev_lag"], as_index=False)
        .agg(pred=("predicted_payment", "sum"), true=("true_incremental_payment", "sum"))
    )
    return g


def _cells_cl(truth_cells: pd.DataFrame) -> pd.DataFrame:
    """Chain-ladder incremental projection per lower-triangle cell, joined with the RBNS truth."""
    cl = pd.read_csv(os.path.join(RESULTS, "chain_ladder_results.csv"))
    rows = []
    for _, r in cl.iterrows():
        ay = int(r["accident_year"])
        cum = np.array([r[f"DY{j}"] for j in range(1, MAX_AY + 1)], dtype=float)
        inc = np.diff(np.concatenate([[0.0], cum]))
        for dy in range(2, MAX_AY + 1):
            if ay + dy - 1 > MAX_AY:
                rows.append({"accident_year": ay, "dev_lag": dy, "pred": inc[dy - 1]})
    pred = pd.DataFrame(rows)
    out = pred.merge(truth_cells, on=["accident_year", "dev_lag"], how="left")
    out["true"] = out["true"].fillna(0.0)
    total_cl = float(cl["CL_reserve"].sum())
    assert abs(out["pred"].sum() - total_cl) < 1.0, (out["pred"].sum(), total_cl)
    return out


def decompose(cells: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Net error by development year (Series indexed by DY) and by horizon (DataFrame)."""
    cells = cells.copy()
    cells["h"] = cells["accident_year"] + cells["dev_lag"] - (MAX_AY + 1)
    by_dy = (cells["pred"] - cells["true"]).groupby(cells["dev_lag"]).sum()
    by_h = cells.groupby("h", as_index=False).agg(pred=("pred", "sum"), true=("true", "sum"))
    by_h["err"] = by_h["pred"] - by_h["true"]
    by_h["err_pct"] = 100.0 * by_h["err"] / by_h["true"]
    return by_dy, by_h


def main() -> None:
    mc = pd.read_csv(os.path.join(RESULTS, "model_comparison.csv")).set_index("name")
    truth_cells = None
    rows_dy, rows_h, table_rows = [], [], []

    for name, label in CONFIGS:
        if name == "chain_ladder":
            cells = _cells_cl(truth_cells)
        else:
            cells = _cells_rf(name)
            if truth_cells is None:
                truth_cells = cells[["accident_year", "dev_lag", "true"]].copy()
            else:  # the truth is the same for every configuration
                chk = cells.merge(truth_cells, on=["accident_year", "dev_lag"], suffixes=("", "_ref"))
                assert np.allclose(chk["true"], chk["true_ref"]), name

        total_pred, total_true = cells["pred"].sum(), cells["true"].sum()
        # consistency with the first submission's totals
        ref = mc.loc[name]
        assert abs(total_pred - ref["total_predicted_reserve"]) < 1.0, (name, total_pred, ref["total_predicted_reserve"])
        assert abs(total_true - ref["total_true_reserve"]) < 1.0

        by_dy, by_h = decompose(cells)
        d26 = by_dy.loc[2:6].sum() / 1e6
        d710 = by_dy.loc[7:10].sum() / 1e6
        tot_pct = 100.0 * (total_pred / total_true - 1.0)
        h1 = by_h.set_index("h")
        h3plus_pred = h1.loc[3:, "pred"].sum()
        h3plus_true = h1.loc[3:, "true"].sum()

        rows_dy.append({"method": name, **{f"DY{int(k)}_M": v / 1e6 for k, v in by_dy.items()},
                        "DY2_6_M": d26, "DY7_10_M": d710, "total_M": (total_pred - total_true) / 1e6,
                        "total_pct": tot_pct})
        for _, r in by_h.iterrows():
            rows_h.append({"method": name, "h": int(r["h"]), "pred_M": r["pred"] / 1e6, "true_M": r["true"] / 1e6,
                           "err_M": r["err"] / 1e6, "err_pct": r["err_pct"],
                           "share_of_true_pct": 100.0 * r["true"] / total_true})
        table_rows.append({
            "label": label, "total_pct": tot_pct, "d26": d26, "d710": d710,
            "h1": h1.loc[1, "err_pct"], "h2": h1.loc[2, "err_pct"],
            "h3p": 100.0 * (h3plus_pred / h3plus_true - 1.0),
        })

        if name in ("random_forest__per_dy__log1p__two_stage", "random_forest__single_model__log1p__include_zeros"):
            print(f"\n{name}: total {tot_pct:+.2f} %  |  DY 2-6 {d26:+.1f} M, DY 7-10 {d710:+.1f} M")
            print(by_h.assign(pred_M=lambda d: d.pred / 1e6, true_M=lambda d: d.true / 1e6,
                              share_pct=lambda d: 100 * d.true / total_true)
                  [["h", "pred_M", "true_M", "err_pct", "share_pct"]].to_string(index=False, float_format="%.1f"))

    pd.DataFrame(rows_dy).to_csv(os.path.join(RESULTS, "v1_decomposition_by_dy.csv"), index=False)
    pd.DataFrame(rows_h).to_csv(os.path.join(RESULTS, "v1_decomposition_by_h.csv"), index=False)

    # --- LaTeX table -------------------------------------------------------------------------------
    def pct(x: float) -> str:
        return f"${x:+.1f}$"

    def mil(x: float) -> str:
        return f"${x:+.1f}$"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Decomposition of the total RBNS reserve error of the first-submission run by development",
        r"year band and by horizon $h$ (calendar years after the valuation date).  Net errors in millions;",
        r"percentages relative to the true payments of the band.  The near-zero total of the leading",
        r"configuration is a net of opposite errors: over-prediction at development years 2--6 against",
        r"under-prediction at development years 7--10, and, by horizon, an almost exact one-year reserve",
        r"($h=1$) against large errors further out.  The chain-ladder row is the standard chain ladder",
        r"compared with the RBNS truth, as in Table~\ref{tab:model_comparison}; see",
        r"Section~\ref{sec:results} for its like-for-like comparison.}",
        r"\label{tab:decomposition}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{Total\,\%} & \textbf{DY 2--6 (M)} & \textbf{DY 7--10 (M)}",
        r"  & $\boldsymbol{h=1}$\,\% & $\boldsymbol{h=2}$\,\% & $\boldsymbol{h\ge 3}$\,\% \\",
        r"\midrule",
    ]
    for r in table_rows:
        lines.append(
            f"{r['label']} & {pct(r['total_pct'])} & {mil(r['d26'])} & {mil(r['d710'])}"
            f" & {pct(r['h1'])} & {pct(r['h2'])} & {pct(r['h3p'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(TABLE_OUT), exist_ok=True)
    with open(TABLE_OUT, "w") as f:
        f.write("\n".join(lines))

    print("\nTable rows:")
    for r in table_rows:
        print(f"  {r['label']:<40s} tot {r['total_pct']:+7.1f}  DY2-6 {r['d26']:+7.1f}  DY7-10 {r['d710']:+7.1f}"
              f"  h1 {r['h1']:+7.1f}  h2 {r['h2']:+7.1f}  h>=3 {r['h3p']:+7.1f}")
    print(f"\nwritten: {TABLE_OUT}")


if __name__ == "__main__":
    sys.exit(main())
