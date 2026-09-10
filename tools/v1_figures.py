"""Regenerate the Paper-A (EAJ) figures that the two-paper split changes, from persisted v1 artefacts.

1. IBNR panels (ibnr_reserves_only.pdf, ibnr_combined.pdf, ibnr_error.pdf) with the reconciled
   truth (payments inside the 10 x 10 square; tools/v1_ibnr_restate.py) and the chain-ladder bars
   labelled for what the standard chain ladder projects -- the TOTAL (RBNS + IBNR) reserve -- instead
   of the first submission's "CL (RBNS only)".  ibnr_counts.pdf is unchanged (counts are unaffected).
   The plotting follows ibnr_model.plot_ibnr_results panel for panel.
2. Aggregate violin figures (violin_aggregate_next_year.pdf, violin_aggregate_ultimate.pdf) from the
   cached bootstrap draws (results/full_data_violin_cache.pkl) WITHOUT the 99.5 % VaR line: the
   per-tree dispersion is descriptive ensemble variability, not a one-year risk measure.
   The drawing follows plot_violin._render_plots panel for panel.

Inputs : results/v1_ibnr_restated.csv (run tools/v1_ibnr_restate.py first),
         results/full_data_violin_cache.pkl
Also writes results/violin_aggregate_summary.csv (true / RF / CL totals and the 2.5-97.5 percentile range of the draws).
Outputs: paper/eaj/figures/{ibnr_reserves_only,ibnr_combined,ibnr_error,
                            violin_aggregate_next_year,violin_aggregate_ultimate}.pdf
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "paper", "eaj", "figures")
HALF_W = 3.15  # half of the 6.3in text width, as in ibnr_model.py


def ibnr_panels(df: pd.DataFrame) -> list[str]:
    paths = []
    d = df[(df["true_ibnr_claims"] > 0) | (df["pred_ibnr_claims"] > 0.5)]
    labels = [f"AY {int(a)}" for a in d["accident_year"]]

    # Panel 2: IBNR reserves
    fig, ax = plt.subplots(figsize=(HALF_W, 3))
    x, w = np.arange(len(d)), 0.35
    ax.bar(x - w / 2, d["pred_ibnr_reserve"] / 1e6, w, label="RF Predicted", color="#4C72B0", alpha=0.8)
    ax.bar(x + w / 2, d["true_ibnr"] / 1e6, w, label="True", color="#2ca02c", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Reserve (millions)", fontsize=10); ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    p = os.path.join(FIG_DIR, "ibnr_reserves_only.pdf"); fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    paths.append(p)

    d2 = df[df["true_total"] > 0]
    labels2 = [f"AY {int(a)}" for a in d2["accident_year"]]

    # Panel 3: combined reserves
    fig, ax = plt.subplots(figsize=(HALF_W, 3))
    x, w = np.arange(len(d2)), 0.25
    ax.bar(x - w, d2["rbns_pred"] / 1e6, w, label="RF RBNS", color="#4C72B0", alpha=0.8)
    ax.bar(x - w, d2["ibnr_pred"] / 1e6, w, bottom=d2["rbns_pred"] / 1e6, label="RF IBNR", color="#ff7f0e", alpha=0.8)
    ax.bar(x, d2["true_total"] / 1e6, w, label="True Total", color="#2ca02c", alpha=0.8)
    ax.bar(x + w, d2["cl_standard"] / 1e6, w, label="CL (total)", color="#d62728", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels2, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Reserve (millions)", fontsize=10); ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    p = os.path.join(FIG_DIR, "ibnr_combined.pdf"); fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    paths.append(p)

    # Panel 4: error comparison
    fig, ax = plt.subplots(figsize=(HALF_W, 3))
    x, w = np.arange(len(d2)), 0.35
    ax.bar(x - w / 2, d2["total_err_pct"], w, label="RF (RBNS+IBNR)", color="#4C72B0", alpha=0.8)
    ax.bar(x + w / 2, d2["cl_err_pct"], w, label="CL (total)", color="#d62728", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels2, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Reserve Error (%)", fontsize=10); ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    p = os.path.join(FIG_DIR, "ibnr_error.pdf"); fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    paths.append(p)
    return paths


def violin_panels() -> list[str]:
    import plot_violin  # the v1 script; only its cache loader is used

    cache = plot_violin._load_violin_cache()
    ny, ult = cache["next_year_draws"], cache["ultimate_draws"]
    all_ays = cache["all_test_ays"]
    n_draws = cache["n_draws"]
    total_ny = np.zeros(n_draws)
    for ay in ny:
        total_ny += ny[ay]
    total_ult = np.zeros(n_draws)
    for ay in all_ays:
        total_ult += ult[ay]
    true_ny = sum(cache["true_next_year"].get(ay, 0.0) for ay in ny)
    true_ult = sum(cache["true_ultimate"].get(ay, 0.0) for ay in all_ays)
    cl_ny = sum(cache["cl_next_year"].get(ay, 0.0) for ay in ny)
    cl_ult = sum(cache["cl_reserves"].get(ay, 0.0) for ay in all_ays)
    specs = [
        (total_ny, true_ny, cl_ny, cache.get("total_rf_calib_ny"), "violin_aggregate_next_year.pdf"),
        (total_ult, true_ult, cl_ult, cache["total_rf_calib_ult"], "violin_aggregate_ultimate.pdf"),
    ]
    paths = []
    summary_rows = []
    for draws, true_val, cl_val, rf_val, fname in specs:
        fig, ax = plt.subplots(figsize=(6, 5))
        parts = ax.violinplot([draws], positions=[0], showmeans=False, showmedians=False,
                              showextrema=False, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0"); pc.set_alpha(0.55); pc.set_edgecolor("#2a4a7f"); pc.set_linewidth(0.8)
        lo, hi = -0.45, 0.45
        ci_lo, ci_hi = np.percentile(draws, [2.5, 97.5])
        ax.hlines([ci_lo, ci_hi], lo, hi, colors="#555555", linewidths=1.2, linestyles=":", zorder=4,
                  label=f"2.5th–97.5th percentiles of tree totals [{ci_lo/1e6:,.0f}M, {ci_hi/1e6:,.0f}M]")
        ax.hlines(true_val, lo, hi, colors="#2ca02c", linewidths=2.5, zorder=5, label=f"True ({true_val/1e6:,.0f}M)")
        if rf_val is not None:
            ax.hlines(rf_val, lo, hi, colors="#ff7f0e", linewidths=2.5, zorder=5,
                      label=f"RF calibrated ({rf_val/1e6:,.0f}M)")
        ax.hlines(cl_val, lo, hi, colors="#d62728", linewidths=2.5, linestyles="--", zorder=5,
                  label=f"CL ({cl_val/1e6:,.0f}M)")
        ax.set_ylabel("Reserve (millions)", fontsize=14)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v/1e6:,.0f}"))
        ax.set_xticks([]); ax.grid(axis="y", alpha=0.3, linestyle=":"); ax.tick_params(axis="y", labelsize=15)
        ax.legend(fontsize=11, loc="upper right", bbox_to_anchor=(1, 0.85))
        fig.tight_layout()
        p = os.path.join(FIG_DIR, fname); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        paths.append(p)
        print(f"  {fname}: true {true_val/1e6:,.1f}M | RF calibrated {rf_val/1e6 if rf_val is not None else float('nan'):,.1f}M"
              f" | CL {cl_val/1e6:,.1f}M | 95% [{ci_lo/1e6:,.1f}, {ci_hi/1e6:,.1f}]M | draws {n_draws}")
        summary_rows.append(dict(quantity="next_year" if "next_year" in fname else "ultimate", true=true_val,
                                 rf_calibrated=rf_val, cl=cl_val, **{"p2.5": ci_lo, "p97.5": ci_hi}, n_draws=n_draws))
    # deposit the numbers quoted in the paper (Section 7 ensemble-dispersion paragraph)
    out = os.path.join(RESULTS, "violin_aggregate_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    print("  Saved:", out)
    return paths


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS, "v1_ibnr_restated.csv"))
    for p in ibnr_panels(df) + violin_panels():
        print("  Saved:", p)


if __name__ == "__main__":
    sys.exit(main())
