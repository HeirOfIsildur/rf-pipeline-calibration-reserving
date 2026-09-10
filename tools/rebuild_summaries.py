"""
Rebuild the summary tables of a results directory from its per-config partials.

Long runs are launched in pieces (a crash, a re-run of a few configs, a new
objective added later), and each launch of main.py overwrites the root summary
tables with only the configs *it* ran.  The per-config partials in
<results>/partial/ are never overwritten, so the union of everything that has
finished lives there.  This script rebuilds

    model_comparison.csv          (+ full_data_ copy)
    correction_comparison.csv     (+ full_data_ copy)
    calibration_factors.csv       (+ full_data_ copy)
    all_models_reserves_by_ay.csv (+ full_data_ copy)
    chain_ladder_results.csv      (+ full_data_ copy)

from the partials, using the same evaluate.py functions main.py uses.  The
large predictions__*.csv copies are NOT re-emitted (they already exist in
partial/).

Usage (from the repository root, with the project interpreter):

    python tools/rebuild_summaries.py --results-dir results_v2
    python tools/rebuild_summaries.py --results-dir results_v2 --exclude hgb_poisson__per_dy__raw__include_zeros
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as cfg  # noqa: E402
from src.evaluate import (  # noqa: E402
    build_comparison_table,
    build_correction_table,
    collect_calibration_diagnostics,
    compute_aggregate_reserves,
)

LABEL_TO_OBJECTIVE = {
    # keyed by (method_label, target): method_label alone is ambiguous because
    # random_forest maps to both mse_raw and mse_log1p (same forest, different
    # target), so keying on the label alone silently collapses the two.
    (spec["method_label"], spec["target"]): (objective, spec["method"])
    for objective, spec in cfg.OBJECTIVE_SPECS.items()
}


def parse_name(name):
    """<method_label>__<structure>__<target>__<zero_handling> -> dict."""
    parts = name.split("__")
    if len(parts) != 4:
        raise ValueError(f"unexpected config name: {name}")
    label, structure, target, zero_handling = parts
    objective, method = LABEL_TO_OBJECTIVE.get((label, target), (None, label))
    return {
        "name": name,
        "objective": objective,
        "method": method,
        "structure": structure,
        "target": target,
        "zero_handling": zero_handling,
    }


def load_results(results_dir, exclude=()):
    partial = os.path.join(results_dir, "partial")
    files = sorted(glob.glob(os.path.join(partial, "predictions__*.csv")))
    if not files:
        sys.exit(f"no partial predictions found in {partial}")
    results = []
    for f in files:
        name = os.path.basename(f)[len("predictions__"):-len(".csv")]
        if name in exclude:
            print(f"  skip   {name}")
            continue
        r = parse_name(name)
        r["predictions"] = pd.read_csv(f)
        calib = os.path.join(partial, f"calibration__{name}.csv")
        r["calibration_diagnostics"] = pd.read_csv(calib) if os.path.exists(calib) else pd.DataFrame()
        r["runtime_seconds"] = np.nan
        results.append(r)
        print(f"  loaded {name}  ({len(r['predictions']):,} rows)")
    return results


def chain_ladder_results(results_dir):
    """Re-run chain ladder (fast) or fall back to the saved projected table."""
    try:
        from src.chain_ladder import run_chain_ladder

        return run_chain_ladder()
    except Exception as e:  # pragma: no cover - depends on data availability
        print(f"  [WARN] run_chain_ladder failed ({e}); rebuilding CL row from CSV")
        path = os.path.join(results_dir, "chain_ladder_results.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        return {
            "reserves_by_ay": df["CL_reserve"],
            "true_reserves_by_ay": df["true_reserve"],
            "projected_df": df,
        }


def emit(df, results_dir, fname, full_data=True):
    df.to_csv(os.path.join(results_dir, fname), index=False)
    if full_data:
        df.to_csv(os.path.join(results_dir, "full_data_" + fname), index=False)
    print(f"  wrote  {fname}" + ("  (+ full_data_ copy)" if full_data else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="config names to leave out")
    ap.add_argument("--no-full-data-copy", action="store_true")
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    cfg.RESULTS_DIR = results_dir
    full = not args.no_full_data_copy

    print(f"Rebuilding summaries in {results_dir}")
    all_results = load_results(results_dir, exclude=set(args.exclude))
    cl = chain_ladder_results(results_dir)

    comparison = build_comparison_table(all_results, cl)
    emit(comparison, results_dir, "model_comparison.csv", full)

    corr = build_correction_table(all_results)
    if not corr.empty:
        emit(corr, results_dir, "correction_comparison.csv", full)

    diag = collect_calibration_diagnostics(all_results)
    if not diag.empty:
        emit(diag, results_dir, "calibration_factors.csv", full)

    if cl is not None and "projected_df" in cl:
        emit(cl["projected_df"], results_dir, "chain_ladder_results.csv", full)

    ay_rows = []
    for r in all_results:
        if r["predictions"].empty:
            continue
        res = compute_aggregate_reserves(r["predictions"])
        res["model_name"] = r["name"]
        ay_rows.append(res)
    if ay_rows:
        emit(pd.concat(ay_rows, ignore_index=True), results_dir, "all_models_reserves_by_ay.csv", full)

    with pd.option_context("display.width", 220, "display.max_columns", 40, "display.max_rows", 200):
        print("\nTotal reserve error (%) — uncorrected pipeline output per config:")
        print(comparison[["name", "total_reserve_error_pct", "max_ay_error_pct", "rmse", "mae"]].round(2).to_string(index=False))
        if not corr.empty:
            pivot = corr.pivot(index="name", columns="arm", values="total_reserve_error_pct")
            print("\nTotal reserve error (%) by correction arm:")
            print(pivot.round(1).to_string())


if __name__ == "__main__":
    main()
