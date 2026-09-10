"""Extract the calibration factors actually applied in the first-submission (v1) flagship run.

The submitted paper never showed the realised calibration factors (referee R2-S16).  The v1 pipeline
stored them on each per-DY regressor as `pipeline_calibration_factors_`: a dict {'__global__': f,
<claim type>: f, ...} of the SHRUNK and CLIPPED factors that multiply the pipeline output
(factor = 1 + Z (raw - 1), clipped to [0.5, 5.0]; see src/models.py at tag astin-submission,
_compute_pipeline_calibration).  The raw cross-validated ratios and the credibility weight Z were not
persisted and cannot be recovered without re-running the calibration CV; this script therefore
reports exactly what was applied, nothing more.

Conventions of the v1 code that show in the table
  * a claim type with fewer than 10 training rows at a DY falls back to the global factor of that DY;
  * a type whose out-of-sample predictions sum to zero gets raw ratio 1 (factor 1.0);
  * a type with predictions but no positive payments has raw ratio 0, hence factor 1 - Z;
  * DY 10 is trained on a single observed accident year, so the CV cannot form two groups and the
    factor is 1 (no calibration).

Inputs : results/persisted_models__random_forest__per_dy__log1p__two_stage.pkl
Outputs: results/v1_calibration_factors.csv
         paper/eaj/tables/table_factors.tex   (label tab:factors)
"""
from __future__ import annotations

import os
import pickle
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # sklearn objects only; no project classes are needed to unpickle

PKL = os.path.join(ROOT, "results", "persisted_models__random_forest__per_dy__log1p__two_stage.pkl")
CSV_OUT = os.path.join(ROOT, "results", "v1_calibration_factors.csv")
TEX_OUT = os.path.join(ROOT, "paper", "eaj", "tables", "table_factors.tex")
GLOBAL = "__global__"
CLIP = (0.5, 5.0)
TYPES = ["Material", "Technical", "Bodily", "Annuity", "Lost", "Remaining"]


def load_factors() -> pd.DataFrame:
    with open(PKL, "rb") as f:
        models = pickle.load(f)
    rows = []
    for dy in sorted(models):
        mdl = models[dy]["model"]
        reg = mdl[1] if isinstance(mdl, (tuple, list)) else mdl
        fac = getattr(reg, "pipeline_calibration_factors_", None)
        if fac is None:
            raise SystemExit(f"DY {dy}: the persisted regressor carries no pipeline_calibration_factors_")
        row = {"dev_lag": dy, "global": fac[GLOBAL]}
        for t in TYPES:
            row[t] = fac.get(t)  # None when the type was not calibrated at all (DY 10)
        rows.append(row)
    return pd.DataFrame(rows)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Mark clipped values (dagger) and global fall-backs (asterisk)."""
    out = df.copy()
    for t in TYPES:
        out[t + "_flag"] = ""
        for i, r in df.iterrows():
            v = r[t]
            if v is None or pd.isna(v):
                out.at[i, t + "_flag"] = "none"
            elif abs(v - CLIP[0]) < 1e-9 or abs(v - CLIP[1]) < 1e-9:
                out.at[i, t + "_flag"] = "clipped"
            elif abs(v - r["global"]) < 1e-9 and r["dev_lag"] != 10:
                out.at[i, t + "_flag"] = "global-fallback"
    return out


def to_tex(df: pd.DataFrame) -> str:
    def cell(v, flag):
        if flag == "none":
            return "---"
        s = f"{v:.2f}"
        if flag == "clipped":
            s += r"$^{\dagger}$"
        elif flag == "global-fallback":
            s += r"$^{*}$"
        return s

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Realised calibration factors of the leading configuration (per-DY, log1p, two-stage)",
        r"in the first-submission run: the shrunk and clipped factors $\tilde\alpha$ that multiply the",
        r"pipeline output, by development year and claim type, as stored with the fitted models.  The",
        r"raw cross-validated ratios and the shrinkage weights were not persisted.",
        r"$^{\dagger}$: at a clip bound ($0.5$ or $5$).  $^{*}$: fewer than ten training rows of that type,",
        r"global factor of the development year applied.  DY~10 is trained on a single observed accident",
        r"year, so no cross-validated ratio exists and the factor is~$1$.}",
        r"\label{tab:factors}",
        r"\small",
        r"\begin{tabular}{c" + "r" * (1 + len(TYPES)) + "}",
        r"\toprule",
        r"\textbf{DY} & \textbf{Global} & " + " & ".join(rf"\textbf{{{t}}}" for t in TYPES) + r" \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        cells = [f"{int(r.dev_lag)}", f"{r['global']:.2f}"] + [cell(r[t], r[t + '_flag']) for t in TYPES]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    df = annotate(load_factors())
    df.to_csv(CSV_OUT, index=False)
    os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)
    with open(TEX_OUT, "w") as f:
        f.write(to_tex(df))
    print(df[["dev_lag", "global"] + TYPES].to_string(index=False, float_format="%.3f"))
    print("\nclip-bound hits:")
    for _, r in df.iterrows():
        hits = [f"{t}={r[t]:.1f}" for t in TYPES if r[t + "_flag"] == "clipped"]
        if hits:
            print(f"  DY {int(r.dev_lag)}: " + ", ".join(hits))
    print("global fall-backs (<10 rows):")
    for _, r in df.iterrows():
        fb = [t for t in TYPES if r[t + "_flag"] == "global-fallback"]
        if fb:
            print(f"  DY {int(r.dev_lag)}: " + ", ".join(fb))
    print(f"\nwritten: {CSV_OUT}\n         {TEX_OUT}")


if __name__ == "__main__":
    sys.exit(main())
