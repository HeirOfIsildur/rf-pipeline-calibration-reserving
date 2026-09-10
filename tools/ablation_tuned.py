#!/usr/bin/env python3
"""Re-run the three robustness checks of Paper A (Sect. 7 'Robustness refits', Sect. S5) on the TUNED flagship.

The earlier numbers (results_v2/ablation_flagship.csv) were measured at one early hyperparameter draw.  This script uses
the hyperparameters persisted with the deployed per-DY models of results_v2/ (the completed tuned recomputation):

  1. deployed models  - reproduces the `none` arm of results_v2 (a check), then sweeps the gate threshold tau over
                        {0, 0.005, 0.01, 0.02, 0.05, 0.1} on the deployed classifier/regressor outputs;
  2. baseline_refit   - refits every per-DY pipeline with the persisted hyperparameters (should reproduce `none`);
  3. no_interactions  - the same refit without the five interaction features;
  4. unlimited_depth  - the same refit with max_depth=None for classifier and regressor (and regressor-only variant).

Every number is the uncorrected lower-triangle total (no calibration), as in the earlier ablation; the horizon-1 error
and the DY 2-6 / DY 7-10 net errors are added so the 'decomposition unchanged' sentence can be checked.

Usage:  python tools/ablation_tuned.py [--results-dir results_v2]   ->  <results-dir>/ablation_flagship_tuned.csv
"""
from __future__ import annotations
import argparse, os, pickle, sys, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import src.config as cfg                                   # noqa: E402
from src.data_prep import prepare_data, get_split_masks    # noqa: E402
from src.feature_engineering import engineer_features      # noqa: E402
from src import models as models_mod                       # noqa: E402

FLAGSHIP = "random_forest__per_dy__log1p__two_stage"
INTERACTIONS = ["cum_paid_x_notidel", "dev_lag_x_cum_paid", "notidel_x_dev_lag", "log_cum_paid_x_dev_lag", "velocity_x_dev_lag"]
TAUS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]


def summarise(df, pred_col):
    """Total, horizon-1 and DY-band errors of a lower-triangle prediction frame."""
    t, p = df["true"].sum(), df[pred_col].sum()
    h = df["accident_year"] + df["dev_lag"] - cfg.MAX_ACCIDENT_YEAR - 1
    h1 = df[h == 1]
    dy26 = df[df["dev_lag"].between(2, 6)]; dy710 = df[df["dev_lag"] >= 7]
    return dict(total_err_pct=100 * (p / t - 1), h1_err_pct=100 * (h1[pred_col].sum() / h1["true"].sum() - 1),
                dy26_net_M=(dy26[pred_col].sum() - dy26["true"].sum()) / 1e6,
                dy710_net_M=(dy710[pred_col].sum() - dy710["true"].sum()) / 1e6, n_test=len(df))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-dir", default="results_v2"); args = ap.parse_args()
    rdir = os.path.join(ROOT, args.results_dir)
    t0 = time.time()
    claim_dy, claims_rbns, transactions_obs, _ = prepare_data()
    claim_dy, feature_cols, feature_cols_per_dy = engineer_features(claim_dy, transactions_obs)
    # reporting-consistent panel and prediction cap, exactly as src/train.py:run_single_experiment
    obs_mask = (claim_dy["accident_year"] + claim_dy["dev_lag"] - 1) <= cfg.MAX_ACCIDENT_YEAR
    pre = obs_mask & (claim_dy["dev_lag"] < claim_dy["report_dy"])
    print(f"excluding {int(pre.sum()):,} pre-reporting cells", flush=True)
    claim_dy = claim_dy.loc[~pre].reset_index(drop=True)
    observed = (claim_dy["accident_year"] + claim_dy["dev_lag"] - 1) <= cfg.MAX_ACCIDENT_YEAR
    models_mod.reset_prediction_cap(float(claim_dy.loc[observed, "incremental_payment"].max()))
    print(f"panel ready in {time.time()-t0:.0f}s; cap {models_mod.PRED_CAP:,.2f}", flush=True)

    persisted = pickle.load(open(os.path.join(rdir, f"persisted_models__{FLAGSHIP}.pkl"), "rb"))
    ref = pd.read_csv(os.path.join(rdir, "partial", f"predictions__{FLAGSHIP}.csv"),
                      usecols=["claim_no", "accident_year", "dev_lag", "true_incremental_payment", "pred__none"])
    rows = []

    # ---- 1. deployed models: reproduce `none`, sweep tau -------------------------------------------------------
    parts = []
    for dy, item in sorted(persisted.items()):
        _, _, te = get_split_masks(claim_dy, dy, structure="per_dy"); test = claim_dy[te]
        if len(test) == 0: continue
        clf, reg = item["model"]; X = test[item["feat_cols"]].values
        proba = clf.predict_proba(X)
        prob = (np.zeros(len(X)) if clf.classes_[0] == 0 else np.ones(len(X))) if proba.shape[1] == 1 else proba[:, 1]
        amt = models_mod._inverse_transform_target(np.asarray(reg.predict(X), dtype=float), "log1p")
        parts.append(pd.DataFrame({"claim_no": test["claim_no"].values, "accident_year": test["accident_year"].values,
                                   "dev_lag": test["dev_lag"].values, "true": test["true_incremental_payment"].values,
                                   "prob": prob, "amt": amt}))
    D = pd.concat(parts, ignore_index=True)
    D["deployed"] = D["prob"] * D["amt"] * (D["prob"] > 0.01)
    m = D.merge(ref, on=["claim_no", "accident_year", "dev_lag"], how="inner")
    print(f"deployed vs results_v2 pred__none: rows {len(m):,}/{len(D):,}, max |diff| {np.nanmax(np.abs(m['deployed']-m['pred__none'])):.3e}, "
          f"max |true diff| {np.nanmax(np.abs(m['true']-m['true_incremental_payment'])):.3e}", flush=True)
    rows.append(dict(variant="deployed (tau=0.01)", **summarise(D, "deployed"), minutes=(time.time() - t0) / 60))
    for tau in TAUS:
        col = f"tau_{tau}"; D[col] = D["prob"] * D["amt"] * (D["prob"] > tau)
        r = summarise(D, col); r["gated_share_pct"] = 100 * float((D["prob"] <= tau).mean())
        rows.append(dict(variant=f"deployed tau={tau}", **r, minutes=np.nan))

    # ---- 2.-4. refits with the persisted hyperparameters ------------------------------------------------------
    def refit(variant, feat_fn, reg_fn, clf_fn):
        t1 = time.time(); parts = []
        for dy, item in sorted(persisted.items()):
            tr, va, te = get_split_masks(claim_dy, dy, structure="per_dy")
            train, val, test = claim_dy[tr], claim_dy[va], claim_dy[te]
            if len(train) == 0 or len(test) == 0: continue
            feat = feat_fn(item["feat_cols"])
            val_in_train = cfg.get_val_ay_for_dy(dy) in cfg.get_train_ays_for_dy(dy)
            full = train if (val_in_train or len(val) == 0) else pd.concat([train, val])
            model = models_mod.train_final_model("random_forest", full[feat].values, full["incremental_payment"].values,
                                                 reg_fn(dict(item["best_params"])), clf_fn(dict(item["best_clf_params"])),
                                                 "log1p", "two_stage", compute_record=False)
            base, _, _ = models_mod._pipeline_components(model, test[feat].values, "log1p", "two_stage")
            parts.append(pd.DataFrame({"accident_year": test["accident_year"].values, "dev_lag": test["dev_lag"].values,
                                       "true": test["true_incremental_payment"].values, "pred": base}))
            print(f"  {variant} DY{dy}: {len(full):,} train rows, {len(feat)} features, done", flush=True)
        R = pd.concat(parts, ignore_index=True)
        r = dict(variant=variant, **summarise(R, "pred"), minutes=(time.time() - t1) / 60); rows.append(r)
        print(f"{variant}: total {r['total_err_pct']:+.3f}%  h1 {r['h1_err_pct']:+.3f}%  DY2-6 {r['dy26_net_M']:+.1f}M  DY7-10 {r['dy710_net_M']:+.1f}M  ({r['minutes']:.1f} min)", flush=True)

    ident = lambda p: p
    nodepth = lambda p: {**p, "max_depth": None}
    refit("baseline_refit", ident, ident, ident)
    refit("no_interactions", lambda f: [c for c in f if c not in INTERACTIONS], ident, ident)
    refit("unlimited_depth", ident, nodepth, nodepth)
    refit("unlimited_depth_regressor_only", ident, nodepth, ident)

    out = pd.DataFrame(rows); path = os.path.join(rdir, "ablation_flagship_tuned.csv"); out.to_csv(path, index=False)
    print(out.round(4).to_string(index=False)); print("wrote", path); print("ABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
