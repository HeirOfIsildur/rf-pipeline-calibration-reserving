"""
Post-hoc balance corrections for a fitted prediction pipeline.

Every correction here is a cheap function of three things the training code
already produces for each fitted pipeline:

  * cross-fitted (out-of-fold) predictions on the training cells,
  * in-sample predictions of the final model on the training cells,
  * uncorrected predictions on the test cells (plus, for two-stage
    pipelines, the payment probability and the regressor output).

Corrections are therefore emitted as additional prediction columns
(``pred__<arm>``) instead of separate experiment configurations: one fitted
pipeline yields every arm.  Notation follows the revised manuscript
(paper/theory.tex):

  alpha_0 = E[Y] / E[f(X)]           calibration functional (Def. 3.1)
  alpha'  = E[1+Y] / E[1+f(X)]       shifted functional, applied as
                                     alpha' (1 + f) - 1 (Prop. 3.5)
  alpha_j fold ratios, s^2 their variance; v = s^2 / K estimates
          Var(alpha_hat) (Thm 3.6 (iii)); delta method as fallback
  Z_k = a / (a + v_k)                Buhlmann-Straub weight (Thm 3.8)

Arms
----
  none            uncorrected pipeline output
  paper           first-submission method: per-type cross-fitted ratio,
                  heuristic Z = min(n_eff / max(n_eff s^2, 50), 1), clip [0.5, 5]
                  (identity for raw targets)
  insample_global ratio of the final model on its own training cells
  cv_global       cross-fitted ratio, unshrunk, unclipped
  cv_dy           per-development-year ratio, unshrunk
  cv_dy_heur      per-DY ratio with the heuristic Z (clipped)
  cv_dy_bs1       per-DY ratio, Buhlmann-Straub toward 1
  cv_dy_bsmean    per-DY ratio, Buhlmann-Straub toward the credibility mean
  cv_type         per-claim-type ratio, unshrunk
  cv_type_bs1     per-claim-type ratio, Buhlmann-Straub toward 1
  shift_global    shifted ratio on 1 + Y, global
  shift_dy_bs1    shifted ratio per DY, Buhlmann-Straub toward 1
  duan            Duan (1983) smearing on the regressor output (log1p only;
                  equals `none` for raw targets)
  iso_global      isotonic recalibration of the out-of-fold predictions
                  (Wuthrich & Ziegel, 2024)

When a correction cannot be computed (no calibration record, too few
groups, raw target for Duan) the arm equals `none`, and the diagnostics
table records why.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.config import (
    CALIB_CLIP,
    CALIB_K_FLOOR,
    CALIB_MIN_GROUP_N,
    CALIB_MIN_GROUP_POS,
    GLOBAL_CALIB_KEY,
)

ARMS = [
    "none",
    "paper",
    "insample_global",
    "cv_global",
    "cv_dy",
    "cv_dy_heur",
    "cv_dy_bs1",
    "cv_dy_bsmean",
    "cv_dy_zcv",
    "cv_type",
    "cv_type_bs1",
    "shift_global",
    "shift_dy_bs1",
    "duan",
    "iso_global",
]


# -----------------------------------------------------------------------
# Elementary statistics
# -----------------------------------------------------------------------


def _safe_ratio(num, den):
    den = float(den)
    return float(num) / den if den > 0 else np.nan


def ratio(y, pred):
    """alpha_hat = sum(y) / sum(pred)."""
    return _safe_ratio(np.nansum(y), np.nansum(pred))


def shifted_ratio(y, pred):
    """alpha'_hat = sum(1 + y) / sum(1 + pred)."""
    return _safe_ratio(np.nansum(1.0 + y), np.nansum(1.0 + pred))


def fold_ratios(y, pred, fold_id, shifted=False):
    """Per-fold ratios alpha_j (folds with non-positive prediction sums are skipped)."""
    out = []
    for f in np.unique(fold_id[fold_id >= 0]):
        m = fold_id == f
        r = shifted_ratio(y[m], pred[m]) if shifted else ratio(y[m], pred[m])
        if np.isfinite(r):
            out.append(r)
    return np.asarray(out, dtype=float)


def between_fold_variance(ratios):
    """s^2 of the fold ratios (ddof = 1); NaN with fewer than two folds."""
    return float(np.var(ratios, ddof=1)) if len(ratios) >= 2 else np.nan


def heuristic_Z(n_eff, s2, floor=CALIB_K_FLOOR):
    """First-submission rule: Z = min(n_eff / max(n_eff s^2, floor), 1)."""
    s2 = 0.0 if not np.isfinite(s2) else float(s2)
    n_rel = max(int(n_eff * s2), floor)
    return float(min(n_eff / n_rel, 1.0)) if n_rel > 0 else 1.0


def delta_variance(y, pred, alpha):
    """Var(alpha_hat) by the delta method: Var(Y - alpha f) / (n mean(f)^2) (Thm 3.6 (i))."""
    n = len(y)
    m = np.nanmean(pred) if n else np.nan
    if n < 2 or not np.isfinite(m) or m <= 0 or not np.isfinite(alpha):
        return np.nan
    return float(np.nanvar(y - alpha * pred, ddof=1) / (n * m * m))


def delta_variance_clustered(y, pred, alpha, cluster):
    """
    Claim-clustered delta-method variance of alpha_hat: the residuals
    e_i = y_i - alpha f_i are summed within each cluster (claim) before the
    sandwich, Var = G/(G-1) * sum_c S_c^2 / (sum_i f_i)^2, so cells of the same
    claim may be arbitrarily correlated.  Equals delta_variance up to the
    finite-sample factor when every cluster holds one cell.
    """
    if cluster is None:
        return np.nan
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 2 or not np.isfinite(alpha):
        return np.nan
    e = y[ok] - alpha * pred[ok]
    _, inv = np.unique(np.asarray(cluster)[ok], return_inverse=True)
    S = np.bincount(inv, weights=e)
    G = len(S)
    denom = float(np.sum(pred[ok]))
    if G < 2 or denom <= 0:
        return np.nan
    return float(G / (G - 1) * np.sum(S * S) / (denom * denom))


def ratio_variance(y, pred, fold_id, alpha, shifted=False):
    """
    Var(alpha_hat) estimate.  Primary: the delta-method estimate of Thm 3.6 (i),
    which uses every row.  The fold-based estimate s^2 / K_eff of Thm 3.6 (iii)
    is returned alongside as a diagnostic: with K = 3 folds it has relative
    standard deviation 1 and is far too noisy to drive the credibility weight.
    When the delta estimate is unavailable the fold estimate is used.

    Returns (v_hat, s2, K_eff, v_fold, v_delta).
    """
    fr = fold_ratios(y, pred, fold_id, shifted=shifted)
    s2 = between_fold_variance(fr)
    v_fold = s2 / len(fr) if (np.isfinite(s2) and len(fr) >= 2) else np.nan
    yy = 1.0 + y if shifted else y
    pp = 1.0 + pred if shifted else pred
    v_delta = delta_variance(yy, pp, alpha)
    v_hat = v_delta if np.isfinite(v_delta) else v_fold
    return v_hat, s2, int(len(fr)), v_fold, v_delta


# -----------------------------------------------------------------------
# Credibility (Thm 3.8) and the first-submission factors
# -----------------------------------------------------------------------


def bs_weights(alpha, v, theta=1.0, n_iter=5):
    """
    Buhlmann-Straub weights Z_k = a / (a + v_k) for group factors alpha_k with
    within-group variances v_k.  The structural variance a is the moment
    estimator max{0, mean((alpha_k - theta)^2) - mean(v_k)} (unbiased before
    truncation, Thm 3.8 plug-in); with theta=None the centre is the
    credibility-weighted mean, iterated, and one degree of freedom is lost.

    Returns (Z array aligned with alpha, a_hat, theta_used).
    """
    alpha = np.asarray(alpha, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(alpha) & np.isfinite(v) & (v >= 0)
    Z = np.full(alpha.shape, np.nan)
    if ok.sum() == 0:
        return Z, np.nan, theta
    al, vv = alpha[ok], v[ok]
    G = len(al)
    estimate_centre = theta is None
    if estimate_centre:
        w = 1.0 / np.maximum(vv, 1e-12)
        th = float(np.sum(w * al) / np.sum(w))
    else:
        th = float(theta)
    Zk = np.zeros_like(vv)
    a_hat = np.nan
    for _ in range(n_iter if estimate_centre else 1):
        ss = float(np.sum((al - th) ** 2))
        dof = (G - 1) if (estimate_centre and G > 1) else G
        a_hat = max(0.0, ss / dof - float(np.mean(vv)))
        Zk = a_hat / (a_hat + vv) if a_hat > 0 else np.zeros_like(vv)
        if estimate_centre and Zk.sum() > 0:
            th = float(np.sum(Zk * al) / np.sum(Zk))
    Z[ok] = Zk
    return Z, a_hat, th


def zcv_scalar(records_by_group, z_grid=None):
    """
    R2-S11: one shrinkage weight Z for the per-group factors, chosen by
    cross-validation on the aggregate balance criterion.  For each fold j the
    per-group ratios are estimated from the other folds and applied, shrunk by
    Z toward 1, to fold j's cross-fitted predictions; Z minimises the summed
    squared aggregate error over folds.  Returns Z (NaN when not estimable).
    """
    z_grid = np.linspace(0.0, 1.0, 21) if z_grid is None else np.asarray(z_grid)
    groups = [(g, r) for g, r in records_by_group.items() if record_is_estimable(r)]
    if not groups:
        return np.nan
    folds = sorted({int(f) for _, r in groups for f in np.unique(r["fold_id"]) if f >= 0})
    if len(folds) < 2:
        return np.nan
    err = np.zeros(len(z_grid))
    for j in folds:
        pred_j = np.zeros(len(z_grid))
        true_j = 0.0
        for _, r in groups:
            y, oos, fid = r["y"], r["oos_pred"], r["fold_id"]
            m_in = fid == j
            m_out = (fid >= 0) & (fid != j)
            if not m_in.any() or not m_out.any():
                continue
            den = oos[m_out].sum()
            a = y[m_out].sum() / den if den > 0 else 1.0
            s_in = oos[m_in].sum()
            true_j += y[m_in].sum()
            pred_j += (1.0 + z_grid * (a - 1.0)) * s_in
        err += (pred_j - true_j) ** 2
    return float(z_grid[int(np.argmin(err))])


def zcv_from_record(record, key="dev_lag"):
    """zcv_scalar for a single record, grouped by record[key]."""
    if record is None or record.get(key) is None:
        return np.nan
    groups = {}
    keys = np.asarray(record[key])
    for g in np.unique(keys):
        m = keys == g
        if m.sum() < CALIB_MIN_GROUP_N or (record["y"][m] > 0).sum() < CALIB_MIN_GROUP_POS:
            continue
        groups[g] = {"y": record["y"][m], "oos_pred": record["oos_pred"][m],
                     "fold_id": record["fold_id"][m]}
    return zcv_scalar(groups)


def add_zcv_column(table, z):
    """factor 1 + Z (alpha - 1) per group; identity when Z is NaN."""
    if table is None or table.empty:
        return table
    zz = 0.0 if not np.isfinite(z) else float(z)
    table["Z_zcv"] = z
    table["factor_cv_zcv"] = 1.0 + zz * (table["alpha_cv"] - 1.0)
    return table


def shrink(alpha, Z, theta):
    """theta + Z (alpha - theta), elementwise; NaN Z leaves alpha unchanged."""
    alpha = np.asarray(alpha, dtype=float)
    Z = np.asarray(Z, dtype=float)
    return np.where(np.isfinite(Z), theta + Z * (alpha - theta), alpha)


def clip_factor(f, bounds=CALIB_CLIP):
    return float(np.clip(f, bounds[0], bounds[1])) if np.isfinite(f) else np.nan


def paper_factors(record):
    """
    The first-submission calibration: cross-fitted global ratio shrunk by the
    heuristic Z toward 1 and clipped to CALIB_CLIP, with per-claim-type factors
    (same global Z, groups of at least CALIB_MIN_GROUP_N rows) and the global
    factor as fallback.  Returns {GLOBAL_CALIB_KEY: factor, type: factor, ...}.
    """
    if record is None:
        return {GLOBAL_CALIB_KEY: 1.0}
    y, oos, fid = record["y"], record["oos_pred"], record["fold_id"]
    n_eff = int((y > 0).sum())
    fr = fold_ratios(y, oos, fid)
    s2 = between_fold_variance(fr) if len(fr) > 1 else 0.0
    Z = heuristic_Z(n_eff, s2)

    def _shrunk(yy, pp):
        raw = ratio(yy, pp)
        raw = 1.0 if not np.isfinite(raw) else raw
        return clip_factor(1.0 + Z * (raw - 1.0))

    factors = {GLOBAL_CALIB_KEY: _shrunk(y, oos)}
    ct = record.get("claim_type")
    if ct is not None:
        types = np.unique(ct)
        if len(types) > 1:
            for t in types:
                m = ct == t
                factors[t] = (
                    _shrunk(y[m], oos[m])
                    if m.sum() >= CALIB_MIN_GROUP_N
                    else factors[GLOBAL_CALIB_KEY]
                )
    return factors


# -----------------------------------------------------------------------
# Group summaries and credibility tables
# -----------------------------------------------------------------------


def summarize_groups(record, key=None):
    """
    One row per group of record[key] (key in {"dev_lag", "claim_type", None}),
    plus a GLOBAL_CALIB_KEY row, holding the cross-fitted statistics every arm
    needs: alpha_cv, s2, K_eff, v_cv, alpha_shift, v_shift, n, n_eff, Z_heur.
    Groups smaller than CALIB_MIN_GROUP_N or without positive payments are
    dropped (the global row serves as fallback).
    """
    y, oos, fid = record["y"], record["oos_pred"], record["fold_id"]
    cid = record.get("claim_id")

    def _row(name, m):
        a = ratio(y[m], oos[m])
        v, s2, K, v_fold, v_delta = ratio_variance(y[m], oos[m], fid[m], a)
        v_delta_cl = delta_variance_clustered(y[m], oos[m], a, cid[m]) if cid is not None else np.nan
        n_cl = int(len(np.unique(cid[m]))) if cid is not None else int(m.sum())
        ash = shifted_ratio(y[m], oos[m])
        vsh, _, _, _, _ = ratio_variance(y[m], oos[m], fid[m], ash, shifted=True)
        n_eff = int((y[m] > 0).sum())
        return {
            "group": name,
            "n": int(m.sum()),
            "n_eff": n_eff,
            "K_eff": K,
            "alpha_cv": a,
            "s2": s2,
            "v_cv": v,
            "v_cv_fold": v_fold,
            "v_cv_delta": v_delta,
            "v_cv_delta_cl": v_delta_cl,
            "n_clusters": n_cl,
            "alpha_shift": ash,
            "v_shift": vsh,
            "Z_heur": heuristic_Z(n_eff, s2),
        }

    rows = [_row(GLOBAL_CALIB_KEY, np.ones(len(y), dtype=bool))]
    groups = record.get(key) if key is not None else None
    if groups is not None:
        for g in np.unique(groups):
            m = groups == g
            if m.sum() >= CALIB_MIN_GROUP_N and (y[m] > 0).sum() >= CALIB_MIN_GROUP_POS:
                rows.append(_row(g, m))
    return pd.DataFrame(rows)


def record_is_estimable(record):
    """A record can carry a calibration factor only with enough positive cells."""
    return record is not None and int((record["y"] > 0).sum()) >= CALIB_MIN_GROUP_POS


def combine_global_rows(records_by_group):
    """
    Build a group summary from the global rows of several records, e.g. the ten
    per-development-year models: {dy: record} -> one row per dy.
    """
    rows = []
    for g, rec in records_by_group.items():
        if not record_is_estimable(rec):
            continue
        r = summarize_groups(rec, None).iloc[0].to_dict()
        r["group"] = g
        rows.append(r)
    return pd.DataFrame(rows)


_CRED_COLS = [
    "factor_cv_heur", "Z_bs1", "a_bs1", "factor_cv_bs1",
    "Z_bsmean", "a_bsmean", "theta_bsmean", "factor_cv_bsmean",
    "Z_shift_bs1", "a_shift_bs1", "factor_shift_bs1",
    "Z_bs1_cl", "a_bs1_cl", "factor_cv_bs1_cl",
]


def credibility_table(summary):
    """
    Add the shrunk factors to a group summary (rows other than the global one):
    factor_cv_heur (first-submission rule, clipped), factor_cv_bs1 and
    factor_cv_bsmean (Thm 3.8 toward 1 / toward the credibility mean), and
    factor_shift_bs1 for the shifted ratio.  Also records the Z's, a_hat, theta.
    """
    if summary is None or summary.empty:
        return summary
    df = summary.copy()
    for col in _CRED_COLS:
        df[col] = np.nan
    grp = df[df["group"] != GLOBAL_CALIB_KEY]
    if grp.empty:
        return df
    idx = grp.index
    a_cv = grp["alpha_cv"].values.astype(float)
    v_cv = grp["v_cv"].values.astype(float)
    df.loc[idx, "factor_cv_heur"] = [
        clip_factor(1.0 + z * (a - 1.0)) if np.isfinite(a) else np.nan
        for a, z in zip(a_cv, grp["Z_heur"].values)
    ]
    Z1, a1, _ = bs_weights(a_cv, v_cv, theta=1.0)
    df.loc[idx, "Z_bs1"] = Z1
    df.loc[idx, "a_bs1"] = a1
    df.loc[idx, "factor_cv_bs1"] = shrink(a_cv, Z1, 1.0)
    # claim-clustered variance where the record carries claim ids (falls back to v_cv)
    v_cl = grp["v_cv_delta_cl"].values.astype(float) if "v_cv_delta_cl" in grp else np.full(len(idx), np.nan)
    v_used = np.where(np.isfinite(v_cl), v_cl, v_cv)
    Zc, ac, _ = bs_weights(a_cv, v_used, theta=1.0)
    df.loc[idx, "Z_bs1_cl"] = Zc
    df.loc[idx, "a_bs1_cl"] = ac
    df.loc[idx, "factor_cv_bs1_cl"] = shrink(a_cv, Zc, 1.0)
    Zm, am, thm = bs_weights(a_cv, v_cv, theta=None)
    df.loc[idx, "Z_bsmean"] = Zm
    df.loc[idx, "a_bsmean"] = am
    df.loc[idx, "theta_bsmean"] = thm
    df.loc[idx, "factor_cv_bsmean"] = shrink(a_cv, Zm, thm if (thm is not None and np.isfinite(thm)) else 1.0)
    a_sh = grp["alpha_shift"].values.astype(float)
    Zs, as_, _ = bs_weights(a_sh, grp["v_shift"].values.astype(float), theta=1.0)
    df.loc[idx, "Z_shift_bs1"] = Zs
    df.loc[idx, "a_shift_bs1"] = as_
    df.loc[idx, "factor_shift_bs1"] = shrink(a_sh, Zs, 1.0)
    return df


# -----------------------------------------------------------------------
# Duan smearing and isotonic recalibration
# -----------------------------------------------------------------------


def duan_smearing_factor(record):
    """S = mean(exp(log(1+y) - z_hat)) over out-of-fold positive cells (log1p only)."""
    if record is None or record.get("oos_reg_log") is None:
        return np.nan
    y, lg = record["y"], record["oos_reg_log"]
    m = np.isfinite(lg) & (y > 0)
    if m.sum() < 2:
        return np.nan
    return float(np.mean(np.exp(np.log1p(y[m]) - lg[m])))


def fit_isotonic(record, min_rows=10):
    """Isotonic regression of y on the out-of-fold prediction (clipped outside the range)."""
    if record is None:
        return None
    y, oos = record["y"], record["oos_pred"]
    m = np.isfinite(oos)
    if m.sum() < min_rows or np.nanstd(oos[m]) == 0:
        return None
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(oos[m], y[m])


# -----------------------------------------------------------------------
# Applying the arms to test predictions
# -----------------------------------------------------------------------


def _one(x):
    return float(x) if np.isfinite(x) else 1.0


def _lookup(table, keys, col, fallback):
    """Factor per test row from table[col] by group, with a scalar fallback."""
    keys = np.asarray(keys)
    if table is None or table.empty or col not in table.columns:
        return np.full(len(keys), fallback, dtype=float)
    mapping = dict(zip(table["group"].tolist(), table[col].astype(float).tolist()))
    out = np.array([mapping.get(k, np.nan) for k in keys.tolist()], dtype=float)
    return np.where(np.isfinite(out), out, fallback)


def apply_shift(pred, factor):
    """alpha' (1 + f) - 1, floored at zero."""
    return np.maximum(np.asarray(factor, dtype=float) * (1.0 + pred) - 1.0, 0.0)


def arm_predictions(record, comps, dy_table, type_table, test_dy, test_type):
    """
    Corrected test predictions for every arm in ARMS, plus an info dict.

    record     calibration record of the fitted pipeline (may be None)
    comps      dict from models.predict(..., return_components=True):
               pred (paper arm), pred_base, prob_pos, reg_out
    dy_table   credibility table whose 'group' values are development years
    type_table credibility table whose 'group' values are claim types, or None
    test_dy    development year per test row
    test_type  claim type per test row, or None
    """
    base = np.asarray(comps["pred_base"], dtype=float)
    n = len(base)
    out = {"none": base.copy(), "paper": np.asarray(comps["pred"], dtype=float)}
    info = {"duan_S": np.nan, "isotonic": False, "record": record is not None, "estimable": False}

    if record is None:
        for a in ARMS:
            out.setdefault(a, base.copy())
        return out, info

    y, oos = record["y"], record["oos_pred"]
    estimable = record_is_estimable(record)
    info["estimable"] = estimable
    a_g = _one(ratio(y, oos)) if estimable else 1.0
    ash_g = _one(shifted_ratio(y, oos)) if estimable else 1.0
    test_dy = np.asarray(test_dy)

    ins = record.get("insample_pred")
    out["insample_global"] = base * (_one(ratio(y, ins)) if ins is not None else 1.0)
    out["cv_global"] = base * a_g
    out["cv_dy"] = base * _lookup(dy_table, test_dy, "alpha_cv", a_g)
    out["cv_dy_heur"] = base * _lookup(dy_table, test_dy, "factor_cv_heur", a_g)
    out["cv_dy_bs1"] = base * _lookup(dy_table, test_dy, "factor_cv_bs1", a_g)
    out["cv_dy_bsmean"] = base * _lookup(dy_table, test_dy, "factor_cv_bsmean", a_g)
    out["cv_dy_zcv"] = base * _lookup(dy_table, test_dy, "factor_cv_zcv", a_g)
    if type_table is not None and test_type is not None:
        tt = np.asarray(test_type)
        out["cv_type"] = base * _lookup(type_table, tt, "alpha_cv", a_g)
        out["cv_type_bs1"] = base * _lookup(type_table, tt, "factor_cv_bs1", a_g)
    else:
        out["cv_type"] = base * a_g
        out["cv_type_bs1"] = base * a_g
    out["shift_global"] = apply_shift(base, ash_g)
    out["shift_dy_bs1"] = apply_shift(base, _lookup(dy_table, test_dy, "factor_shift_bs1", ash_g))

    # Duan smearing needs the stage-2 regressor of a two-stage pipeline: with
    # zero-inflated targets the log-scale residuals are bimodal and the smearing
    # factor is meaningless (observed S ~ 10^3 in include_zeros fits).
    S = duan_smearing_factor(record) if record.get("zero_handling") == "two_stage" else np.nan
    reg_out = comps.get("reg_out")
    if estimable and np.isfinite(S) and reg_out is not None and record.get("target_type") == "log1p":
        ro = np.asarray(reg_out, dtype=float)
        prob = comps.get("prob_pos")
        prob = np.ones(n) if prob is None else np.asarray(prob, dtype=float)
        amt = np.where(np.isfinite(ro), np.maximum(np.exp(np.nan_to_num(ro)) * S - 1.0, 0.0), 0.0)
        out["duan"] = prob * amt
        info["duan_S"] = S
    else:
        out["duan"] = base.copy()

    iso = fit_isotonic(record) if estimable else None
    if iso is not None:
        out["iso_global"] = np.asarray(iso.predict(base), dtype=float)
        info["isotonic"] = True
    else:
        out["iso_global"] = base.copy()

    return out, info


def diagnostics_rows(name, level, table, extra=None):
    """Long-format diagnostics: one row per group of a credibility table."""
    if table is None or table.empty:
        return pd.DataFrame()
    df = table.copy()
    df.insert(0, "level", level)
    df.insert(0, "name", name)
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df
