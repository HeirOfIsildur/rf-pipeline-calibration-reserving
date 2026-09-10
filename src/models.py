"""
Model definitions, Optuna hyperparameter tuning, and prediction logic.

Supports ML methods (random forest), two-stage models, and log1p target transforms.

Includes pipeline-level calibration for log-retransformation bias:
when training on log1p(Y), naive expm1(prediction) underestimates E[Y|X]
because of Jensen's inequality.  We correct this via cross-validated
ratio calibration computed on the *full* two-stage pipeline output
(classifier soft-gating × regressor expm1), so the factor accounts for
variance absorbed by the classifier stage.

The calibration approach draws on several established methods:

- **Balance property correction** (Wüthrich, 2020; Wüthrich & Merz, 2023):
  The ratio Σy/Σŷ restores the balance property — ensuring aggregate
  predictions match aggregate actuals — a standard post-hoc correction
  for models that do not satisfy it inherently (tree ensembles, NNs).
  Ref: Wüthrich, M.V. (2020). "Bias regularization in neural network
  models for general insurance pricing." European Actuarial Journal, 10,
  179-202.

- **Ratio estimation for log-retransformed costs** (Manning & Mullahy, 2001):
  In health economics, the same ratio Σy/Σŷ is standard practice for
  recalibrating exponentiated predictions from log-linear cost models.
  Ref: Manning, W.G. & Mullahy, J. (2001). "Estimating log models: to
  transform or not to transform?" Journal of Health Economics, 20(4),
  461-494.

- **Bühlmann credibility shrinkage** (Bühlmann, 1967):
  The calibration factor is shrunk toward 1.0 (the prior of "no
  correction needed") using a linear credibility formula, with the
  credibility weight derived from effective sample size.  This is the
  actuarial analogue of James-Stein shrinkage (James & Stein, 1961).
  Ref: Bühlmann, H. (1967). "Experience Rating and Credibility."
  ASTIN Bulletin, 4(3), 199-207.

Note: an earlier version used Duan's (1983) nonparametric smearing
estimate (mean of exponentiated residuals).  The current pipeline-level
ratio approach supersedes it — it calibrates the *full* two-stage
pipeline output rather than regressor residuals alone, avoiding the
overcorrection that arises when the classifier absorbs variance.
Ref: Duan, N. (1983). "Smearing Estimate: A Nonparametric
Retransformation Method." JASA, 78(383), 605-610.
"""

import numpy as np
import gc
import os
import optuna
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    RandomForestClassifier,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import KFold, GroupKFold
import warnings

from src.config import (
    HP_SPACES,
    hp_classifier,
    RANDOM_STATE,
    OPTUNA_N_TRIALS,
    GLOBAL_CALIB_KEY,
    OBJECTIVE_SPECS,
    CALIB_N_FOLDS,
    CALIB_CV_N_ESTIMATORS_CAP,
)
from src import corrections

try:  # optional: Tweedie objective via LightGBM
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    HAS_LIGHTGBM = False

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Quick mode flag — set from main.py to reduce subsample size during tuning
QUICK_MODE = False

# Cap parallel jobs to avoid OOM on machines with many cores but limited RAM
_N_JOBS = min(4, os.cpu_count() or 4)


# -----------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------


def build_regressor(method, params, objective=None):
    """
    Build a regressor instance for the given method and hyperparameters.

    Parameters
    ----------
    method : str
        One of: cart, bagging, random_forest, adaboost, ridge, hgb, lightgbm
    params : dict
        Hyperparameters (from Optuna or defaults).
    objective : str or None
        Key of config.OBJECTIVE_SPECS; its reg_kwargs (split criterion or
        loss) are merged into the parameters.

    Returns
    -------
    sklearn-compatible regressor
    """
    p = params.copy()
    if objective is not None:
        p.update(OBJECTIVE_SPECS[objective].get("reg_kwargs", {}))

    if method == "cart":
        return DecisionTreeRegressor(random_state=RANDOM_STATE, **p)

    elif method == "bagging":
        base_params = {
            k.replace("base__", ""): v for k, v in p.items() if k.startswith("base__")
        }
        bag_params = {k: v for k, v in p.items() if not k.startswith("base__")}
        base_tree = DecisionTreeRegressor(random_state=RANDOM_STATE, **base_params)
        return BaggingRegressor(
            estimator=base_tree,
            random_state=RANDOM_STATE,
            n_jobs=1,
            **bag_params,
        )

    elif method == "random_forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=_N_JOBS, **p)

    elif method == "adaboost":
        base_params = {
            k.replace("base__", ""): v for k, v in p.items() if k.startswith("base__")
        }
        ada_params = {k: v for k, v in p.items() if not k.startswith("base__")}
        base_tree = DecisionTreeRegressor(random_state=RANDOM_STATE, **base_params)
        return AdaBoostRegressor(
            estimator=base_tree,
            random_state=RANDOM_STATE,
            **ada_params,
        )

    elif method == "ridge":
        return Ridge(**p)

    elif method == "hgb":
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE, **p)

    elif method == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed")
        if p.get("objective") != "tweedie":
            p.pop("tweedie_variance_power", None)  # tuned by hp_lightgbm; Tweedie-only
        return LGBMRegressor(random_state=RANDOM_STATE, n_jobs=_N_JOBS, verbosity=-1, **p)

    else:
        raise ValueError(f"Unknown method: {method}")


def build_classifier(params):
    """Build a RandomForestClassifier for the two-stage model."""
    return RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=_N_JOBS,
        **params,
    )


def fit_regressor(reg, X, y_t):
    """
    Fit `reg` and return it; if a deviance-based objective rejects the target
    (all-zero payments: Poisson/Gamma need sum(y) > 0, Gamma needs y > 0), fall
    back to a constant predictor equal to the target mean.  This happens for
    late development years on small training sets and must not fail a config.
    """
    try:
        return reg.fit(X, y_t)
    except ValueError as e:
        msg = str(e)
        if "requires non-negative y" in msg or "sum(y) > 0" in msg or "strictly positive" in msg or "y > 0" in msg:
            from sklearn.dummy import DummyRegressor

            return DummyRegressor(strategy="mean").fit(X, y_t)
        raise


# -----------------------------------------------------------------------
# Default parameters (used when Optuna is skipped or as fallback)
# -----------------------------------------------------------------------

DEFAULT_PARAMS = {
    "cart": {
        "max_depth": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": None,
    },
    "bagging": {
        "n_estimators": 100,
        "max_samples": 0.8,
        "max_features": 0.8,
        "base__max_depth": 15,
        "base__min_samples_leaf": 5,
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
    "adaboost": {"n_estimators": 200, "learning_rate": 0.1, "base__max_depth": 5},
    "ridge": {"alpha": 1.0},
    "hgb": {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
    },
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tweedie_variance_power": 1.5,
    },
}

DEFAULT_CLF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
}


# -----------------------------------------------------------------------
# Optuna tuning
# -----------------------------------------------------------------------


def tune_regressor(
    method, X_train, y_train, X_val, y_val, n_trials=None, max_tune_samples=50000, objective=None
):
    """
    Tune a regressor using Optuna, optimizing RMSE on validation set.

    To speed up tuning on large datasets, the training data is subsampled
    to `max_tune_samples` rows during the search. The final model is always
    trained on the full data.

    Returns best_params dict.
    """
    if n_trials is None:
        n_trials = OPTUNA_N_TRIALS

    if QUICK_MODE:
        max_tune_samples = min(max_tune_samples, 20000)

    # Memory-intensive methods get smaller subsample
    if method in ("bagging", "adaboost"):
        max_tune_samples = min(max_tune_samples, 30000)

    hp_func = HP_SPACES[method]

    # Subsample for faster tuning
    X_t, y_t = _subsample(X_train, y_train, max_tune_samples)

    # The callback must not be named `objective`: that would shadow the loss
    # identifier passed in as `objective` (found in review, 9 Sep 2026).
    def _trial(trial):
        params = hp_func(trial)
        try:
            model = build_regressor(method, params, objective)
            model = fit_regressor(model, X_t, y_t)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            del model
            gc.collect()
            return rmse
        except Exception:
            return float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(_trial, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    del study
    gc.collect()
    return best


def tune_classifier(
    X_train, y_train, X_val, y_val, n_trials=None, max_tune_samples=50000
):
    """
    Tune the binary classifier (payment > 0) for the two-stage model.
    Optimizes F1 score on the positive class.

    Returns best_params dict.
    """
    if n_trials is None:
        n_trials = max(OPTUNA_N_TRIALS // 2, 10)  # fewer trials for classifier

    if QUICK_MODE:
        max_tune_samples = min(max_tune_samples, 20000)

    # Subsample for faster tuning
    X_t, y_t = _subsample(X_train, y_train, max_tune_samples)

    def objective(trial):
        params = hp_classifier(trial)
        try:
            clf = build_classifier(params)
            clf.fit(X_t, y_t)
            preds = clf.predict(X_val)
            score = f1_score(y_val, preds, zero_division=0)
            del clf
            gc.collect()
            return score
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Strip 'clf_' prefix from Optuna suggest names to get actual param names
    best = {}
    for k, v in study.best_params.items():
        clean_key = k.replace("clf_", "", 1) if k.startswith("clf_") else k
        best[clean_key] = v

    del study
    gc.collect()
    return best


def tune_model(
    method, X_train, y_train, X_val, y_val, target_type, zero_handling, n_trials=None,
    objective=None,
):
    """
    Full tuning pipeline for a single model configuration.

    Handles:
      - target_type: "raw" or "log1p" (transforms y before training)
      - zero_handling: "include_zeros" or "two_stage"

    Returns:
      best_params : dict (for regressor)
      best_clf_params : dict or None (for classifier in two-stage)
    """
    # Transform target if needed
    y_train_t, y_val_t = _transform_target(y_train, y_val, target_type)

    if zero_handling == "include_zeros":
        best_params = tune_regressor(
            method, X_train, y_train_t, X_val, y_val_t, n_trials, objective=objective
        )
        return best_params, None

    elif zero_handling == "two_stage":
        # Stage 1: binary classifier (payment > 0)
        y_train_bin = (y_train > 0).astype(int)
        y_val_bin = (y_val > 0).astype(int)

        # Only tune classifier if there are both classes in train AND val
        if (
            y_train_bin.sum() > 0
            and y_val_bin.sum() > 0
            and (1 - y_train_bin).sum() > 0
        ):
            best_clf_params = tune_classifier(
                X_train, y_train_bin, X_val, y_val_bin, n_trials
            )
        else:
            best_clf_params = DEFAULT_CLF_PARAMS.copy()

        # Stage 2: regressor on non-zero payments only
        nz_train = y_train > 0
        nz_val = y_val > 0

        if nz_train.sum() > 10 and nz_val.sum() > 0:
            y_train_nz = _transform_target_single(y_train[nz_train], target_type)
            y_val_nz = _transform_target_single(y_val[nz_val], target_type)
            best_params = tune_regressor(
                method, X_train[nz_train], y_train_nz, X_val[nz_val], y_val_nz, n_trials,
                objective=objective,
            )
        else:
            best_params = DEFAULT_PARAMS[method].copy()

        return best_params, best_clf_params

    else:
        raise ValueError(f"Unknown zero_handling: {zero_handling}")


# -----------------------------------------------------------------------
# Training (final model on train+val)
# -----------------------------------------------------------------------


def train_final_model(
    method,
    X,
    y,
    params,
    clf_params,
    target_type,
    zero_handling,
    claim_types=None,
    accident_years=None,
    objective=None,
    dev_lags=None,
    n_folds=None,
    compute_record=True,
    claim_ids=None,
):
    """
    Train the final pipeline on the full training data and attach the
    cross-fitted calibration record used by every post-hoc correction arm
    (src/corrections.py).

    Returns the fitted regressor, or (classifier, regressor) for two-stage.
    The regressor carries two attributes:

      calibration_record_            dict from compute_calibration_record
                                     (None when not computed / too few rows)
      pipeline_calibration_factors_  the first-submission per-type factors
                                     (identity for raw targets); predict()
                                     applies them so `predicted_payment`
                                     reproduces the first-submission method.
    """
    n_folds = n_folds or CALIB_N_FOLDS
    y_t = _transform_target_single(y, target_type)

    if zero_handling == "include_zeros":
        reg = fit_regressor(build_regressor(method, params, objective), X, y_t)
        model = reg

    elif zero_handling == "two_stage":
        y_bin = (y > 0).astype(int)
        if clf_params is None:
            clf_params = DEFAULT_CLF_PARAMS.copy()
        clf = build_classifier(clf_params)
        clf.fit(X, y_bin)
        nz = y > 0
        reg = build_regressor(method, params, objective)
        if nz.sum() > 0:
            reg = fit_regressor(reg, X[nz], _transform_target_single(y[nz], target_type))
        else:
            # Fallback: train on everything (should not happen in practice)
            reg = fit_regressor(reg, X, y_t)
        model = (clf, reg)

    else:
        raise ValueError(f"Unknown zero_handling: {zero_handling}")

    record = None
    if compute_record:
        record = compute_calibration_record(
            method,
            X,
            y,
            params,
            clf_params,
            target_type,
            zero_handling,
            objective=objective,
            n_folds=n_folds,
            claim_types=claim_types,
            accident_years=accident_years,
            dev_lags=dev_lags,
            claim_ids=claim_ids,
        )
        if record is not None:
            base, _, _ = _pipeline_components(model, X, target_type, zero_handling)
            record["insample_pred"] = base
    reg.calibration_record_ = record
    if target_type == "log1p":
        reg.pipeline_calibration_factors_ = corrections.paper_factors(record)
    else:
        reg.pipeline_calibration_factors_ = {GLOBAL_CALIB_KEY: 1.0}
    return model


# -----------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------


def predict(model, X, target_type, zero_handling, claim_types=None, return_components=False):
    """
    Predictions in the ORIGINAL scale (payment amounts).

    The returned array applies the first-submission calibration factors
    (`pipeline_calibration_factors_`, identity for raw targets), so it equals
    the `paper` arm.  With return_components=True a dict is returned instead:

      pred       the same array
      pred_base  uncorrected pipeline output (two-stage: p_hat * amount)
      prob_pos   payment probability per row (None for include_zeros)
      reg_out    regressor output on its own scale (log scale for log1p)
                 per row, NaN where the regressor was not evaluated

    Two-stage pipelines use soft gating: rows with p_hat <= 0.01 get zero.
    """
    base, prob_pos, reg_out = _pipeline_components(model, X, target_type, zero_handling)
    reg = model[1] if zero_handling == "two_stage" else model
    pred = base * _paper_multipliers(reg, target_type, claim_types, len(base))
    if return_components:
        return {"pred": pred, "pred_base": base, "prob_pos": prob_pos, "reg_out": reg_out}
    return pred


def _pipeline_components(model, X, target_type, zero_handling):
    """Uncorrected pipeline output and its components; see predict()."""
    n = len(X)
    if zero_handling == "include_zeros":
        reg_out = np.asarray(model.predict(X), dtype=float)
        base = np.clip(_inverse_transform_target(reg_out, target_type), 0, None)
        return base, None, reg_out

    elif zero_handling == "two_stage":
        clf, reg = model
        proba = clf.predict_proba(X)
        if proba.shape[1] == 1:
            # Only one class was seen during training (e.g. all zeros at DY=10)
            prob_pos = np.zeros(n) if clf.classes_[0] == 0 else np.ones(n)
        else:
            prob_pos = proba[:, 1]
        likely = prob_pos > 0.01
        base = np.zeros(n)
        reg_out = np.full(n, np.nan)
        if likely.sum() > 0:
            X_pos = X[likely]
            ro = np.asarray(reg.predict(X_pos), dtype=float)
            reg_out[likely] = ro
            base[likely] = prob_pos[likely] * np.clip(
                _inverse_transform_target(ro, target_type), 0, None
            )
        return base, prob_pos, reg_out

    else:
        raise ValueError(f"Unknown zero_handling: {zero_handling}")


def _paper_multipliers(reg, target_type, claim_types, n):
    """Per-row multipliers of the first-submission calibration (1 for raw targets)."""
    if target_type != "log1p":
        return np.ones(n)
    calib = getattr(reg, "pipeline_calibration_factors_", None)
    if not calib:
        return np.ones(n)
    g = calib.get(GLOBAL_CALIB_KEY, 1.0)
    if claim_types is not None and len(calib) > 1:
        return np.array([calib.get(ct, g) for ct in claim_types], dtype=float)
    return np.full(n, g)


def _predict_with_corrections(model, X, target_type, claim_types=None):
    """Regressor-only prediction with the first-submission factors (kept for callers)."""
    if target_type != "log1p":
        return model.predict(X)
    preds = np.expm1(model.predict(X))
    return preds * _paper_multipliers(model, target_type, claim_types, len(preds))


# -----------------------------------------------------------------------
# Target transform helpers
# -----------------------------------------------------------------------


def _transform_target(y_train, y_val, target_type):
    """Transform both train and val targets."""
    return (
        _transform_target_single(y_train, target_type),
        _transform_target_single(y_val, target_type),
    )


def _transform_target_single(y, target_type):
    """Transform a single target array."""
    if target_type == "raw":
        return y
    elif target_type == "log1p":
        neg_mask = y < 0
        if np.any(neg_mask):
            n_neg = int(np.sum(neg_mask))
            warnings.warn(
                f"log1p transform: clipping {n_neg} negative target value(s) "
                f"to 0 (min was {y[neg_mask].min():.2f}). "
                f"Negative payments (recoveries) cannot be log-transformed."
            )
            y = np.where(neg_mask, 0, y)
        return np.log1p(y)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")


# Largest payment a model may predict for one cell.  train.run_single_experiment
# sets this to the largest payment observed in the training triangle, so no
# cell is predicted above anything ever seen.  Mean-leaf forests and log1p
# forests cannot exceed that bound anyway; the cap bites only deviance-loss
# boosters, which predict through exp() and can run away on zero-inflated,
# heavy-tailed targets (sklearn HGB has no leaf-step cap; LightGBM's
# poisson_max_delta_step tames but does not bound the tail).  The number of
# capped cells is counted in _N_CAPPED and reported per configuration.
PRED_CAP = 1e12
_N_CAPPED = 0


def reset_prediction_cap(cap):
    """Set the per-cell prediction cap and zero the capped-cell counter."""
    global PRED_CAP, _N_CAPPED
    PRED_CAP = float(cap)
    _N_CAPPED = 0


def n_capped():
    return _N_CAPPED


def _inverse_transform_target(preds, target_type):
    """Inverse-transform predictions back to original scale, finite, non-negative, <= PRED_CAP."""
    global _N_CAPPED
    preds = np.asarray(preds, dtype=float)
    if target_type == "raw":
        out = preds
    elif target_type == "log1p":
        with np.errstate(over="ignore"):
            out = np.expm1(np.minimum(preds, 700.0))
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    over = out > PRED_CAP  # +inf counts as capped
    if over.any():
        _N_CAPPED += int(over.sum())
        out = np.where(over, PRED_CAP, out)
    return np.nan_to_num(out, nan=0.0, neginf=0.0)


# -----------------------------------------------------------------------
# Cross-fitted calibration record (Section 4 of the revised manuscript)
# -----------------------------------------------------------------------


def compute_calibration_record(
    method,
    X,
    y,
    params,
    clf_params,
    target_type,
    zero_handling,
    objective=None,
    n_folds=None,
    claim_types=None,
    accident_years=None,
    dev_lags=None,
    claim_ids=None,
):
    """
    Out-of-fold predictions of the full pipeline on the training cells.

    For each of K folds (GroupKFold by accident year when available, so folds
    respect the run-off structure; otherwise unshuffled KFold) the complete
    pipeline (classifier + regressor for two-stage) is refitted on the other
    folds and evaluated on the held-out fold.  Every post-hoc correction arm
    in src/corrections.py is a function of the returned record:

      y            training targets (original scale)
      oos_pred     out-of-fold pipeline prediction per training row
      oos_reg_log  out-of-fold regressor output on the log scale (log1p only,
                   NaN where the regressor was not evaluated), for Duan smearing
      fold_id      fold index per row (-1 if never held out)
      claim_type   per row or None;  dev_lag  per row or None
      claim_id     per row or None (cluster label for the claim-clustered variance)
      n_folds_eff  folds actually used;  n_estimators_cap  cap applied in folds
      target_type, zero_handling

    Returns None when there are too few rows or groups for at least two folds.
    """
    n_folds = n_folds or CALIB_N_FOLDS
    n = len(y)
    if n < n_folds * 2:
        return None

    # Fold models are capped in size for speed; the cap is recorded so the
    # manuscript can state it (referee R1-A9).
    cv_params = params.copy()
    cap = CALIB_CV_N_ESTIMATORS_CAP
    if cap is not None:
        for key in ("n_estimators", "max_iter"):
            if key in cv_params:
                cv_params[key] = min(cv_params[key], cap)

    if accident_years is not None:
        n_groups = len(np.unique(accident_years))
        effective_folds = min(n_folds, n_groups)
        if effective_folds < 2:
            return None
        kf = GroupKFold(n_splits=effective_folds)
        split_iter = kf.split(X, y, groups=accident_years)
    else:
        effective_folds = n_folds
        kf = KFold(n_splits=n_folds, shuffle=False)
        split_iter = kf.split(X)

    oos_preds = np.full(n, np.nan)
    oos_reg = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=int)

    for j, (train_idx, val_idx) in enumerate(split_iter):
        X_tr, y_tr, X_va = X[train_idx], y[train_idx], X[val_idx]

        if zero_handling == "two_stage":
            y_bin = (y_tr > 0).astype(int)
            fold_clf = build_classifier((clf_params or DEFAULT_CLF_PARAMS).copy())
            fold_clf.fit(X_tr, y_bin)
            nz = y_tr > 0
            fold_reg = build_regressor(method, cv_params, objective)
            if nz.sum() > 0:
                fold_reg = fit_regressor(fold_reg, X_tr[nz], _transform_target_single(y_tr[nz], target_type))
            else:
                fold_reg = fit_regressor(fold_reg, X_tr, _transform_target_single(y_tr, target_type))
            proba = fold_clf.predict_proba(X_va)
            if proba.shape[1] == 1:
                prob_pos = np.zeros(len(val_idx)) if fold_clf.classes_[0] == 0 else np.ones(len(val_idx))
            else:
                prob_pos = proba[:, 1]
            likely = prob_pos > 0.01
            fold_preds = np.zeros(len(val_idx))
            if likely.sum() > 0:
                ro = np.asarray(fold_reg.predict(X_va[likely]), dtype=float)
                oos_reg[val_idx[likely]] = ro
                fold_preds[likely] = prob_pos[likely] * np.clip(
                    _inverse_transform_target(ro, target_type), 0, None
                )
            del fold_clf, fold_reg

        elif zero_handling == "include_zeros":
            fold_reg = build_regressor(method, cv_params, objective)
            fold_reg = fit_regressor(fold_reg, X_tr, _transform_target_single(y_tr, target_type))
            ro = np.asarray(fold_reg.predict(X_va), dtype=float)
            oos_reg[val_idx] = ro
            fold_preds = np.clip(_inverse_transform_target(ro, target_type), 0, None)
            del fold_reg

        else:
            raise ValueError(f"Unknown zero_handling: {zero_handling}")

        oos_preds[val_idx] = fold_preds
        fold_id[val_idx] = j
        gc.collect()

    record = {
        "y": np.asarray(y, dtype=float),
        "oos_pred": oos_preds,
        "oos_reg_log": oos_reg if target_type == "log1p" else None,
        "fold_id": fold_id,
        "claim_type": np.asarray(claim_types) if claim_types is not None else None,
        "dev_lag": np.asarray(dev_lags) if dev_lags is not None else None,
        "claim_id": np.asarray(claim_ids) if claim_ids is not None else None,
        "n_folds_eff": int(effective_folds),
        "n_estimators_cap": cap,
        "target_type": target_type,
        "zero_handling": zero_handling,
    }
    if os.environ.get("CALIB_DEBUG", ""):
        f = corrections.paper_factors(record)
        print(
            f"  [CALIB] n={n}, nz_true={int((y > 0).sum())}, folds={effective_folds}, "
            f"raw_ratio={corrections.ratio(record['y'], oos_preds):.3f}, "
            f"paper_global={f[GLOBAL_CALIB_KEY]:.3f}"
        )
    return record


def _compute_pipeline_calibration(*args, **kwargs):
    """Backward-compatible wrapper returning the first-submission factor dict."""
    return corrections.paper_factors(compute_calibration_record(*args, **kwargs))


# -----------------------------------------------------------------------
# Subsampling helper
# -----------------------------------------------------------------------


def _subsample(X, y, max_samples):
    """
    Subsample arrays to at most `max_samples` rows for faster tuning.
    Uses uniform random sampling (not stratified by zero/non-zero ratio).
    """
    n = len(y)
    if n <= max_samples:
        return X, y

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(n, size=max_samples, replace=False)
    if hasattr(X, "iloc"):
        return X.iloc[idx], y[idx] if hasattr(y, "__getitem__") else y.iloc[idx]
    return X[idx], y[idx]
