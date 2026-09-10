"""
Training orchestration for all experiment configurations.

Iterates through the full experiment matrix:
  methods × structures × targets × zero_handling combinations

For each config, tunes with Optuna, retrains on full train+val, predicts.
"""

import os
import pickle
import time
import gc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import src.config as cfg

from src.config import (
    METHODS,
    STRUCTURES,
    TARGETS,
    ZERO_HANDLING,
    generate_experiment_configs,
)
from src.data_prep import get_split_masks
from src.models import (
    tune_model,
    train_final_model,
    predict,
    DEFAULT_PARAMS,
    DEFAULT_CLF_PARAMS,
    QUICK_MODE,
)
from src import corrections
from src import models as models_mod
from src.feature_engineering import horizon_shifted_features

PRED_COLS = ["claim_no", "accident_year", "dev_lag", "true_incremental_payment"]

# Optional fresh-replicate evaluation (simulation coverage experiment): when set to
# {"claim_dy": DataFrame, "feat_cols": list}, every per-DY final model also predicts
# the upper-triangle cells of that independent portfolio; rows accumulate in ["preds"].
FRESH_EVAL = None


def _objective_of(config):
    """Objective key of a config; legacy configs map to the mse_* objectives."""
    return config.get("objective") or ("mse_log1p" if config["target"] == "log1p" else "mse_raw")


def _record_of(model, zero_handling):
    reg = model[1] if zero_handling == "two_stage" else model
    return getattr(reg, "calibration_record_", None)


def _attach_arms(pred_df, arms):
    """Write `predicted_payment` (paper arm) and one `pred__<arm>` column per arm."""
    pred_df["predicted_payment"] = arms["paper"]
    for a in corrections.ARMS:
        pred_df[f"pred__{a}"] = arms[a]
    return pred_df


def run_single_experiment(
    config, claim_dy, feature_cols, feature_cols_per_dy, n_trials=50,
    save_models_path=None, n_folds=None, partial_dir=None,
):
    """
    Run a single experiment configuration.

    Parameters
    ----------
    config : dict
        Keys: method, structure, target, zero_handling, name
    claim_dy : pd.DataFrame
        The full claim × DY dataset with features and ground truth.
    feature_cols : list[str]
        Feature columns for single_model.
    feature_cols_per_dy : list[str]
        Feature columns for per_dy models.
    n_trials : int
        Optuna trials per model.

    Returns
    -------
    result : dict with keys:
        name, method, structure, target, zero_handling,
        predictions (DataFrame), val_metrics, best_params, runtime_seconds
    """
    method = config["method"]
    structure = config["structure"]
    target_type = config["target"]
    zero_handling = config["zero_handling"]
    name = config["name"]
    objective = _objective_of(config)
    # multi_step uses the single-model feature set (dev_lag is a feature)
    if structure == "multi_step":
        pass  # feat_cols chosen below

    # Reporting-consistent risk set (default since 9 Sep 2026): observed cells that precede the claim's
    # reporting year leave the panel, so that the training and calibration risk set at development year k
    # holds only claims reported by k (features are already engineered; target cells are unaffected).
    # RESERVING_KEEP_PREREPORT=1 restores the earlier panel for the sensitivity comparison.
    if os.environ.get("RESERVING_KEEP_PREREPORT") != "1" and "report_dy" in claim_dy.columns:
        obs_mask = (claim_dy["accident_year"] + claim_dy["dev_lag"] - 1) <= cfg.MAX_ACCIDENT_YEAR
        pre = obs_mask & (claim_dy["dev_lag"] < claim_dy["report_dy"])
        print(f"  [{name}] excluding {int(pre.sum()):,} pre-reporting cells from the observed panel", flush=True)
        claim_dy = claim_dy.loc[~pre].reset_index(drop=True)  # contiguous index: the horizon shift re-indexes
    # Per-cell prediction cap = largest payment in the observed (training) triangle.
    observed = (claim_dy["accident_year"] + claim_dy["dev_lag"] - 1) <= cfg.MAX_ACCIDENT_YEAR
    y_max_train = float(claim_dy.loc[observed, "incremental_payment"].max())
    models_mod.reset_prediction_cap(y_max_train)

    start_time = time.time()
    feat_cols = feature_cols_per_dy if structure == "per_dy" else feature_cols

    if structure == "per_dy":
        result = _run_per_dy(
            method, target_type, zero_handling, name, claim_dy, feat_cols, n_trials,
            save_models_path=save_models_path, objective=objective, n_folds=n_folds,
        )
    elif structure == "multi_step":
        result = _run_multi_step(
            method, target_type, zero_handling, name, claim_dy, feat_cols, n_trials,
            objective=objective, n_folds=n_folds,
        )
    elif structure == "single_model":
        result = _run_single_model(
            method, target_type, zero_handling, name, claim_dy, feat_cols, n_trials,
            objective=objective, n_folds=n_folds,
        )
    else:
        raise ValueError(f"Unknown structure: {structure}")

    result["objective"] = objective
    result["runtime_seconds"] = time.time() - start_time
    result["n_capped"] = models_mod.n_capped()
    if result["n_capped"]:
        print(f"  [{name}] {result['n_capped']:,} predicted cells capped at the largest "
              f"training payment ({y_max_train:,.0f})", flush=True)

    # Incremental persistence: long runs should not lose finished configs.
    if partial_dir:
        os.makedirs(partial_dir, exist_ok=True)
        result["predictions"].to_csv(os.path.join(partial_dir, f"predictions__{name}.csv"), index=False)
        diag = result.get("calibration_diagnostics")
        if isinstance(diag, pd.DataFrame) and not diag.empty:
            diag.to_csv(os.path.join(partial_dir, f"calibration__{name}.csv"), index=False)
        if os.environ.get("RESERVING_DUMP_CALIB_RECORD") == "1":
            frames = []
            for g, rec in (result.get("calibration_records") or {}).items():
                if rec is None:
                    continue
                n = len(rec["y"])
                frames.append(pd.DataFrame({
                    "group": g,
                    "claim_id": rec["claim_id"] if rec.get("claim_id") is not None else np.full(n, -1),
                    "dev_lag": rec["dev_lag"] if rec.get("dev_lag") is not None else np.full(n, g),
                    "fold_id": rec["fold_id"], "y": rec["y"], "oos_pred": rec["oos_pred"],
                }))
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(
                    os.path.join(partial_dir, f"calibration_record__{name}.csv"), index=False
                )
    return result


# -----------------------------------------------------------------------
# Per-DY training: one model per development year
# -----------------------------------------------------------------------


def _run_per_dy(
    method, target_type, zero_handling, name, claim_dy, feat_cols, n_trials,
    save_models_path=None, objective=None, n_folds=None,
):
    """Train a separate model for each development year."""
    all_predictions = []
    all_val_metrics = {}
    all_best_params = {}
    fi = None  # feature importances (from DY=1 model)
    persisted_models = {} if save_models_path else None
    per_dy_items = {}  # dy -> record, test components, test rows (for the correction arms)

    # Check if claim_type is available for per-type calibration
    has_claim_type = "claim_type" in claim_dy.columns

    # Use fewer trials per DY since we train 9 separate models
    per_dy_trials = max(n_trials // 2, 5)

    for dy in range(1, cfg.MAX_DEV_LAG + 1):
        # Get data splits for this DY
        train_mask, val_mask, test_mask = get_split_masks(
            claim_dy, dy, structure="per_dy"
        )

        # Check we have enough data
        train_data = claim_dy[train_mask]
        val_data = claim_dy[val_mask]
        test_data = claim_dy[test_mask]

        if len(train_data) == 0:
            # Even when we can't train, we must still record test rows so
            # that true_reserve totals remain complete across all DYs.
            if len(test_data) > 0:
                pred_df = test_data[PRED_COLS].copy()
                pred_df["predicted_payment"] = np.nan
                for a in corrections.ARMS:
                    pred_df[f"pred__{a}"] = np.nan
                all_predictions.append(pred_df)
            continue

        X_train = train_data[feat_cols].values
        y_train = train_data["incremental_payment"].values
        X_val = val_data[feat_cols].values if len(val_data) > 0 else X_train[:0]
        y_val = (
            val_data["incremental_payment"].values if len(val_data) > 0 else y_train[:0]
        )

        # Extract claim_types for per-type calibration
        ct_train = train_data["claim_type"].values if has_claim_type else None
        ct_val = (
            val_data["claim_type"].values
            if has_claim_type and len(val_data) > 0
            else None
        )
        ct_test = (
            test_data["claim_type"].values
            if has_claim_type and len(test_data) > 0
            else None
        )

        # Tune hyperparameters (if we have validation data)
        if len(val_data) > 10:
            try:
                best_params, best_clf_params = tune_model(
                    method,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    target_type,
                    zero_handling,
                    per_dy_trials,
                    objective=objective,
                )
            except Exception as e:
                print(f"  [WARN] Tuning failed for {name} DY={dy}: {e}")
                best_params = DEFAULT_PARAMS[method].copy()
                best_clf_params = (
                    DEFAULT_CLF_PARAMS.copy() if zero_handling == "two_stage" else None
                )
        else:
            best_params = DEFAULT_PARAMS[method].copy()
            best_clf_params = (
                DEFAULT_CLF_PARAMS.copy() if zero_handling == "two_stage" else None
            )

        all_best_params[f"DY{dy}"] = {
            "regressor": best_params,
            "classifier": best_clf_params,
        }

        # Compute validation metrics using a model trained ONLY on train
        # (before retraining on train+val), so val is truly out-of-sample
        if len(val_data) > 0:
            try:
                tuned_model = train_final_model(
                    method,
                    X_train,
                    y_train,
                    best_params,
                    best_clf_params,
                    target_type,
                    zero_handling,
                    claim_types=ct_train,
                    accident_years=train_data["accident_year"].values,
                    objective=objective,
                    n_folds=n_folds,
                    compute_record=(target_type == "log1p"),
                )
                val_preds = predict(
                    tuned_model,
                    X_val,
                    target_type,
                    zero_handling,
                    claim_types=ct_val,
                )
                val_rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
                val_mae = np.mean(np.abs(y_val - val_preds))
                all_val_metrics[f"DY{dy}"] = {"rmse": val_rmse, "mae": val_mae}
                del tuned_model
                gc.collect()
            except Exception as e:
                print(f"  [WARN] Val metrics failed for {name} DY={dy}: {e}")

        # Retrain on train + val with best params, then predict test
        if len(test_data) > 0:
            # Avoid double-stacking when val AY is already in train
            # (happens for DY=10 where only AY=1 is available)
            val_ay = cfg.get_val_ay_for_dy(dy)
            train_ays = cfg.get_train_ays_for_dy(dy)
            val_in_train = val_ay in train_ays

            if val_in_train or len(X_val) == 0:
                X_full = X_train
                y_full = y_train
                ct_full = ct_train
                ay_full = train_data["accident_year"].values
                cid_full = train_data["claim_no"].values
            else:
                X_full = np.vstack([X_train, X_val])
                y_full = np.concatenate([y_train, y_val])
                ct_full = (
                    np.concatenate([ct_train, ct_val])
                    if ct_train is not None and ct_val is not None
                    else ct_train
                )
                ay_full = np.concatenate([
                    train_data["accident_year"].values,
                    val_data["accident_year"].values,
                ])
                cid_full = np.concatenate([train_data["claim_no"].values, val_data["claim_no"].values])

            try:
                final_model = train_final_model(
                    method,
                    X_full,
                    y_full,
                    best_params,
                    best_clf_params,
                    target_type,
                    zero_handling,
                    claim_types=ct_full,
                    accident_years=ay_full,
                    objective=objective,
                    dev_lags=np.full(len(y_full), dy),
                    n_folds=n_folds,
                    claim_ids=cid_full,
                )
                X_test = test_data[feat_cols].values
                comps = predict(
                    final_model,
                    X_test,
                    target_type,
                    zero_handling,
                    claim_types=ct_test,
                    return_components=True,
                )

                if FRESH_EVAL is not None:
                    fdf = FRESH_EVAL["claim_dy"]
                    fm = (fdf["dev_lag"] == dy) & ((fdf["accident_year"] + fdf["dev_lag"] - 1) <= cfg.MAX_ACCIDENT_YEAR)
                    if os.environ.get("RESERVING_KEEP_PREREPORT") != "1" and "report_dy" in fdf.columns:
                        fm &= fdf["dev_lag"] >= fdf["report_dy"]  # same reporting-consistent risk set as the calibration record
                    if fm.any():
                        fbase, _, _ = models_mod._pipeline_components(
                            final_model, fdf.loc[fm, feat_cols].values, target_type, zero_handling
                        )
                        FRESH_EVAL.setdefault("preds", []).append(pd.DataFrame({
                            "config": name, "dev_lag": dy, "claim_no": fdf.loc[fm, "claim_no"].values,
                            "accident_year": fdf.loc[fm, "accident_year"].values,
                            "y": fdf.loc[fm, "incremental_payment"].values, "pred": np.asarray(fbase, dtype=float),
                        }))
                # Arms are attached after the loop, once every DY's calibration
                # record is available for the cross-DY credibility step.
                per_dy_items[dy] = {
                    "record": _record_of(final_model, zero_handling),
                    "comps": comps,
                    "test_df": test_data[PRED_COLS].copy(),
                    "ct_test": ct_test,
                }

                # Extract feature importances (only first DY for per_dy)
                if dy == 1:
                    fi = _extract_feature_importances(
                        final_model, zero_handling, feat_cols
                    )

                if persisted_models is not None:
                    persisted_models[dy] = {
                        "model": final_model,
                        "best_params": best_params,
                        "best_clf_params": best_clf_params,
                        "feat_cols": list(feat_cols),
                    }
                else:
                    del final_model
            except Exception as e:
                print(f"  [WARN] Prediction failed for {name} DY={dy}: {e}")
                # Still record test rows with NaN predictions so that
                # true_reserve totals are not affected by prediction failures.
                pred_df = test_data[PRED_COLS].copy()
                pred_df["predicted_payment"] = np.nan
                for a in corrections.ARMS:
                    pred_df[f"pred__{a}"] = np.nan
                all_predictions.append(pred_df)

        # Free memory after each DY
        del train_data, val_data, test_data, X_train, y_train
        gc.collect()

    # Correction arms: credibility across the per-DY models (Thm 3.8), then
    # per-DY type tables, then one set of arm columns per DY.
    diag_frames = []
    if per_dy_items:
        dy_table = corrections.credibility_table(
            corrections.combine_global_rows({dy: it["record"] for dy, it in per_dy_items.items()})
        )
        dy_table = corrections.add_zcv_column(
            dy_table, corrections.zcv_scalar({dy: it["record"] for dy, it in per_dy_items.items()})
        )
        diag_frames.append(corrections.diagnostics_rows(name, "dy", dy_table))
        for dy, it in per_dy_items.items():
            rec = it["record"]
            type_table = None
            if rec is not None and rec.get("claim_type") is not None:
                type_table = corrections.credibility_table(
                    corrections.summarize_groups(rec, "claim_type")
                )
                diag_frames.append(
                    corrections.diagnostics_rows(name, "type", type_table, {"dev_lag": dy})
                )
            n_test = len(it["test_df"])
            arms, info = corrections.arm_predictions(
                rec, it["comps"], dy_table, type_table, np.full(n_test, dy), it["ct_test"]
            )
            diag_frames.append(
                pd.DataFrame(
                    [{"name": name, "level": "arms", "group": dy, "dev_lag": dy,
                      "duan_S": info["duan_S"], "isotonic": info["isotonic"],
                      "has_record": info["record"],
                      "K_eff": rec["n_folds_eff"] if rec else np.nan,
                      "n_estimators_cap": rec["n_estimators_cap"] if rec else np.nan}]
                )
            )
            all_predictions.append(_attach_arms(it["test_df"], arms))
    calibration_diagnostics = (
        pd.concat([d for d in diag_frames if d is not None and not d.empty], ignore_index=True)
        if diag_frames else pd.DataFrame()
    )

    # Combine predictions across all DYs
    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_df = predictions_df.sort_values(["dev_lag", "accident_year", "claim_no"]).reset_index(drop=True)
    else:
        predictions_df = pd.DataFrame(
            columns=PRED_COLS + ["predicted_payment"] + [f"pred__{a}" for a in corrections.ARMS]
        )

    if persisted_models is not None and save_models_path:
        os.makedirs(os.path.dirname(save_models_path) or ".", exist_ok=True)
        with open(save_models_path, "wb") as f:
            pickle.dump(persisted_models, f)
        print(f"  Persisted per-DY models to {save_models_path}")

    return {
        "name": name,
        "method": method,
        "structure": "per_dy",
        "calibration_records": {dy: it["record"] for dy, it in per_dy_items.items()},
        "target": target_type,
        "zero_handling": zero_handling,
        "predictions": predictions_df,
        "val_metrics": all_val_metrics,
        "best_params": all_best_params,
        "feature_importances": fi,
        "calibration_diagnostics": calibration_diagnostics,
    }


# -----------------------------------------------------------------------
# Direct multi-step training: one model per horizon h, history frozen
# h years before the target (Remark "Two remedies", variant (ii)).
# -----------------------------------------------------------------------


def _run_multi_step(
    method, target_type, zero_handling, name, claim_dy, feat_cols, n_trials,
    objective=None, n_folds=None,
):
    """
    One model per horizon h = 1..MAX_DEV_LAG-1.  The model for horizon h is
    trained on upper-triangle cells (a, k) with k >= h + 1 whose features are
    computed from history frozen at k - h (horizon_shifted_features), so that
    training and prediction covariates have the same construction; it predicts
    the lower-triangle cells at exactly horizon h = a + k - 1 - MAX_AY.
    Calibration is cross-fitted within each horizon model; the credibility
    step runs across horizons (the per-group tables are keyed by h).
    """
    all_predictions = []
    all_val_metrics = {}
    all_best_params = {}
    fi = None
    per_h_items = {}  # h -> record, comps, test rows (arms attached after the loop)

    has_claim_type = "claim_type" in claim_dy.columns
    per_h_trials = max(n_trials // 2, 5)
    I = cfg.MAX_ACCIDENT_YEAR

    # single_model-style splits: per-DY latest observed AY is validation
    base_train, base_val, base_test = get_split_masks(
        claim_dy, dy=None, structure="single_model"
    )
    cell_h = claim_dy["accident_year"] + claim_dy["dev_lag"] - 1 - I  # test horizon

    keep_pre = os.environ.get("RESERVING_KEEP_PREREPORT") == "1" or "report_dy" not in claim_dy.columns
    for h in range(1, cfg.MAX_DEV_LAG):
        usable = claim_dy["dev_lag"] >= h + 1
        if not keep_pre:  # a training cell at horizon h needs the claim reported by the freeze date k - h
            usable &= claim_dy["report_dy"] <= claim_dy["dev_lag"] - h
        df_h = horizon_shifted_features(claim_dy, h)

        train_data = df_h[base_train & usable]
        val_data = df_h[base_val & usable]
        test_data = df_h[base_test & (cell_h == h)]

        if len(train_data) == 0:
            if len(test_data) > 0:
                pred_df = test_data[PRED_COLS].copy()
                pred_df["predicted_payment"] = np.nan
                for a in corrections.ARMS:
                    pred_df[f"pred__{a}"] = np.nan
                all_predictions.append(pred_df)
            continue

        X_train = train_data[feat_cols].values
        y_train = train_data["incremental_payment"].values
        X_val = val_data[feat_cols].values if len(val_data) > 0 else X_train[:0]
        y_val = val_data["incremental_payment"].values if len(val_data) > 0 else y_train[:0]

        ct_train = train_data["claim_type"].values if has_claim_type else None
        ct_val = val_data["claim_type"].values if has_claim_type and len(val_data) > 0 else None
        ct_test = test_data["claim_type"].values if has_claim_type and len(test_data) > 0 else None

        if len(val_data) > 10:
            try:
                best_params, best_clf_params = tune_model(
                    method, X_train, y_train, X_val, y_val,
                    target_type, zero_handling, per_h_trials, objective=objective,
                )
            except Exception as e:
                print(f"  [WARN] Tuning failed for {name} h={h}: {e}")
                best_params = DEFAULT_PARAMS[method].copy()
                best_clf_params = DEFAULT_CLF_PARAMS.copy() if zero_handling == "two_stage" else None
        else:
            best_params = DEFAULT_PARAMS[method].copy()
            best_clf_params = DEFAULT_CLF_PARAMS.copy() if zero_handling == "two_stage" else None

        all_best_params[f"h{h}"] = {"regressor": best_params, "classifier": best_clf_params}

        # validation metrics on the tuning split
        if len(val_data) > 0:
            try:
                tuned_model = train_final_model(
                    method, X_train, y_train, best_params, best_clf_params,
                    target_type, zero_handling, claim_types=ct_train,
                    accident_years=train_data["accident_year"].values,
                    objective=objective, n_folds=n_folds, compute_record=False,
                )
                val_preds = predict(tuned_model, X_val, target_type, zero_handling, claim_types=ct_val)
                all_val_metrics[f"h{h}"] = {
                    "rmse": float(np.sqrt(np.mean((y_val - val_preds) ** 2))),
                    "mae": float(np.mean(np.abs(y_val - val_preds))),
                }
                del tuned_model
                gc.collect()
            except Exception as e:
                print(f"  [WARN] Val metrics failed for {name} h={h}: {e}")

        # final model on the union of train and validation cells (the latest observed accident year can
        # serve as both; it must not be counted twice), cross-fitted record, test predictions
        val_data = df_h[base_val & ~base_train & usable]
        X_val = val_data[feat_cols].values if len(val_data) > 0 else X_train[:0]
        y_val = val_data["incremental_payment"].values if len(val_data) > 0 else y_train[:0]
        ct_val = val_data["claim_type"].values if has_claim_type and len(val_data) > 0 else None
        try:
            X_full = np.vstack([X_train, X_val]) if len(X_val) else X_train
            y_full = np.concatenate([y_train, y_val]) if len(y_val) else y_train
            ct_full = (
                np.concatenate([ct_train, ct_val])
                if ct_train is not None and ct_val is not None else ct_train
            )
            ay_full = (
                np.concatenate([train_data["accident_year"].values, val_data["accident_year"].values])
                if len(val_data) else train_data["accident_year"].values
            )
            dl_full = (
                np.concatenate([train_data["dev_lag"].values, val_data["dev_lag"].values])
                if len(val_data) else train_data["dev_lag"].values
            )
            cid_full = (
                np.concatenate([train_data["claim_no"].values, val_data["claim_no"].values])
                if len(val_data) else train_data["claim_no"].values
            )
            final_model = train_final_model(
                method, X_full, y_full, best_params, best_clf_params,
                target_type, zero_handling, claim_types=ct_full,
                accident_years=ay_full, objective=objective,
                dev_lags=dl_full, n_folds=n_folds, claim_ids=cid_full,
            )
            if len(test_data) > 0:
                comps = predict(
                    final_model, test_data[feat_cols].values, target_type,
                    zero_handling, claim_types=ct_test, return_components=True,
                )
                per_h_items[h] = {
                    "record": _record_of(final_model, zero_handling),
                    "comps": comps,
                    "test_df": test_data[PRED_COLS].copy(),
                    "ct_test": ct_test,
                }
            if h == 1:
                fi = _extract_feature_importances(final_model, zero_handling, feat_cols)
            del final_model
        except Exception as e:
            print(f"  [WARN] Prediction failed for {name} h={h}: {e}")
            if len(test_data) > 0:
                pred_df = test_data[PRED_COLS].copy()
                pred_df["predicted_payment"] = np.nan
                for a in corrections.ARMS:
                    pred_df[f"pred__{a}"] = np.nan
                all_predictions.append(pred_df)

        del df_h, train_data, val_data, test_data, X_train, y_train
        gc.collect()

    # Correction arms: credibility ACROSS HORIZONS (the group key is h).
    diag_frames = []
    if per_h_items:
        h_table = corrections.credibility_table(
            corrections.combine_global_rows({h: it["record"] for h, it in per_h_items.items()})
        )
        h_table = corrections.add_zcv_column(
            h_table, corrections.zcv_scalar({h: it["record"] for h, it in per_h_items.items()})
        )
        diag_frames.append(corrections.diagnostics_rows(name, "dy", h_table))
        for h, it in per_h_items.items():
            rec = it["record"]
            type_table = None
            if rec is not None and rec.get("claim_type") is not None:
                type_table = corrections.credibility_table(
                    corrections.summarize_groups(rec, "claim_type")
                )
                diag_frames.append(
                    corrections.diagnostics_rows(name, "type", type_table, {"dev_lag": h})
                )
            n_test = len(it["test_df"])
            arms, info = corrections.arm_predictions(
                rec, it["comps"], h_table, type_table, np.full(n_test, h), it["ct_test"]
            )
            diag_frames.append(
                pd.DataFrame(
                    [{"name": name, "level": "arms", "group": h, "dev_lag": h,
                      "duan_S": info["duan_S"], "isotonic": info["isotonic"],
                      "has_record": info["record"],
                      "K_eff": rec["n_folds_eff"] if rec else np.nan,
                      "n_estimators_cap": rec["n_estimators_cap"] if rec else np.nan}]
                )
            )
            all_predictions.append(_attach_arms(it["test_df"], arms))
    calibration_diagnostics = (
        pd.concat([d for d in diag_frames if d is not None and not d.empty], ignore_index=True)
        if diag_frames else pd.DataFrame()
    )

    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_df = predictions_df.sort_values(["dev_lag", "accident_year", "claim_no"]).reset_index(drop=True)
    else:
        predictions_df = pd.DataFrame(
            columns=PRED_COLS + ["predicted_payment"] + [f"pred__{a}" for a in corrections.ARMS]
        )

    return {
        "name": name,
        "method": method,
        "structure": "multi_step",
        "calibration_records": {h: it["record"] for h, it in per_h_items.items()},
        "target": target_type,
        "zero_handling": zero_handling,
        "predictions": predictions_df,
        "val_metrics": all_val_metrics,
        "best_params": all_best_params,
        "feature_importances": fi,
        "calibration_diagnostics": calibration_diagnostics,
    }


# -----------------------------------------------------------------------
# Single-model training: one model with dev_lag as feature
# -----------------------------------------------------------------------


def _run_single_model(
    method, target_type, zero_handling, name, claim_dy, feat_cols, n_trials,
    objective=None, n_folds=None,
):
    """Train a single model using dev_lag as a feature, across all DYs."""
    calibration_diagnostics = pd.DataFrame()

    # Check if claim_type is available for per-type calibration
    has_claim_type = "claim_type" in claim_dy.columns

    # Build combined train / val / test masks
    train_mask, val_mask, test_mask = get_split_masks(
        claim_dy, dy=None, structure="single_model"
    )

    train_data = claim_dy[train_mask]
    val_data = claim_dy[val_mask]
    test_data = claim_dy[test_mask]

    X_train = train_data[feat_cols].values
    y_train = train_data["incremental_payment"].values
    X_val = val_data[feat_cols].values
    y_val = val_data["incremental_payment"].values

    # Extract claim_types for per-type calibration
    ct_train = train_data["claim_type"].values if has_claim_type else None
    ct_val = val_data["claim_type"].values if has_claim_type else None
    ct_test = (
        test_data["claim_type"].values
        if has_claim_type and len(test_data) > 0
        else None
    )

    # Note: Optuna tuning subsamples internally (50K rows) for speed.
    # Final model trains on the FULL train+val data — no subsampling.

    # Tune hyperparameters
    if len(val_data) > 10:
        try:
            best_params, best_clf_params = tune_model(
                method,
                X_train,
                y_train,
                X_val,
                y_val,
                target_type,
                zero_handling,
                n_trials,
                objective=objective,
            )
        except Exception as e:
            print(f"  [WARN] Tuning failed for {name}: {e}")
            best_params = DEFAULT_PARAMS[method].copy()
            best_clf_params = (
                DEFAULT_CLF_PARAMS.copy() if zero_handling == "two_stage" else None
            )
    else:
        best_params = DEFAULT_PARAMS[method].copy()
        best_clf_params = (
            DEFAULT_CLF_PARAMS.copy() if zero_handling == "two_stage" else None
        )

    # Retrain on the union of train and validation cells (no duplicates) and predict
    val_only = val_data[~val_data.index.isin(train_data.index)]
    if len(val_only) != len(val_data):
        val_data = val_only
        X_val = val_data[feat_cols].values if len(val_data) > 0 else X_train[:0]
        y_val = val_data["incremental_payment"].values if len(val_data) > 0 else y_train[:0]
        ct_val = val_data["claim_type"].values if (ct_train is not None and len(val_data) > 0) else None
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    ct_full = (
        np.concatenate([ct_train, ct_val])
        if ct_train is not None and ct_val is not None
        else ct_train
    )
    ay_full = np.concatenate([
        train_data["accident_year"].values,
        val_data["accident_year"].values,
    ])

    val_metrics = {}
    fi = None  # feature importances
    predictions_df = pd.DataFrame(
        columns=PRED_COLS + ["predicted_payment"] + [f"pred__{a}" for a in corrections.ARMS]
    )

    # Compute validation metrics using a model trained ONLY on train
    # (before retraining on train+val), so val is truly out-of-sample
    if len(val_data) > 0:
        try:
            tuned_model = train_final_model(
                method,
                X_train,
                y_train,
                best_params,
                best_clf_params,
                target_type,
                zero_handling,
                claim_types=ct_train,
                accident_years=train_data["accident_year"].values,
                objective=objective,
                n_folds=n_folds,
                compute_record=(target_type == "log1p"),
            )
            val_preds = predict(
                tuned_model,
                X_val,
                target_type,
                zero_handling,
                claim_types=ct_val,
            )
            val_rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
            val_mae = np.mean(np.abs(y_val - val_preds))
            val_metrics["combined"] = {"rmse": val_rmse, "mae": val_mae}
            del tuned_model
            gc.collect()
        except Exception as e:
            print(f"  [WARN] Val metrics failed for {name}: {e}")

    if len(test_data) > 0:
        try:
            dl_full = np.concatenate([
                train_data["dev_lag"].values,
                val_data["dev_lag"].values,
            ])
            final_model = train_final_model(
                method,
                X_full,
                y_full,
                best_params,
                best_clf_params,
                target_type,
                zero_handling,
                claim_types=ct_full,
                accident_years=ay_full,
                objective=objective,
                dev_lags=dl_full,
                n_folds=n_folds,
                claim_ids=np.concatenate([train_data["claim_no"].values, val_data["claim_no"].values]),
            )

            X_test = test_data[feat_cols].values
            comps = predict(
                final_model,
                X_test,
                target_type,
                zero_handling,
                claim_types=ct_test,
                return_components=True,
            )

            # Correction arms: groups by development year and by claim type
            # within the single model's own calibration record.
            rec = _record_of(final_model, zero_handling)
            dy_table = type_table = None
            if rec is not None:
                dy_table = corrections.add_zcv_column(
            corrections.credibility_table(corrections.summarize_groups(rec, "dev_lag")),
            corrections.zcv_from_record(rec, "dev_lag"),
        )
                if rec.get("claim_type") is not None:
                    type_table = corrections.credibility_table(
                        corrections.summarize_groups(rec, "claim_type")
                    )
            arms, info = corrections.arm_predictions(
                rec, comps, dy_table, type_table, test_data["dev_lag"].values, ct_test
            )
            predictions_df = _attach_arms(test_data[PRED_COLS].copy(), arms)
            calibration_diagnostics = pd.concat(
                [
                    corrections.diagnostics_rows(name, "dy", dy_table),
                    corrections.diagnostics_rows(name, "type", type_table),
                    pd.DataFrame([{"name": name, "level": "arms", "group": "all",
                                   "duan_S": info["duan_S"], "isotonic": info["isotonic"],
                                   "has_record": info["record"],
                                   "K_eff": rec["n_folds_eff"] if rec else np.nan,
                                   "n_estimators_cap": rec["n_estimators_cap"] if rec else np.nan}]),
                ],
                ignore_index=True,
            )

            # Extract feature importances before deleting model
            fi = _extract_feature_importances(final_model, zero_handling, feat_cols)

            del final_model
            gc.collect()
        except Exception as e:
            print(f"  [WARN] Prediction failed for {name}: {e}")
            # Still record test rows with NaN predictions so that
            # true_reserve totals are always consistent.
            predictions_df = test_data[PRED_COLS].copy()
            predictions_df["predicted_payment"] = np.nan
            for a in corrections.ARMS:
                predictions_df[f"pred__{a}"] = np.nan

    return {
        "name": name,
        "method": method,
        "structure": "single_model",
        "target": target_type,
        "zero_handling": zero_handling,
        "predictions": predictions_df,
        "val_metrics": val_metrics,
        "best_params": {
            "regressor": best_params,
            "classifier": best_clf_params,
        },
        "feature_importances": fi,
        "calibration_diagnostics": calibration_diagnostics,
    }


# -----------------------------------------------------------------------
# Run all experiments
# -----------------------------------------------------------------------


def run_all_experiments(
    claim_dy,
    feature_cols,
    feature_cols_per_dy,
    n_trials=50,
    methods=None,
    n_jobs=1,
    structures=None,
    persist_models_for=None,
    persist_dir=None,
    objectives=None,
    n_folds=None,
    partial_dir=None,
    zero_handling=None,
):
    """
    Run the full experiment matrix.

    Parameters
    ----------
    claim_dy : pd.DataFrame
        Full dataset with features and ground truth.
    feature_cols : list[str]
        Features for single_model.
    feature_cols_per_dy : list[str]
        Features for per_dy.
    n_trials : int
        Optuna trials per model.
    methods : list[str] or None
        If provided, only run these methods. Otherwise run all.
    n_jobs : int
        Number of parallel jobs. Use 1 for sequential (safer).
    structures : list[str] or None
        If provided, only run these structures (e.g., ["single_model"]).

    Returns
    -------
    all_results : list of result dicts
    """
    configs = generate_experiment_configs(objectives, structures)

    # Filter methods if specified
    if methods is not None:
        configs = [c for c in configs if c["method"] in methods]

    # Filter zero handling if specified (e.g. ["two_stage"])
    if zero_handling is not None:
        configs = [c for c in configs if c["zero_handling"] in zero_handling]

    total = len(configs)
    print(f"\n{'=' * 60}")
    print(f"Running {total} experiment configurations")
    print(f"Objectives: {sorted(set(_objective_of(c) for c in configs))}")
    print(f"Optuna trials per model: {n_trials}; calibration folds: {n_folds or cfg.CALIB_N_FOLDS}")
    print(f"{'=' * 60}\n")

    def _save_path(config):
        if (
            persist_models_for
            and persist_dir
            and config["name"] == persist_models_for
        ):
            return os.path.join(persist_dir, f"persisted_models__{config['name']}.pkl")
        return None

    if n_jobs == 1:
        # Sequential execution with progress
        all_results = []
        for i, config in enumerate(configs):
            print(f"[{i + 1}/{total}] {config['name']} ...", end=" ", flush=True)
            result = run_single_experiment(
                config, claim_dy, feature_cols, feature_cols_per_dy, n_trials,
                save_models_path=_save_path(config), n_folds=n_folds, partial_dir=partial_dir,
            )
            runtime = result.get("runtime_seconds", 0)
            print(f"done ({runtime:.1f}s)")
            all_results.append(result)
            gc.collect()  # Free memory between experiments
    else:
        # Parallel execution
        print(f"Running in parallel with {n_jobs} jobs...")
        all_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_single_experiment)(
                config, claim_dy, feature_cols, feature_cols_per_dy, n_trials,
                save_models_path=_save_path(config), n_folds=n_folds, partial_dir=partial_dir,
            )
            for config in configs
        )

    print(f"\nCompleted {len(all_results)} experiments.")
    return all_results


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _empty_result(config, reason):
    """Return an empty result dict when a config can't be run."""
    return {
        "name": config["name"],
        "objective": _objective_of(config),
        "method": config["method"],
        "structure": config["structure"],
        "target": config["target"],
        "zero_handling": config["zero_handling"],
        "predictions": pd.DataFrame(
            columns=PRED_COLS + ["predicted_payment"] + [f"pred__{a}" for a in corrections.ARMS]
        ),
        "val_metrics": {"error": reason},
        "best_params": {},
        "feature_importances": None,
        "calibration_diagnostics": pd.DataFrame(),
        "runtime_seconds": 0,
    }


def _extract_feature_importances(model, zero_handling, feat_cols):
    """
    Extract feature importances from a trained model.

    For two-stage models, extracts from the regressor component.
    Returns a dict mapping feature_name -> importance, or None if unavailable.
    """
    try:
        if zero_handling == "two_stage":
            _, reg = model
        else:
            reg = model

        if hasattr(reg, "feature_importances_"):
            importances = reg.feature_importances_
            return dict(zip(feat_cols, importances))
        elif hasattr(reg, "coef_"):
            # Ridge: use absolute coefficient values
            coefs = np.abs(reg.coef_)
            return dict(zip(feat_cols, coefs))
    except Exception:
        pass
    return None


