"""
Configuration for the ML-based claims reserving pipeline.

Defines experiment matrix, hyperparameter search spaces, constants, and paths.
"""

import os
from itertools import product

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "converted")
DATA_DIR_QUARTERLY = os.path.join(BASE_DIR, "data", "converted_quarterly")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_DIR_ANNUAL = RESULTS_DIR
RESULTS_DIR_QUARTERLY = os.path.join(BASE_DIR, "results_quarterly")

# ---------------------------------------------------------------------------
# Data constants (annual defaults; quarterly overrides via set_quarterly_mode)
# ---------------------------------------------------------------------------
MAX_DEV_LAG = 10
MAX_ACCIDENT_YEAR = 10
RANDOM_STATE = 42

# Track whether we're in quarterly mode
_QUARTERLY_MODE = False


def set_quarterly_mode(quarterly=True):
    """
    Switch global config to quarterly (40 periods) or annual (10 periods).

    Must be called before any data loading. Modifies the module-level
    MAX_DEV_LAG, MAX_ACCIDENT_YEAR, and DATA_DIR constants.
    """
    import src.config as cfg

    cfg._QUARTERLY_MODE = quarterly
    if quarterly:
        cfg.MAX_DEV_LAG = 40
        cfg.MAX_ACCIDENT_YEAR = 40
        cfg.DATA_DIR = cfg.DATA_DIR_QUARTERLY
        cfg.RESULTS_DIR = cfg.RESULTS_DIR_QUARTERLY
    else:
        cfg.MAX_DEV_LAG = 10
        cfg.MAX_ACCIDENT_YEAR = 10
        cfg.DATA_DIR = os.path.join(cfg.BASE_DIR, "data", "converted")
        cfg.RESULTS_DIR = cfg.RESULTS_DIR_ANNUAL


def is_quarterly():
    """Return True if config is in quarterly mode."""
    return _QUARTERLY_MODE


# ---------------------------------------------------------------------------
# Experiment dimensions
# ---------------------------------------------------------------------------
METHODS = ["random_forest"]
STRUCTURES = ["single_model", "per_dy"]
TARGETS = ["raw", "log1p"]
ZERO_HANDLING = ["include_zeros", "two_stage"]

# ---------------------------------------------------------------------------
# Training objectives (revision v2).  Each objective fixes the regression
# method, the target transform and the loss / split criterion.  The
# experiment matrix is OBJECTIVES x STRUCTURES x ZERO_HANDLING.  The two
# mse_* objectives are the first-submission configurations and keep their
# legacy names (random_forest__<structure>__<target>__<zero_handling>);
# the others are named <method_label>__<structure>__<target>__<zero_handling>.
# ---------------------------------------------------------------------------
OBJECTIVE_SPECS = {
    "mse_raw": {
        "method": "random_forest", "method_label": "random_forest",
        "target": "raw", "reg_kwargs": {"criterion": "squared_error"},
    },
    "mse_log1p": {
        "method": "random_forest", "method_label": "random_forest",
        "target": "log1p", "reg_kwargs": {"criterion": "squared_error"},
    },
    # proper scoring rules on the original scale (referee point D2)
    "poisson_rf": {
        "method": "random_forest", "method_label": "random_forest_poisson",
        "target": "raw", "reg_kwargs": {"criterion": "poisson"},
    },
    "poisson_hgb": {
        "method": "hgb", "method_label": "hgb_poisson",
        "target": "raw", "reg_kwargs": {"loss": "poisson"},
    },
    "gamma_hgb": {  # gamma deviance needs y > 0: two-stage only
        "method": "hgb", "method_label": "hgb_gamma",
        "target": "raw", "reg_kwargs": {"loss": "gamma"}, "requires_positive": True,
    },
    # LightGBM deviance objectives (optional: only if lightgbm is importable).
    # sklearn's HistGradientBoosting has no cap on the Newton leaf step, and
    # with a zero-inflated heavy-tailed target the Poisson step
    # sum(y)/sum(exp(raw)) - 1 diverges (results_v2, 29 Aug 2026: 7k-17k test
    # cells predicted above the largest observed payment).  LightGBM caps the
    # step at poisson_max_delta_step (default 0.7; stated explicitly here) for
    # both the Poisson and the Tweedie loss.
    "poisson_lgbm": {
        "method": "lightgbm", "method_label": "lightgbm_poisson",
        "target": "raw", "reg_kwargs": {"objective": "poisson", "poisson_max_delta_step": 0.7},
        "optional": True,
    },
    "tweedie_lgbm": {
        "method": "lightgbm", "method_label": "lightgbm_tweedie",
        "target": "raw", "reg_kwargs": {"objective": "tweedie", "poisson_max_delta_step": 0.7},
        "optional": True,
    },
}
OBJECTIVES = ["mse_raw", "mse_log1p", "poisson_rf", "poisson_hgb", "gamma_hgb"]

# ---------------------------------------------------------------------------
# Cross-fitted calibration (Section 4 of the revised manuscript)
# ---------------------------------------------------------------------------
CALIB_N_FOLDS = 3                 # K; overridable via --n-folds
CALIB_CV_N_ESTIMATORS_CAP = 100   # trees/iterations per fold model; None = no cap
CALIB_CLIP = (0.5, 5.0)           # first-submission clip bounds
CALIB_K_FLOOR = 50                # first-submission floor in the heuristic Z
CALIB_MIN_GROUP_N = 10            # smallest group (rows) given its own factor
CALIB_MIN_GROUP_POS = 10          # ... and it needs at least this many positive cells

# Default model name that ibnr_model.combine_with_rbns pulls from the RBNS
# results when building the combined RBNS+IBNR report. Kept here so a change
# to the experiment matrix surfaces in one place instead of silently skipping
# the combine step.
DEFAULT_BEST_MODEL = "random_forest__per_dy__log1p__two_stage"

# Dict key used in pipeline_calibration_factors_ to hold the global (non
# per-claim-type) calibration factor. Lives with the config constants so the
# key is defined once rather than repeated across models, ibnr, evaluate,
# and plotting code.
GLOBAL_CALIB_KEY = "__global__"

# ---------------------------------------------------------------------------
# Optuna defaults
# ---------------------------------------------------------------------------
OPTUNA_N_TRIALS = 50  # per model/config; overridable via CLI


def generate_experiment_configs(objectives=None, structures=None):
    """
    Generate all experiment configuration dicts: OBJECTIVES x STRUCTURES x
    ZERO_HANDLING, skipping objectives whose loss needs positive targets when
    zero_handling is include_zeros.  `structures` may include "multi_step"
    (direct multi-step models, one per horizon with history frozen h years
    before the target — Remark "Two remedies" (ii)); it is not in STRUCTURES,
    so it runs only when requested explicitly.
    """
    objectives = list(objectives) if objectives else list(OBJECTIVES)
    structures = list(structures) if structures else list(STRUCTURES)
    configs = []
    for objective, structure, zero_hdl in product(objectives, structures, ZERO_HANDLING):
        spec = OBJECTIVE_SPECS[objective]
        if spec.get("requires_positive") and zero_hdl != "two_stage":
            continue
        configs.append(
            {
                "objective": objective,
                "method": spec["method"],
                "structure": structure,
                "target": spec["target"],
                "zero_handling": zero_hdl,
                "name": f"{spec['method_label']}__{structure}__{spec['target']}__{zero_hdl}",
            }
        )
    return configs


# ---------------------------------------------------------------------------
# Hyperparameter search spaces  (consumed by Optuna objectives in models.py)
# Each function receives an optuna.Trial and returns a param dict.
# ---------------------------------------------------------------------------


def hp_cart(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 100, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 80, log=True),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
    }


def hp_bagging(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=50),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        "base__max_depth": trial.suggest_int("base__max_depth", 5, 20),
        "base__min_samples_leaf": trial.suggest_int(
            "base__min_samples_leaf", 1, 50, log=True
        ),
    }


def hp_random_forest(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 60, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 40, log=True),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 0.75, None]
        ),
    }


def hp_adaboost(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "base__max_depth": trial.suggest_int("base__max_depth", 2, 10),
    }


def hp_ridge(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True),
    }


def hp_hgb(trial):
    """sklearn HistGradientBoostingRegressor (Poisson / Gamma deviance losses)."""
    return {
        "max_iter": trial.suggest_int("max_iter", 100, 500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 64, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
    }


def hp_lightgbm(trial):
    """LightGBM with the Tweedie objective (only used when lightgbm is installed)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
    }


# Mapping method name -> hp function
HP_SPACES = {
    "cart": hp_cart,
    "bagging": hp_bagging,
    "random_forest": hp_random_forest,
    "adaboost": hp_adaboost,
    "ridge": hp_ridge,
    "hgb": hp_hgb,
    "lightgbm": hp_lightgbm,
}


# ---------------------------------------------------------------------------
# Two-stage classifier hyperparameter space
# ---------------------------------------------------------------------------


def hp_classifier(trial):
    """Hyperparameters for the binary classifier in two-stage models."""
    return {
        "n_estimators": trial.suggest_int("clf_n_estimators", 100, 300, step=100),
        "max_depth": trial.suggest_int("clf_max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("clf_min_samples_leaf", 1, 30, log=True),
        "max_features": trial.suggest_categorical(
            "clf_max_features", ["sqrt", "log2", None]
        ),
    }



# ---------------------------------------------------------------------------
# Train / Validation / Test split logic
# ---------------------------------------------------------------------------


def get_available_ays_for_dy(dy):
    """
    For development year `dy`, return the accident years that have
    observed data for that DY.

    AY `a` has observed DY `k` if a + k - 1 <= MAX_ACCIDENT_YEAR (=10).
    So available AYs for DY k: 1 .. (10 - k + 1).
    """
    return list(range(1, MAX_ACCIDENT_YEAR - dy + 2))


def get_val_ay_for_dy(dy):
    """The latest available AY for DY `dy` — used as validation."""
    return MAX_ACCIDENT_YEAR - dy + 1


def get_train_ays_for_dy(dy):
    """All AYs except the validation AY.
    When only one AY is available, use it for both training and validation."""
    available = get_available_ays_for_dy(dy)
    if len(available) <= 1:
        return available
    val_ay = get_val_ay_for_dy(dy)
    return [ay for ay in available if ay != val_ay]


def get_test_ays_for_dy(dy):
    """AYs that need DY `dy` predicted (unobservable)."""
    return list(range(MAX_ACCIDENT_YEAR - dy + 2, MAX_ACCIDENT_YEAR + 1))
