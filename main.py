#!/usr/bin/env python3
"""
ML-Based Claims Reserving Pipeline — Entry Point.

Runs:
  objectives (mse_raw, mse_log1p, poisson_rf, poisson_hgb, gamma_hgb)
  × 2 model structures × 2 zero-handling  = 20 fitted pipelines (gamma is
  two-stage only), each evaluated under every post-hoc correction arm of
  src/corrections.py, + the Chain Ladder aggregate baseline (RBNS-only triangle).

Usage:
  python main.py                           # full run (50 Optuna trials each)
  python main.py --n-trials 10             # quick tuning (10 trials)
  python main.py --quick                   # minimal run for testing (5 trials)
  python main.py --n-jobs 4                # parallel execution
  python main.py --objectives mse_log1p poisson_rf   # subset of objectives
  python main.py --n-folds 10              # calibration folds (default 3)
"""

import argparse
import time
import sys
import os
import warnings
import pandas as pd
import src.config as cfg

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _apply_quick_hp_caps():
    """
    Override HP search spaces to cap n_estimators at 100 for fast testing.
    This mutates the HP_SPACES dict in-place so Optuna uses smaller models.
    """
    import src.config as cfg

    _orig_hp_rf = cfg.HP_SPACES["random_forest"]
    _orig_hp_bag = cfg.HP_SPACES["bagging"]
    _orig_hp_ada = cfg.HP_SPACES["adaboost"]

    def hp_rf_quick(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 60, log=True
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 40, log=True),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

    def hp_bag_quick(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100, step=50),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "base__max_depth": trial.suggest_int("base__max_depth", 5, 20),
            "base__min_samples_leaf": trial.suggest_int(
                "base__min_samples_leaf", 1, 50, log=True
            ),
        }

    def hp_ada_quick(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "base__max_depth": trial.suggest_int("base__max_depth", 2, 6),
        }

    def hp_hgb_quick(trial):
        return {
            "max_iter": trial.suggest_int("max_iter", 50, 100, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 31, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        }

    cfg.HP_SPACES["random_forest"] = hp_rf_quick
    cfg.HP_SPACES["bagging"] = hp_bag_quick
    cfg.HP_SPACES["adaboost"] = hp_ada_quick
    cfg.HP_SPACES["hgb"] = hp_hgb_quick

    # Also cap default params in models module (already imported at module level)
    import src.models as mod

    for method in mod.DEFAULT_PARAMS:
        for key in ("n_estimators", "max_iter"):
            if key in mod.DEFAULT_PARAMS[method]:
                mod.DEFAULT_PARAMS[method][key] = min(mod.DEFAULT_PARAMS[method][key], 100)
    if "n_estimators" in mod.DEFAULT_CLF_PARAMS:
        mod.DEFAULT_CLF_PARAMS["n_estimators"] = min(
            mod.DEFAULT_CLF_PARAMS["n_estimators"], 100
        )

    # Reduce tuning subsample size for quick mode
    mod.QUICK_MODE = True


def parse_args():
    parser = argparse.ArgumentParser(description="ML-Based Claims Reserving Pipeline")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model (default: 50)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to run (default: random_forest only)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run: 5 Optuna trials, only random_forest",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1 for sequential)",
    )
    parser.add_argument(
        "--quarterly",
        action="store_true",
        help="Use quarterly data (40x40 triangle) instead of annual (10x10)",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=None,
        help="Subset of training objectives (default: all in config.OBJECTIVES)",
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        default=None,
        choices=list(cfg.STRUCTURES) + ["multi_step"],
        help="Subset of model structures (default: all in config.STRUCTURES)",
    )
    parser.add_argument(
        "--zero-handling",
        nargs="+",
        default=None,
        choices=cfg.ZERO_HANDLING,
        help="Subset of zero-handling schemes (default: all in config.ZERO_HANDLING)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help=f"Calibration folds K (default: config.CALIB_N_FOLDS = {cfg.CALIB_N_FOLDS})",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Write outputs here instead of config.RESULTS_DIR (keeps first-submission artefacts intact)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip all computation; re-render plots from full_data_* CSVs "
        "produced by a previous full (non-quick) run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quarterly:
        cfg.set_quarterly_mode(True)
    if args.results_dir:
        cfg.RESULTS_DIR = os.path.abspath(args.results_dir)
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # Suppress some noisy warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    total_start = time.time()

    if args.plot_only:
        from src.evaluate import plot_from_full_data

        print("Re-rendering plots from full_data_* cache...")
        plot_from_full_data()
        print(
            f"\nPIPELINE COMPLETE — Total time: "
            f"{time.time() - total_start:.1f}s"
        )
        return

    # Quick mode overrides
    if args.quick:
        args.n_trials = 5
        if args.methods is None:
            args.methods = ["random_forest"]
        # Cap n_estimators in HP spaces for faster tuning
        _apply_quick_hp_caps()
        # Limit structures for quick mode (per_dy trains 9 models per config)
        args._quick_structures = ["single_model"]

    from src.data_prep import prepare_data
    from src.feature_engineering import engineer_features
    from src.chain_ladder import run_chain_ladder
    from src.train import run_all_experiments
    from src.evaluate import (
        build_comparison_table,
        build_correction_table,
        plot_results,
        save_results,
        save_feature_importances,
        print_summary,
    )

    print("=" * 60)
    print("ML-BASED CLAIMS RESERVING PIPELINE")
    print(f"Data mode: {'QUARTERLY (40x40)' if args.quarterly else 'ANNUAL (10x10)'}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Data preparation
    # ------------------------------------------------------------------
    print("\n[1/5] Loading and preparing data...")
    step_start = time.time()
    claim_dy, claims_rbns, transactions_obs, triangles = prepare_data()
    print(f"  Data preparation: {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Feature engineering
    # ------------------------------------------------------------------
    print("\n[2/5] Engineering features...")
    step_start = time.time()
    claim_dy, feature_cols, feature_cols_per_dy = engineer_features(
        claim_dy, transactions_obs
    )
    print(f"  Feature engineering: {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Chain Ladder baseline
    # ------------------------------------------------------------------
    print("\n[3/5] Running Chain Ladder baseline...")
    step_start = time.time()
    chain_ladder_results = run_chain_ladder()
    print(f"  Chain Ladder: {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Run all ML experiments
    # ------------------------------------------------------------------
    print("\n[4/5] Running ML experiments...")
    step_start = time.time()
    all_results = run_all_experiments(
        claim_dy,
        feature_cols,
        feature_cols_per_dy,
        n_trials=args.n_trials,
        methods=args.methods,
        n_jobs=args.n_jobs,
        structures=args.structures or getattr(args, "_quick_structures", None),
        zero_handling=args.zero_handling,
        persist_models_for=(cfg.DEFAULT_BEST_MODEL if not args.quick else None),
        persist_dir=cfg.RESULTS_DIR,
        objectives=args.objectives,
        n_folds=args.n_folds,
        partial_dir=os.path.join(cfg.RESULTS_DIR, "partial"),
        # (structures / zero_handling filters are applied inside run_all_experiments)
    )
    print(f"  ML experiments: {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # Step 5: Evaluate, compare, and save
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating and saving results...")
    step_start = time.time()

    comparison_df = build_comparison_table(all_results, chain_ladder_results)

    # Create results directory
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # Save results (full runs also emit full_data_* copies for --plot-only)
    is_full_run = not args.quick
    save_results(comparison_df, all_results, chain_ladder_results, is_full_run=is_full_run)

    # Save feature importances
    save_feature_importances(all_results, is_full_run=is_full_run)

    # Correction arms: total reserve error (%) by configuration x arm
    corr_df = build_correction_table(all_results)
    if not corr_df.empty:
        pivot = corr_df.pivot(index="name", columns="arm", values="total_reserve_error_pct")
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print("\nTotal reserve error (%) by correction arm:")
            print(pivot.round(1).to_string())

    # Generate plots
    if not args.no_plots:
        try:
            plot_results(all_results, chain_ladder_results, comparison_df)
        except Exception as e:
            print(f"  [WARN] Plot generation failed: {e}")

    # Print summary
    print_summary(comparison_df, chain_ladder_results)

    print(f"\n  Evaluation: {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(
        f"PIPELINE COMPLETE — Total time: {total_time:.1f}s ({total_time / 60:.1f} min)"
    )
    print(f"Results saved to: {cfg.RESULTS_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
