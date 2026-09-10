#!/usr/bin/env python3
"""
Run the Gabrielli (2021) subnet NN on MTPL data.

Implements the full pipeline:
  1. Data preparation (MTPL -> Gabrielli wide format)
  2. Step 1: Train embeddings
  3. Step 2: Train dense layers (embeddings frozen)
  4. Predict reserves
  5. Compare with Chain Ladder and RF results

Usage:
  python run_gabrielli_nn.py
  python run_gabrielli_nn.py --epochs1 100 --epochs2 150
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category=FutureWarning)

import src.config as cfg
from src.gabrielli_data_prep import (
    N_DY,
    load_and_prepare,
    build_prediction_data,
    compute_starting_values,
    compute_loss_weights,
)
from src.gabrielli_model import GabrielliNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Gabrielli NN on MTPL data")
    p.add_argument(
        "--epochs1", type=int, default=80, help="Epochs for step 1 (embedding training)"
    )
    p.add_argument(
        "--epochs2", type=int, default=120, help="Epochs for step 2 (dense training)"
    )
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument(
        "--avg-epochs",
        type=int,
        default=5,
        help="Number of neighbouring epochs to average predictions over",
    )
    return p.parse_args()


# -----------------------------------------------------------------------
# Build PyTorch tensors from the DataFrame
# -----------------------------------------------------------------------


def df_to_tensors(data):
    """Convert Gabrielli-format DataFrame to dict of tensors."""
    B = len(data)

    claim_type = torch.tensor(data["claim_type_code"].values, dtype=torch.long)
    ay = torch.tensor(data["AY_code"].values, dtype=torch.long)
    aq = torch.tensor(data["AQ_code"].values, dtype=torch.long)
    report_dy = torch.tensor(data["report_dy_code"].values, dtype=torch.long)

    # (B,D) Numeric features: notidel_raw, log_cum_paid_obs, no_payments_raw
    numeric_cols = ["notidel_raw", "log_cum_paid_obs", "no_payments_raw"]
    numeric_features = torch.tensor(data[numeric_cols].values, dtype=torch.float32)

    # Past payment info: (B, N_DY-1)
    pay_info_cols = [f"Pay{t:02d}Info" for t in range(1, N_DY)]
    pay_info = torch.tensor(data[pay_info_cols].values, dtype=torch.long)

    # Time_Known: (B, N_DY)
    tk_cols = [f"Time_Known{t:02d}" for t in range(1, N_DY + 1)]
    time_known = torch.tensor(data[tk_cols].values, dtype=torch.float32)

    # Time_Predict_Indicator: (B, N_DY)
    tpi_cols = [f"Time_Predict_Indicator{t:02d}" for t in range(1, N_DY + 1)]
    time_pred_ind = torch.tensor(data[tpi_cols].values, dtype=torch.float32)

    # Time_Predict_Payment: (B, N_DY)
    tpp_cols = [f"Time_Predict_Payment{t:02d}" for t in range(1, N_DY + 1)]
    time_pred_pay = torch.tensor(data[tpp_cols].values, dtype=torch.float32)

    # Response: PayInd (B, N_DY) and LogPay (B, N_DY)
    ind_cols = [f"PayInd{t:02d}" for t in range(1, N_DY + 1)]
    log_cols = [f"LogPay{t:02d}" for t in range(1, N_DY + 1)]
    pay_ind = torch.tensor(data[ind_cols].values, dtype=torch.float32)
    log_pay = torch.tensor(data[log_cols].values, dtype=torch.float32)

    return {
        "claim_type": claim_type,
        "ay": ay,
        "aq": aq,
        "report_dy": report_dy,
        "numeric_features": numeric_features,
        "pay_info": pay_info,
        "time_known": time_known,
        "time_pred_ind": time_pred_ind,
        "time_pred_pay": time_pred_pay,
        "pay_ind": pay_ind,
        "log_pay": log_pay,
    }


def make_dataloader(tensors, batch_size, shuffle=True):
    """Create a DataLoader from the tensor dict."""
    ds = TensorDataset(
        tensors["claim_type"],
        tensors["ay"],
        tensors["aq"],
        tensors["report_dy"],
        tensors["numeric_features"],
        tensors["pay_info"],
        tensors["time_known"],
        tensors["time_pred_ind"],
        tensors["time_pred_pay"],
        tensors["pay_ind"],
        tensors["log_pay"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# -----------------------------------------------------------------------
# Loss function
# -----------------------------------------------------------------------


def gabrielli_loss(
    pred_ind, pred_mean, true_ind, true_log, time_pred_ind, time_pred_pay, loss_weights
):
    """
    Weighted multi-task loss: BCE for indicators + MSE for log-payments.

    Parameters
    ----------
    pred_ind : (B, N_DY)    predicted payment probabilities
    pred_mean : (B, N_DY)   predicted log-payment means
    true_ind : (B, N_DY)    true payment indicators (0/1)
    true_log : (B, N_DY)    true log-payments (0 where no payment)
    time_pred_ind : (B, N_DY)  masks for indicator loss
    time_pred_pay : (B, N_DY)  masks for payment loss
    loss_weights : (N_DY, 2)   per-DY weights

    Returns
    -------
    total_loss : scalar tensor
    """
    total = torch.tensor(0.0, device=pred_ind.device)
    eps = 1e-7

    for t in range(N_DY):
        mask_ind = time_pred_ind[:, t]
        mask_pay = time_pred_pay[:, t]
        w_ind = loss_weights[t, 0]
        w_pay = loss_weights[t, 1]

        # BCE for indicator — divide by batch size n (not eligible count),
        # matching Gabrielli's R code which pads with zeros to divide by n.
        B = float(pred_ind.size(0))
        if mask_ind.sum() > 0:
            p = torch.clamp(pred_ind[:, t], eps, 1 - eps)
            bce = -(
                true_ind[:, t] * torch.log(p) + (1 - true_ind[:, t]) * torch.log(1 - p)
            )
            bce = (bce * mask_ind).sum() / B
            total = total + w_ind * bce

        # MSE for log-payment — same: divide by batch size n
        if mask_pay.sum() > 0:
            mse = (pred_mean[:, t] - true_log[:, t]) ** 2
            mse = (mse * mask_pay).sum() / B
            total = total + w_pay * mse

    return total


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------


def train_epoch(model, loader, optimizer, loss_weights_t):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        (ct, ay, aq, rd, nf, pi, tk, tpi, tpp, true_ind, true_log) = [
            b.to(device) for b in batch
        ]

        optimizer.zero_grad()
        pred_ind, pred_mean = model(ct, ay, aq, rd, nf, pi, tk, tpi, tpp)
        loss = gabrielli_loss(
            pred_ind, pred_mean, true_ind, true_log, tpi, tpp, loss_weights_t
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# -----------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------


@torch.no_grad()
def predict_all(model, data_tensors, batch_size=10000):
    """Run prediction and return (indicators, means) as numpy arrays."""
    model.eval()
    loader = make_dataloader(data_tensors, batch_size, shuffle=False)

    all_ind = []
    all_mean = []
    for batch in loader:
        (ct, ay, aq, rd, nf, pi, tk, tpi, tpp, _, _) = [b.to(device) for b in batch]
        pred_ind, pred_mean = model(
            ct, ay, aq, rd, nf, pi, tk, tpi, tpp, use_dropout=False
        )
        all_ind.append(pred_ind.cpu().numpy())
        all_mean.append(pred_mean.cpu().numpy())

    return np.vstack(all_ind), np.vstack(all_mean)


def compute_sigma_squared(model, data, data_tensors, batch_size=10000):
    """
    Compute log-normal variance correction sigma^2 per DY.

    Per Gabrielli (2021) eq (5.7), the balance-property formula:
        sigma^2_t = max(2 * log(sum_actual / sum_exp_mu), 1e-9)

    where sum_actual = sum of actual payments for known DYs with predicted
    p>0, and sum_exp_mu = sum of p_hat * exp(mu_hat) over known DYs.
    This ensures aggregate predictions match aggregate actuals per DY.
    """
    pred_ind, pred_mean = predict_all(model, data_tensors, batch_size)
    sigma_sq = np.full(N_DY, 1e-9)

    for t in range(N_DY):
        col_pay = f"Pay{t + 1:02d}"
        col_known = f"Time_Known{t + 1:02d}"

        known = data[col_known].values == 1.0
        actual = data[col_pay].values
        p_pred = pred_ind[:, t]
        m_pred = pred_mean[:, t]

        # Balance-property formula (eq 5.7):
        # sigma^2 = 2 * log(sum(actual[known & p>0]) / sum(p * exp(mu) [known]))
        numerator = actual[known & (p_pred > 0)].sum()
        denominator = (p_pred[known] * np.exp(m_pred[known])).sum()

        if denominator > 0 and numerator > 0:
            s2 = 2.0 * np.log(numerator / denominator)
            sigma_sq[t] = max(s2, 1e-9)

    return sigma_sq


def compute_reserves(data_pred, pred_ind, pred_mean, sigma_sq):
    """
    Compute individual RBNS reserves from NN predictions.

    Reserve for claim i at DY t = P(pay>0) * exp(mu + sigma^2/2)
    Summed over all unobserved DYs.

    Returns per-claim reserves and per-AY aggregates.
    """
    n = len(data_pred)
    claim_reserves = np.zeros(n)

    for t in range(N_DY):
        col_tpi = f"Time_Predict_Indicator{t + 1:02d}"
        mask = data_pred[col_tpi].values == 1.0  # unobserved DYs

        p = pred_ind[:, t]
        mu = pred_mean[:, t]
        s2 = sigma_sq[t]

        # E[payment] = P(pay>0) * exp(mu + sigma^2/2)
        expected = p * np.exp(mu + s2 / 2)
        claim_reserves += expected * mask

    # Per-AY aggregation
    ay_reserves = (
        pd.DataFrame(
            {
                "accident_year": data_pred["accident_year"].values,
                "predicted_reserve": claim_reserves,
                "true_reserve": data_pred["true_reserve"].values,
            }
        )
        .groupby("accident_year")
        .sum()
        .reset_index()
    )
    ay_reserves["error"] = (
        ay_reserves["predicted_reserve"] - ay_reserves["true_reserve"]
    )
    ay_reserves["error_pct"] = (
        ay_reserves["error"] / ay_reserves["true_reserve"].clip(lower=1) * 100
    )

    return claim_reserves, ay_reserves


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    args = parse_args()
    if args.seed is not None:
        import random
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    total_start = time.time()

    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_gabrielli_nn"
    )
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("GABRIELLI (2021) SUBNET NN ON MTPL DATA")
    print(f"  Development years: {N_DY}")
    print(f"  Step 1 epochs: {args.epochs1}")
    print(f"  Step 2 epochs: {args.epochs2}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {device}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data preparation
    # ------------------------------------------------------------------
    print("\n[1/5] Preparing data...")
    step_start = time.time()
    data = load_and_prepare(verbose=True)

    starting_values = compute_starting_values(data)
    loss_weights = compute_loss_weights(data)
    loss_weights_t = torch.tensor(loss_weights, dtype=torch.float32, device=device)

    print(f"\n  Starting values (P(pay>0), mean log-pay) per DY:")
    for t in range(N_DY):
        print(
            f"    DY {t + 1:2d}: P={starting_values[t, 0]:.4f}, "
            f"mu={starting_values[t, 1]:.2f}"
        )

    # Build tensors
    tensors_train = df_to_tensors(data)
    loader_train = make_dataloader(tensors_train, args.batch_size)
    print(f"  Data prep: {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # 2. Step 1: Train with embeddings
    # ------------------------------------------------------------------
    print(f"\n[2/5] Step 1: Training embeddings ({args.epochs1} epochs)...")
    step_start = time.time()

    model = GabrielliNN(
        starting_values=starting_values,
    ).to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    best_epoch1 = 0
    for epoch in range(1, args.epochs1 + 1):
        loss = train_epoch(model, loader_train, optimizer, loss_weights_t)
        if loss < best_loss:
            best_loss = loss
            best_epoch1 = epoch
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{args.epochs1}: loss={loss:.4f}")

    print(f"  Step 1 best: epoch {best_epoch1}, loss={best_loss:.4f}")
    print(f"  Step 1 time: {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # 3. Step 2: Freeze embeddings, train dense layers
    # ------------------------------------------------------------------
    print(f"\n[3/5] Step 2: Training dense layers ({args.epochs2} epochs)...")
    step_start = time.time()

    model.freeze_embeddings()
    # New optimizer for only trainable params
    optimizer2 = torch.optim.NAdam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    best_loss2 = float("inf")
    best_epoch2 = 0
    checkpoints = {}
    for epoch in range(1, args.epochs2 + 1):
        loss = train_epoch(model, loader_train, optimizer2, loss_weights_t)
        if loss < best_loss2:
            best_loss2 = loss
            best_epoch2 = epoch
        # Save checkpoint for last N epochs (for averaging)
        if epoch >= args.epochs2 - args.avg_epochs:
            checkpoints[epoch] = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{args.epochs2}: loss={loss:.4f}")

    print(f"  Step 2 best: epoch {best_epoch2}, loss={best_loss2:.4f}")
    print(f"  Step 2 time: {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # 4. Compute sigma^2 and reserves
    # ------------------------------------------------------------------
    print("\n[4/5] Computing reserves...")
    step_start = time.time()

    # Average predictions over the last avg_epochs checkpoints
    data_pred = build_prediction_data(data)
    tensors_pred = df_to_tensors(data_pred)

    avg_ind = np.zeros((len(data_pred), N_DY))
    avg_mean = np.zeros((len(data_pred), N_DY))
    avg_sigma = np.zeros(N_DY)

    n_ckpts = len(checkpoints)
    if n_ckpts == 0:
        # No checkpoints — use current model state
        n_ckpts = 1
        sigma_sq = compute_sigma_squared(model, data, tensors_train)
        p_ind, p_mean = predict_all(model, tensors_pred)
        avg_ind = p_ind
        avg_mean = p_mean
        avg_sigma = sigma_sq
    else:
        for ep, state in checkpoints.items():
            model.load_state_dict(state)
            model.to(device)
            sigma_sq = compute_sigma_squared(model, data, tensors_train)
            p_ind, p_mean = predict_all(model, tensors_pred)
            avg_ind += p_ind
            avg_mean += p_mean
            avg_sigma += sigma_sq
        avg_ind /= n_ckpts
        avg_mean /= n_ckpts
        avg_sigma /= n_ckpts

    print(f"  Averaged over {n_ckpts} checkpoints")
    print(f"  Sigma^2 per DY: {np.round(avg_sigma, 4)}")

    # Compute reserves
    claim_reserves, ay_reserves = compute_reserves(
        data_pred, avg_ind, avg_mean, avg_sigma
    )

    total_predicted = ay_reserves["predicted_reserve"].sum()
    total_true = ay_reserves["true_reserve"].sum()
    total_error = total_predicted - total_true
    total_error_pct = total_error / total_true * 100

    print(f"\n  Reserve computation: {time.time() - step_start:.1f}s")

    # ------------------------------------------------------------------
    # 5. Results comparison
    # ------------------------------------------------------------------
    print("\n[5/5] Results...")

    print(f"\n{'=' * 70}")
    print(f"GABRIELLI NN RESULTS (MTPL Data)")
    print(f"{'=' * 70}")
    print(f"\n  Total predicted reserve:  {total_predicted:>15,.0f}")
    print(f"  Total true RBNS reserve:  {total_true:>15,.0f}")
    print(
        f"  Total error:              {total_error:>15,.0f} ({total_error_pct:+.2f}%)"
    )

    print(f"\n  Per Accident Year:")
    print(
        f"  {'AY':>4s}  {'Predicted':>14s}  {'True':>14s}  {'Error':>14s}  {'Error%':>8s}"
    )
    print(f"  {'-' * 60}")
    for _, row in ay_reserves.iterrows():
        print(
            f"  {int(row['accident_year']):4d}  {row['predicted_reserve']:14,.0f}  "
            f"{row['true_reserve']:14,.0f}  {row['error']:14,.0f}  "
            f"{row['error_pct']:+8.1f}%"
        )

    # Save results
    ay_reserves.to_csv(
        os.path.join(results_dir, "gabrielli_nn_reserves_by_ay.csv"), index=False
    )

    # Load existing RF and CL results for comparison
    rf_results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "model_comparison.csv"
    )
    cl_results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "chain_ladder_results.csv",
    )

    print(f"\n{'=' * 70}")
    print(f"COMPARISON (on same MTPL data)")
    print(f"{'=' * 70}")

    comparison_rows = [
        {
            "model": "Gabrielli NN",
            "total_predicted": total_predicted,
            "total_true": total_true,
            "total_error": total_error,
            "error_pct": total_error_pct,
        }
    ]

    if os.path.exists(cl_results_path):
        cl = pd.read_csv(cl_results_path)
        cl_reserve = cl["CL_reserve"].sum()
        cl_true = cl["true_reserve"].sum()
        cl_err = cl_reserve - cl_true
        cl_pct = cl_err / cl_true * 100 if cl_true != 0 else 0
        comparison_rows.append(
            {
                "model": "Chain Ladder",
                "total_predicted": cl_reserve,
                "total_true": cl_true,
                "total_error": cl_err,
                "error_pct": cl_pct,
            }
        )

    if os.path.exists(rf_results_path):
        rf = pd.read_csv(rf_results_path)
        for _, row in rf.iterrows():
            if row["method"] == "random_forest":
                comparison_rows.append(
                    {
                        "model": f"RF ({row['structure']}/{row['target']}/{row['zero_handling']})",
                        "total_predicted": row["total_predicted_reserve"],
                        "total_true": row["total_true_reserve"],
                        "total_error": row["total_reserve_error"],
                        "error_pct": row["total_reserve_error_pct"],
                    }
                )

    comp_df = pd.DataFrame(comparison_rows)
    comp_df["abs_error_pct"] = comp_df["error_pct"].abs()
    comp_df = comp_df.sort_values("abs_error_pct")

    print(f"\n  {'Model':<50s}  {'Predicted':>14s}  {'True':>14s}  {'Error%':>8s}")
    print(f"  {'-' * 90}")
    for _, row in comp_df.iterrows():
        print(
            f"  {row['model']:<50s}  {row['total_predicted']:14,.0f}  "
            f"{row['total_true']:14,.0f}  {row['error_pct']:+8.2f}%"
        )

    comp_df.to_csv(os.path.join(results_dir, "comparison.csv"), index=False)

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"COMPLETE — Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Results saved to: {results_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
