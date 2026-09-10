#!/usr/bin/env python3
"""
Convert a raw claims-transaction CSV to the format expected by the
ML reserving pipeline.

Input:  data/raw_claims.csv
        Semicolon-delimited, columns: ID, Type, Accident, Reporting, Payment, Amount
        Dates are Excel serial numbers (base: 1899-12-30).

Output (written to data/converted/):
    - claims_individual.csv
    - transactions_individual.csv
    - triangle_observed.csv, triangle_full.csv, triangle_incremental.csv
    - triangle_observed_rbns.csv, triangle_full_rbns.csv, triangle_incremental_rbns.csv

The conversion maps 10 calendar years (2000-2009) to accident years 1-10,
with an evaluation date of 2009-12-31 (end of AY 10).  AY 1-10 all have
complete 10-year development histories (payments through 2018), so the
full data is a true 10×10 square.  The evaluation cutoff creates a triangle
within this square; the held-out lower-right cells have ground truth.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXCEL_EPOCH = datetime(1899, 12, 30)  # Excel serial date base
# FIRST_YEAR is coupled to src.config.BASE_YEAR — keep them in sync.
# (This script is standalone and intentionally does not import src/.)
FIRST_YEAR = 2000  # Maps to accident_year = 1
LAST_YEAR = 2009  # Maps to accident_year = 10  (only AY 1-10)
I = LAST_YEAR - FIRST_YEAR + 1  # 10 accident years
MAX_DY = 10  # Maximum development lag for triangles
EVAL_DATE = datetime(2009, 12, 31)  # Evaluation date (end of AY 10)
# We read ALL payments (including post-2010) for claims in AY 1-11
# to get ground truth in the held-out lower-right triangle.
LAST_PAYMENT_YEAR = 2019  # Latest payment year in data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "raw_claims.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "converted")


def excel_serial_to_date(serial):
    """Convert Excel serial number to Python datetime."""
    return EXCEL_EPOCH + timedelta(days=float(serial))


def build_triangles(transactions, claims_agg):
    """
    Build all 6 triangle CSVs from the transactions table.

    Returns dict of name -> DataFrame.
    """
    dy_names = [f"DY{d}" for d in range(1, MAX_DY + 1)]

    def tri_to_df(tri_matrix):
        df_tri = pd.DataFrame(tri_matrix, columns=dy_names)
        df_tri.insert(0, "accident_year", range(1, I + 1))
        return df_tri

    # --- Full triangles (all payments, including unobserved) ---
    tri_full = np.zeros((I, MAX_DY))
    for ay in range(1, I + 1):
        for dy in range(1, MAX_DY + 1):
            mask = (transactions["accident_year"] == ay) & (
                transactions["dev_lag"] <= dy
            )
            tri_full[ay - 1, dy - 1] = transactions.loc[mask, "payment_size"].sum()

    # --- Observed triangle (only payments that are observable) ---
    tri_obs = np.full((I, MAX_DY), np.nan)
    for ay in range(1, I + 1):
        for dy in range(1, MAX_DY + 1):
            if ay + dy - 1 <= I:
                mask = (
                    (transactions["accident_year"] == ay)
                    & (transactions["dev_lag"] <= dy)
                    & (transactions["observed_flag"] == 1)
                )
                tri_obs[ay - 1, dy - 1] = transactions.loc[mask, "payment_size"].sum()

    # --- Incremental triangle (from full) ---
    tri_inc = np.zeros((I, MAX_DY))
    for ay in range(I):
        tri_inc[ay, 0] = tri_full[ay, 0]
        for dy in range(1, MAX_DY):
            tri_inc[ay, dy] = tri_full[ay, dy] - tri_full[ay, dy - 1]

    # --- RBNS-only triangles (exclude IBNR claims) ---
    rbns_claims = set(claims_agg[claims_agg["ibnr_flag"] == 0]["claim_no"])
    txn_rbns = transactions[transactions["claim_no"].isin(rbns_claims)]

    tri_full_rbns = np.zeros((I, MAX_DY))
    tri_obs_rbns = np.full((I, MAX_DY), np.nan)
    for ay in range(1, I + 1):
        for dy in range(1, MAX_DY + 1):
            mask_full = (txn_rbns["accident_year"] == ay) & (txn_rbns["dev_lag"] <= dy)
            tri_full_rbns[ay - 1, dy - 1] = txn_rbns.loc[
                mask_full, "payment_size"
            ].sum()

            if ay + dy - 1 <= I:
                mask_obs = mask_full & (txn_rbns["observed_flag"] == 1)
                tri_obs_rbns[ay - 1, dy - 1] = txn_rbns.loc[
                    mask_obs, "payment_size"
                ].sum()

    tri_inc_rbns = np.zeros((I, MAX_DY))
    for ay in range(I):
        tri_inc_rbns[ay, 0] = tri_full_rbns[ay, 0]
        for dy in range(1, MAX_DY):
            tri_inc_rbns[ay, dy] = tri_full_rbns[ay, dy] - tri_full_rbns[ay, dy - 1]

    return {
        "triangle_observed": tri_to_df(tri_obs),
        "triangle_full": tri_to_df(tri_full),
        "triangle_incremental": tri_to_df(tri_inc),
        "triangle_observed_rbns": tri_to_df(tri_obs_rbns),
        "triangle_full_rbns": tri_to_df(tri_full_rbns),
        "triangle_incremental_rbns": tri_to_df(tri_inc_rbns),
    }


def main():
    print("=" * 60)
    print("Raw Claims Data Conversion")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==================================================================
    # 1. LOAD AND PARSE RAW DATA
    # ==================================================================
    print("\n1. Loading raw claims data...")

    raw = pd.read_csv(
        INPUT_FILE,
        sep=";",
        encoding="utf-8-sig",  # handles BOM
        skip_blank_lines=True,
    )

    # Strip whitespace from column names
    raw.columns = raw.columns.str.strip()
    print(f"   Raw rows: {len(raw):,}")
    print(f"   Columns: {list(raw.columns)}")

    # Drop any fully-NaN rows (from blank lines)
    raw = raw.dropna(how="all").reset_index(drop=True)
    print(f"   After dropping blank rows: {len(raw):,}")

    # Validate required columns
    required = {"ID", "Type", "Accident", "Reporting", "Payment", "Amount"}
    missing = required - set(raw.columns)
    if missing:
        print(f"   ERROR: Missing columns: {missing}")
        sys.exit(1)

    # ==================================================================
    # 2. CONVERT DATES AND COMPUTE PERIODS
    # ==================================================================
    print("\n2. Converting dates...")

    raw["accident_date"] = raw["Accident"].apply(excel_serial_to_date)
    raw["reporting_date"] = raw["Reporting"].apply(excel_serial_to_date)
    raw["payment_date"] = raw["Payment"].apply(excel_serial_to_date)

    raw["accident_cal_year"] = raw["accident_date"].dt.year
    raw["reporting_cal_year"] = raw["reporting_date"].dt.year
    raw["payment_cal_year"] = raw["payment_date"].dt.year

    print(
        f"   Accident year range: {raw['accident_cal_year'].min()} - "
        f"{raw['accident_cal_year'].max()}"
    )
    print(
        f"   Payment year range:  {raw['payment_cal_year'].min()} - "
        f"{raw['payment_cal_year'].max()}"
    )
    print(f"   Claim types: {raw['Type'].value_counts().to_dict()}")

    # ==================================================================
    # 3. FILTER TO TARGET PERIOD
    # ==================================================================
    print(f"\n3. Filtering to accident years {FIRST_YEAR}-{LAST_YEAR}...")

    df = raw[
        (raw["accident_cal_year"] >= FIRST_YEAR)
        & (raw["accident_cal_year"] <= LAST_YEAR)
    ].copy()
    print(f"   Rows with accident in [{FIRST_YEAR}, {LAST_YEAR}]: {len(df):,}")

    # Keep payments through LAST_PAYMENT_YEAR (2019) for ground truth,
    # but drop anything beyond that (shouldn't exist in this dataset)
    n_future = (df["payment_cal_year"] > LAST_PAYMENT_YEAR).sum()
    if n_future > 0:
        print(f"   Dropping {n_future:,} payments after {LAST_PAYMENT_YEAR}")
        df = df[df["payment_cal_year"] <= LAST_PAYMENT_YEAR].copy()

    n_post_eval = (df["payment_cal_year"] > LAST_YEAR).sum()
    print(f"   Payments after {LAST_YEAR} (ground truth): {n_post_eval:,}")

    # ==================================================================
    # 4. MAP TO PIPELINE PERIODS
    # ==================================================================
    print("\n4. Mapping to pipeline periods...")

    df["accident_year"] = df["accident_cal_year"] - FIRST_YEAR + 1  # 1..10
    df["occurrence_period"] = df["accident_year"]  # same for annual

    df["payment_period"] = df["payment_cal_year"] - FIRST_YEAR + 1
    df["dev_lag"] = df["payment_period"] - df["accident_year"] + 1

    # Filter out dev_lag < 1 (should not happen, but be safe)
    n_bad_devlag = (df["dev_lag"] < 1).sum()
    if n_bad_devlag > 0:
        print(f"   WARNING: {n_bad_devlag} rows with dev_lag < 1 (dropped)")
        df = df[df["dev_lag"] >= 1].copy()

    print(f"   Dev lag range: {df['dev_lag'].min()} - {df['dev_lag'].max()}")
    print(
        f"   Accident year range: {df['accident_year'].min()} - "
        f"{df['accident_year'].max()}"
    )

    # observed_flag: payment is observed if it happened within the evaluation window
    df["observed_flag"] = (df["accident_year"] + df["dev_lag"] - 1 <= I).astype(int)

    n_obs = df["observed_flag"].sum()
    n_unobs = len(df) - n_obs
    print(f"   Observed payments: {n_obs:,}")
    print(f"   Unobserved payments: {n_unobs:,}")

    # ==================================================================
    # 5. CREATE CLAIM ID MAPPING
    # ==================================================================
    print("\n5. Creating claim ID mapping...")

    unique_ids = sorted(df["ID"].unique())
    id_map = {orig_id: new_id for new_id, orig_id in enumerate(unique_ids, start=1)}
    df["claim_no"] = df["ID"].map(id_map)
    n_claims = len(unique_ids)
    print(f"   Unique claims: {n_claims:,}")

    # ==================================================================
    # 6. BUILD TRANSACTIONS TABLE
    # ==================================================================
    print("\n6. Building transactions table...")

    df = df.sort_values(["claim_no", "payment_date"]).reset_index(drop=True)

    # Compute claim_size = total of ALL payments for each claim
    claim_totals = df.groupby("claim_no")["Amount"].sum().reset_index()
    claim_totals.columns = ["claim_no", "claim_size"]
    df = df.merge(claim_totals, on="claim_no", how="left")

    # Build the transactions output
    transactions = df[
        [
            "claim_no",
            "accident_year",
            "occurrence_period",
            "dev_lag",
            "payment_period",
            "claim_size",
            "Amount",
            "observed_flag",
        ]
    ].copy()
    transactions.columns = [
        "claim_no",
        "accident_year",
        "occurrence_period",
        "dev_lag",
        "payment_period",
        "claim_size",
        "payment_size",
        "observed_flag",
    ]

    print(f"   Transactions: {len(transactions):,} rows")
    print(f"   Columns: {list(transactions.columns)}")

    # ==================================================================
    # 7. BUILD CLAIMS TABLE
    # ==================================================================
    print("\n7. Building claims table...")

    # Compute per-claim date-based info
    claim_dates = (
        df.groupby("claim_no")
        .agg(
            accident_year=("accident_year", "first"),
            occurrence_period=("occurrence_period", "first"),
            claim_size=("claim_size", "first"),
            accident_date=("accident_date", "first"),
            reporting_date=("reporting_date", "first"),
            last_payment_date=("payment_date", "max"),
            claim_type=("Type", "first"),
        )
        .reset_index()
    )

    # Notification delay in years
    claim_dates["notidel"] = (
        (
            claim_dates["reporting_date"] - claim_dates["accident_date"]
        ).dt.total_seconds()
        / (365.25 * 24 * 3600)
    ).clip(lower=0)

    # Reporting development year (calendar-based).
    # This must use the same calendar-year convention as payment dev_lag
    # so that the observability rule (AY + DY - 1 <= MAX_AY) correctly
    # identifies which claims have been reported by the evaluation date.
    # Using ceil(notidel) instead would mis-classify IBNR claims whose
    # notidel < 1 year but whose reporting_date falls in the next calendar
    # year (e.g. accident Dec 15 2009, reported Jan 2 2010 → notidel ≈ 0.05
    # → ceil gives DY 1, but calendar gives DY 2, and only DY 2 correctly
    # places it outside the observed triangle for AY 10).
    claim_dates["report_dy"] = (
        claim_dates["reporting_date"].dt.year - claim_dates["accident_date"].dt.year + 1
    ).clip(lower=1)

    # Settlement delay in years
    claim_dates["setldel"] = (
        (
            claim_dates["last_payment_date"] - claim_dates["accident_date"]
        ).dt.total_seconds()
        / (365.25 * 24 * 3600)
    ).clip(lower=0)

    # Cumulative paid: observed vs full
    obs_paid = (
        transactions[transactions["observed_flag"] == 1]
        .groupby("claim_no")["payment_size"]
        .sum()
        .reset_index()
        .rename(columns={"payment_size": "cum_paid_observed"})
    )

    full_paid = (
        transactions.groupby("claim_no")["payment_size"]
        .sum()
        .reset_index()
        .rename(columns={"payment_size": "cum_paid_full"})
    )

    claims_agg = claim_dates.merge(obs_paid, on="claim_no", how="left")
    claims_agg = claims_agg.merge(full_paid, on="claim_no", how="left")
    claims_agg["cum_paid_observed"] = claims_agg["cum_paid_observed"].fillna(0)
    claims_agg["cum_paid_full"] = claims_agg["cum_paid_full"].fillna(0)

    # Reserve = full - observed
    claims_agg["reserve_true"] = (
        claims_agg["cum_paid_full"] - claims_agg["cum_paid_observed"]
    )

    # IBNR flag: claim is IBNR if reporting date > evaluation date
    claims_agg["ibnr_flag"] = (claims_agg["reporting_date"] > EVAL_DATE).astype(int)

    n_ibnr = claims_agg["ibnr_flag"].sum()
    n_rbns = len(claims_agg) - n_ibnr
    print(f"   Total claims: {len(claims_agg):,}")
    print(f"   RBNS claims: {n_rbns:,}")
    print(f"   IBNR claims: {n_ibnr:,}")
    print(f"   Claim types: {claims_agg['claim_type'].value_counts().to_dict()}")

    # Select and order output columns
    claims_out = claims_agg[
        [
            "claim_no",
            "accident_year",
            "occurrence_period",
            "claim_size",
            "notidel",
            "setldel",
            "report_dy",
            "cum_paid_observed",
            "cum_paid_full",
            "reserve_true",
            "ibnr_flag",
            "claim_type",
        ]
    ]

    # ==================================================================
    # 8. SUMMARY STATISTICS
    # ==================================================================
    print("\n8. Summary statistics...")

    print(f"\n   Claims per accident year:")
    ay_counts = claims_out.groupby("accident_year").size()
    for ay in range(1, I + 1):
        cnt = ay_counts.get(ay, 0)
        cal_year = FIRST_YEAR + ay - 1
        print(f"     AY {ay:2d} ({cal_year}): {cnt:,} claims")

    print(f"\n   Transactions per dev lag (observed only):")
    obs_txn = transactions[transactions["observed_flag"] == 1]
    for dy in range(1, MAX_DY + 1):
        cnt = (obs_txn["dev_lag"] == dy).sum()
        print(f"     DY {dy:2d}: {cnt:,} payments")

    beyond_max_dy = (transactions["dev_lag"] > MAX_DY).sum()
    print(f"\n   Payments with dev_lag > {MAX_DY}: {beyond_max_dy:,}")

    avg_claim_size = claims_out["claim_size"].mean()
    median_claim_size = claims_out["claim_size"].median()
    print(f"\n   Average claim size: {avg_claim_size:,.0f}")
    print(f"   Median claim size: {median_claim_size:,.0f}")

    avg_notidel = claims_out["notidel"].mean()
    avg_setldel = claims_out["setldel"].mean()
    print(f"   Average notification delay: {avg_notidel:.3f} years")
    print(f"   Average settlement delay: {avg_setldel:.3f} years")

    # ==================================================================
    # 9. BUILD TRIANGLES
    # ==================================================================
    print("\n9. Building triangles...")

    triangles = build_triangles(transactions, claims_agg)

    tri_obs_df = triangles["triangle_observed"]
    print(f"   Triangle dimensions: {I} AY x {MAX_DY} DY")
    print(f"   Observed cells (non-NaN): {tri_obs_df.iloc[:, 1:].notna().sum().sum()}")

    print(f"\n   Observed triangle diagonal (cumulative to last observed DY):")
    for ay in range(1, I + 1):
        max_obs_dy = min(I - ay + 1, MAX_DY)
        if max_obs_dy >= 1:
            val = tri_obs_df.loc[ay - 1, f"DY{max_obs_dy}"]
            cal_year = FIRST_YEAR + ay - 1
            print(f"     AY {ay:2d} ({cal_year}), DY {max_obs_dy:2d}: {val:,.0f}")

    # ==================================================================
    # 10. SAVE OUTPUT FILES
    # ==================================================================
    print(f"\n10. Saving to {OUTPUT_DIR}...")

    claims_out.to_csv(os.path.join(OUTPUT_DIR, "claims_individual.csv"), index=False)
    transactions.to_csv(
        os.path.join(OUTPUT_DIR, "transactions_individual.csv"), index=False
    )

    for name, tri_df in triangles.items():
        tri_df.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)

    print(f"\n   Files written:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"     {f}: {size_kb:.1f} KB")

    # ==================================================================
    # 11. VALIDATION CHECKS
    # ==================================================================
    print("\n11. Validation checks...")

    # Check: claim_no is sequential 1..N
    assert claims_out["claim_no"].min() == 1
    assert claims_out["claim_no"].max() == n_claims
    print(f"   [OK] claim_no is sequential 1..{n_claims}")

    # Check: all transaction claim_nos exist in claims table
    txn_claims = set(transactions["claim_no"].unique())
    claims_set = set(claims_out["claim_no"])
    assert txn_claims.issubset(claims_set)
    print("   [OK] All transaction claim_nos exist in claims table")

    # Check: cum_paid_full >= cum_paid_observed for all claims
    assert (claims_out["cum_paid_full"] >= claims_out["cum_paid_observed"] - 0.01).all()
    print("   [OK] cum_paid_full >= cum_paid_observed for all claims")

    # Check: reserve_true >= 0 (or very close)
    assert (claims_out["reserve_true"] >= -0.01).all()
    print("   [OK] reserve_true >= 0 for all claims")

    # Check: triangle observed matches sum of observed transactions
    for ay in range(1, I + 1):
        max_obs_dy = min(I - ay + 1, MAX_DY)
        if max_obs_dy >= 1:
            tri_val = tri_obs_df.loc[ay - 1, f"DY{max_obs_dy}"]
            txn_val = transactions[
                (transactions["accident_year"] == ay)
                & (transactions["dev_lag"] <= max_obs_dy)
                & (transactions["observed_flag"] == 1)
            ]["payment_size"].sum()
            assert abs(tri_val - txn_val) < 0.01, (
                f"Triangle mismatch at AY {ay}, DY {max_obs_dy}: {tri_val} vs {txn_val}"
            )
    print("   [OK] Observed triangle matches transaction sums")

    # Final summary
    total_observed = claims_out["cum_paid_observed"].sum()
    total_full = claims_out["cum_paid_full"].sum()
    total_reserve = claims_out["reserve_true"].sum()
    print(f"\n   Total observed paid: {total_observed:,.0f}")
    print(f"   Total full paid:     {total_full:,.0f}")
    print(f"   Total true reserve:  {total_reserve:,.0f}")
    if total_full > 0:
        print(f"   Reserve ratio:       {total_reserve / total_full * 100:.1f}%")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
