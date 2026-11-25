import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def event_matches_row(row, ev):
    if not (ev["Start_Date_YM"] <= row["Date"] <= ev["End_Date_YM"]):
        return False
    stype = ev["Scope_Type"]
    sval = str(ev["Scope_Value"]).strip()
    if stype == "Global" or sval.lower() == "all":
        return True
    if stype == "Region":
        return row["Region"] == sval
    if stype == "Product_Family":
        return row["Product_Family"] == sval
    if stype == "Channel":
        return row["Sales_Channel"] == sval
    if stype == "Customer":
        if "|" in sval:
            cust, reg = [x.strip() for x in sval.split("|", 1)]
            return (row["Customer_Name"] == cust) and (row["Region"] == reg)
        return row["Customer_Name"] == sval
    return False


def run_step2():
    train = pd.read_parquet(CLEAN_DIR / "train_clean.parquet")
    sku = pd.read_parquet(CLEAN_DIR / "sku_master_clean.parquet")
    events = pd.read_parquet(CLEAN_DIR / "events_clean.parquet")
    stockouts = pd.read_parquet(CLEAN_DIR / "stockouts_clean.parquet")

    so_agg = (
        stockouts.groupby(["Date", "SKU_ID", "Region"], as_index=False)
        .agg(
            {
                "Requested_Units": "sum",
                "Shipped_Units": "sum",
                "Backordered_Units": "sum",
                "Lost_Units": "sum",
                "Stockout_Days": "max",
                "Reason": lambda s: " | ".join(sorted(set(s.dropna().astype(str)))),
            }
        )
    )

    df = train.merge(so_agg, on=["Date", "SKU_ID", "Region"], how="left")
    df = df.merge(
        sku[["SKU_ID", "Launch_Date", "EOL_Date", "Lifecycle_Status", "Powertrain_Application"]],
        on="SKU_ID",
        how="left",
    )

    df = df.sort_values(["SKU_ID", "Region", "Customer_Name", "Date"]).reset_index(drop=True)
    grp_cols = ["SKU_ID", "Region", "Customer_Name"]

    rolling_med = df.groupby(grp_cols)["Units_Sold"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=2).median()
    )
    rolling_std = df.groupby(grp_cols)["Units_Sold"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=2).std()
    ).fillna(0.0)

    df["Unconstrained_Units"] = df["Units_Sold"]
    known_so_mask = df["Requested_Units"].notna() & (df["Requested_Units"] > df["Units_Sold"])
    df.loc[known_so_mask, "Unconstrained_Units"] = df.loc[known_so_mask, "Requested_Units"]

    inferred_so_mask = (
        (~known_so_mask)
        & (rolling_med > 20)
        & (df["Units_Sold"] < 0.35 * rolling_med)
        & (df["Month"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
    )
    df.loc[inferred_so_mask, "Unconstrained_Units"] = rolling_med.loc[inferred_so_mask]

    df["Unconstrained_Revenue_USD"] = df["Unconstrained_Units"] * df["Unit_Price_USD"]

    anomaly_records = []

    for idx, r in df[known_so_mask].iterrows():
        anomaly_records.append(
            {
                "Date": r["Date"],
                "SKU_ID": r["SKU_ID"],
                "Product_Family": r["Product_Family"],
                "Region": r["Region"],
                "Customer_Name": r["Customer_Name"],
                "Anomaly_Type": "Recorded Stockout",
                "Observed_Units": r["Units_Sold"],
                "Adjusted_Units": r["Unconstrained_Units"],
                "Reason": r["Reason"] if pd.notna(r["Reason"]) else "Supply constraint",
                "Action": "Replace Units_Sold with Requested_Units for unconstrained demand",
            }
        )

    for idx, r in df[inferred_so_mask].iterrows():
        anomaly_records.append(
            {
                "Date": r["Date"],
                "SKU_ID": r["SKU_ID"],
                "Product_Family": r["Product_Family"],
                "Region": r["Region"],
                "Customer_Name": r["Customer_Name"],
                "Anomaly_Type": "Inferred Stockout / Drop",
                "Observed_Units": r["Units_Sold"],
                "Adjusted_Units": r["Unconstrained_Units"],
                "Reason": "Sharp drop (<35% of 6M median) outside normal baseline",
                "Action": "Impute unconstrained demand from rolling 6M median",
            }
        )

    spike_mask = (
        (~known_so_mask)
        & (~inferred_so_mask)
        & (rolling_med > 0)
        & (df["Units_Sold"] > rolling_med + 3.0 * np.maximum(rolling_std, 10.0))
    )
    for idx, r in df[spike_mask].iterrows():
        atype = "Pre-Price-Hike Forward Buying" if r["Month"] == 12 else "Demand Spike / One-Off Order"
        action = "Keep in history with event/seasonality flag; inspect for one-off surge"
        anomaly_records.append(
            {
                "Date": r["Date"],
                "SKU_ID": r["SKU_ID"],
                "Product_Family": r["Product_Family"],
                "Region": r["Region"],
                "Customer_Name": r["Customer_Name"],
                "Anomaly_Type": atype,
                "Observed_Units": r["Units_Sold"],
                "Adjusted_Units": r["Unconstrained_Units"],
                "Reason": "Exceeds 6M rolling median + 3*SD",
                "Action": action,
            }
        )

    lifecycle_viol = (
        (df["Launch_Date"].notna() & (df["Date"] < df["Launch_Date"]))
        | (df["EOL_Date"].notna() & (df["Date"] > df["EOL_Date"]))
    )
    for idx, r in df[lifecycle_viol].iterrows():
        anomaly_records.append(
            {
                "Date": r["Date"],
                "SKU_ID": r["SKU_ID"],
                "Product_Family": r["Product_Family"],
                "Region": r["Region"],
                "Customer_Name": r["Customer_Name"],
                "Anomaly_Type": "Lifecycle Window Mismatch",
                "Observed_Units": r["Units_Sold"],
                "Adjusted_Units": r["Unconstrained_Units"],
                "Reason": f"Sale outside Launch ({r['Launch_Date']}) - EOL ({r['EOL_Date']})",
                "Action": "Audit master data dates vs. legacy/run-out shipments",
            }
        )

    anomaly_df = pd.DataFrame(anomaly_records)
    anomaly_df.to_parquet(OUTPUT_DIR / "anomaly_register.parquet", index=False)

    drop_cols = [
        "Requested_Units",
        "Shipped_Units",
        "Backordered_Units",
        "Lost_Units",
        "Stockout_Days",
        "Reason",
    ]
    train_unconstrained = df.drop(columns=drop_cols)
    train_unconstrained.to_parquet(CLEAN_DIR / "train_unconstrained.parquet", index=False)

    summary = {
        "total_clean_rows": len(df),
        "known_stockouts_adjusted": int(known_so_mask.sum()),
        "inferred_stockouts_adjusted": int(inferred_so_mask.sum()),
        "demand_spikes_flagged": int(spike_mask.sum()),
        "lifecycle_mismatches_flagged": int(lifecycle_viol.sum()),
        "total_anomaly_register_rows": len(anomaly_df),
        "historical_units_before": float(df["Units_Sold"].sum()),
        "historical_units_unconstrained": float(df["Unconstrained_Units"].sum()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_step2()
