from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ym_to_period_num(ym_series):
    dt = pd.to_datetime(ym_series + "-01", errors="coerce")
    return dt.dt.year * 12 + dt.dt.month


def run_step3():
    train = pd.read_parquet(CLEAN_DIR / "train_unconstrained.parquet")
    test = pd.read_parquet(CLEAN_DIR / "test_clean.parquet")
    sku = pd.read_parquet(CLEAN_DIR / "sku_master_clean.parquet")
    macro = pd.read_parquet(CLEAN_DIR / "macro_clean.parquet")

    train["is_test"] = 0
    train["target"] = train["Unconstrained_Units"].astype(float)

    test["is_test"] = 1
    test["target"] = np.nan

    keep_cols = [
        "Date",
        "Year",
        "Month",
        "Product_Family",
        "Business_Unit",
        "SKU_ID",
        "SKU_Description",
        "Customer_Name",
        "Sales_Channel",
        "Region",
        "is_test",
        "target",
    ]
    full = pd.concat([train[keep_cols], test[keep_cols]], ignore_index=True)

    sku_meta_cols = [
        "SKU_ID",
        "Size_Class",
        "Powertrain_Application",
        "Launch_Date",
        "EOL_Date",
        "Replaces_SKU",
        "Lifecycle_Status",
    ]
    full = full.merge(sku[sku_meta_cols].drop_duplicates("SKU_ID"), on="SKU_ID", how="left")

    # Inherit baseline demand from Replaces_SKU for planned/new SKUs
    sku_hist_avg = (
        train.groupby("SKU_ID")["Unconstrained_Units"]
        .mean()
        .rename("sku_hist_avg_units")
    )
    full = full.merge(sku_hist_avg, on="SKU_ID", how="left")
    full = full.merge(
        sku_hist_avg.rename("replaced_sku_avg_units"),
        left_on="Replaces_SKU",
        right_index=True,
        how="left",
    )
    full["effective_baseline_units"] = full["sku_hist_avg_units"].fillna(full["replaced_sku_avg_units"]).fillna(0.0)

    # Lifecycle window indicator
    cur_p = ym_to_period_num(full["Date"])
    launch_p = ym_to_period_num(full["Launch_Date"])
    eol_p = ym_to_period_num(full["EOL_Date"])

    after_launch = launch_p.isna() | (cur_p >= launch_p)
    before_eol = eol_p.isna() | (cur_p <= eol_p)
    full["in_lifecycle_window"] = (after_launch & before_eol).astype(int)

    # Merge macro indicators and compute lags (1, 2, 3 months)
    macro_cols = [
        "Date",
        "ISM_Manufacturing_PMI",
        "Auto_Production_YoY_Pct",
        "Vehicle_Inventory_Days",
        "Raw_Material_Cost_Index",
        "New_Vehicle_Orders_Index",
        "Industrial_Util_Rate_Pct",
        "Fuel_Price_Index",
        "EV_Adoption_Index",
    ]
    macro_sorted = macro[macro_cols].sort_values("Date").reset_index(drop=True)
    for col in macro_cols:
        if col != "Date":
            for lag in [1, 2, 3]:
                macro_sorted[f"{col}_lag{lag}"] = macro_sorted[col].shift(lag).bfill()

    full = full.merge(macro_sorted, on="Date", how="left")

    # EV powertrain shift interaction
    pt_map = {"ICE": -1.0, "Universal": 0.0, "EV": 1.0}
    full["pt_score"] = full["Powertrain_Application"].map(pt_map).fillna(0.0)
    full["ev_shift_interaction"] = full["EV_Adoption_Index"] * full["pt_score"]

    # Sort by series and time for lag & intermittency features
    series_cols = ["SKU_ID", "Region", "Customer_Name", "Sales_Channel"]
    full = full.sort_values(series_cols + ["Date"]).reset_index(drop=True)

    # Intermittency & ordering-cycle stats from historical rows only
    hist_only = full[full["is_test"] == 0].copy()
    hist_only["is_pos"] = (hist_only["target"] > 0).astype(int)

    interm = (
        hist_only.groupby(["SKU_ID", "Region", "Customer_Name"], as_index=False)
        .agg(
            obs_months=("target", "count"),
            pos_months=("is_pos", "sum"),
            hist_mean=("target", "mean"),
            hist_std=("target", "std"),
        )
    )
    interm["nonzero_rate"] = interm["pos_months"] / np.maximum(interm["obs_months"], 1)
    interm["adi"] = np.where(interm["pos_months"] > 0, interm["obs_months"] / np.maximum(interm["pos_months"], 1), 99.0)
    interm["cv2"] = np.where(interm["hist_mean"] > 0, (interm["hist_std"].fillna(0.0) / interm["hist_mean"]) ** 2, 0.0)

    def classify_intermittency(row):
        if row["adi"] < 1.32 and row["cv2"] < 0.49:
            return "Smooth"
        if row["adi"] < 1.32 and row["cv2"] >= 0.49:
            return "Erratic"
        if row["adi"] >= 1.32 and row["cv2"] < 0.49:
            return "Intermittent"
        return "Lumpy"

    interm["intermittency_class"] = interm.apply(classify_intermittency, axis=1)
    full = full.merge(
        interm[["SKU_ID", "Region", "Customer_Name", "nonzero_rate", "adi", "cv2", "intermittency_class"]],
        on=["SKU_ID", "Region", "Customer_Name"],
        how="left",
    )
    full["nonzero_rate"] = full["nonzero_rate"].fillna(0.0)
    full["adi"] = full["adi"].fillna(1.0)
    full["cv2"] = full["cv2"].fillna(0.0)
    full["intermittency_class"] = full["intermittency_class"].fillna("Smooth")

    # Cyclical calendar features
    full["month_sin"] = np.sin(2 * np.pi * full["Month"] / 12.0)
    full["month_cos"] = np.cos(2 * np.pi * full["Month"] / 12.0)
    full["is_dec_pre_hike"] = (full["Month"] == 12).astype(int)

    # Historical lag & rolling features (computed on known history so no future leakage)
    hist_sorted = full[full["is_test"] == 0].copy()
    g_hist = hist_sorted.groupby(series_cols)["target"]
    for lag in [1, 2, 3, 6, 12]:
        hist_sorted[f"lag_{lag}"] = g_hist.shift(lag)
    hist_sorted["roll_mean_3"] = g_hist.shift(1).rolling(3, min_periods=1).mean()
    hist_sorted["roll_mean_6"] = g_hist.shift(1).rolling(6, min_periods=1).mean()
    hist_sorted["roll_std_6"] = g_hist.shift(1).rolling(6, min_periods=1).std().fillna(0.0)

    # Carry forward last known lag/rolling values into test rows per series
    lag_cols = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "roll_mean_3", "roll_mean_6", "roll_std_6"]
    last_vals = hist_sorted.groupby(series_cols, as_index=False)[lag_cols].last()

    test_part = full[full["is_test"] == 1].drop(columns=lag_cols, errors="ignore").merge(
        last_vals, on=series_cols, how="left"
    )
    for col in lag_cols:
        test_part[col] = test_part[col].fillna(test_part["effective_baseline_units"])

    featured = pd.concat([hist_sorted, test_part], ignore_index=True)
    featured.to_parquet(OUTPUT_DIR / "features_full.parquet", index=False)

    print("Step 3 Complete!")
    print("Total rows in features_full:", len(featured))
    print("Intermittency breakdown:\n", featured[featured["is_test"] == 0]["intermittency_class"].value_counts())


if __name__ == "__main__":
    run_step3()
