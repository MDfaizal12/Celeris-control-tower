import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_wape_bias(df_eval, y_true_col="y_true", y_pred_col="y_pred"):
    y_true = df_eval[y_true_col].to_numpy(dtype=float)
    y_pred = df_eval[y_pred_col].to_numpy(dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0, 0.0
    wape = float(np.sum(np.abs(y_true - y_pred)) / denom)
    bias = float(np.sum(y_pred - y_true) / denom)
    return wape, bias


def run_step4():
    df = pd.read_parquet(OUTPUT_DIR / "features_full.parquet")
    sample_sub = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "sample_submission.csv")

    hist = df[df["is_test"] == 0].copy()
    test = df[df["is_test"] == 1].copy()

    # Target encoding for high-cardinality IDs (>255 categories limit in HistGradientBoosting)
    global_mean = float(hist["target"].mean())
    sku_enc = hist.groupby("SKU_ID")["target"].mean()
    cust_enc = hist.groupby("Customer_Name")["target"].mean()

    for part in (hist, test):
        part["sku_target_enc"] = part["SKU_ID"].map(sku_enc).fillna(global_mean)
        part["cust_target_enc"] = part["Customer_Name"].map(cust_enc).fillna(global_mean)

    # Keep only categorical features with <= 255 unique categories
    candidate_cats = [
        "Product_Family",
        "Business_Unit",
        "Sales_Channel",
        "Region",
        "Size_Class",
        "Powertrain_Application",
        "Lifecycle_Status",
        "intermittency_class",
    ]
    cat_cols = [c for c in candidate_cats if hist[c].astype(str).nunique() <= 255]

    num_cols = [
        "Year",
        "Month",
        "in_lifecycle_window",
        "effective_baseline_units",
        "sku_target_enc",
        "cust_target_enc",
        "ISM_Manufacturing_PMI",
        "Auto_Production_YoY_Pct",
        "Vehicle_Inventory_Days",
        "Raw_Material_Cost_Index",
        "New_Vehicle_Orders_Index",
        "Industrial_Util_Rate_Pct",
        "Fuel_Price_Index",
        "EV_Adoption_Index",
        "ISM_Manufacturing_PMI_lag1",
        "ISM_Manufacturing_PMI_lag2",
        "ISM_Manufacturing_PMI_lag3",
        "Auto_Production_YoY_Pct_lag1",
        "Auto_Production_YoY_Pct_lag2",
        "Auto_Production_YoY_Pct_lag3",
        "New_Vehicle_Orders_Index_lag1",
        "New_Vehicle_Orders_Index_lag2",
        "New_Vehicle_Orders_Index_lag3",
        "EV_Adoption_Index_lag1",
        "EV_Adoption_Index_lag2",
        "EV_Adoption_Index_lag3",
        "Raw_Material_Cost_Index_lag1",
        "Raw_Material_Cost_Index_lag2",
        "Raw_Material_Cost_Index_lag3",
        "pt_score",
        "ev_shift_interaction",
        "nonzero_rate",
        "adi",
        "cv2",
        "month_sin",
        "month_cos",
        "is_dec_pre_hike",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "roll_mean_3",
        "roll_mean_6",
        "roll_std_6",
    ]

    for c in cat_cols:
        hist[c] = hist[c].astype(str).astype("category")
        test[c] = test[c].astype(str).astype("category")

    feature_cols = cat_cols + num_cols
    cat_idx = list(range(len(cat_cols)))

    # Rolling-origin 19-month validation split (train <= 2025-03, val 2025-04 .. 2026-10)
    val_cutoff = "2025-04"
    tr_sub = hist[hist["Date"] < val_cutoff]
    va_sub = hist[hist["Date"] >= val_cutoff].copy()

    X_tr, y_tr = tr_sub[feature_cols], tr_sub["target"].to_numpy(dtype=float)
    X_va, y_va = va_sub[feature_cols], va_sub["target"].to_numpy(dtype=float)

    model_p50_val = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=150,
        learning_rate=0.06,
        max_depth=8,
        min_samples_leaf=30,
        categorical_features=cat_idx,
        random_state=42,
    )
    model_p50_val.fit(X_tr, y_tr)

    va_sub["y_true"] = y_va
    va_sub["pred_base1"] = np.clip(va_sub["roll_mean_6"].fillna(0.0), 0, None) * va_sub["in_lifecycle_window"]
    va_sub["pred_base2"] = np.clip(
        0.5 * va_sub["lag_12"].fillna(va_sub["roll_mean_3"]) + 0.5 * va_sub["roll_mean_3"].fillna(0.0),
        0,
        None,
    ) * va_sub["in_lifecycle_window"]
    va_sub["pred_ml"] = np.clip(model_p50_val.predict(X_va), 0, None) * va_sub["in_lifecycle_window"]

    wape_b1, bias_b1 = compute_wape_bias(va_sub, "y_true", "pred_base1")
    wape_b2, bias_b2 = compute_wape_bias(va_sub, "y_true", "pred_base2")
    wape_ml, bias_ml = compute_wape_bias(va_sub, "y_true", "pred_ml")

    metrics_rows = []
    for (fam, reg), grp in va_sub.groupby(["Product_Family", "Region"], observed=False):
        w, b = compute_wape_bias(grp, "y_true", "pred_ml")
        metrics_rows.append(
            {
                "Product_Family": str(fam),
                "Region": str(reg),
                "WAPE": w,
                "Bias": b,
                "Actual_Units": float(grp["y_true"].sum()),
                "Forecast_Units": float(grp["pred_ml"].sum()),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_parquet(OUTPUT_DIR / "backtest_metrics.parquet", index=False)

    # Train final P10, P50, P90 models on full training set
    X_all, y_all = hist[feature_cols], hist["target"].to_numpy(dtype=float)
    X_test = test[feature_cols]

    model_p10 = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.10,
        max_iter=150,
        learning_rate=0.06,
        max_depth=8,
        min_samples_leaf=30,
        categorical_features=cat_idx,
        random_state=42,
    )
    model_p50 = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=180,
        learning_rate=0.06,
        max_depth=8,
        min_samples_leaf=30,
        categorical_features=cat_idx,
        random_state=42,
    )
    model_p90 = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.90,
        max_iter=150,
        learning_rate=0.06,
        max_depth=8,
        min_samples_leaf=30,
        categorical_features=cat_idx,
        random_state=42,
    )

    model_p10.fit(X_all, y_all)
    model_p50.fit(X_all, y_all)
    model_p90.fit(X_all, y_all)

    p10_raw = np.clip(model_p10.predict(X_test), 0, None) * test["in_lifecycle_window"].to_numpy()
    p50_raw = np.clip(model_p50.predict(X_test), 0, None) * test["in_lifecycle_window"].to_numpy()
    p90_raw = np.clip(model_p90.predict(X_test), 0, None) * test["in_lifecycle_window"].to_numpy()

    p10_int = np.rint(np.minimum(p10_raw, p50_raw)).astype(int)
    p50_int = np.rint(p50_raw).astype(int)
    p90_int = np.rint(np.maximum(p90_raw, p50_raw)).astype(int)

    forecast_out = test[
        [
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
            "Powertrain_Application",
            "Lifecycle_Status",
            "effective_baseline_units",
        ]
    ].copy()
    forecast_out["P10"] = p10_int
    forecast_out["Units_Sold_Forecast"] = p50_int
    forecast_out["P90"] = p90_int
    forecast_out.to_parquet(OUTPUT_DIR / "forecast_full.parquet", index=False)

    sub_agg = (
        forecast_out.groupby(["Date", "SKU_ID", "Region"], as_index=False)["Units_Sold_Forecast"]
        .sum()
    )
    sub_agg["Units_Sold_Forecast"] = np.clip(np.rint(sub_agg["Units_Sold_Forecast"]), 0, None).astype(int)

    sub_final = sample_sub[["Date", "SKU_ID", "Region"]].merge(
        sub_agg, on=["Date", "SKU_ID", "Region"], how="left"
    )
    sub_final["Units_Sold_Forecast"] = sub_final["Units_Sold_Forecast"].fillna(0).astype(int)
    sub_final.to_csv(OUTPUT_DIR / "submission.csv", index=False)

    summary = {
        "val_rows": len(va_sub),
        "baseline_1_roll6_WAPE": round(wape_b1, 4),
        "baseline_2_lag12_roll3_WAPE": round(wape_b2, 4),
        "ml_p50_WAPE": round(wape_ml, 4),
        "ml_p50_Bias": round(bias_ml, 4),
        "test_forecast_rows": len(forecast_out),
        "submission_rows": len(sub_final),
        "submission_nulls": int(sub_final["Units_Sold_Forecast"].isna().sum()),
        "total_19m_forecast_units_p50": int(forecast_out["Units_Sold_Forecast"].sum()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_step4()
