from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def normalize_sku(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper().replace("_", "-").replace(" ", "")
    m = re.match(r"^([A-Z]{3})-?(\d+)$", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):03d}"
    return s


def normalize_date(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    dt = pd.to_datetime(s, format="mixed", errors="coerce")
    return dt.strftime("%Y-%m") if pd.notna(dt) else np.nan


def normalize_region(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    mapping = {
        "north america": "North America",
        "n. america": "North America",
        "na": "North America",
        "europe": "Europe",
        "eu": "Europe",
        "asia pacific": "Asia Pacific",
        "asia-pacific": "Asia Pacific",
        "apac": "Asia Pacific",
        "india": "India",
        "latin america": "Latin America",
        "latin-america": "Latin America",
        "latam": "Latin America",
        "middle east & africa": "Middle East & Africa",
        "middle east and africa": "Middle East & Africa",
        "me&a": "Middle East & Africa",
        "mea": "Middle East & Africa",
    }
    return mapping.get(s, val)


def normalize_channel(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace("-", " ")
    mapping = {
        "direct oem": "Direct OEM",
        "oem direct": "Direct OEM",
        "tier 1 distributor": "Tier-1 Distributor",
        "t1 distributor": "Tier-1 Distributor",
        "aftermarket": "Aftermarket",
        "after market": "Aftermarket",
        "export": "Export",
    }
    return mapping.get(s, val)


def build_customer_cleaner(canonical_names):
    canon_list = sorted({str(c).strip() for c in canonical_names if pd.notna(c) and str(c).strip()})
    canon_norm = [
        (c, re.sub(r"[^a-z0-9]+", " ", c.lower()).strip(), re.sub(r"[^a-z0-9]+", " ", c.lower()).split())
        for c in canon_list
    ]
    cache = {}

    def _clean(val):
        if pd.isna(val):
            return "Unknown Customer"
        key = str(val)
        if key in cache:
            return cache[key]
        s = re.sub(r"\binc\.?\b", " ", key.strip().lower())
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        if not s:
            cache[key] = "Unknown Customer"
            return "Unknown Customer"
        tokens = s.split()
        best_name = key.strip()
        best_score = -1.0
        for cand, c_str, c_tokens in canon_norm:
            token_hits = 0.0
            for t in tokens:
                if t in c_tokens:
                    token_hits += 1.5
                else:
                    best_t = max((SequenceMatcher(None, t, ct).ratio() for ct in c_tokens), default=0.0)
                    if best_t >= 0.72:
                        token_hits += best_t
            score = token_hits + 2.0 * SequenceMatcher(None, s, c_str).ratio()
            if score > best_score:
                best_score = score
                best_name = cand
        res = best_name if best_score >= 1.8 else key.strip()
        cache[key] = res
        return res

    return _clean


def clean_numeric_units(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", "")
    if s in ["-", "", "nan", "None", "null"]:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def run_step1():
    audit = {"files": {}}

    sku = pd.read_csv(RAW_DIR / "sku_master.csv")
    sku["SKU_ID"] = sku["SKU_ID"].apply(normalize_sku)
    sku["Launch_Date"] = sku["Launch_Date"].apply(normalize_date)
    sku["EOL_Date"] = sku["EOL_Date"].apply(normalize_date)
    sku.to_parquet(CLEAN_DIR / "sku_master_clean.parquet", index=False)

    macro = pd.read_csv(RAW_DIR / "macro_indicators.csv")
    macro_nulls_before = int(macro.isna().sum().sum())
    macro["Date"] = macro["Date"].apply(normalize_date)
    macro = macro.sort_values("Date").reset_index(drop=True)
    macro["Year"] = macro["Date"].str[:4].astype(int)
    macro["Month"] = macro["Date"].str[5:7].astype(int)
    num_cols = macro.select_dtypes(include=[np.number]).columns
    macro[num_cols] = macro[num_cols].interpolate(method="linear").bfill().ffill()
    macro.to_parquet(CLEAN_DIR / "macro_clean.parquet", index=False)
    audit["files"]["macro_indicators.csv"] = {
        "rows": len(macro),
        "nulls_fixed": macro_nulls_before,
    }

    events = pd.read_csv(RAW_DIR / "events_calendar.csv")
    events["Start_Date_YM"] = events["Start_Date"].apply(normalize_date)
    events["End_Date_YM"] = events["End_Date"].apply(normalize_date).fillna("2028-05")
    events.to_parquet(CLEAN_DIR / "events_clean.parquet", index=False)

    stockouts = pd.read_csv(RAW_DIR / "inventory_stockouts.csv")
    stockouts["SKU_ID"] = stockouts["SKU_ID"].apply(normalize_sku)
    stockouts["Date"] = stockouts["Date"].apply(normalize_date)
    stockouts["Region"] = stockouts["Region"].apply(normalize_region)
    stockouts.to_parquet(CLEAN_DIR / "stockouts_clean.parquet", index=False)

    test = pd.read_csv(RAW_DIR / "test.csv")
    test["Date"] = test["Date"].apply(normalize_date)
    test["Year"] = test["Date"].str[:4].astype(int)
    test["Month"] = test["Date"].str[5:7].astype(int)
    test["SKU_ID"] = test["SKU_ID"].apply(normalize_sku)
    test["Region"] = test["Region"].apply(normalize_region)
    test["Sales_Channel"] = test["Sales_Channel"].apply(normalize_channel)

    clean_customer = build_customer_cleaner(test["Customer_Name"].unique())
    test["Customer_Name"] = test["Customer_Name"].apply(clean_customer)
    test.to_parquet(CLEAN_DIR / "test_clean.parquet", index=False)

    train = pd.read_csv(RAW_DIR / "train.csv", low_memory=False)
    raw_rows = len(train)
    raw_customers = int(train["Customer_Name"].nunique())

    train["Load_Timestamp"] = pd.to_datetime(train["Load_Timestamp"], errors="coerce")
    train = train.sort_values("Load_Timestamp").drop_duplicates(subset=["Record_ID"], keep="last")
    dedup_removed = raw_rows - len(train)

    train["Date"] = train["Date"].apply(normalize_date)
    train["Year"] = train["Date"].str[:4].astype(int)
    train["Month"] = train["Date"].str[5:7].astype(int)
    train["SKU_ID"] = train["SKU_ID"].apply(normalize_sku)
    train["Region"] = train["Region"].apply(normalize_region)
    train["Sales_Channel"] = train["Sales_Channel"].apply(normalize_channel)
    train["Customer_Name"] = train["Customer_Name"].apply(clean_customer)

    sku_lookup = sku.set_index("SKU_ID")[["Product_Family", "Business_Unit", "SKU_Description"]]
    for col in ["Product_Family", "Business_Unit", "SKU_Description"]:
        train[col] = train[col].fillna(train["SKU_ID"].map(sku_lookup[col]))

    train["Region"] = train["Region"].fillna(
        train.groupby(["SKU_ID", "Customer_Name"])["Region"].transform(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        )
    )

    train["Units_Sold"] = train["Units_Sold"].apply(clean_numeric_units)
    train["Unit_Price_USD"] = pd.to_numeric(train["Unit_Price_USD"], errors="coerce")
    train["Revenue_USD"] = pd.to_numeric(train["Revenue_USD"], errors="coerce")

    sku_year_price = train.groupby(["SKU_ID", "Year"])["Unit_Price_USD"].transform("median")
    price_ratio = train["Unit_Price_USD"] / sku_year_price
    currency_fixed = int(((price_ratio > 5) | (price_ratio < 0.2)).sum())
    train.loc[(price_ratio > 5) | (price_ratio < 0.2), "Unit_Price_USD"] = sku_year_price

    calc_rev = train["Units_Sold"] * train["Unit_Price_USD"]
    rev_mismatch = int((train["Revenue_USD"].notna() & (np.abs(train["Revenue_USD"] - calc_rev) > 1.0)).sum())
    train["Revenue_USD"] = calc_rev

    missing_units = int(train["Units_Sold"].isna().sum())
    train["Units_Sold"] = train["Units_Sold"].fillna(0.0)

    train = (
        train.groupby(
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
            ],
            as_index=False,
        )
        .agg({"Units_Sold": "sum", "Unit_Price_USD": "median", "Revenue_USD": "sum"})
    )

    train.to_parquet(CLEAN_DIR / "train_clean.parquet", index=False)

    audit["files"]["train.csv"] = {
        "raw_rows": raw_rows,
        "duplicate_record_ids_removed": dedup_removed,
        "customer_variants_collapsed": raw_customers - int(train["Customer_Name"].nunique()),
        "clean_customer_count": int(train["Customer_Name"].nunique()),
        "currency_or_price_outliers_fixed": currency_fixed,
        "revenue_reconciled_rows": rev_mismatch,
        "missing_units_imputed": missing_units,
        "final_clean_rows": len(train),
    }

    with open(CLEAN_DIR / "data_health_report.json", "w") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    run_step1()
