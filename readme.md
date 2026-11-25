# 🏭 Celeris Automobili Industrial — Demand Forecasting & Planning Control Tower

An end-to-end data engineering, probabilistic demand forecasting, and interactive planning application built for **Celeris Automobili Industrial** (MKS AI/ML Hackathon).

This system cleans noisy ERP historical data (`2019-01` to `2026-10`), reconstructs unconstrained customer demand across stockout periods, forecasts 19 months ahead (`2026-11` to `2028-05`) with P10/P50/P90 uncertainty intervals, and serves insights through an interactive **Control Tower** web dashboard.

---

## 📁 Project Structure

```text
Celeris control tower/
├── data/
│   ├── raw/                  # Original hackathon CSV files
│   ├── clean/                # Cleaned Parquet datasets & data quality audit JSON
│   └── outputs/              # Feature tables, anomaly register, backtest metrics & forecasts
├── src/
│   ├── st1_clean.py          # Step 1: Data cleaning, deduplication & audit report
│   ├── st2_anomalies.py      # Step 2: Anomaly detection & unconstrained demand recovery
│   ├── st3_features.py       # Step 3: Macro lags, EV shift, lifecycle & intermittency features
│   └── st4_forecast.py       # Step 4: Backtesting, P10/P50/P90 ML models & submission export
├── app.py                    # Step 5: Interactive Streamlit + Plotly Control Tower dashboard
└── README.md                 # Project documentation & setup guide
```

---

## ✨ Key Capabilities

1. **Automated Data Quality & Audit Engine (`st1_clean.py`)**

   * Standardizes mixed date formats (`YYYY-MM`), SKU codes (`AAA-NNN`), regions, sales channels, and fuzzy-matches 220+ customer spelling variants into 12 canonical accounts.
   * Deduplicates repeated `Record_ID` transactions using the latest `Load_Timestamp`.
   * Detects and repairs currency/unit scaling anomalies and reconciles `Units_Sold × Unit_Price_USD = Revenue_USD`.
   * Outputs a machine-readable `data_health_report.json` tracking every fix.

2. **Anomaly Register & Unconstrained Demand Reconstruction (`st2_anomalies.py`)**

   * Recovers true customer demand (`Unconstrained_Units`) during recorded stockouts (`inventory_stockouts.csv`) and inferred supply disruptions.
   * Flags pre-price-hike forward buying (December surges), one-off demand spikes, and product lifecycle window mismatches into `anomaly_register.parquet`.

3. **Domain Feature Engineering (`st3_features.py`)**

   * **Macroeconomic Lead/Lag Drivers:** Incorporates 1–3 month lagged `ISM_Manufacturing_PMI`, `Auto_Production_YoY_Pct`, `New_Vehicle_Orders_Index`, and `Raw_Material_Cost_Index`.
   * **EV Transition Modeling:** Interacts `EV_Adoption_Index` with SKU `Powertrain_Application` (`ICE`, `EV`, `Universal`).
   * **Lifecycle & Replacement Inheritance:** Enforces `Launch_Date` and `EOL_Date` windows and transfers historical demand baselines via `Replaces_SKU` for newly launched parts.
   * **Intermittency Profiling:** Classifies SKU-customer series into *Smooth*, *Erratic*, *Intermittent*, or *Lumpy* using ADI and $CV^2$.

4. **Probabilistic Forecasting & Backtesting (`st4_forecast.py`)**

   * Evaluates performance on a 19-month time-based validation window using **WAPE** and **Forecast Bias** against rolling and seasonal baselines.
   * Generates **P10, P50, and P90** demand intervals using Gradient Boosted Quantile Regressors and exports `submission.csv`.

5. **Interactive Planning Control Tower (`app.py`)**

   * **Overview:** Unified continuous timeline (`2019-01` → `2028-05`) for both Units and Revenue with P10–P90 bands, plus Family, Region, and Channel breakdowns.
   * **Data Health:** Before-and-after cleaning metrics and issue-by-issue audit summary.
   * **Demand & Lifecycle:** Multi-level drill-down (`Family → SKU → Region → Customer`), seasonality profile, zero-order months, and SKU Launch/EOL tracking.
   * **Anomalies:** Interactive time-series markers displaying anomaly types, root causes, and recommended planner actions.
   * **Forecast & Accuracy:** Backtest WAPE/Bias scorecards by Family/Region, horizon breakdown (`M1–M6`, `M7–M12`, `M13–M19`), and one-click CSV export.
   * **Macro & What-If:** Interactive sliders to simulate shifts in **PMI** and **EV Adoption**, plus a macro correlation heatmap.
   * **Procurement & Alerts:** P90 safety-buffer procurement plan and historical stockout/lost-sales tracker.

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites

* Python **3.10+**
* Place the 7 raw competition CSV files inside `data/raw/`.

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn pyarrow streamlit plotly
```

### 3. Run the Data & Forecasting Pipeline

Run the four pipeline scripts in sequence from the project root:

```bash
python src/st1_clean.py
python src/st2_anomalies.py
python src/st3_features.py
python src/st4_forecast.py
```

### 4. Launch the Control Tower Dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal (typically `http://localhost:8501`).

---

## 📊 Key Outputs Generated

* `data/clean/train_unconstrained.parquet` — Cleaned historical demand adjusted for stockouts.
* `data/clean/data_health_report.json` — Automated audit log of data cleaning fixes.
* `data/outputs/anomaly_register.parquet` — Flagged historical anomalies with reasons and actions.
* `data/outputs/backtest_metrics.parquet` — Validation WAPE and Bias across families and regions.
* `data/outputs/forecast_full.parquet` — Granular 19-month P10/P50/P90 forecasts.
* `data/outputs/submission.csv` — Final competition forecast file.