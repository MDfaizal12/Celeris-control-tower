import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

st.set_page_config(
    page_title="Celeris Control Tower",
    page_icon="🏭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1.5rem;}
    [data-testid="stHeaderActionElements"],
    .stMarkdown a.anchor-link,
    h1 a, h2 a, h3 a {
        display: none !important;
        visibility: hidden !important;
    }
    .kpi-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px 14px;
    }
    .kpi-title {color: #9ca3af; font-size: 0.75rem; text-transform: uppercase;}
    .kpi-val {color: #f9fafb; font-size: 1.35rem; font-weight: 700; margin-top: 2px;}
    .kpi-sub {color: #34d399; font-size: 0.75rem; margin-top: 2px;}
    </style>
    """,
    unsafe_allow_html=True,
)



@st.cache_data(show_spinner="Loading Control Tower data...")
def load_data():
    hist = pd.read_parquet(CLEAN_DIR / "train_unconstrained.parquet")
    forecast = pd.read_parquet(OUTPUT_DIR / "forecast_full.parquet")
    anomalies = pd.read_parquet(OUTPUT_DIR / "anomaly_register.parquet")
    metrics = pd.read_parquet(OUTPUT_DIR / "backtest_metrics.parquet")
    macro = pd.read_parquet(CLEAN_DIR / "macro_clean.parquet")
    sku = pd.read_parquet(CLEAN_DIR / "sku_master_clean.parquet")
    stockouts = pd.read_parquet(CLEAN_DIR / "stockouts_clean.parquet")
    with open(CLEAN_DIR / "data_health_report.json", "r") as f:
        audit = json.load(f)

    latest_prices = (
        hist.sort_values("Date")
        .groupby("SKU_ID", as_index=False, observed=True)["Unit_Price_USD"]
        .last()
    )
    price_map = dict(zip(latest_prices["SKU_ID"], latest_prices["Unit_Price_USD"]))
    med_price = float(hist["Unit_Price_USD"].median())

    forecast = forecast.copy()
    est_price = forecast["SKU_ID"].map(price_map).fillna(med_price)
    forecast["Revenue_P10"] = forecast["P10"] * est_price
    forecast["Revenue_P50"] = forecast["Units_Sold_Forecast"] * est_price
    forecast["Revenue_P90"] = forecast["P90"] * est_price

    if "Product_Family" not in anomalies.columns:
        sku_fam = sku[["SKU_ID", "Product_Family"]].drop_duplicates("SKU_ID")
        anomalies = anomalies.merge(sku_fam, on="SKU_ID", how="left")

    return hist, forecast, anomalies, metrics, macro, sku, stockouts, audit


@st.cache_data(show_spinner=False)
def get_filter_options(hist, sel_fam, sel_reg, sel_ch, sel_cust):
    fams = ["All"] + sorted(hist["Product_Family"].dropna().astype(str).unique().tolist())
    regs = ["All"] + sorted(hist["Region"].dropna().astype(str).unique().tolist())
    chs = ["All"] + sorted(hist["Sales_Channel"].dropna().astype(str).unique().tolist())

    mask = np.ones(len(hist), dtype=bool)
    if sel_fam != "All":
        mask &= hist["Product_Family"].astype(str).to_numpy() == sel_fam
    if sel_reg != "All":
        mask &= hist["Region"].astype(str).to_numpy() == sel_reg
    if sel_ch != "All":
        mask &= hist["Sales_Channel"].astype(str).to_numpy() == sel_ch

    sub = hist.loc[mask]
    custs = ["All"] + sorted(sub["Customer_Name"].dropna().astype(str).unique().tolist())

    if sel_cust != "All":
        sub = sub[sub["Customer_Name"].astype(str) == sel_cust]
    skus = ["All"] + sorted(sub["SKU_ID"].dropna().astype(str).unique().tolist())
    return fams, regs, chs, custs, skus


def fast_mask(df, sel_fam, sel_reg, sel_ch, sel_cust, sel_sku):
    m = np.ones(len(df), dtype=bool)
    if sel_fam != "All" and "Product_Family" in df.columns:
        m &= df["Product_Family"].astype(str).to_numpy() == sel_fam
    if sel_reg != "All" and "Region" in df.columns:
        m &= df["Region"].astype(str).to_numpy() == sel_reg
    if sel_ch != "All" and "Sales_Channel" in df.columns:
        m &= df["Sales_Channel"].astype(str).to_numpy() == sel_ch
    if sel_cust != "All" and "Customer_Name" in df.columns:
        m &= df["Customer_Name"].astype(str).to_numpy() == sel_cust
    if sel_sku != "All" and "SKU_ID" in df.columns:
        m &= df["SKU_ID"].astype(str).to_numpy() == sel_sku
    return df.loc[m]


@st.cache_data(show_spinner=False)
def get_filtered_slices(
    hist, forecast, anomalies, stockouts, sel_fam, sel_reg, sel_ch, sel_cust, sel_sku
):
    return (
        fast_mask(hist, sel_fam, sel_reg, sel_ch, sel_cust, sel_sku),
        fast_mask(forecast, sel_fam, sel_reg, sel_ch, sel_cust, sel_sku),
        fast_mask(anomalies, sel_fam, sel_reg, sel_ch, sel_cust, sel_sku),
        fast_mask(stockouts, sel_fam, sel_reg, sel_ch, sel_cust, sel_sku),
    )


def timeline_chart(h_df, p_df, h_col, p10_col, p50_col, p90_col, y_label, title):
    h_ts = h_df.groupby("Date", as_index=False, observed=True)[h_col].sum().sort_values("Date")
    p_ts = (
        p_df.groupby("Date", as_index=False, observed=True)[[p10_col, p50_col, p90_col]]
        .sum()
        .sort_values("Date")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h_ts["Date"],
            y=h_ts[h_col],
            mode="lines",
            name="History",
            line=dict(color="#38bdf8", width=2.4),
        )
    )
    if len(p_ts) > 0:
        fig.add_trace(
            go.Scatter(
                x=p_ts["Date"],
                y=p_ts[p50_col],
                mode="lines+markers",
                name="Forecast P50",
                line=dict(color="#f59e0b", width=2.4, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(p_ts["Date"]) + list(p_ts["Date"])[::-1],
                y=list(p_ts[p90_col]) + list(p_ts[p10_col])[::-1],
                fill="toself",
                fillcolor="rgba(245, 158, 11, 0.16)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="P10–P90 Band",
            )
        )
        fig.add_vline(x="2026-11", line_width=1.2, line_dash="dot", line_color="#a78bfa")

    fig.update_layout(height=360, title=title, xaxis_title="Month", yaxis_title=y_label, hovermode="x unified")
    return fig


hist, forecast, anomalies, metrics, macro, sku, stockouts, audit = load_data()

st.title("🏭 Celeris Control Tower")
st.caption("History (2019-01 → 2026-10) & 19-Month Forecast (2026-11 → 2028-05)")

st.sidebar.header("Filters")
fams, regs, chs, _, _ = get_filter_options(hist, "All", "All", "All", "All")
sel_fam = st.sidebar.selectbox("Family", fams)
sel_reg = st.sidebar.selectbox("Region", regs)
sel_ch = st.sidebar.selectbox("Channel", chs)
_, _, _, custs, _ = get_filter_options(hist, sel_fam, sel_reg, sel_ch, "All")
sel_cust = st.sidebar.selectbox("Customer", custs)
_, _, _, _, skus = get_filter_options(hist, sel_fam, sel_reg, sel_ch, sel_cust)
sel_sku = st.sidebar.selectbox("SKU", skus)

f_hist, f_fore, f_anom, f_so = get_filtered_slices(
    hist, forecast, anomalies, stockouts, sel_fam, sel_reg, sel_ch, sel_cust, sel_sku
)

f_met = metrics
if sel_fam != "All":
    f_met = f_met[f_met["Product_Family"] == sel_fam]
if sel_reg != "All":
    f_met = f_met[f_met["Region"] == sel_reg]

tabs = st.tabs(
    [
        "Overview",
        "Data Health",
        "Demand & Lifecycle",
        "Anomalies",
        "Forecast & Accuracy",
        "Macro & What-If",
        "Procurement & Alerts",
    ]
)

with tabs[0]:
    h_u = float(f_hist["Unconstrained_Units"].sum())
    h_r = float(f_hist["Unconstrained_Revenue_USD"].sum())
    p_u = float(f_fore["Units_Sold_Forecast"].sum())
    p_r = float(f_fore["Revenue_P50"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-title">History Units</div><div class="kpi-val">{h_u:,.0f}</div><div class="kpi-sub">Unconstrained</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-title">History Revenue</div><div class="kpi-val">${h_r/1e6:,.2f}M</div><div class="kpi-sub">Reconciled USD</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-title">19M Forecast (P50)</div><div class="kpi-val">{p_u:,.0f}</div><div class="kpi-sub">2026-11 → 2028-05</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-title">19M Revenue (P50)</div><div class="kpi-val">${p_r/1e6:,.2f}M</div><div class="kpi-sub">P90: ${f_fore["Revenue_P90"].sum()/1e6:,.2f}M</div></div>', unsafe_allow_html=True)

    st.plotly_chart(timeline_chart(f_hist, f_fore, "Unconstrained_Units", "P10", "Units_Sold_Forecast", "P90", "Units", "Units: History → Forecast"), use_container_width=True)
    st.plotly_chart(timeline_chart(f_hist, f_fore, "Unconstrained_Revenue_USD", "Revenue_P10", "Revenue_P50", "Revenue_P90", "USD", "Revenue: History → Forecast"), use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.plotly_chart(px.bar(f_fore.groupby("Product_Family", as_index=False, observed=True)["Units_Sold_Forecast"].sum(), x="Units_Sold_Forecast", y="Product_Family", orientation="h", title="By Family"), use_container_width=True)
    m2.plotly_chart(px.pie(f_fore.groupby("Region", as_index=False, observed=True)["Units_Sold_Forecast"].sum(), values="Units_Sold_Forecast", names="Region", title="By Region"), use_container_width=True)
    m3.plotly_chart(px.bar(f_fore.groupby("Sales_Channel", as_index=False, observed=True)["Units_Sold_Forecast"].sum(), x="Sales_Channel", y="Units_Sold_Forecast", title="By Channel"), use_container_width=True)

with tabs[1]:
    st.subheader("Cleaning Summary")
    tr = audit["files"].get("train.csv", {})
    mc = audit["files"].get("macro_indicators.csv", {})
    health_df = pd.DataFrame(
        [
            {"File": "train.csv", "Issue": "Duplicate Record_IDs", "Rows Fixed": tr.get("duplicate_record_ids_removed", 0)},
            {"File": "train.csv", "Issue": "Customer Variants Collapsed", "Rows Fixed": tr.get("customer_variants_collapsed", 0)},
            {"File": "train.csv", "Issue": "Price / Currency Outliers", "Rows Fixed": tr.get("currency_or_price_outliers_fixed", 0)},
            {"File": "train.csv", "Issue": "Revenue Mismatches", "Rows Fixed": tr.get("revenue_reconciled_rows", 0)},
            {"File": "train.csv", "Issue": "Missing Units", "Rows Fixed": tr.get("missing_units_imputed", 0)},
            {"File": "macro_indicators.csv", "Issue": "Missing Values", "Rows Fixed": mc.get("nulls_fixed", 0)},
        ]
    )
    st.dataframe(health_df, use_container_width=True, hide_index=True)
    comp = pd.DataFrame(
        [
            {"Stage": "Before Cleaning", "Rows": tr.get("raw_rows", 0)},
            {"Stage": "After Cleaning", "Rows": len(hist)},
        ]
    )
    st.plotly_chart(px.bar(comp, x="Stage", y="Rows", title="Rows Before vs. After"), use_container_width=True)

with tabs[2]:
    st.subheader("Drill-Down & Seasonality")
    drill = (
        f_hist.groupby(["Product_Family", "SKU_ID", "Region", "Customer_Name"], as_index=False, observed=True)
        .agg(
            Units=("Unconstrained_Units", "sum"),
            Months=("Date", "nunique"),
            Zero_Months=("Unconstrained_Units", lambda s: int((s == 0).sum())),
        )
        .sort_values("Units", ascending=False)
        .head(250)
    )
    st.dataframe(drill, use_container_width=True, hide_index=True)

    seas = f_hist.groupby("Month", as_index=False, observed=True)["Unconstrained_Units"].mean().sort_values("Month")
    st.plotly_chart(px.bar(seas, x="Month", y="Unconstrained_Units", title="Average Units by Month"), use_container_width=True)

    st.subheader("SKU Lifecycle")
    lc_cols = [c for c in ["SKU_ID", "Product_Family", "Powertrain_Application", "Launch_Date", "EOL_Date", "Replaces_SKU", "Lifecycle_Status"] if c in sku.columns]
    lc_view = sku[lc_cols]
    if sel_fam != "All":
        lc_view = lc_view[lc_view["Product_Family"] == sel_fam]
    if sel_sku != "All":
        lc_view = lc_view[lc_view["SKU_ID"] == sel_sku]
    st.dataframe(lc_view, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Anomaly Points & Register")
    h_line = f_hist.groupby("Date", as_index=False, observed=True)["Unconstrained_Units"].sum().sort_values("Date")
    a_line = (
        f_anom.groupby(["Date", "Anomaly_Type", "Reason", "Action"], as_index=False, observed=True)["Adjusted_Units"]
        .sum()
        .sort_values("Date")
    )
    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(x=h_line["Date"], y=h_line["Unconstrained_Units"], mode="lines", name="Unconstrained Units", line=dict(color="#38bdf8", width=2)))
    if len(a_line) > 0:
        fig_a.add_trace(
            go.Scatter(
                x=a_line["Date"],
                y=a_line["Adjusted_Units"],
                mode="markers",
                name="Flagged Points",
                marker=dict(size=8, color="#f43f5e", symbol="circle-open"),
                customdata=np.stack([a_line["Anomaly_Type"], a_line["Reason"], a_line["Action"]], axis=-1),
                hovertemplate="<b>%{x}</b><br>Units: %{y:,.0f}<br>Type: %{customdata[0]}<br>Reason: %{customdata[1]}<br>Action: %{customdata[2]}<extra></extra>",
            )
        )
    fig_a.update_layout(height=360, xaxis_title="Month", yaxis_title="Units", hovermode="closest")
    st.plotly_chart(fig_a, use_container_width=True)
    st.dataframe(f_anom.head(300), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Backtest Accuracy & Horizon")
    if len(f_met) > 0:
        w_wape = float(np.average(f_met["WAPE"], weights=np.maximum(f_met["Actual_Units"], 1.0)))
        w_bias = float(np.average(f_met["Bias"], weights=np.maximum(f_met["Actual_Units"], 1.0)))
        k1, k2 = st.columns(2)
        k1.metric("Weighted WAPE", f"{w_wape:.2%}")
        k2.metric("Weighted Bias", f"{w_bias:+.2%}")
        st.dataframe(
            f_met.sort_values("WAPE", ascending=False).style.format(
                {"WAPE": "{:.2%}", "Bias": "{:+.2%}", "Actual_Units": "{:,.0f}", "Forecast_Units": "{:,.0f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    months_sorted = sorted(f_fore["Date"].unique())
    h_map = {m: i + 1 for i, m in enumerate(months_sorted)}
    f_h = f_fore.copy()
    h_num = f_h["Date"].map(h_map)
    f_h["Horizon_Group"] = np.where(h_num <= 6, "M1–M6", np.where(h_num <= 12, "M7–M12", "M13–M19"))
    hb = f_h.groupby("Horizon_Group", as_index=False, observed=True)[["P10", "Units_Sold_Forecast", "P90"]].sum()
    st.plotly_chart(px.bar(hb, x="Horizon_Group", y=["P10", "Units_Sold_Forecast", "P90"], barmode="group", title="Forecast by Horizon"), use_container_width=True)

    csv_bytes = f_fore[["Date", "SKU_ID", "Region", "Customer_Name", "Sales_Channel", "P10", "Units_Sold_Forecast", "P90"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast CSV", data=csv_bytes, file_name="celeris_forecast.csv", mime="text/csv")

with tabs[5]:
    st.subheader("What-If Simulator & Macro Correlations")
    pmi_shift = st.slider("PMI Shift", min_value=-10.0, max_value=10.0, value=-3.0, step=0.5)
    ev_shift = st.slider("EV Index Shift", min_value=-20.0, max_value=30.0, value=0.0, step=1.0)

    pt_mult = np.where(
        f_fore.get("Powertrain_Application", "Universal") == "EV",
        1.0 + 0.008 * ev_shift,
        np.where(f_fore.get("Powertrain_Application", "Universal") == "ICE", 1.0 - 0.004 * ev_shift, 1.0),
    )
    sim_p50 = np.clip(f_fore["Units_Sold_Forecast"].to_numpy() * (1.0 + 0.012 * pmi_shift) * pt_mult, 0, None)

    c_a, c_b, c_c = st.columns(3)
    c_a.metric("Baseline P50", f"{f_fore['Units_Sold_Forecast'].sum():,.0f}")
    c_b.metric("Simulated P50", f"{sim_p50.sum():,.0f}")
    c_c.metric("Change", f"{sim_p50.sum() - f_fore['Units_Sold_Forecast'].sum():+,.0f}")

    num_macro = macro.select_dtypes(include=[np.number]).drop(columns=["Year", "Month"], errors="ignore")
    st.plotly_chart(px.imshow(num_macro.corr(), text_auto=".2f", aspect="auto", title="Macro Correlation Heatmap"), use_container_width=True)

with tabs[6]:
    st.subheader("P90 Procurement & Stockouts")
    proc = (
        f_fore.groupby(["SKU_ID", "Product_Family", "Region"], as_index=False, observed=True)[["P10", "Units_Sold_Forecast", "P90"]]
        .sum()
    )
    proc["Buffer_Units"] = proc["P90"] - proc["Units_Sold_Forecast"]
    st.dataframe(proc.sort_values("Buffer_Units", ascending=False).head(200), use_container_width=True, hide_index=True)

    st.subheader("Stockouts & Lost Units")
    if len(f_so) > 0:
        so_cols = [c for c in ["Date", "SKU_ID", "Region", "Requested_Units", "Shipped_Units", "Backordered_Units", "Lost_Units", "Fill_Rate_Pct", "Reason"] if c in f_so.columns]
        st.dataframe(f_so[so_cols].head(250), use_container_width=True, hide_index=True)
    else:
        st.write("No stockout records for this filter.")
