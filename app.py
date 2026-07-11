"""
Sales Forecasting & Demand Intelligence Dashboard
====================================================
Streamlit app (Task 7). Reads the pre-computed outputs from analysis.py
(outputs/*.csv and charts/*.png) so the dashboard loads instantly without
retraining any model.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud by pointing it at this file, with
requirements.txt in the same repo.
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
DATA_PATH = os.path.join(BASE_DIR, "Sales_data.csv")


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_raw_sales():
    df = pd.read_csv(DATA_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.month_name()
    return df


@st.cache_data
def load_csv(filename, parse_dates=None):
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


def chart_path(filename):
    path = os.path.join(CHARTS_DIR, filename)
    return path if os.path.exists(path) else None


sales = load_raw_sales()
monthly_sales = load_csv("monthly_sales.csv", parse_dates=["Order Date"])
weekly_sales = load_csv("weekly_sales.csv", parse_dates=["Order Date"])
metrics = load_csv("metrics.csv")
forecast = load_csv("forecast.csv", parse_dates=["Date"])
segment_forecasts = load_csv("segment_forecasts.csv", parse_dates=["Date"])
segment_growth = load_csv("segment_growth.csv")
anomalies = load_csv("anomalies.csv", parse_dates=["Date"])
cluster_data = load_csv("cluster_data.csv")

best_model = metrics.sort_values("RMSE").iloc[0]["Model"] if metrics is not None else "N/A"

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📦 Sales Intelligence")
page = st.sidebar.radio(
    "Navigate",
    [
        "1. Sales Overview",
        "2. Forecast Explorer",
        "3. Anomaly Report",
        "4. Product Demand Segments",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Superstore Sales (train.csv) · Models: SARIMA, Prophet, XGBoost "
    "· Best model by RMSE: **{}**".format(best_model)
)


# ---------------------------------------------------------------------------
# PAGE 1 — Sales Overview Dashboard
# ---------------------------------------------------------------------------
if page == "1. Sales Overview":
    st.title("📈 Sales Overview Dashboard")
    st.caption("A Monday-morning snapshot of overall sales performance.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${sales['Sales'].sum():,.0f}")
    col2.metric("Total Orders", f"{sales['Order ID'].nunique():,}")
    col3.metric("Avg Order Value", f"${sales['Sales'].mean():,.2f}")
    col4.metric("Date Range", f"{sales['Order Date'].dt.year.min()}–{sales['Order Date'].dt.year.max()}")

    st.markdown("### Total Sales by Year")
    yearly = sales.groupby("Year", as_index=False)["Sales"].sum()
    fig_year = px.bar(yearly, x="Year", y="Sales", text_auto=".2s", color="Sales", color_continuous_scale="Blues")
    fig_year.update_layout(showlegend=False)
    st.plotly_chart(fig_year, use_container_width=True)

    st.markdown("### Monthly Sales Trend")
    if monthly_sales is not None:
        fig_trend = px.line(monthly_sales, x="Order Date", y="Sales", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### Sales by Region and Category")
    filt_col1, filt_col2 = st.columns(2)
    with filt_col1:
        regions = st.multiselect(
            "Filter by Region", options=sorted(sales["Region"].unique()),
            default=sorted(sales["Region"].unique()),
        )
    with filt_col2:
        categories = st.multiselect(
            "Filter by Category", options=sorted(sales["Category"].unique()),
            default=sorted(sales["Category"].unique()),
        )

    filtered = sales[sales["Region"].isin(regions) & sales["Category"].isin(categories)]

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        region_sales = filtered.groupby("Region", as_index=False)["Sales"].sum()
        fig_region = px.bar(region_sales, x="Region", y="Sales", color="Region", title="Sales by Region")
        st.plotly_chart(fig_region, use_container_width=True)
    with chart_col2:
        cat_sales = filtered.groupby("Category", as_index=False)["Sales"].sum()
        fig_cat = px.pie(cat_sales, names="Category", values="Sales", title="Sales Share by Category")
        st.plotly_chart(fig_cat, use_container_width=True)

    with st.expander("📊 Saved EDA charts from the notebook (Task 1 & 2)"):
        img_cols = st.columns(3)
        for i, name in enumerate(["category_sales.png", "shipping_time_region.png", "heatmap.png"]):
            p = chart_path(name)
            if p:
                img_cols[i % 3].image(p, use_container_width=True)
        img_cols2 = st.columns(2)
        for i, name in enumerate(["monthly_sales_trend.png", "decomposition.png"]):
            p = chart_path(name)
            if p:
                img_cols2[i % 2].image(p, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 2 — Forecast Explorer
# ---------------------------------------------------------------------------
elif page == "2. Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.caption(f"Forecasts generated by the recommended production model: **{best_model}**")

    col1, col2 = st.columns([1, 2])

    with col1:
        scope = st.selectbox("Forecast scope", ["Overall Company", "Category", "Region"])

        if scope == "Overall Company":
            selection = "Overall"
        elif scope == "Category":
            selection = st.selectbox("Select Category", ["Furniture", "Technology", "Office Supplies"])
        else:
            selection = st.selectbox("Select Region", ["West", "East"])

        horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3)

    with col2:
        if scope == "Overall Company" and forecast is not None:
            plot_df = forecast.head(horizon).rename(columns={"Date": "ds", "Forecast": "yhat"})
            history_df = monthly_sales.rename(columns={"Order Date": "ds", "Sales": "yhat"}).tail(12)
        elif segment_forecasts is not None:
            seg_hist_mask = segment_forecasts["Segment"] == selection
            plot_df = segment_forecasts[seg_hist_mask].head(horizon).rename(columns={"Forecast": "yhat"})[["Date", "yhat"]].rename(columns={"Date": "ds"})
            history_df = None
        else:
            plot_df = pd.DataFrame()
            history_df = None

        fig = px.line()
        if history_df is not None and not history_df.empty:
            fig.add_scatter(x=history_df["ds"], y=history_df["yhat"], mode="lines+markers", name="Recent Actuals")
        if not plot_df.empty:
            fig.add_scatter(x=plot_df["ds"], y=plot_df["yhat"], mode="lines+markers", name=f"{selection} Forecast", line=dict(dash="dash", color="crimson"))
        fig.update_layout(title=f"{selection} — {horizon}-Month Forecast", xaxis_title="Date", yaxis_title="Predicted Sales")
        st.plotly_chart(fig, use_container_width=True)

        if not plot_df.empty:
            st.dataframe(plot_df.rename(columns={"ds": "Date", "yhat": "Forecast"}), use_container_width=True)

    st.markdown("### Model Performance")
    if metrics is not None:
        best_row = metrics.sort_values("RMSE").iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{best_row['MAE']:.2f}")
        m2.metric("RMSE", f"{best_row['RMSE']:.2f}")
        m3.metric("MAPE", f"{best_row['MAPE (%)']:.2f}%")
        st.caption(f"Metrics shown for the recommended model: **{best_row['Model']}** (lowest RMSE on the 3-month holdout).")
        with st.expander("Compare all 3 models"):
            st.dataframe(metrics.round(2), use_container_width=True)

    with st.expander("📊 Forecast comparison chart (all 3 models)"):
        p = chart_path("forecast_comparison.png")
        if p:
            st.image(p, use_container_width=True)
        p2 = chart_path("segment_forecast.png")
        if p2:
            st.image(p2, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 3 — Anomaly Report
# ---------------------------------------------------------------------------
elif page == "3. Anomaly Report":
    st.title("🚨 Anomaly Report")
    st.caption("Weekly sales weeks flagged as unusually high or low by Isolation Forest and/or rolling Z-Score.")

    p = chart_path("anomaly_detection.png")
    if p:
        st.image(p, use_container_width=True)

    if anomalies is not None:
        st.markdown("### Detected Anomalies")
        m1, m2, m3 = st.columns(3)
        m1.metric("Isolation Forest Flags", int(anomalies["Isolation_Anomaly"].sum()))
        m2.metric("Z-Score Flags", int(anomalies["Z_Anomaly"].sum()))
        m3.metric("Flagged by Both", int((anomalies["Isolation_Anomaly"] & anomalies["Z_Anomaly"]).sum()))

        method_filter = st.multiselect(
            "Filter by detection method",
            ["Isolation Forest", "Z-Score"],
            default=["Isolation Forest", "Z-Score"],
        )
        mask = pd.Series(False, index=anomalies.index)
        if "Isolation Forest" in method_filter:
            mask |= anomalies["Isolation_Anomaly"]
        if "Z-Score" in method_filter:
            mask |= anomalies["Z_Anomaly"]

        st.dataframe(
            anomalies[mask].sort_values("Date").reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.info("outputs/anomalies.csv not found. Run analysis.py first.")


# ---------------------------------------------------------------------------
# PAGE 4 — Product Demand Segments
# ---------------------------------------------------------------------------
elif page == "4. Product Demand Segments":
    st.title("🧩 Product Demand Segments")
    st.caption("Sub-categories grouped by demand behavior using K-Means clustering + PCA.")

    p = chart_path("cluster_pca.png")
    if p:
        st.image(p, use_container_width=True)

    p_elbow = chart_path("elbow_method.png")
    if p_elbow:
        with st.expander("How was the number of clusters chosen? (Elbow Method)"):
            st.image(p_elbow, use_container_width=True)

    if cluster_data is not None:
        st.markdown("### Sub-Categories by Demand Segment")
        segment_filter = st.multiselect(
            "Filter by demand segment",
            options=sorted(cluster_data["Demand_Segment"].unique()),
            default=sorted(cluster_data["Demand_Segment"].unique()),
        )
        display_df = cluster_data[cluster_data["Demand_Segment"].isin(segment_filter)]
        st.dataframe(
            display_df[[c for c in cluster_data.columns if c != "Cluster"]].round(2),
            use_container_width=True,
        )

        st.markdown("### Recommended Stocking Strategy per Segment")
        if "Stocking Strategy" in cluster_data.columns:
            strategy_summary = cluster_data[["Demand_Segment", "Stocking Strategy"]].drop_duplicates()
            st.table(strategy_summary.reset_index(drop=True))
        else:
            default_strategy = {
                "High Volume, Stable Demand": "Maintain high base stock and use frequent replenishment.",
                "Low Volume, High Volatility": "Keep conservative safety stock and monitor demand frequently.",
                "Growing Demand": "Gradually increase reorder levels and supplier capacity.",
                "Declining / Slow Demand": "Reduce inventory exposure and avoid overstocking.",
            }
            st.table(pd.DataFrame(default_strategy.items(), columns=["Demand_Segment", "Stocking Strategy"]))
    else:
        st.info("outputs/cluster_data.csv not found. Run analysis.py first.")

st.sidebar.markdown("---")
st.sidebar.caption("Built for the Week 3 & 4 internship project · End-to-End Sales Forecasting & Demand Intelligence System")
