"""
End-to-End Sales Forecasting & Demand Intelligence System
============================================================
analysis.py

Script version of analysis.ipynb. Running this file reproduces every
chart in charts/, every CSV in outputs/, and every trained model in
saved_models/ that power the Streamlit dashboard (app.py).

Usage:
    python analysis.py

Datasets required in the same folder:
    - Sales_data.csv   (Superstore sales dataset)
    - vgsales.csv      (Video game sales dataset - multi-source demo)
"""

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so the script can run headless
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from xgboost import XGBRegressor
from prophet import Prophet

import joblib

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Folder setup
# ---------------------------------------------------------------------------
CHARTS_DIR = "charts"
OUTPUTS_DIR = "outputs"
MODELS_DIR = "saved_models"

for folder in (CHARTS_DIR, OUTPUTS_DIR, MODELS_DIR):
    os.makedirs(folder, exist_ok=True)


def savefig(name):
    """Save the current matplotlib figure into charts/ and close it."""
    path = os.path.join(CHARTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart -> {path}")


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def calculate_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    non_zero = actual != 0
    mape = np.mean(
        np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])
    ) * 100

    return mae, rmse, mape


def adf_test(series, name="Series"):
    result = adfuller(series.dropna())
    print(f"ADF Test for: {name}")
    print("-" * 50)
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"P-value       : {result[1]:.4f}")
    for key, value in result[4].items():
        print(f"  Critical Value ({key}): {value:.4f}")
    if result[1] < 0.05:
        print("Conclusion: The series is stationary.\n")
    else:
        print("Conclusion: The series is non-stationary; differencing may be required.\n")
    return result


print("=" * 70)
print("TASK 1 - DATA LOADING, MERGING & DEEP EXPLORATION")
print("=" * 70)

sales = pd.read_csv("Sales_data.csv")
vgsales = pd.read_csv("vgsales.csv")
print("Primary Sales Dataset Shape:", sales.shape)
print("Video Game Sales Dataset Shape:", vgsales.shape)

# --- Parse dates -----------------------------------------------------------
sales["Order Date"] = pd.to_datetime(sales["Order Date"], dayfirst=True, errors="coerce")
sales["Ship Date"] = pd.to_datetime(sales["Ship Date"], dayfirst=True, errors="coerce")

# --- Missing values / duplicates / dtypes ----------------------------------
missing_values = pd.DataFrame({
    "Missing Count": sales.isnull().sum(),
    "Missing Percentage": (sales.isnull().mean() * 100).round(2),
})
print("\nMissing values:\n", missing_values[missing_values["Missing Count"] > 0])
print("\nExact duplicate rows:", sales.duplicated().sum())

sales = sales.drop_duplicates().copy()
sales = sales.dropna(subset=["Order Date", "Ship Date", "Sales"])
sales["Sales"] = pd.to_numeric(sales["Sales"], errors="coerce")
sales = sales.dropna(subset=["Sales"])
print("Cleaned Shape:", sales.shape)

# --- Time features -----------------------------------------------------------
sales["Year"] = sales["Order Date"].dt.year
sales["Month"] = sales["Order Date"].dt.month
sales["Month Name"] = sales["Order Date"].dt.month_name()
sales["Week Number"] = sales["Order Date"].dt.isocalendar().week.astype(int)
sales["Day of Week"] = sales["Order Date"].dt.day_name()
sales["Quarter"] = sales["Order Date"].dt.quarter
sales["Season"] = sales["Month"].apply(get_season)
sales["Shipping Days"] = (sales["Ship Date"] - sales["Order Date"]).dt.days

# --- Multi-source analysis with vgsales.csv ---------------------------------
vgsales["Year"] = pd.to_numeric(vgsales["Year"], errors="coerce")
vgsales["Global_Sales"] = pd.to_numeric(vgsales["Global_Sales"], errors="coerce")
vgsales_clean = vgsales.dropna(subset=["Year", "Global_Sales"]).copy()
vgsales_clean["Year"] = vgsales_clean["Year"].astype(int)

superstore_yearly = (
    sales.groupby("Year", as_index=False)["Sales"].sum()
    .rename(columns={"Sales": "Superstore_Sales"})
)
gaming_yearly = vgsales_clean.groupby("Year", as_index=False)["Global_Sales"].sum()
multi_source = pd.merge(superstore_yearly, gaming_yearly, on="Year", how="left")
multi_source["Superstore_Sales_Index"] = (
    multi_source["Superstore_Sales"] / multi_source["Superstore_Sales"].max() * 100
)
multi_source["Gaming_Sales_Index"] = (
    multi_source["Global_Sales"] / multi_source["Global_Sales"].max() * 100
)
print("\nMulti-source yearly comparison:\n", multi_source)

# --- Weekly / monthly aggregation -------------------------------------------
weekly_sales = (
    sales.set_index("Order Date").resample("W")["Sales"].sum().reset_index()
)
monthly_sales = (
    sales.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
)

weekly_sales.to_csv(os.path.join(OUTPUTS_DIR, "weekly_sales.csv"), index=False)
monthly_sales.to_csv(os.path.join(OUTPUTS_DIR, "monthly_sales.csv"), index=False)
print("Saved outputs/weekly_sales.csv and outputs/monthly_sales.csv")

# --- Business Question 1: highest revenue category --------------------------
category_revenue = sales.groupby("Category")["Sales"].sum().sort_values(ascending=False)
print("\nRevenue by category:\n", category_revenue)

plt.figure(figsize=(8, 5))
category_revenue.plot(kind="bar")
plt.title("Total Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.tight_layout()
savefig("category_sales.png")

# --- Business Question 2: most consistent regional growth -------------------
regional_yearly = sales.groupby(["Region", "Year"])["Sales"].sum().unstack()
growth_consistency = pd.DataFrame({
    "Std Dev of YoY Growth": regional_yearly.pct_change(axis=1).std(axis=1),
    "Total Sales": regional_yearly.sum(axis=1),
})
print("\nRegional growth consistency (lower std = more consistent):\n", growth_consistency.sort_values("Std Dev of YoY Growth"))

# --- Business Question 3: average shipping time by region -------------------
shipping_by_region = sales.groupby("Region")["Shipping Days"].agg(["mean", "std"])
print("\nShipping time by region:\n", shipping_by_region)

plt.figure(figsize=(8, 5))
shipping_by_region["mean"].plot(kind="bar")
plt.title("Average Shipping Time by Region")
plt.xlabel("Region")
plt.ylabel("Average Shipping Days")
plt.xticks(rotation=0)
plt.tight_layout()
savefig("shipping_time_region.png")

# --- Business Question 4: seasonal monthly spikes ----------------------------
monthly_by_year = sales.groupby(["Year", "Month"])["Sales"].sum().unstack()

plt.figure(figsize=(12, 6))
sns.heatmap(monthly_by_year, cmap="YlGnBu", annot=True, fmt=".0f")
plt.title("Monthly Sales Across Years")
plt.xlabel("Month")
plt.ylabel("Year")
plt.tight_layout()
savefig("heatmap.png")

average_monthly_pattern = sales.groupby("Month")["Sales"].mean().sort_values(ascending=False)
print("\nStrongest average sales month:", average_monthly_pattern.index[0])


print("\n" + "=" * 70)
print("TASK 2 - TIME SERIES ANALYSIS & DECOMPOSITION")
print("=" * 70)

monthly_ts = sales.set_index("Order Date").resample("MS")["Sales"].sum()
monthly_ts = monthly_ts.asfreq("MS", fill_value=0)

plt.figure(figsize=(14, 6))
plt.plot(monthly_ts.index, monthly_ts.values, marker="o")
plt.title("Overall Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Monthly Sales")
plt.grid(alpha=0.3)
plt.tight_layout()
savefig("monthly_sales_trend.png")

decomposition = seasonal_decompose(monthly_ts, model="additive", period=12)
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.tight_layout()
savefig("decomposition.png")

residuals = decomposition.resid.dropna()
highest_noise = residuals.abs().sort_values(ascending=False).head(5)
print("\nMonths with highest residual noise:\n", highest_noise)

adf_original = adf_test(monthly_ts, "Original Monthly Sales")
monthly_diff = monthly_ts.diff().dropna()
adf_differenced = adf_test(monthly_diff, "First-Differenced Monthly Sales")


print("\n" + "=" * 70)
print("TASK 3 - SALES FORECASTING USING 3 MODELS")
print("=" * 70)

train_ts = monthly_ts.iloc[:-3]
test_ts = monthly_ts.iloc[-3:]
print("Training Months:", len(train_ts), " | Testing Months:", len(test_ts))

# ---------------- Model 1: SARIMA -------------------------------------------
sarima_model = SARIMAX(
    train_ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False, enforce_invertibility=False,
)
sarima_fit = sarima_model.fit(disp=False)

sarima_test_result = sarima_fit.get_forecast(steps=3)
sarima_test_pred = sarima_test_result.predicted_mean
sarima_mae, sarima_rmse, sarima_mape = calculate_metrics(test_ts.values, sarima_test_pred.values)
print(f"\nSARIMA  -> MAE: {sarima_mae:.2f}  RMSE: {sarima_rmse:.2f}  MAPE: {sarima_mape:.2f}%")

plt.figure(figsize=(14, 6))
plt.plot(train_ts.index, train_ts.values, label="Training Data")
plt.plot(test_ts.index, test_ts.values, marker="o", label="Actual Test Sales")
plt.plot(test_ts.index, sarima_test_pred.values, marker="o", linestyle="--", label="SARIMA Prediction")
plt.title("SARIMA: Actual vs Predicted Sales")
plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout()
savefig("sarima_actual_vs_predicted.png")

final_sarima_model = SARIMAX(
    monthly_ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False, enforce_invertibility=False,
)
final_sarima_fit = final_sarima_model.fit(disp=False)
sarima_future_result = final_sarima_fit.get_forecast(steps=3)
sarima_future = sarima_future_result.predicted_mean
joblib.dump(final_sarima_fit, os.path.join(MODELS_DIR, "sarima.pkl"))
print("Saved saved_models/sarima.pkl")

# ---------------- Model 2: Prophet ------------------------------------------
prophet_data = monthly_ts.reset_index()
prophet_data.columns = ["ds", "y"]
prophet_train = prophet_data.iloc[:-3].copy()
prophet_test = prophet_data.iloc[-3:].copy()

prophet_model = Prophet(
    yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
    interval_width=0.95,
)
prophet_model.fit(prophet_train)
prophet_test_forecast = prophet_model.predict(prophet_test[["ds"]].copy())
prophet_mae, prophet_rmse, prophet_mape = calculate_metrics(
    prophet_test["y"].values, prophet_test_forecast["yhat"].values
)
print(f"Prophet -> MAE: {prophet_mae:.2f}  RMSE: {prophet_rmse:.2f}  MAPE: {prophet_mape:.2f}%")

plt.figure(figsize=(14, 6))
plt.plot(prophet_train["ds"], prophet_train["y"], label="Training Data")
plt.plot(prophet_test["ds"], prophet_test["y"], marker="o", label="Actual Test Sales")
plt.plot(prophet_test_forecast["ds"], prophet_test_forecast["yhat"], marker="o", linestyle="--", label="Prophet Prediction")
plt.title("Prophet: Actual vs Predicted")
plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout()
savefig("prophet_actual_vs_predicted.png")

final_prophet = Prophet(
    yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
    interval_width=0.95,
)
final_prophet.fit(prophet_data)
future_dates = final_prophet.make_future_dataframe(periods=3, freq="MS")
final_prophet_forecast = final_prophet.predict(future_dates)
prophet_future = final_prophet_forecast.tail(3).reset_index(drop=True)

final_prophet.plot_components(final_prophet_forecast)
plt.tight_layout()
savefig("prophet_components.png")

joblib.dump(final_prophet, os.path.join(MODELS_DIR, "prophet.pkl"))
print("Saved saved_models/prophet.pkl")


# ---------------- Model 3: XGBoost ------------------------------------------
def create_time_features(series):
    df = series.to_frame(name="Sales").copy()
    df["Lag_1"] = df["Sales"].shift(1)
    df["Lag_2"] = df["Sales"].shift(2)
    df["Lag_3"] = df["Sales"].shift(3)
    df["Rolling_Mean_3"] = df["Sales"].shift(1).rolling(window=3).mean()
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter
    df["Season"] = df["Month"].apply(get_season)
    df = pd.get_dummies(df, columns=["Season"], drop_first=False)
    return df.dropna()


xgb_data = create_time_features(monthly_ts)
xgb_train = xgb_data.iloc[:-3].copy()
xgb_test = xgb_data.iloc[-3:].copy()

X_train, y_train = xgb_train.drop(columns="Sales"), xgb_train["Sales"]
X_test, y_test = xgb_test.drop(columns="Sales"), xgb_test["Sales"]

xgb_model = XGBRegressor(
    n_estimators=300, learning_rate=0.03, max_depth=3,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", random_state=42,
)
xgb_model.fit(X_train, y_train)
xgb_test_pred = xgb_model.predict(X_test)
xgb_mae, xgb_rmse, xgb_mape = calculate_metrics(y_test.values, xgb_test_pred)
print(f"XGBoost -> MAE: {xgb_mae:.2f}  RMSE: {xgb_rmse:.2f}  MAPE: {xgb_mape:.2f}%")

plt.figure(figsize=(14, 6))
plt.plot(xgb_train.index, xgb_train["Sales"], label="Training Data")
plt.plot(y_test.index, y_test.values, marker="o", label="Actual")
plt.plot(y_test.index, xgb_test_pred, marker="o", linestyle="--", label="XGBoost Prediction")
plt.title("XGBoost: Actual vs Predicted")
plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout()
savefig("xgboost_actual_vs_predicted.png")


def recursive_xgb_forecast(model, history, feature_columns, steps=3):
    history = history.copy()
    predictions = []
    for _ in range(steps):
        next_date = history.index[-1] + pd.offsets.MonthBegin(1)
        row = pd.DataFrame(index=[next_date])
        row["Lag_1"] = history.iloc[-1]
        row["Lag_2"] = history.iloc[-2]
        row["Lag_3"] = history.iloc[-3]
        row["Rolling_Mean_3"] = history.iloc[-3:].mean()
        row["Month"] = next_date.month
        row["Quarter"] = next_date.quarter
        season = get_season(next_date.month)
        for col in feature_columns:
            if col.startswith("Season_"):
                row[col] = 0
        season_col = f"Season_{season}"
        if season_col in row.columns:
            row[season_col] = 1
        row = row.reindex(columns=feature_columns, fill_value=0)
        prediction = model.predict(row)[0]
        predictions.append({"Date": next_date, "Forecast": prediction})
        history.loc[next_date] = prediction
    return pd.DataFrame(predictions)


final_xgb_data = create_time_features(monthly_ts)
X_full, y_full = final_xgb_data.drop(columns="Sales"), final_xgb_data["Sales"]
final_xgb_model = XGBRegressor(
    n_estimators=300, learning_rate=0.03, max_depth=3,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", random_state=42,
)
final_xgb_model.fit(X_full, y_full)
xgb_future = recursive_xgb_forecast(final_xgb_model, monthly_ts, X_full.columns, steps=3)
joblib.dump(final_xgb_model, os.path.join(MODELS_DIR, "xgb.pkl"))
print("Saved saved_models/xgb.pkl")

# ---------------- Model comparison table ------------------------------------
comparison_table = pd.DataFrame({
    "Model": ["SARIMA", "Prophet", "XGBoost"],
    "MAE": [sarima_mae, prophet_mae, xgb_mae],
    "RMSE": [sarima_rmse, prophet_rmse, xgb_rmse],
    "MAPE (%)": [sarima_mape, prophet_mape, xgb_mape],
    "Forecast Month 1": [sarima_future.iloc[0], prophet_future.iloc[0]["yhat"], xgb_future.iloc[0]["Forecast"]],
    "Forecast Month 2": [sarima_future.iloc[1], prophet_future.iloc[1]["yhat"], xgb_future.iloc[1]["Forecast"]],
    "Forecast Month 3": [sarima_future.iloc[2], prophet_future.iloc[2]["yhat"], xgb_future.iloc[2]["Forecast"]],
}).sort_values("RMSE").reset_index(drop=True)

print("\nModel comparison table:\n", comparison_table.round(2))
comparison_table.to_csv(os.path.join(OUTPUTS_DIR, "metrics.csv"), index=False)
print("Saved outputs/metrics.csv")

best_model_name = comparison_table.iloc[0]["Model"]
print("\nRecommended production model (lowest RMSE):", best_model_name)

# combined forecast-vs-actual comparison chart for all 3 models
plt.figure(figsize=(14, 6))
plt.plot(train_ts.index, train_ts.values, label="Training Data", color="gray")
plt.plot(test_ts.index, test_ts.values, marker="o", label="Actual", color="black")
plt.plot(test_ts.index, sarima_test_pred.values, marker="o", linestyle="--", label="SARIMA")
plt.plot(prophet_test["ds"], prophet_test_forecast["yhat"], marker="o", linestyle="--", label="Prophet")
plt.plot(y_test.index, xgb_test_pred, marker="o", linestyle="--", label="XGBoost")

future_map = {"SARIMA": sarima_future, "Prophet": prophet_future.set_index("ds")["yhat"], "XGBoost": xgb_future.set_index("Date")["Forecast"]}
best_future = future_map[best_model_name]
plt.plot(best_future.index, best_future.values, marker="*", markersize=14, linestyle=":", label=f"{best_model_name} 3-Month Forecast", color="red")

plt.title("Forecast Comparison — SARIMA vs Prophet vs XGBoost")
plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
savefig("forecast_comparison.png")

# save the best model's 3-month forecast as the primary forecast.csv
best_forecast_df = pd.DataFrame({
    "Date": pd.to_datetime(list(best_future.index)),
    "Forecast": list(best_future.values),
})
best_forecast_df.to_csv(os.path.join(OUTPUTS_DIR, "forecast.csv"), index=False)
print("Saved outputs/forecast.csv (best model:", best_model_name, ")")


print("\n" + "=" * 70)
print("TASK 4 - PRODUCT CATEGORY & REGION LEVEL FORECASTING")
print("=" * 70)


def prophet_segment_forecast(dataframe, filter_column, filter_value, periods=3):
    segment = dataframe[dataframe[filter_column] == filter_value].copy()
    segment_monthly = (
        segment.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
    )
    segment_monthly.columns = ["ds", "y"]
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(segment_monthly)
    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)
    return segment_monthly, forecast.tail(periods), model


segments = [
    ("Category", "Furniture"),
    ("Category", "Technology"),
    ("Category", "Office Supplies"),
    ("Region", "West"),
    ("Region", "East"),
]

segment_forecasts = {}
segment_rows = []
for column, value in segments:
    history, forecast, model = prophet_segment_forecast(sales, column, value, periods=3)
    segment_forecasts[value] = {"history": history, "forecast": forecast, "model": model}
    for _, r in forecast.iterrows():
        segment_rows.append({
            "Segment Type": column, "Segment": value,
            "Date": r["ds"], "Forecast": r["yhat"],
            "Lower CI": r["yhat_lower"], "Upper CI": r["yhat_upper"],
        })

segment_forecast_df = pd.DataFrame(segment_rows)
segment_forecast_df.to_csv(os.path.join(OUTPUTS_DIR, "segment_forecasts.csv"), index=False)
print("Saved outputs/segment_forecasts.csv")

plt.figure(figsize=(14, 7))
for name, result in segment_forecasts.items():
    forecast = result["forecast"]
    plt.plot(forecast["ds"], forecast["yhat"], marker="o", label=name)
plt.title("Three-Month Forecast Comparison Across Categories and Regions")
plt.xlabel("Forecast Date"); plt.ylabel("Predicted Sales"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
savefig("segment_forecast.png")

growth_results = []
for name, result in segment_forecasts.items():
    forecast = result["forecast"]
    first_value = forecast["yhat"].iloc[0]
    last_value = forecast["yhat"].iloc[-1]
    growth_pct = ((last_value - first_value) / abs(first_value)) * 100
    growth_results.append({"Segment": name, "Forecast Growth (%)": growth_pct})

growth_df = pd.DataFrame(growth_results).sort_values("Forecast Growth (%)", ascending=False)
growth_df.to_csv(os.path.join(OUTPUTS_DIR, "segment_growth.csv"), index=False)
print("\nSegment growth ranking:\n", growth_df.round(2))
print("Strongest upcoming growth:", growth_df.iloc[0]["Segment"])


print("\n" + "=" * 70)
print("TASK 5 - ANOMALY DETECTION IN WEEKLY SALES")
print("=" * 70)

weekly_anomaly = weekly_sales.copy()
weekly_anomaly.columns = ["Date", "Sales"]

iso_forest = IsolationForest(contamination=0.05, random_state=42)
weekly_anomaly["Isolation_Label"] = iso_forest.fit_predict(weekly_anomaly[["Sales"]])
weekly_anomaly["Isolation_Anomaly"] = weekly_anomaly["Isolation_Label"] == -1
print("Isolation Forest anomalies:", weekly_anomaly["Isolation_Anomaly"].sum())

rolling_window = 8
weekly_anomaly["Rolling_Mean"] = weekly_anomaly["Sales"].rolling(window=rolling_window, min_periods=4).mean()
weekly_anomaly["Rolling_Std"] = weekly_anomaly["Sales"].rolling(window=rolling_window, min_periods=4).std()
weekly_anomaly["Z_Score"] = (weekly_anomaly["Sales"] - weekly_anomaly["Rolling_Mean"]) / weekly_anomaly["Rolling_Std"]
weekly_anomaly["Z_Anomaly"] = weekly_anomaly["Z_Score"].abs() > 2
print("Z-Score anomalies:", weekly_anomaly["Z_Anomaly"].sum())

weekly_anomaly["Both_Methods"] = weekly_anomaly["Isolation_Anomaly"] & weekly_anomaly["Z_Anomaly"]
print("Anomalies flagged by both methods:", weekly_anomaly["Both_Methods"].sum())

# combined anomaly plot
plt.figure(figsize=(15, 6))
plt.plot(weekly_anomaly["Date"], weekly_anomaly["Sales"], label="Weekly Sales", color="steelblue", zorder=1)
iso_points = weekly_anomaly[weekly_anomaly["Isolation_Anomaly"]]
z_points = weekly_anomaly[weekly_anomaly["Z_Anomaly"]]
plt.scatter(iso_points["Date"], iso_points["Sales"], color="orange", marker="o", s=80, label="Isolation Forest Anomaly", zorder=2)
plt.scatter(z_points["Date"], z_points["Sales"], color="red", marker="x", s=100, label="Z-Score Anomaly", zorder=3)
plt.title("Weekly Sales Anomaly Detection")
plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
savefig("anomaly_detection.png")

detected_anomalies = weekly_anomaly[
    weekly_anomaly["Isolation_Anomaly"] | weekly_anomaly["Z_Anomaly"]
][["Date", "Sales", "Isolation_Anomaly", "Z_Anomaly", "Z_Score"]].sort_values("Date").copy()


def anomaly_explanation(row):
    month = row["Date"].month
    if month in [11, 12]:
        return "Possible festive or year-end promotion demand spike."
    elif month in [1, 2]:
        return "Possible post-holiday demand correction or seasonal slowdown."
    elif row["Sales"] > weekly_anomaly["Sales"].median():
        return "Possible major bulk order, promotion, or temporary demand surge."
    else:
        return "Possible stock shortage, demand decline, or operational disruption."


detected_anomalies["Possible Explanation"] = detected_anomalies.apply(anomaly_explanation, axis=1)
detected_anomalies.to_csv(os.path.join(OUTPUTS_DIR, "anomalies.csv"), index=False)
print("Saved outputs/anomalies.csv")


print("\n" + "=" * 70)
print("TASK 6 - PRODUCT DEMAND SEGMENTATION USING K-MEANS")
print("=" * 70)

subcat_basic = sales.groupby("Sub-Category").agg(
    Total_Sales=("Sales", "sum"), Average_Order_Value=("Sales", "mean")
)

subcat_monthly = (
    sales.groupby([pd.Grouper(key="Order Date", freq="MS"), "Sub-Category"])["Sales"]
    .sum().reset_index()
)
subcat_volatility = subcat_monthly.groupby("Sub-Category")["Sales"].std().rename("Sales_Volatility")

subcat_yearly = sales.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack()
first_year, last_year = subcat_yearly.columns.min(), subcat_yearly.columns.max()
subcat_growth = (
    (subcat_yearly[last_year] - subcat_yearly[first_year]) / subcat_yearly[first_year].replace(0, np.nan)
) * 100
subcat_growth = subcat_growth.rename("YoY_Growth")

cluster_features = subcat_basic.join(subcat_volatility).join(subcat_growth)
cluster_features = cluster_features.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

inertias = []
k_values = range(2, 9)
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    model.fit(scaled_features)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters"); plt.ylabel("Inertia"); plt.tight_layout()
savefig("elbow_method.png")

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_features["Cluster"] = kmeans.fit_predict(scaled_features)

cluster_profiles = cluster_features.groupby("Cluster").mean()


def assign_cluster_labels(profile_df):
    labels = {}
    sales_median = profile_df["Total_Sales"].median()
    volatility_median = profile_df["Sales_Volatility"].median()
    growth_median = profile_df["YoY_Growth"].median()
    for cluster_id, row in profile_df.iterrows():
        if row["Total_Sales"] >= sales_median and row["Sales_Volatility"] < volatility_median:
            label = "High Volume, Stable Demand"
        elif row["Sales_Volatility"] >= volatility_median and row["Total_Sales"] < sales_median:
            label = "Low Volume, High Volatility"
        elif row["YoY_Growth"] >= growth_median:
            label = "Growing Demand"
        else:
            label = "Declining / Slow Demand"
        labels[cluster_id] = label
    return labels


cluster_labels = assign_cluster_labels(cluster_profiles)
cluster_features["Demand_Segment"] = cluster_features["Cluster"].map(cluster_labels)

pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame({
    "PC1": pca_components[:, 0], "PC2": pca_components[:, 1],
    "Cluster": cluster_features["Cluster"].values,
    "Sub-Category": cluster_features.index,
})

plt.figure(figsize=(12, 7))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", s=150)
for _, row in pca_df.iterrows():
    plt.text(row["PC1"] + 0.05, row["PC2"] + 0.05, row["Sub-Category"], fontsize=9)
plt.title("Product Sub-Category Clusters (PCA Projection)")
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2"); plt.tight_layout()
savefig("cluster_pca.png")

stocking_strategy = {
    "High Volume, Stable Demand": "Maintain high base stock and use frequent replenishment.",
    "Low Volume, High Volatility": "Keep conservative safety stock and monitor demand frequently.",
    "Growing Demand": "Gradually increase reorder levels and supplier capacity.",
    "Declining / Slow Demand": "Reduce inventory exposure and avoid overstocking.",
}

cluster_export = cluster_features.reset_index().rename(columns={"index": "Sub-Category"})
cluster_export["Stocking Strategy"] = cluster_export["Demand_Segment"].map(stocking_strategy)
cluster_export.to_csv(os.path.join(OUTPUTS_DIR, "cluster_data.csv"), index=False)
print("Saved outputs/cluster_data.csv")

print("\n" + "=" * 70)
print("ALL TASKS COMPLETE — charts/, outputs/, and saved_models/ are up to date.")
print("=" * 70)
