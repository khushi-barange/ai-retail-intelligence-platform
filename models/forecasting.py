# models/forecasting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet

BASE        = Path(__file__).resolve().parent.parent
PROCESSED   = BASE / "data" / "processed"
SCREENSHOTS = BASE / "screenshots"

def prepare_timeseries():
    df = pd.read_csv(PROCESSED / "master_orders.csv")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    daily = (
        df[df["order_status"] == "delivered"]
        .groupby(df["order_purchase_timestamp"].dt.date)["total_revenue"]
        .sum()
        .reset_index()
    )
    daily.columns = ["ds", "y"]
    daily["ds"] = pd.to_datetime(daily["ds"])
    upper = daily["y"].quantile(0.99)
    daily = daily[daily["y"] <= upper]
    print(f"  Time series: {len(daily)} days")
    print(f"  Range: {daily['ds'].min().date()} to {daily['ds'].max().date()}")
    print(f"  Avg daily revenue: R${daily['y'].mean():,.0f}")
    return daily

def train_prophet(daily):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    model.add_country_holidays(country_name="BR")
    model.fit(daily)
    print(f"  Prophet model trained")
    return model

def generate_forecast(model, daily, periods=90):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
    forecast_out.columns = ["date", "predicted_revenue", "lower_bound", "upper_bound"]
    forecast_out.to_csv(PROCESSED / "sales_forecast.csv", index=False)
    total = forecast_out["predicted_revenue"].sum()
    print(f"  90-day forecasted revenue: R${total:,.0f}")
    print(f"  Avg daily forecast: R${forecast_out['predicted_revenue'].mean():,.0f}")
    return forecast

def plot_forecast(model, forecast, daily):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ax1 = axes[0]
    ax1.plot(daily["ds"], daily["y"], color="#185FA5", linewidth=1.5, label="Actual", alpha=0.8)
    future_mask = forecast["ds"] > daily["ds"].max()
    ax1.plot(forecast[~future_mask]["ds"], forecast[~future_mask]["yhat"],
             color="#1D9E75", linewidth=1.5, label="Fitted", alpha=0.7)
    ax1.plot(forecast[future_mask]["ds"], forecast[future_mask]["yhat"],
             color="#E24B4A", linewidth=2, label="Forecast (90 days)")
    ax1.fill_between(forecast[future_mask]["ds"],
                     forecast[future_mask]["yhat_lower"],
                     forecast[future_mask]["yhat_upper"],
                     alpha=0.2, color="#E24B4A", label="Confidence interval")
    ax1.set_title("RetailX 90-Day Sales Forecast", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Daily Revenue (R$)")
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x/1000:.0f}k"))

    ax2 = axes[1]
    ff = forecast[future_mask]
    ax2.plot(ff["ds"], ff["yhat"], color="#E24B4A", linewidth=2.5, label="Predicted")
    ax2.fill_between(ff["ds"], ff["yhat_lower"], ff["yhat_upper"],
                     alpha=0.25, color="#E24B4A", label="Confidence interval")
    ax2.set_title("Next 90 Days Zoomed", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Daily Revenue (R$)")
    ax2.legend()
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x/1000:.0f}k"))

    plt.tight_layout()
    plt.savefig(SCREENSHOTS / "sales_forecast.png", dpi=150)
    plt.show()
    print("  Saved sales_forecast.png")

def plot_seasonality(model, forecast):
    fig = model.plot_components(forecast)
    plt.suptitle("Prophet Seasonality Components", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS / "seasonality_components.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved seasonality_components.png")

if __name__ == "__main__":
    print("="*55)
    print("  RETAILX - SALES FORECASTING (PROPHET)")
    print("="*55)
    daily    = prepare_timeseries()
    model    = train_prophet(daily)
    forecast = generate_forecast(model, daily, periods=90)
    plot_forecast(model, forecast, daily)
    plot_seasonality(model, forecast)
    print("\nForecasting complete. Ready for AI insights.\n")
