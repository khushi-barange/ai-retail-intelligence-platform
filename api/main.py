# api/main.py
# Purpose: FastAPI backend — serves all analytics data via REST endpoints

import pandas as pd
import pickle
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

BASE      = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
MODELS    = BASE / "models"
REPORTS   = BASE / "reports"

app = FastAPI(
    title="RetailX Analytics API",
    description="AI-powered retail analytics platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load data on startup ──────────────────────────────────────────────────────
@app.on_event("startup")
async def load_data():
    global master, customers, forecast, churn_scores, products
    master       = pd.read_csv(PROCESSED / "master_orders.csv")
    customers    = pd.read_csv(PROCESSED / "customer_features.csv")
    forecast     = pd.read_csv(PROCESSED / "sales_forecast.csv")
    churn_scores = pd.read_csv(PROCESSED / "customer_churn_scores.csv")
    products     = pd.read_csv(PROCESSED / "products_clean.csv")
    print("Data loaded successfully")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "online",
        "platform": "RetailX Analytics API",
        "version": "1.0.0",
        "endpoints": ["/sales", "/customers", "/churn", "/forecast", "/products", "/kpis"]
    }


# ── KPIs ──────────────────────────────────────────────────────────────────────
@app.get("/kpis")
def get_kpis():
    delivered = master[master["order_status"] == "delivered"]
    return {
        "total_orders":       len(master),
        "delivered_orders":   len(delivered),
        "total_revenue":      round(master["total_revenue"].sum(), 2),
        "avg_order_value":    round(master["total_revenue"].mean(), 2),
        "total_customers":    len(customers),
        "repeat_rate_pct":    round((customers["total_orders"] > 1).mean() * 100, 2),
        "churn_rate_pct":     round(customers["is_churned"].mean() * 100, 2),
        "avg_delivery_days":  round(delivered["delivery_days"].mean(), 1),
        "late_delivery_pct":  round(delivered["is_late"].mean() * 100, 1),
        "avg_review_score":   round(master["review_score"].mean(), 2),
    }


# ── Sales ─────────────────────────────────────────────────────────────────────
@app.get("/sales")
def get_sales(period: str = "monthly"):
    df = master[master["order_status"] == "delivered"].copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    if period == "monthly":
        df["period"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    else:
        df["period"] = df["order_purchase_timestamp"].dt.to_period("W").astype(str)

    result = (
        df.groupby("period")
        .agg(
            total_orders=("order_id", "count"),
            total_revenue=("total_revenue", "sum"),
            avg_order_value=("total_revenue", "mean"),
        )
        .reset_index()
        .round(2)
    )
    return result.to_dict(orient="records")


# ── Customers ─────────────────────────────────────────────────────────────────
@app.get("/customers")
def get_customers(limit: int = 100):
    result = customers.head(limit).fillna(0)
    return result.to_dict(orient="records")


@app.get("/customers/segments")
def get_customer_segments():
    df = customers.copy()
    df["segment"] = pd.cut(
        df["total_orders"],
        bins=[0, 1, 2, 3, 5, 100],
        labels=["One-time", "Returning", "Loyal", "Champion", "VIP"]
    )
    result = df.groupby("segment", observed=True).agg(
        count=("customer_unique_id", "count"),
        avg_spent=("total_spent", "mean"),
        avg_orders=("total_orders", "mean"),
    ).reset_index().round(2)
    return result.to_dict(orient="records")


# ── Churn ─────────────────────────────────────────────────────────────────────
@app.get("/churn")
def get_churn_summary():
    df = churn_scores.copy()
    risk_dist = df["churn_risk"].value_counts().to_dict()
    high_risk = df[df["churn_risk"] == "High"].nlargest(10, "churn_probability")
    return {
        "churn_rate_pct":    round(df["is_churned"].mean() * 100, 2),
        "risk_distribution": risk_dist,
        "top_at_risk_customers": high_risk[
            ["customer_unique_id", "total_spent", "recency_days", "churn_probability"]
        ].round(3).to_dict(orient="records"),
    }


# ── Forecast ──────────────────────────────────────────────────────────────────
@app.get("/forecast")
def get_forecast():
    result = forecast.copy()
    total  = round(result["predicted_revenue"].sum(), 2)
    daily  = round(result["predicted_revenue"].mean(), 2)
    return {
        "total_forecasted_revenue": total,
        "avg_daily_revenue":        daily,
        "forecast_days":            len(result),
        "data": result.round(2).to_dict(orient="records"),
    }


# ── Products ──────────────────────────────────────────────────────────────────
@app.get("/products")
def get_top_products(limit: int = 10):
    result = products.head(limit).fillna("unknown")
    return result.to_dict(orient="records")


@app.get("/products/categories")
def get_categories():
    result = (
        products.groupby("product_category_name_english")
        .size()
        .reset_index(name="product_count")
        .sort_values("product_count", ascending=False)
        .head(15)
    )
    return result.to_dict(orient="records")


# ── Shipping ──────────────────────────────────────────────────────────────────
@app.get("/shipping")
def get_shipping_analysis():
    delivered = master[master["order_status"] == "delivered"].copy()
    result = (
        delivered.groupby("customer_state")
        .agg(
            total_orders=("order_id", "count"),
            avg_delivery_days=("delivery_days", "mean"),
            late_orders=("is_late", "sum"),
        )
        .reset_index()
    )
    result["late_rate_pct"] = (result["late_orders"] / result["total_orders"] * 100).round(1)
    result = result.sort_values("late_rate_pct", ascending=False).round(2)
    return result.to_dict(orient="records")


# ── AI Summary (reads saved report) ──────────────────────────────────────────
@app.get("/ai-summary")
def get_ai_summary():
    report_path = REPORTS / "ai_business_report.txt"
    if not report_path.exists():
        return {
            "status": "not_generated",
            "message": "Run python ai/ai_insights.py to generate the AI report first"
        }
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"status": "ready", "report": content}
