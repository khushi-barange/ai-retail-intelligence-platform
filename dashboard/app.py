# dashboard/app.py
# Purpose: Streamlit multi-page analytics dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import requests

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailX Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE      = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
REPORTS   = BASE / "reports"

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    master    = pd.read_csv(PROCESSED / "master_orders.csv", parse_dates=["order_purchase_timestamp"])
    customers = pd.read_csv(PROCESSED / "customer_features.csv")
    forecast  = pd.read_csv(PROCESSED / "sales_forecast.csv", parse_dates=["date"])
    churn     = pd.read_csv(PROCESSED / "customer_churn_scores.csv")
    products  = pd.read_csv(PROCESSED / "products_clean.csv")
    items     = pd.read_csv(PROCESSED / "category_recommendations.csv")
    return master, customers, forecast, churn, products, items

master, customers, forecast, churn, products, items = load_data()
delivered = master[master["order_status"] == "delivered"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/shop.png", width=60)
st.sidebar.title("RetailX Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Executive Dashboard", "Customer Analytics", "Product Analytics", "Sales Forecast", "AI Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Summary**")
st.sidebar.metric("Total Orders", f"{len(master):,}")
st.sidebar.metric("Total Revenue", f"R${master['total_revenue'].sum():,.0f}")
st.sidebar.metric("Total Customers", f"{len(customers):,}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Executive Dashboard":
    st.title("📊 Executive Dashboard")
    st.markdown("Real-time business performance overview")
    st.markdown("---")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Revenue",    f"R${master['total_revenue'].sum():,.0f}")
    col2.metric("Total Orders",     f"{len(master):,}")
    col3.metric("Avg Order Value",  f"R${master['total_revenue'].mean():.2f}")
    col4.metric("Churn Rate",       f"{customers['is_churned'].mean()*100:.1f}%")
    col5.metric("Avg Review Score", f"{master['review_score'].mean():.2f}/5")

    st.markdown("---")

    # Monthly revenue trend
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Monthly Revenue Trend")
        monthly = (
            delivered.set_index("order_purchase_timestamp")
            .resample("ME")["total_revenue"]
            .sum()
            .reset_index()
        )
        fig = px.area(
            monthly, x="order_purchase_timestamp", y="total_revenue",
            color_discrete_sequence=["#185FA5"],
            labels={"order_purchase_timestamp": "Month", "total_revenue": "Revenue (R$)"}
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Order Status")
        status_counts = master["order_status"].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Review scores + delivery
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Review Score Distribution")
        score_counts = master["review_score"].value_counts().sort_index()
        colors = ["#E24B4A", "#E24B4A", "#EF9F27", "#1D9E75", "#1D9E75"]
        fig = px.bar(
            x=score_counts.index, y=score_counts.values,
            color=score_counts.index.astype(str),
            color_discrete_sequence=colors,
            labels={"x": "Score", "y": "Count"}
        )
        fig.update_layout(showlegend=False, height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Delivery Performance by State")
        state_perf = (
            delivered.groupby("customer_state")
            .agg(avg_delivery=("delivery_days", "mean"), late_rate=("is_late", "mean"))
            .reset_index()
            .sort_values("late_rate", ascending=False)
            .head(10)
        )
        state_perf["late_rate"] = (state_perf["late_rate"] * 100).round(1)
        fig = px.bar(
            state_perf, x="customer_state", y="late_rate",
            color="late_rate", color_continuous_scale="Reds",
            labels={"customer_state": "State", "late_rate": "Late Rate (%)"}
        )
        fig.update_layout(height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Customer Analytics":
    st.title("👥 Customer Analytics")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers",    f"{len(customers):,}")
    col2.metric("Repeat Customers",   f"{(customers['total_orders']>1).sum():,}")
    col3.metric("Repeat Rate",        f"{(customers['total_orders']>1).mean()*100:.1f}%")
    col4.metric("Churn Rate",         f"{customers['is_churned'].mean()*100:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Segments")
        customers["segment"] = pd.cut(
            customers["total_orders"],
            bins=[0,1,2,3,5,100],
            labels=["One-time","Returning","Loyal","Champion","VIP"]
        )
        seg = customers["segment"].value_counts()
        fig = px.pie(
            values=seg.values, names=seg.index,
            color_discrete_sequence=["#E24B4A","#EF9F27","#185FA5","#1D9E75","#534AB7"]
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Avg Spend by Segment")
        seg_spend = customers.groupby("segment", observed=True)["total_spent"].mean().sort_values(ascending=False)
        fig = px.bar(
            x=seg_spend.index, y=seg_spend.values,
            color=seg_spend.index.astype(str),
            color_discrete_sequence=["#534AB7","#1D9E75","#185FA5","#EF9F27","#E24B4A"],
            labels={"x": "Segment", "y": "Avg Spend (R$)"}
        )
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🚨 High Churn Risk Customers")
    high_risk = churn[churn["churn_risk"] == "High"].nlargest(20, "churn_probability")[
        ["customer_unique_id", "total_orders", "total_spent", "recency_days", "churn_probability"]
    ].round(3)
    high_risk.columns = ["Customer ID", "Orders", "Total Spent", "Days Since Purchase", "Churn Probability"]
    st.dataframe(high_risk, use_container_width=True)

    st.subheader("Churn Risk Distribution")
    risk_counts = churn["churn_risk"].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High Risk",   f"{risk_counts.get('High', 0):,}")
    col2.metric("🟡 Medium Risk", f"{risk_counts.get('Medium', 0):,}")
    col3.metric("🟢 Low Risk",    f"{risk_counts.get('Low', 0):,}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRODUCT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Product Analytics":
    st.title("📦 Product Analytics")
    st.markdown("---")

    cat_revenue = (
    pd.read_csv(PROCESSED / "recommendation_lookup.csv")
    .groupby("if_customer_bought")["co_purchase_count"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)
cat_revenue.columns = ["Category", "Revenue"]

    st.subheader("Top 15 Categories by Revenue")
    fig = px.bar(
        cat_revenue, x="Revenue", y="Category",
        orientation="h",
        color="Revenue",
        color_continuous_scale="Blues",
        labels={"Revenue": "Total Revenue (R$)", "Category": ""}
    )
    fig.update_layout(height=500, showlegend=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Categories — Market Share")
        top10 = cat_revenue.head(10)
        fig = px.pie(
            top10, values="Revenue", names="Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Category Stats")
        st.dataframe(cat_revenue.head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SALES FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sales Forecast":
    st.title("📈 Sales Forecast")
    st.markdown("Prophet ML model — 90-day revenue forecast")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("90-Day Forecast",   f"R${forecast['predicted_revenue'].sum():,.0f}")
    col2.metric("Avg Daily Revenue", f"R${forecast['predicted_revenue'].mean():,.0f}")
    col3.metric("Forecast Period",   f"{len(forecast)} days")

    st.markdown("---")

    # Historical + forecast
    monthly_hist = (
        delivered.set_index("order_purchase_timestamp")
        .resample("ME")["total_revenue"]
        .sum()
        .reset_index()
    )
    monthly_hist.columns = ["date", "revenue"]
    monthly_hist["type"] = "Historical"

    monthly_fc = forecast.resample("ME", on="date")["predicted_revenue"].sum().reset_index()
    monthly_fc.columns = ["date", "revenue"]
    monthly_fc["type"] = "Forecast"

    combined = pd.concat([monthly_hist, monthly_fc])

    fig = px.line(
        combined, x="date", y="revenue", color="type",
        color_discrete_map={"Historical": "#185FA5", "Forecast": "#E24B4A"},
        labels={"date": "Date", "revenue": "Revenue (R$)", "type": ""}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Forecast with Confidence Interval")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["predicted_revenue"],
        name="Predicted", line=dict(color="#E24B4A", width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=pd.concat([forecast["date"], forecast["date"][::-1]]),
        y=pd.concat([forecast["upper_bound"], forecast["lower_bound"][::-1]]),
        fill="toself", fillcolor="rgba(226,75,74,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval"
    ))
    fig2.update_layout(height=350, xaxis_title="Date", yaxis_title="Revenue (R$)")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Forecast Data")
    st.dataframe(forecast.round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "AI Insights":
    st.title("🤖 AI Insights")
    st.markdown("Powered by Google Gemini")
    st.markdown("---")

    report_path = REPORTS / "ai_business_report.txt"

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report = f.read()

        sections = report.split("---") if "---" in report else [report]

        st.subheader("📋 AI Executive Summary")
        st.info(report)

    else:
        st.warning("AI report not generated yet.")
        st.markdown("""
        **To generate the AI report:**
        1. Make sure your Gemini API key is in `.env`
        2. Run: `python ai/ai_insights.py`
        3. Refresh this page
        """)

    st.markdown("---")
    st.subheader("📊 Key Business Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue",    f"R${master['total_revenue'].sum():,.0f}")
    col1.metric("Churn Rate",       f"{customers['is_churned'].mean()*100:.1f}%")
    col2.metric("Repeat Rate",      f"{(customers['total_orders']>1).mean()*100:.1f}%")
    col2.metric("Avg Delivery",     f"{delivered['delivery_days'].mean():.1f} days")
    col3.metric("90-Day Forecast",  f"R${forecast['predicted_revenue'].sum():,.0f}")
    col3.metric("Avg Review Score", f"{master['review_score'].mean():.2f}/5")
