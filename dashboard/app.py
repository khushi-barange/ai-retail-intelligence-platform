# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="RetailX Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border-left: 4px solid #4da6ff;
        margin-bottom: 10px;
    }
    .metric-label { color: #a8c6e8; font-size: 13px; font-weight: 500; margin-bottom: 4px; }
    .metric-value { color: #ffffff; font-size: 24px; font-weight: 700; }
    .section-header {
        color: #4da6ff;
        font-size: 16px;
        font-weight: 600;
        border-bottom: 2px solid #4da6ff;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a 0%, #1a2f45 100%); }
    .stMetric { background: #1a2f45; border-radius: 10px; padding: 12px; border-left: 3px solid #4da6ff; }
</style>
""", unsafe_allow_html=True)

BASE      = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
REPORTS   = BASE / "reports"

COLORS = {
    "primary":   "#4da6ff",
    "secondary": "#1D9E75",
    "warning":   "#EF9F27",
    "danger":    "#E24B4A",
    "purple":    "#7C5CBF",
    "bg":        "#0d1b2a",
}

@st.cache_data
def load_data():
    master    = pd.read_csv(PROCESSED / "master_orders.csv", parse_dates=["order_purchase_timestamp"])
    customers = pd.read_csv(PROCESSED / "customer_features.csv")
    forecast  = pd.read_csv(PROCESSED / "sales_forecast.csv", parse_dates=["date"])
    churn     = pd.read_csv(PROCESSED / "customer_churn_scores.csv")
    products  = pd.read_csv(PROCESSED / "products_clean.csv")
    recs      = pd.read_csv(PROCESSED / "recommendation_lookup.csv")
    return master, customers, forecast, churn, products, recs

master, customers, forecast, churn, products, recs = load_data()
delivered = master[master["order_status"] == "delivered"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛒 RetailX Analytics")
st.sidebar.markdown("*AI-Powered Retail Intelligence*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Executive Dashboard", "👥 Customer Analytics",
     "📦 Product Analytics", "📈 Sales Forecast", "🤖 AI Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Platform Stats")
st.sidebar.metric("Total Orders",    f"{len(master):,}")
st.sidebar.metric("Total Revenue",   f"R${master['total_revenue'].sum()/1e6:.2f}M")
st.sidebar.metric("Total Customers", f"{len(customers):,}")
st.sidebar.markdown("---")
st.sidebar.markdown("*Data: Brazilian E-Commerce*")
st.sidebar.markdown("*Model: XGBoost + Prophet*")
st.sidebar.markdown("*AI: Groq LLaMA 3.1*")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":
    st.title("📊 Executive Dashboard")
    st.caption("Real-time business performance overview — RetailX Brazil")
    st.markdown("---")

    # KPI row — only once
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("💰 Total Revenue",    f"R${master['total_revenue'].sum():,.0f}")
    col2.metric("📦 Total Orders",     f"{len(master):,}")
    col3.metric("🛒 Avg Order Value",  f"R${master['total_revenue'].mean():.2f}")
    col4.metric("👥 Customers",        f"{len(customers):,}")
    col5.metric("⚠️ Churn Rate",       f"{customers['is_churned'].mean()*100:.1f}%")
    col6.metric("⭐ Avg Review",       f"{master['review_score'].mean():.2f}/5")

    st.markdown("---")

    # Row 1: Revenue trend + Order status
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<p class="section-header">Monthly Revenue Trend</p>', unsafe_allow_html=True)
        monthly = (
            delivered.set_index("order_purchase_timestamp")
            .resample("ME")["total_revenue"]
            .sum()
            .reset_index()
        )
        fig = px.area(
            monthly, x="order_purchase_timestamp", y="total_revenue",
            color_discrete_sequence=[COLORS["primary"]],
            labels={"order_purchase_timestamp": "Month", "total_revenue": "Revenue (R$)"}
        )
        fig.update_traces(fillcolor="rgba(77,166,255,0.15)")
        fig.update_layout(
            showlegend=False, height=300,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", xaxis=dict(gridcolor="#2d4a6b"), yaxis=dict(gridcolor="#2d4a6b")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Order Status</p>', unsafe_allow_html=True)
        status_counts = master["order_status"].value_counts()
        # Clean labels — remove "label=" prefix
        clean_labels = [s.replace("_", " ").title() for s in status_counts.index]
        fig = px.pie(
            values=status_counts.values,
            names=clean_labels,
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            height=300, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Review scores + Delivery by state

    with col1:
        st.markdown('<p class="section-header">Review Score Distribution</p>', unsafe_allow_html=True)
    
    # Round scores and filter to 1-5 only
    review_df = master.copy()
    review_df["review_score"] = review_df["review_score"].round(0)
    review_df = review_df[review_df["review_score"].isin([1.0, 2.0, 3.0, 4.0, 5.0])]
    score_counts = review_df["review_score"].value_counts().sort_index()
    
    score_labels = ["1", "2", "3", "4", "5"]
    bar_colors   = [COLORS["danger"], "#ff7043", COLORS["warning"], "#66bb6a", COLORS["secondary"]]
    
    fig = px.bar(
        x=score_labels,
        y=score_counts.values,
        color=score_labels,
        color_discrete_sequence=bar_colors,
        labels={"x": "Rating", "y": "Number of Reviews", "color": "Rating"},
        text=score_counts.values
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        showlegend=True, height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", xaxis=dict(gridcolor="#2d4a6b"), yaxis=dict(gridcolor="#2d4a6b")
    )
    st.plotly_chart(fig, use_container_width=True)
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Analytics":
    st.title("👥 Customer Analytics")
    st.caption("Customer segmentation, churn risk, and lifetime value analysis")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers",  f"{len(customers):,}")
    col2.metric("🔄 Repeat Customers", f"{(customers['total_orders']>1).sum():,}")
    col3.metric("📊 Repeat Rate",      f"{(customers['total_orders']>1).mean()*100:.1f}%")
    col4.metric("⚠️ Churn Rate",       f"{customers['is_churned'].mean()*100:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        
        st.markdown('<p class="section-header">Customer Segments</p>', unsafe_allow_html=True)
        customers["segment"] = pd.cut(
            customers["total_orders"],
            bins=[0,1,2,3,5,100],
            labels=["One-time","Returning","Loyal","Champion","VIP"]
        )
        seg = customers["segment"].value_counts()
        fig = px.pie(
            values=seg.values, names=seg.index,
            color_discrete_sequence=[COLORS["danger"], COLORS["warning"],
                                      COLORS["primary"], COLORS["secondary"], COLORS["purple"]],
            hole=0.35
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Average Lifetime Value by Segment</p>', unsafe_allow_html=True)
        seg_spend = customers.groupby("segment", observed=True)["total_spent"].mean().sort_values(ascending=False)
        fig = px.bar(
            x=seg_spend.index, y=seg_spend.values,
            color=seg_spend.index.astype(str),
            color_discrete_sequence=[COLORS["purple"], COLORS["secondary"],
                                      COLORS["primary"], COLORS["warning"], COLORS["danger"]],
            labels={"x": "Segment", "y": "Avg Lifetime Value (R$)", "color": "Segment"},
            text=seg_spend.values
        )
        fig.update_traces(texttemplate="R$%{text:.0f}", textposition="outside")
        fig.update_layout(
            showlegend=False, height=350,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", xaxis=dict(gridcolor="#2d4a6b"), yaxis=dict(gridcolor="#2d4a6b")
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">🚨 High Churn Risk Customers (Top 20)</p>', unsafe_allow_html=True)

    high_risk = churn[churn["churn_risk"] == "High"].nlargest(20, "churn_probability")[
        ["customer_unique_id", "total_orders", "total_spent", "recency_days", "churn_probability"]
    ].round(3)
    high_risk.columns = ["Customer ID", "Orders", "Total Spent (R$)", "Days Inactive", "Churn Risk Score"]
    st.dataframe(
        high_risk.style.background_gradient(subset=["Churn Risk Score"], cmap="Reds"),
        use_container_width=True
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    risk_counts = churn["churn_risk"].value_counts()
    col1.metric("🔴 High Risk",   f"{risk_counts.get('High', 0):,}", delta="Needs immediate attention")
    col2.metric("🟡 Medium Risk", f"{risk_counts.get('Medium', 0):,}", delta="Monitor closely")
    col3.metric("🟢 Low Risk",    f"{risk_counts.get('Low', 0):,}", delta="Healthy customers")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRODUCT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Product Analytics":
    st.title("📦 Product Analytics")
    st.caption("Category performance and product recommendation insights")
    st.markdown("---")

    # KPIs for consistency
    cat_revenue = (
        recs.groupby("if_customer_bought")["co_purchase_count"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    cat_revenue.columns = ["Category", "Co-purchases"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📂 Total Categories",    f"{products['product_category_name_english'].nunique():,}")
    col2.metric("📦 Total Products",      f"{len(products):,}")
    col3.metric("🔗 Recommendation Pairs", f"{len(recs):,}")
    col4.metric("🏆 Top Category",        f"{cat_revenue.iloc[0]['Category']}")

    st.markdown("---")

    st.markdown('<p class="section-header">Top 15 Categories by Co-purchase Frequency</p>', unsafe_allow_html=True)
    fig = px.bar(
        cat_revenue, x="Co-purchases", y="Category",
        orientation="h",
        color="Co-purchases",
        color_continuous_scale=[[0, "#1e3a5f"], [0.5, COLORS["primary"]], [1, COLORS["secondary"]]],
        labels={"Co-purchases": "Co-purchase Count", "Category": ""},
        text="Co-purchases"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=500, showlegend=False,
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", xaxis=dict(gridcolor="#2d4a6b"),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Top 10 — Market Share</p>', unsafe_allow_html=True)
        top10 = cat_revenue.head(10)
        fig = px.pie(
            top10, values="Co-purchases", names="Category",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.3
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Category Data</p>', unsafe_allow_html=True)
        st.dataframe(
            cat_revenue.style.background_gradient(subset=["Co-purchases"], cmap="Blues"),
            use_container_width=True, height=380
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SALES FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Sales Forecast":
    st.title("📈 Sales Forecast")
    st.caption("Prophet ML model — 90-day revenue forecast with seasonality")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 90-Day Forecast",   f"R${forecast['predicted_revenue'].sum():,.0f}")
    col2.metric("📊 Avg Daily Revenue", f"R${forecast['predicted_revenue'].mean():,.0f}")
    col3.metric("📆 Forecast Days",     f"{len(forecast)}")
    col4.metric("📈 Peak Day Forecast", f"R${forecast['predicted_revenue'].max():,.0f}")

    st.markdown("---")

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

    st.markdown('<p class="section-header">Historical Revenue vs 90-Day Forecast</p>', unsafe_allow_html=True)
    fig = px.line(
        combined, x="date", y="revenue", color="type",
        color_discrete_map={"Historical": COLORS["primary"], "Forecast": COLORS["danger"]},
        labels={"date": "Date", "revenue": "Revenue (R$)", "type": ""},
        markers=True
    )
    fig.update_layout(
        height=400,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", xaxis=dict(gridcolor="#2d4a6b"), yaxis=dict(gridcolor="#2d4a6b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Daily Forecast with Confidence Interval</p>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=pd.concat([forecast["date"], forecast["date"][::-1]]),
        y=pd.concat([forecast["upper_bound"], forecast["lower_bound"][::-1]]),
        fill="toself", fillcolor="rgba(226,75,74,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval"
    ))
    fig2.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["predicted_revenue"],
        name="Predicted Revenue",
        line=dict(color=COLORS["danger"], width=2.5)
    ))
    fig2.update_layout(
        height=350, xaxis_title="Date", yaxis_title="Revenue (R$)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", xaxis=dict(gridcolor="#2d4a6b"), yaxis=dict(gridcolor="#2d4a6b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("View Raw Forecast Data"):
        st.dataframe(forecast.round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Insights":
    st.title("🤖 AI Insights")
    st.caption("Executive intelligence powered by Groq LLaMA 3.1")
    st.markdown("---")

    # KPIs at top for consistency
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue",   f"R${master['total_revenue'].sum():,.0f}")
    col2.metric("⚠️ Churn Rate",      f"{customers['is_churned'].mean()*100:.1f}%")
    col3.metric("📅 90-Day Forecast", f"R${forecast['predicted_revenue'].sum():,.0f}")
    col4.metric("⭐ Avg Review",      f"{master['review_score'].mean():.2f}/5")

    st.markdown("---")

    report_path = REPORTS / "ai_business_report.txt"

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report = f.read()

        # Parse and display sections
        lines = report.strip().split("\n")
        in_summary = False
        in_metrics = False
        in_recs = False
        summary_lines = []
        metrics_lines = []
        recs_lines = []

        for line in lines:
            if "EXECUTIVE SUMMARY" in line:
                in_summary = True; in_metrics = False; in_recs = False
            elif "KEY METRICS" in line:
                in_summary = False; in_metrics = True; in_recs = False
            elif "AI RECOMMENDATIONS" in line:
                in_summary = False; in_metrics = False; in_recs = True
            elif in_summary and line.strip() and "---" not in line:
                summary_lines.append(line)
            elif in_metrics and line.strip() and "---" not in line:
                metrics_lines.append(line)
            elif in_recs and line.strip():
                recs_lines.append(line)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<p class="section-header">📋 AI Executive Summary</p>', unsafe_allow_html=True)
            st.info("\n".join(summary_lines) if summary_lines else report)

        with col2:
            st.markdown('<p class="section-header">📊 Key Metrics</p>', unsafe_allow_html=True)
            for line in metrics_lines:
                if ":" in line:
                    parts = line.split(":")
                    st.markdown(f"**{parts[0].strip()}:** {':'.join(parts[1:]).strip()}")

        st.markdown("---")
        st.markdown('<p class="section-header">🎯 AI Recommendations</p>', unsafe_allow_html=True)
        for line in recs_lines:
            if line.strip():
                if "PRIORITY" in line.upper():
                    st.markdown(f"✅ {line}")
                else:
                    st.markdown(f"  {line}")

        with st.expander("📄 View Full AI Report"):
            st.text(report)

    else:
        st.warning("AI report not yet generated.")
        st.markdown("""
        **To generate the AI report, run locally:**
        ```bash
        python ai/ai_insights.py
        ```
        Then commit the generated `reports/ai_business_report.txt` file.
        """)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Churn Risk Breakdown</p>', unsafe_allow_html=True)
        risk_counts = churn["churn_risk"].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color_discrete_map={"High": COLORS["danger"], "Medium": COLORS["warning"], "Low": COLORS["secondary"]},
            hole=0.4
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            height=300,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Delivery Performance</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.metric("Avg Delivery",    f"{delivered['delivery_days'].mean():.1f} days")
        col_a.metric("Late Rate",       f"{delivered['is_late'].mean()*100:.1f}%")
        col_b.metric("Repeat Rate",     f"{(customers['total_orders']>1).mean()*100:.1f}%")
        col_b.metric("5-Star Reviews",  f"{(master['review_score']==5).sum():,}")
