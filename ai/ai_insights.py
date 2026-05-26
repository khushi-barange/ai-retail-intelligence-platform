# ai/ai_insights.py
# Purpose: Generate AI-powered business insights using Google Gemini
# Uses: google-genai package (new SDK)

import pandas as pd
from google import genai
from pathlib import Path
from dotenv import load_dotenv
import os
import json
import time

load_dotenv()

BASE      = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
REPORTS   = BASE / "reports"
REPORTS.mkdir(exist_ok=True)

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL  = "gemini-2.0-flash-lite"


# ── Load business data ────────────────────────────────────────────────────────
def load_business_data():
    master    = pd.read_csv(PROCESSED / "master_orders.csv")
    customers = pd.read_csv(PROCESSED / "customer_features.csv")
    forecast  = pd.read_csv(PROCESSED / "sales_forecast.csv")
    delivered = master[master["order_status"] == "delivered"]

    stats = {
        "total_orders":         len(master),
        "delivered_orders":     len(delivered),
        "total_revenue":        round(master["total_revenue"].sum(), 2),
        "avg_order_value":      round(master["total_revenue"].mean(), 2),
        "total_customers":      len(customers),
        "repeat_customers":     int((customers["total_orders"] > 1).sum()),
        "repeat_rate_pct":      round((customers["total_orders"] > 1).mean() * 100, 2),
        "churn_rate_pct":       round(customers["is_churned"].mean() * 100, 2),
        "avg_delivery_days":    round(delivered["delivery_days"].mean(), 1),
        "late_delivery_pct":    round(delivered["is_late"].mean() * 100, 1),
        "avg_review_score":     round(master["review_score"].mean(), 2),
        "five_star_reviews":    int((master["review_score"] == 5).sum()),
        "one_star_reviews":     int((master["review_score"] == 1).sum()),
        "forecast_90d_revenue": round(forecast["predicted_revenue"].sum(), 2),
        "forecast_daily_avg":   round(forecast["predicted_revenue"].mean(), 2),
    }
    return stats


# ── Core Gemini call ──────────────────────────────────────────────────────────
def ask_gemini(prompt):
    time.sleep(3)  # avoid rate limits
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return response.text


# ── Executive summary ─────────────────────────────────────────────────────────
def generate_executive_summary(stats):
    prompt = f"""
You are a senior data analyst presenting to the CEO of RetailX, a Brazilian e-commerce company.

Business performance data:
- Total orders: {stats['total_orders']:,}
- Total revenue: R${stats['total_revenue']:,.0f}
- Average order value: R${stats['avg_order_value']:.2f}
- Total customers: {stats['total_customers']:,}
- Repeat customers: {stats['repeat_customers']:,} ({stats['repeat_rate_pct']}%)
- Churn rate: {stats['churn_rate_pct']}%
- Average delivery time: {stats['avg_delivery_days']} days
- Late delivery rate: {stats['late_delivery_pct']}%
- Average review score: {stats['avg_review_score']}/5
- 5-star reviews: {stats['five_star_reviews']:,}
- 1-star reviews: {stats['one_star_reviews']:,}
- 90-day revenue forecast: R${stats['forecast_90d_revenue']:,.0f}
- Forecasted daily average: R${stats['forecast_daily_avg']:,.0f}

Write a concise executive summary (3 paragraphs):
1. Key business wins with specific numbers
2. Top 3 risks or problems
3. Three specific actionable recommendations + revenue forecast outlook

Be direct and professional. Use specific numbers.
"""
    print("  Generating executive summary...")
    return ask_gemini(prompt)


# ── Recommendations ───────────────────────────────────────────────────────────
def generate_recommendations(stats):
    prompt = f"""
RetailX e-commerce performance metrics:
- Churn rate: {stats['churn_rate_pct']}%
- Repeat purchase rate: {stats['repeat_rate_pct']}%
- Late delivery rate: {stats['late_delivery_pct']}%
- Average delivery: {stats['avg_delivery_days']} days
- 1-star reviews: {stats['one_star_reviews']:,}
- 90-day revenue forecast: R${stats['forecast_90d_revenue']:,.0f}

Generate 5 specific business recommendations.
Format each as:
PRIORITY [1-5]: [Action] - [Expected impact with numbers]

Be specific and data-driven.
"""
    print("  Generating recommendations...")
    return ask_gemini(prompt)


# ── Natural language Q&A ──────────────────────────────────────────────────────
def answer_question(question, stats):
    prompt = f"""
You are a data analyst for RetailX. Answer this question using the data below.
Keep your answer under 100 words and include specific numbers.

Data: {json.dumps(stats)}

Question: {question}
"""
    return ask_gemini(prompt)


# ── Save report ───────────────────────────────────────────────────────────────
def save_report(summary, recommendations, stats):
    report = f"""
RETAILX - AI BUSINESS INTELLIGENCE REPORT
==========================================

EXECUTIVE SUMMARY
-----------------
{summary}

KEY METRICS SNAPSHOT
--------------------
Total Revenue      : R${stats['total_revenue']:,.0f}
Total Orders       : {stats['total_orders']:,}
Total Customers    : {stats['total_customers']:,}
Repeat Rate        : {stats['repeat_rate_pct']}%
Churn Rate         : {stats['churn_rate_pct']}%
Avg Delivery       : {stats['avg_delivery_days']} days
Late Deliveries    : {stats['late_delivery_pct']}%
Avg Review Score   : {stats['avg_review_score']}/5
90-Day Forecast    : R${stats['forecast_90d_revenue']:,.0f}

AI RECOMMENDATIONS
------------------
{recommendations}
"""
    path = REPORTS / "ai_business_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved to {path}")
    return report


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  RETAILX - AI INSIGHTS ENGINE")
    print("=" * 55)

    # Load data
    stats = load_business_data()
    print(f"  Business data loaded: {len(stats)} metrics")

    # Executive summary
    print("\n--- EXECUTIVE SUMMARY ---")
    summary = generate_executive_summary(stats)
    print(summary)

    # Recommendations
    print("\n--- AI RECOMMENDATIONS ---")
    recommendations = generate_recommendations(stats)
    print(recommendations)

    # Q&A demo
    print("\n--- Q&A DEMO ---")
    questions = [
        "Why is our churn rate so high and what should we do about it?",
        "What is our revenue forecast for the next 90 days?",
        "Which area needs the most urgent attention?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {answer_question(q, stats)}")

    # Save report
    save_report(summary, recommendations, stats)

    print("\nAI insights complete.\n")
