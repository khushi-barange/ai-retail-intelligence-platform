# etl/transform.py
# Purpose: Clean data, fix nulls, engineer new features

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_PATH.mkdir(exist_ok=True)


# ── Clean orders ──────────────────────────────────────────────────────────────
def clean_orders(df):
    print("\n  Cleaning orders...")

    # Convert all date columns to datetime
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Feature: delivery time in days (actual)
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # Feature: whether order was delivered late
    df["is_late"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype(int)

    # Keep only delivered orders for ML (others have no delivery data)
    df_delivered = df[df["order_status"] == "delivered"].copy()

    print(f"    ✓ Date columns converted")
    print(f"    ✓ delivery_days engineered")
    print(f"    ✓ is_late flag created")
    print(f"    ✓ Delivered orders: {len(df_delivered):,} / {len(df):,}")

    return df, df_delivered


# ── Clean products ────────────────────────────────────────────────────────────
def clean_products(df, translations):
    print("\n  Cleaning products...")

    # Merge English category names
    df = df.merge(translations, on="product_category_name", how="left")

    # Fill missing category with 'unknown'
    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    df["product_category_name_english"] = df[
        "product_category_name_english"
    ].fillna("unknown")

    # Fill missing dimensions with median
    dim_cols = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
    ]
    for col in dim_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    print(f"    ✓ English category names merged")
    print(f"    ✓ Null categories filled with 'unknown'")
    print(f"    ✓ Null dimensions filled with median")

    return df


# ── Clean geolocation ─────────────────────────────────────────────────────────
def clean_geolocation(df):
    print("\n  Cleaning geolocation...")

    before = len(df)
    df = df.drop_duplicates(subset=["geolocation_zip_code_prefix"])
    after = len(df)

    print(f"    ✓ Duplicates removed: {before - after:,} rows dropped")
    print(f"    ✓ Unique zip codes: {after:,}")

    return df


# ── Clean reviews ─────────────────────────────────────────────────────────────
def clean_reviews(df):
    print("\n  Cleaning reviews...")

    # Fill null comment text — expected, not an error
    df["review_comment_title"] = df["review_comment_title"].fillna("")
    df["review_comment_message"] = df["review_comment_message"].fillna("")

    # Convert dates
    df["review_creation_date"] = pd.to_datetime(
        df["review_creation_date"], errors="coerce"
    )

    print(f"    ✓ Null comment text filled with empty string")
    print(f"    ✓ Review dates converted")

    return df


# ── Build master orders table ─────────────────────────────────────────────────
def build_orders_master(orders, order_items, payments, reviews, customers, products, sellers):
    """
    Join all datasets into one master analytics table.
    This is the core fact table for all our SQL and ML work.
    """
    print("\n  Building master orders table...")

    # Aggregate order_items per order
    items_agg = order_items.groupby("order_id").agg(
        total_items=("order_item_id", "count"),
        total_revenue=("price", "sum"),
        total_freight=("freight_value", "sum"),
    ).reset_index()

    # Aggregate payments per order
    payments_agg = payments.groupby("order_id").agg(
        payment_type=("payment_type", "first"),
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
    ).reset_index()

    # Aggregate reviews per order (take first review score)
    reviews_agg = reviews.groupby("order_id").agg(
        review_score=("review_score", "mean"),
    ).reset_index()

    # Start joining
    master = orders.merge(customers, on="customer_id", how="left")
    master = master.merge(items_agg, on="order_id", how="left")
    master = master.merge(payments_agg, on="order_id", how="left")
    master = master.merge(reviews_agg, on="order_id", how="left")

    # Feature: profit margin (assume 20% margin on revenue)
    master["profit"] = master["total_revenue"] * 0.20

    # Feature: total order value (revenue + freight)
    master["order_value"] = master["total_revenue"] + master["total_freight"]

    print(f"    ✓ Master table built: {master.shape[0]:,} rows x {master.shape[1]} cols")

    return master


# ── Build customer features ───────────────────────────────────────────────────
def build_customer_features(master):
    """
    Build customer-level features for churn prediction.
    RFM: Recency, Frequency, Monetary
    """
    print("\n  Building customer features (RFM)...")

    # Use latest purchase date as reference
    reference_date = master["order_purchase_timestamp"].max()

    customer_features = master.groupby("customer_unique_id").agg(
        total_orders=("order_id", "count"),
        total_spent=("order_value", "sum"),
        avg_order_value=("order_value", "mean"),
        avg_review_score=("review_score", "mean"),
        last_purchase=("order_purchase_timestamp", "max"),
        first_purchase=("order_purchase_timestamp", "min"),
    ).reset_index()

    # Recency: days since last purchase
    customer_features["recency_days"] = (
        reference_date - customer_features["last_purchase"]
    ).dt.days

    # Customer lifetime in days
    customer_features["customer_lifetime_days"] = (
        customer_features["last_purchase"] - customer_features["first_purchase"]
    ).dt.days

    # Churn label: inactive for 90+ days
    customer_features["is_churned"] = (
        customer_features["recency_days"] >= 90
    ).astype(int)

    churn_rate = customer_features["is_churned"].mean() * 100
    print(f"    ✓ RFM features built")
    print(f"    ✓ Churn label created (90-day threshold)")
    print(f"    ✓ Churn rate: {churn_rate:.1f}%")
    print(f"    ✓ Total customers: {len(customer_features):,}")

    return customer_features


# ── Save processed files ──────────────────────────────────────────────────────
def save(dataframes_dict):
    print("\n  Saving processed files...")
    for name, df in dataframes_dict.items():
        filepath = PROCESSED_PATH / f"{name}.csv"
        df.to_csv(filepath, index=False)
        print(f"    ✓ Saved {name}.csv — {len(df):,} rows")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Import extract functions
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from etl.extract import extract_all

    print("=" * 55)
    print("  RETAILX — DATA TRANSFORMATION")
    print("=" * 55)

    # Load raw data
    raw = extract_all()

    # Clean each dataset
    orders_full, orders_delivered = clean_orders(raw["orders"])
    products_clean = clean_products(raw["products"], raw["translations"])
    geolocation_clean = clean_geolocation(raw["geolocation"])
    reviews_clean = clean_reviews(raw["reviews"])

    # Build master table
    master = build_orders_master(
        orders_full,
        raw["order_items"],
        raw["payments"],
        reviews_clean,
        raw["customers"],
        products_clean,
        raw["sellers"],
    )

    # Build customer features
    customer_features = build_customer_features(master)

    # Save all processed files
    save({
        "orders_clean":       orders_full,
        "orders_delivered":   orders_delivered,
        "products_clean":     products_clean,
        "geolocation_clean":  geolocation_clean,
        "reviews_clean":      reviews_clean,
        "master_orders":      master,
        "customer_features":  customer_features,
    })

    print("\n✅  Transformation complete. Ready for database load.\n")