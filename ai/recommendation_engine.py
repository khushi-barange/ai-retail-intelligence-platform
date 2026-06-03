# ai/recommendation_engine.py
# Purpose: Product recommendation engine using collaborative filtering
# Output: Top N product recommendations per customer

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

BASE      = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
RAW       = BASE / "data" / "raw"


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    orders    = pd.read_csv(PROCESSED / "orders_delivered.csv")
    items     = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    products  = pd.read_csv(PROCESSED / "products_clean.csv")
    customers = pd.read_csv(RAW / "olist_customers_dataset.csv")

    # Join orders with items and customers
    df = orders.merge(items[["order_id","product_id"]], on="order_id", how="inner")
    df = df.merge(customers[["customer_id","customer_unique_id"]], on="customer_id", how="left")
    df = df.merge(products[["product_id","product_category_name_english"]], on="product_id", how="left")

    print(f"  Loaded {len(df):,} order-item records")
    print(f"  Unique customers: {df['customer_unique_id'].nunique():,}")
    print(f"  Unique products: {df['product_id'].nunique():,}")
    print(f"  Unique categories: {df['product_category_name_english'].nunique():,}")

    return df


# ── Category-level recommendations ───────────────────────────────────────────
def build_category_matrix(df):
    """
    Build a customer-category matrix.
    Each cell = number of times customer bought from that category.
    We use categories instead of products because most customers
    only buy 1 product — categories give better signal.
    """
    # Filter customers with at least 2 purchases for better recommendations
    customer_counts = df.groupby("customer_unique_id")["order_id"].nunique()
    active_customers = customer_counts[customer_counts >= 1].index

    df_filtered = df[df["customer_unique_id"].isin(active_customers)]

    # Build matrix
    matrix = df_filtered.pivot_table(
        index="customer_unique_id",
        columns="product_category_name_english",
        values="order_id",
        aggfunc="count",
        fill_value=0
    )

    print(f"\n  Customer-category matrix: {matrix.shape}")
    return matrix


# ── Compute category co-occurrence ────────────────────────────────────────────
def category_cooccurrence(df):
    """
    Find categories frequently bought together.
    This powers the 'customers who bought X also bought Y' feature.
    """
    # Group by customer — get list of categories they bought
    customer_cats = df.groupby("customer_unique_id")["product_category_name_english"].apply(list)

    # Count co-occurrences
    cooccurrence = {}
    for cats in customer_cats:
        cats = list(set(cats))  # unique categories per customer
        for i, cat1 in enumerate(cats):
            for cat2 in cats:
                if cat1 != cat2:
                    if cat1 not in cooccurrence:
                        cooccurrence[cat1] = {}
                    cooccurrence[cat1][cat2] = cooccurrence[cat1].get(cat2, 0) + 1

    return cooccurrence


# ── Get recommendations for a category ───────────────────────────────────────
def recommend_for_category(category, cooccurrence, top_n=5):
    """
    Given a category, return top N categories frequently bought with it.
    """
    if category not in cooccurrence:
        return []

    related = cooccurrence[category]
    sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
    return [(cat, count) for cat, count in sorted_related[:top_n]]


# ── Top category combinations ─────────────────────────────────────────────────
def top_category_pairs(cooccurrence, top_n=10):
    """Find the most common category combinations across all customers."""
    pairs = []
    seen = set()

    for cat1, related in cooccurrence.items():
        for cat2, count in related.items():
            pair = tuple(sorted([cat1, cat2]))
            if pair not in seen:
                seen.add(pair)
                pairs.append({"category_1": cat1, "category_2": cat2, "co_purchases": count})

    pairs_df = pd.DataFrame(pairs).sort_values("co_purchases", ascending=False).head(top_n)
    return pairs_df


# ── Save recommendations ──────────────────────────────────────────────────────
def save_recommendations(pairs_df, cooccurrence):
    # Save top pairs
    pairs_df.to_csv(PROCESSED / "category_recommendations.csv", index=False)
    print(f"\n  Saved category_recommendations.csv")

    # Build a simple lookup table: for each category, top 3 recommendations
    lookup = []
    for cat in cooccurrence:
        recs = recommend_for_category(cat, cooccurrence, top_n=3)
        for rec_cat, count in recs:
            lookup.append({
                "if_customer_bought": cat,
                "recommend": rec_cat,
                "co_purchase_count": count
            })

    lookup_df = pd.DataFrame(lookup)
    lookup_df.to_csv(PROCESSED / "recommendation_lookup.csv", index=False)
    print(f"  Saved recommendation_lookup.csv — {len(lookup_df):,} recommendations")

    return lookup_df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  RETAILX - RECOMMENDATION ENGINE")
    print("=" * 55)

    # Load
    df = load_data()

    # Build co-occurrence
    print("\n  Building category co-occurrence matrix...")
    cooccurrence = category_cooccurrence(df)
    print(f"  Categories with recommendations: {len(cooccurrence)}")

    # Top pairs
    print("\n  Top category combinations:")
    pairs_df = top_category_pairs(cooccurrence, top_n=10)
    print(pairs_df.to_string(index=False))

    # Example recommendations
    print("\n  Example recommendations:")
    test_categories = ["health_beauty", "computers_accessories", "watches_gifts"]
    for cat in test_categories:
        recs = recommend_for_category(cat, cooccurrence, top_n=3)
        print(f"\n  If customer bought '{cat}':")
        for rec_cat, count in recs:
            print(f"    → {rec_cat} ({count} co-purchases)")

    # Save
    lookup_df = save_recommendations(pairs_df, cooccurrence)

    print("\n✅  Recommendation engine complete.\n")
