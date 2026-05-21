# etl/extract.py
# Purpose: Read all raw CSV files, validate data quality, report issues

import pandas as pd
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw"

# ── File map ─────────────────────────────────────────────────────────────────
FILES = {
    "orders":       "olist_orders_dataset.csv",
    "customers":    "olist_customers_dataset.csv",
    "order_items":  "olist_order_items_dataset.csv",
    "products":     "olist_products_dataset.csv",
    "sellers":      "olist_sellers_dataset.csv",
    "payments":     "olist_order_payments_dataset.csv",
    "reviews":      "olist_order_reviews_dataset.csv",
    "geolocation":  "olist_geolocation_dataset.csv",
    "translations": "product_category_name_translation.csv",
}

# ── Extract ───────────────────────────────────────────────────────────────────
def extract_all():
    """
    Read all CSV files from data/raw/.
    Returns a dict of DataFrames keyed by dataset name.
    """
    dataframes = {}

    print("=" * 55)
    print("  RETAILX — DATA EXTRACTION")
    print("=" * 55)

    for name, filename in FILES.items():
        filepath = RAW_DATA_PATH / filename

        if not filepath.exists():
            print(f"  ✗  MISSING: {filename}")
            continue

        df = pd.read_csv(filepath)
        dataframes[name] = df
        print(f"  ✓  {name:<15} {df.shape[0]:>7,} rows  {df.shape[1]:>3} cols")

    return dataframes


# ── Validate ──────────────────────────────────────────────────────────────────
def validate(dataframes):
    """
    Check each DataFrame for:
    - Missing values
    - Duplicate rows
    """
    print("\n" + "=" * 55)
    print("  DATA VALIDATION REPORT")
    print("=" * 55)

    for name, df in dataframes.items():
        null_count  = df.isnull().sum().sum()
        dupe_count  = df.duplicated().sum()
        null_pct    = (null_count / (df.shape[0] * df.shape[1])) * 100

        status = "✓" if null_count == 0 and dupe_count == 0 else "!"

        print(f"\n  [{status}] {name}")
        print(f"      Rows      : {df.shape[0]:,}")
        print(f"      Columns   : {df.shape[1]}")
        print(f"      Nulls     : {null_count:,}  ({null_pct:.1f}%)")
        print(f"      Duplicates: {dupe_count:,}")

        # Show which columns have nulls
        if null_count > 0:
            null_cols = df.isnull().sum()
            null_cols = null_cols[null_cols > 0]
            for col, count in null_cols.items():
                print(f"        → {col}: {count:,} nulls")


# ── Preview ───────────────────────────────────────────────────────────────────
def preview(dataframes):
    """Print first 2 rows of each dataset so we can see the structure."""
    print("\n" + "=" * 55)
    print("  DATA PREVIEW")
    print("=" * 55)

    for name, df in dataframes.items():
        print(f"\n── {name} ──")
        print(df.head(2).to_string())


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataframes = extract_all()
    validate(dataframes)
    preview(dataframes)
    print("\n✅  Extraction complete. Ready for transform.\n")