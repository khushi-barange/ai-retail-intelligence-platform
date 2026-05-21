# etl/load.py
# Purpose: Load cleaned data into PostgreSQL database

import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "retailx_db")
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed"

# ── Database connection ───────────────────────────────────────────────────────
def get_engine():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url)
    return engine


# ── Table definitions ─────────────────────────────────────────────────────────
# Maps: csv filename → PostgreSQL table name
TABLES = {
    "orders_clean":      "orders_raw",
    "orders_delivered":  "orders_delivered",
    "products_clean":    "products",
    "geolocation_clean": "geolocation",
    "reviews_clean":     "reviews",
    "master_orders":     "orders_fact",
    "customer_features": "customer_features",
}


# ── Load ──────────────────────────────────────────────────────────────────────
def load_all(engine):
    print("\n" + "=" * 55)
    print("  RETAILX — DATABASE LOAD")
    print("=" * 55)

    for csv_name, table_name in TABLES.items():
        filepath = PROCESSED_PATH / f"{csv_name}.csv"

        if not filepath.exists():
            print(f"  ✗  MISSING file: {csv_name}.csv — run transform.py first")
            continue

        print(f"\n  Loading {csv_name}.csv → {table_name}...")

        df = pd.read_csv(filepath)

        # Push to PostgreSQL
        # if_exists='replace' drops and recreates the table each run
        # This is fine for our project — in production you'd use 'append'
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=1000,      # write in batches of 1000 rows
            method="multi",      # faster bulk insert
        )

        # Verify row count in DB matches what we loaded
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            db_count = result.scalar()

        status = "✓" if db_count == len(df) else "✗ MISMATCH"
        print(f"    {status}  {db_count:,} rows in DB  |  {len(df):,} rows in CSV")


# ── Verify ────────────────────────────────────────────────────────────────────
def verify(engine):
    print("\n" + "=" * 55)
    print("  DATABASE VERIFICATION")
    print("=" * 55)

    query = """
        SELECT table_name, 
               pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """

    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()

    print(f"\n  {'Table':<25} {'Size':>10}")
    print(f"  {'-'*25} {'-'*10}")
    for row in rows:
        print(f"  {row[0]:<25} {row[1]:>10}")

    print(f"\n  Total tables: {len(rows)}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  RETAILX — CONNECTING TO DATABASE")
    print("=" * 55)

    engine = get_engine()

    # Test connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"\n  ✓ Connected to {DB_NAME} on {DB_HOST}:{DB_PORT}")
    except Exception as e:
        print(f"\n  ✗ Connection failed: {e}")
        exit(1)

    # Load all tables
    load_all(engine)

    # Verify
    verify(engine)

    print("\n✅  All data loaded into PostgreSQL. Ready for SQL analytics.\n")