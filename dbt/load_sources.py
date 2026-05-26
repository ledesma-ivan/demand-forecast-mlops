"""
Loads raw Walmart CSVs into a local DuckDB database so dbt can use them as sources.
Run once before `dbt run`:

    python dbt/load_sources.py
"""
import os

import duckdb

DB_PATH = os.getenv("DBT_DB_PATH", "dbt/walmart.duckdb")
DATA_PATH = os.getenv("DATA_PATH", "data")

conn = duckdb.connect(DB_PATH)

tables = {
    "raw_train": f"{DATA_PATH}/train.csv",
    "raw_features": f"{DATA_PATH}/features.csv",
    "raw_stores": f"{DATA_PATH}/stores.csv",
}

for table, path in tables.items():
    conn.execute(
        f"CREATE OR REPLACE TABLE {table} AS "
        f"SELECT * FROM read_csv_auto('{path}', header=true)"
    )
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table}: {count:,} rows loaded from {path}")

conn.close()
print(f"\nDuckDB database ready at: {DB_PATH}")
