"""
Exports mart_training_set from DuckDB to Parquet for consumption by run_pipeline.py.
Run after `dbt run`:

    python dbt/export_mart.py
"""
import os

import duckdb

DB_PATH = os.getenv("DBT_DB_PATH", "dbt/walmart.duckdb")
OUTPUT_PATH = os.getenv(
    "DBT_EXPORT_PATH",
    "data/processed/dbt_features/mart_training_set.parquet",
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

conn = duckdb.connect(DB_PATH, read_only=True)
conn.execute(
    f"COPY (SELECT * FROM mart_training_set) TO '{OUTPUT_PATH}' (FORMAT PARQUET)"
)
count = conn.execute("SELECT COUNT(*) FROM mart_training_set").fetchone()[0]
conn.close()

print(f"Exported {count:,} rows to {OUTPUT_PATH}")
