-- Final training dataset for MLflow models.
-- Drops rows where lag_1 is null (first week per store/dept has no history).
-- Materialized as a DuckDB table; exported to Parquet by dbt/export_mart.py.
select *
from {{ ref('int_features') }}
where weekly_sales is not null
  and sales_lag_1  is not null
order by store_id, dept_id, sale_date
