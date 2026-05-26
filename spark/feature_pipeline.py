"""
PySpark re-implementation of src/features/build_features.py.

Same transformations, same output schema — different engine.
Output: Parquet partitioned by Store and Week, readable by the existing MLflow pipeline.

Run via docker-compose.spark.yml or locally:
    spark-submit spark/feature_pipeline.py
"""
import os
from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import ByteType, IntegerType

INPUT_PATH = os.getenv("INPUT_PATH", "data")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "data/processed/spark_features")


def load_raw(spark: SparkSession):
    train = spark.read.csv(f"{INPUT_PATH}/train.csv", header=True, inferSchema=True)
    features = spark.read.csv(f"{INPUT_PATH}/features.csv", header=True, inferSchema=True)
    stores = spark.read.csv(f"{INPUT_PATH}/stores.csv", header=True, inferSchema=True)
    return train, features, stores


def merge_data(train, features, stores):
    df = train.join(stores, on="Store", how="left")
    df = df.join(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.withColumn("Date", F.to_date(F.col("Date")))
    return df


def add_temporal_features(df):
    return (
        df
        .withColumn("Week", F.weekofyear("Date").cast(IntegerType()))
        .withColumn("Month", F.month("Date").cast(IntegerType()))
        .withColumn("Quarter", F.quarter("Date").cast(IntegerType()))
        .withColumn("Is_Year_End", (F.month("Date") == 12).cast(ByteType()))
    )


def add_context_features(df):
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

    for col in markdown_cols:
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))

    active_markdowns = reduce(
        lambda a, b: a + b,
        [F.when(F.col(c) > 0, F.lit(1)).otherwise(F.lit(0)) for c in markdown_cols],
    )
    df = df.withColumn("Active_Markdowns", active_markdowns.cast(ByteType()))
    df = df.withColumn("IsHoliday", F.col("IsHoliday").cast(ByteType()))

    type_encoding = (
        F.when(F.col("Type") == "A", F.lit(3))
        .when(F.col("Type") == "B", F.lit(2))
        .otherwise(F.lit(1))
    )
    df = df.withColumn("Type_Num", type_encoding.cast(ByteType()))

    return df


def add_lag_features(df):
    """Window lag(n) over (partition by Store, Dept order by Date)."""
    w = Window.partitionBy("Store", "Dept").orderBy("Date")
    for lag in [1, 2, 4, 8, 52]:
        df = df.withColumn(f"Sales_Lag_{lag}", F.lag("Weekly_Sales", lag).over(w))
    return df


def add_rolling_features(df):
    """Rolling mean/std/max — rowsBetween(-w, -1) matches Pandas shift(1).rolling(w)."""
    for size in [4, 8, 12]:
        w = Window.partitionBy("Store", "Dept").orderBy("Date").rowsBetween(-size, -1)
        df = (
            df
            .withColumn(f"Rolling_Mean_{size}", F.avg("Weekly_Sales").over(w))
            .withColumn(f"Rolling_Std_{size}", F.stddev("Weekly_Sales").over(w))
            .withColumn(f"Rolling_Max_{size}", F.max("Weekly_Sales").over(w))
        )
    return df


def add_cross_series_features(df):
    """Cross-store aggregations within each (Date, Dept) group."""
    dept_w = Window.partitionBy("Date", "Dept")
    rank_w = Window.partitionBy("Date", "Dept").orderBy(F.desc("Weekly_Sales"))

    df = df.withColumn("Dept_Avg_Sales_All_Stores", F.avg("Weekly_Sales").over(dept_w))
    df = df.withColumn("Store_Rank_In_Dept", F.dense_rank().over(rank_w))
    return df


def run_pipeline(spark: SparkSession):
    train, features, stores = load_raw(spark)

    df = merge_data(train, features, stores)
    df = add_temporal_features(df)
    df = add_context_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_cross_series_features(df)

    (
        df.write
        .mode("overwrite")
        .partitionBy("Store", "Week")
        .parquet(OUTPUT_PATH)
    )

    row_count = df.count()
    print(f"Feature pipeline complete.")
    print(f"  Rows processed : {row_count:,}")
    print(f"  Output path    : {OUTPUT_PATH}")
    print(f"  Partitions     : Store x Week")


def main():
    spark = (
        SparkSession.builder
        .appName("walmart-demand-feature-engineering")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        run_pipeline(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
