"""Sample input for the reviewer — a small ETL job with a few real problems."""
from pyspark.sql import SparkSession, functions as F


def load_events(spark, path):
    return spark.read.json(path)


def enrich(events, dim_users):
    joined = events.join(dim_users, on="user_id", how="left")
    return joined.withColumn("rate", F.col("clicks") / F.col("impressions"))


def summarize(df):
    return df.groupBy("country").agg(F.avg("rate").alias("avg_rate")).collect()


def main(path):
    spark = SparkSession.builder.appName("events").getOrCreate()
    events = load_events(spark, path)
    users = spark.read.parquet("s3://warehouse/dim_users")
    return summarize(enrich(events, users))
