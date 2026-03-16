import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from schema import flow_schema

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC")

OUTPUT_PATH = os.getenv("OUTPUT_PATH")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")

spark = (
    SparkSession.builder
    .appName("UNSW Streaming")
    .getOrCreate()
)

raw = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", KAFKA_TOPIC)
    .load()
)

json_df = raw.selectExpr("CAST(value AS STRING)")

flows = (
    json_df
    .select(from_json(col("value"), flow_schema).alias("data"))
    .select("data.*")
)

flows = flows.withColumn(
    "event_time",
    to_timestamp("event_time")
)

features = flows.withColumn(
    "bytes_per_packet",
    col("bytes") / (col("packets") + 1)
)

agg = (
    features
    .groupBy(
        window("event_time", "1 minute"),
        col("proto")
    )
    .agg(
        count("*").alias("flow_count"),
        avg("bytes").alias("avg_bytes")
    )
)

query = (
    agg.writeStream
    .format("parquet")
    .option("path", OUTPUT_PATH)
    .option("checkpointLocation", CHECKPOINT_PATH)
    .outputMode("append")
    .start()
)

query.awaitTermination()