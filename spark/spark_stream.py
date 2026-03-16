from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp, date_format
from schema import flow_schema

# Create Spark session
spark = SparkSession.builder \
    .appName("KafkaSparkStreaming") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Read stream from Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "unsw-events") \
    .option("startingOffsets", "earliest") \
    .option("maxOffsetsPerTrigger", 100) \
    .option("failOnDataLoss", "false") \
    .load()

# Convert Kafka value to string
json_df = kafka_df.selectExpr("CAST(value AS STRING)")

# Parse JSON using schema
parsed_df = json_df.select(
    from_json(col("value"), flow_schema).alias("data")
).select("data.*")

# Add processing timestamp
with_time_df = parsed_df.withColumn(
    "processing_time",
    current_timestamp()
)

# Derive partition columns
final_df = with_time_df \
    .withColumn("date", date_format(col("processing_time"), "yyyy-MM-dd")) \
    .withColumn("hour", date_format(col("processing_time"), "HH"))

# Write stream to parquet
query = final_df.writeStream \
    .format("parquet") \
    .outputMode("append") \
    .option("path", "/app/output/unsw_stream") \
    .option("checkpointLocation", "/app/checkpoints/unsw_stream") \
    .partitionBy("date", "hour") \
    .trigger(processingTime="5 seconds") \
    .start()

query.awaitTermination()