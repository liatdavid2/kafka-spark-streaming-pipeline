from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("UNSWStreaming") \
    .getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "unsw-events") \
    .option("startingOffsets", "earliest") \
    .option("maxOffsetsPerTrigger", 100) \
    .load()

events = df.selectExpr("CAST(value AS STRING)")

query = events.writeStream \
    .format("parquet") \
    .option("path", "/output") \
    .option("checkpointLocation", "/checkpoints") \
    .outputMode("append") \
    .trigger(processingTime="5 seconds") \
    .start()

query.awaitTermination()