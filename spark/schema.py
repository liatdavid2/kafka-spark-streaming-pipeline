from pyspark.sql.types import *

flow_schema = StructType([
    StructField("event_time", StringType()),
    StructField("srcip", StringType()),
    StructField("dstip", StringType()),
    StructField("sport", IntegerType()),
    StructField("dport", IntegerType()),
    StructField("proto", StringType()),
    StructField("bytes", DoubleType()),
    StructField("packets", DoubleType())
])