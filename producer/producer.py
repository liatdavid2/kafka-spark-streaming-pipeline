import json
import time
import pandas as pd

from kafka import KafkaProducer

df = pd.read_parquet("/data/UNSW_Flow.parquet")

producer = KafkaProducer(
    bootstrap_servers="kafka:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

topic = "unsw-events"

for _, row in df.iterrows():

    producer.send(topic, row.to_dict())

    time.sleep(0.01)

producer.flush()