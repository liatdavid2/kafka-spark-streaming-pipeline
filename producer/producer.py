import os
import json
import time
import pandas as pd
from kafka import KafkaProducer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))
STREAM_DELAY = float(os.getenv("STREAM_DELAY", 1))

DATA_PATH = "/app/data/UNSW_Flow.parquet"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

df = pd.read_parquet(DATA_PATH)

def build_event(row):

    return {
        "event_time": str(row["stime"]),
        "srcip": row["srcip"],
        "dstip": row["dstip"],
        "sport": int(row["sport"]),
        "dport": int(row["dport"]),
        "proto": row["proto"],
        "bytes": float(row["bytes"]),
        "packets": float(row["packets"])
    }

for i in range(0, len(df), BATCH_SIZE):

    batch = df.iloc[i:i+BATCH_SIZE]

    for _, row in batch.iterrows():

        event = build_event(row)

        key = f"{event['srcip']}-{event['dstip']}".encode()

        producer.send(KAFKA_TOPIC, key=key, value=event)

    producer.flush()

    print(f"sent batch {i}")

    time.sleep(STREAM_DELAY)