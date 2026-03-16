# Kafka–Spark Streaming Pipeline

## Problem

Modern security systems generate massive volumes of network events that must be processed reliably in real time for monitoring, analytics, and machine learning.

## Dataset

This project uses the **UNSW-NB15 network flow dataset**, a cybersecurity dataset containing detailed network traffic features such as ports, packet counts, bytes, and flow duration.

## Input

Network flow records streamed as events into a Kafka topic.

## Output

Processed events stored as **partitioned Parquet files** by date and hour for scalable analytics.

---

# Architecture

```
Dataset → Kafka Producer → Kafka Topic → Spark Streaming → Parquet Storage
```

---

# Pipeline Stages

**Dataset**
Structured network flow records used to simulate real-time network telemetry.

**Kafka Producer**
Reads rows from the dataset and streams them as JSON events to Kafka.

**Kafka Topic**
Acts as a durable event queue buffering incoming data.

**Spark Structured Streaming**
Consumes events from Kafka and processes them in micro-batches.

**Parquet Storage**
Writes processed events into a partitioned data lake for efficient querying.

---

# Streaming Configuration

**Batch size**
maxOffsetsPerTrigger = 100 events per batch.

**Processing interval**
trigger(processingTime = 5 seconds).

**Fault tolerance**
Spark checkpoints store offsets to allow recovery after failures.

---
Why This Architecture Scales to Millions of Events per Minute

**Kafka ingestion layer**
Kafka can ingest very large streams of events using distributed brokers and topic partitions.

**Decoupled producer and consumer**
Kafka buffers events so producers and processors can scale independently.

**Parallel stream processing**
Spark processes events in parallel across multiple cores or machines.

**Micro-batch streaming**
Spark handles data in small batches which stabilizes processing under high load.

**Backpressure control**
`maxOffsetsPerTrigger` limits how many events are processed per batch.

**Fault tolerance**
Checkpointing allows Spark to recover from failures without losing data.

**Scalable storage**
Partitioned Parquet storage supports efficient querying on large datasets.
