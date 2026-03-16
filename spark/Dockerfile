FROM apache/spark:3.5.0

WORKDIR /app

COPY spark_stream.py .
COPY schema.py .

CMD ["/opt/bitnami/spark/bin/spark-submit","spark_stream.py"]