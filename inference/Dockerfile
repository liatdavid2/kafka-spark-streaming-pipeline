FROM python:3.10-slim

WORKDIR /app

# copy inference code
COPY inference /app

# 🔥 copy training code
COPY training /app/training

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]