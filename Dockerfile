
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Increase timeout and retry downloads
RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=10 \
    -r requirements.txt

# Copy application files
COPY app.py .
COPY src/ ./src/

# Copy MLflow database and artifacts
COPY mlflow.db ./mlflow.db
COPY mlruns/ ./mlruns/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
