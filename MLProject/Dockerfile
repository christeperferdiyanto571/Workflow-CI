FROM python:3.12.7-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model dan kode
COPY MLProject/artifacts/model ./model
COPY MLProject/modelling.py .

# Expose port untuk MLflow serving
EXPOSE 5000

# Command untuk run model serving
CMD ["mlflow", "models", "serve", "-m", "model", "--host", "0.0.0.0", "--port", "5000"]
