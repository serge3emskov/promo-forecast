FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py app.py data_preparation.py feature_engineering.py train_model.py ./

# Create data directory for model artifacts and data
RUN mkdir -p /app/data /app/models

# Expose ports
EXPOSE 8000 8501

# Default command runs training pipeline and starts API
CMD if [ -f "sales_history.csv" ]; then \
        echo "Using existing sales_history.csv"; \
    else \
        python data_preparation.py; \
    fi && \
    python feature_engineering.py && \
    if [ -f "promo_model.pkl" ]; then \
        echo "Using existing model"; \
    else \
        python train_model.py; \
    fi && \
    uvicorn api:app --host 0.0.0.0 --port 8000
