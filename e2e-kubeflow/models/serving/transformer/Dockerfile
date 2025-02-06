FROM python:3.8-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy preprocessor code
COPY preprocessor.py .

# Set environment variables
ENV MODEL_NAME=xgboost-preprocessor
ENV STORAGE_URI=/mnt/models

# Run the transformer
ENTRYPOINT ["python", "preprocessor.py"]