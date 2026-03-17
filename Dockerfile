# Dockerfile for WSDP

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Install WSDP
RUN pip install -e .

# Create directories for data and output
RUN mkdir -p /data /output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["wsdp", "--help"]
