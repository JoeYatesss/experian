FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV API_KEY=test_api_key_123

# Expose the port the app runs on
EXPOSE 8000

# Create an entrypoint script
COPY setup.sh .
RUN chmod +x setup.sh
COPY test_api.sh .
RUN chmod +x test_api.sh

# Default command (can be overridden)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 