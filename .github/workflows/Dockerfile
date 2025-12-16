# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Upgrade pip to latest
RUN pip install --upgrade pip

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run your training script
CMD ["python", "src/train.py"]
