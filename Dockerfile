# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (important for tensorflow & pillow)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
