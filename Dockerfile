FROM python:3.11-slim

WORKDIR /app

# Install system dependencies with error handling
RUN apt-get update || true && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* || true

# Upgrade pip first
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || \
    (pip install --no-cache-dir --no-deps -r requirements.txt && \
     pip install --no-cache-dir opencv-python-headless ultralytics fastapi uvicorn)

# Copy application code
COPY . .

# Set environment variable for port
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
