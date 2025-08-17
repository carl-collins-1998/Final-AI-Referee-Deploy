FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is installed
RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir --upgrade pip

# Copy project files
COPY . .

# Install Python dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Expose the port (Railway will override with $PORT)
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]