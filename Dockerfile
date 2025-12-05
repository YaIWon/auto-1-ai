FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    openssl \
    tor \
    proxychains \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -s /bin/bash autonomous

# Set up working directory
WORKDIR /autonomous_system

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /autonomous_system/{training_data,logs,temp,backups,servers,extensions}

# Set permissions
RUN chown -R autonomous:autonomous /autonomous_system
USER autonomous

# Expose ports
EXPOSE 80 443 21 22 8080 8000 8545

# Start command
CMD ["python", "autonomous_core.py"]
