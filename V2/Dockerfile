# RobustMPC-Pharma V2 - Production Container
# 
# This Dockerfile creates a self-contained, portable image of the entire
# V2 industrial-grade MPC application for deployment in any environment.
#
# Build: docker build -t robust-mpc-pharma:v2 .
# Run:   docker run -it --rm robust-mpc-pharma:v2

# Use official Python runtime with slim variant for smaller size
FROM python:3.11-slim

# Set metadata
LABEL maintainer="PharmaControl-Pro Development Team <dev@pharmacontrol.ai>"
LABEL version="2.0.0"
LABEL description="Industrial-grade Model Predictive Control for Pharmaceutical Manufacturing"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:/app/robust_mpc"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Copy the robust_mpc library
COPY robust_mpc/ ./robust_mpc/

# Copy application files
COPY run_controller.py .
COPY config.yaml .
COPY pyproject.toml .

# Copy any pre-trained models and data files (if they exist)
# Note: In production, these might be mounted as volumes or downloaded
COPY data/ ./data/ 2>/dev/null || true
COPY models/ ./models/ 2>/dev/null || true

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash pharma && \
    chown -R pharma:pharma /app

# Switch to non-root user
USER pharma

# Health check to ensure the application can start
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import robust_mpc; print('OK')" || exit 1

# Set default configuration
ENV CONFIG_PATH="/app/config.yaml"

# Expose port for potential web interface (optional)
EXPOSE 8080

# Default command - run the controller
CMD ["python", "run_controller.py", "--config", "config.yaml", "--no-realtime"]

# Alternative entrypoints for different use cases:
# 
# Interactive mode:
# CMD ["python", "run_controller.py", "--config", "config.yaml"]
#
# Web API mode (if implemented):
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
#
# Jupyter notebook mode:
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]