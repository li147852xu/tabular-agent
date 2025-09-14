# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY conf/ ./conf/

# Create necessary directories
RUN mkdir -p /app/runs /app/data /app/output

# Set environment variables for resource limits
ENV TABULAR_AGENT_MAX_MEMORY=8G
ENV TABULAR_AGENT_MAX_THREADS=4

# Create non-root user
RUN useradd --create-home --shell /bin/bash tabular-agent && \
    chown -R tabular-agent:tabular-agent /app
USER tabular-agent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD tabular-agent --help > /dev/null || exit 1

# Default command
ENTRYPOINT ["tabular-agent"]

# Default arguments
CMD ["--help"]
