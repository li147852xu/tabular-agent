# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# Copy source code
COPY src/ ./src/

# Default command
ENTRYPOINT ["python", "-m", "tabular_agent.cli.entry"]

# Default arguments
CMD ["--help"]