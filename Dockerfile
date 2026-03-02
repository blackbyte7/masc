FROM python:3.10-slim

# Prevent Python from writing pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for psycopg
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the entrypoint script has executable permissions inside the container
RUN chmod +x docker-entrypoint.sh

# Expose ports for UI and API
EXPOSE 7860 8000

# Use the custom entrypoint script to handle pre-flight checks
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default to API server, can be overridden by docker-compose
CMD ["python", "main.py", "--mode", "api"]