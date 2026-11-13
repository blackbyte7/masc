# syntax=docker/dockerfile:1

# ---------- Stage 1: Builder ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Resolve the full dependency tree into wheels so the runtime stage needs no compiler.
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# ---------- Stage 2: Production runtime ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime-only libpq5 (no gcc / build headers) and an unprivileged user
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && rm -rf /wheels

# Copy only what the application needs at runtime
COPY --chown=appuser:appgroup main.py docker-entrypoint.sh custom_personas.json ./
COPY --chown=appuser:appgroup src ./src

RUN chmod +x docker-entrypoint.sh

USER appuser

EXPOSE 7860 8000

# Probes the API by default; docker-compose overrides the port for the UI service.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('MASC_HEALTH_PORT', '8000') + '/health', timeout=4)"]

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "main.py", "--mode", "api"]
