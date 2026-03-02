#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "🔍 Checking database readiness..."

# We use an inline Python script with psycopg to poll the database.
# This avoids needing to install 'postgresql-client' or 'netcat' in the final Docker image.
python -c "
import os, sys, time
try:
    import psycopg
except ImportError:
    sys.exit(0)  # Skip if psycopg isn't installed yet (e.g., during builds)

db_url = os.environ.get('DATABASE_URL')
if not db_url or not db_url.startswith('postgres'):
    sys.exit(0)  # Skip if not using PostgreSQL

retries = 30
while retries > 0:
    try:
        psycopg.connect(db_url).close()
        print('✅ PostgreSQL is ready and accepting connections!')
        sys.exit(0)
    except psycopg.OperationalError:
        print('⏳ Waiting for PostgreSQL to become available...')
        time.sleep(2)
        retries -= 1

print('❌ Timeout waiting for PostgreSQL.')
sys.exit(1)
"

# Execute the main container command (e.g., uvicorn, python main.py, etc.)
exec "$@"