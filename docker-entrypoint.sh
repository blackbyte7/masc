#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "🔍 Checking database readiness..."

# Inline Python script with psycopg to poll the database
python -c "
import os, sys, time
try:
    import psycopg
except ImportError:
    sys.exit(0)

db_url = os.environ.get('DATABASE_URL')
if not db_url or not db_url.startswith('postgres'):
    sys.exit(0)

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

# Execute the main container command
exec "$@"