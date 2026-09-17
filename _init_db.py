import os

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

url = os.getenv("DATABASE_URL")
if not url:
    raise SystemExit("DATABASE_URL is not set")

admin_url = url.rsplit("/", 1)[0] + "/postgres"
conn = psycopg2.connect(admin_url)
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()
cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", ("tripdb",))
if cur.fetchone() is None:
    cur.execute("CREATE DATABASE tripdb")
    print("created tripdb")
else:
    print("tripdb already exists")
cur.close()
conn.close()

conn = psycopg2.connect(url)
cur = conn.cursor()
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS trips (
        id SERIAL PRIMARY KEY,
        destination TEXT NOT NULL,
        days INTEGER NOT NULL,
        style TEXT NOT NULL,
        itinerary TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    )
    """
)
conn.commit()
cur.execute(
    """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = %s
    ORDER BY ordinal_position
    """,
    ("trips",),
)
print("columns:", [row[0] for row in cur.fetchall()])
cur.close()
conn.close()
print("ready")
