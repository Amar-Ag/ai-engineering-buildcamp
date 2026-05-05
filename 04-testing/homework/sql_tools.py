import os
import urllib.request

import duckdb

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
PARQUET_FILE = "yellow_tripdata_2024-01.parquet"

con = duckdb.connect("taxi.db")


def setup_database():
    """Download the parquet file and load it into DuckDB."""
    if not os.path.exists(PARQUET_FILE):
        print(f"Downloading {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, PARQUET_FILE)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS trips AS
        SELECT * FROM '{PARQUET_FILE}'
    """)
    count = con.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    print(f"Loaded {count} rows")
    return count

class SQLTools:
    def __init__(self, con):
        self.con = con

    def get_schema(self) -> str:
        results = self.con.execute("DESCRIBE trips").fetchall()
        lines = [f"{row[0]}: {row[1]}" for row in results]
        return "\n".join(lines)

    def run_sql(self, query: str) -> str:
        cursor = self.con.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchmany(50)
        lines = [", ".join(columns)]
        for row in rows:
            lines.append(", ".join(str(v) for v in row))
        return "\n".join(lines)

