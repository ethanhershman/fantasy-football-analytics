"""
Shared database connection helper.
Reads connection config from environment variables (via .env).
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


def get_engine():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "ffdb")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    return create_engine(url)


def test_connection():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        print("Connected:", result.fetchone()[0])


if __name__ == "__main__":
    test_connection()
