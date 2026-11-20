import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row


load_dotenv(Path(__file__).with_name(".env"))


def get_connection():
    settings = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

    if any(value is None for value in settings.values()):
        raise RuntimeError(".env dosyasında eksik database ayarı var.")

    return psycopg.connect(**settings, row_factory=dict_row)
