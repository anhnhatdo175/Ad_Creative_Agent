import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import DATABASE_PATH


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY, url TEXT, url_hash TEXT, facts_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS product_cache (
                url_hash TEXT PRIMARY KEY, url TEXT NOT NULL, facts_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(id),
                idempotency_key TEXT UNIQUE, status TEXT NOT NULL,
                prompt_version TEXT NOT NULL, created_at TEXT NOT NULL,
                total_cost REAL NOT NULL DEFAULT 0, error TEXT
            );
            CREATE TABLE IF NOT EXISTS job_steps (
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                name TEXT NOT NULL, status TEXT NOT NULL, input_hash TEXT,
                output_json TEXT, error TEXT, started_at TEXT, finished_at TEXT
            );
            CREATE TABLE IF NOT EXISTS variants (
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                angle TEXT NOT NULL, caption TEXT NOT NULL, script_json TEXT NOT NULL,
                storyboard_json TEXT NOT NULL, flags_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY, variant_id TEXT NOT NULL REFERENCES variants(id),
                kind TEXT NOT NULL, prompt_hash TEXT, path TEXT, status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT '', error TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, variant_id TEXT NOT NULL REFERENCES variants(id),
                type TEXT NOT NULL CHECK(type IN ('view', 'click')),
                visitor_hash TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS event_dedupe_idx
                ON events(variant_id, type, visitor_hash, created_at);
            CREATE TABLE IF NOT EXISTS llm_calls (
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                step TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
                cost REAL NOT NULL DEFAULT 0, latency_ms INTEGER NOT NULL DEFAULT 0, error TEXT,
                valid INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        llm_call_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(llm_calls)").fetchall()
        }
        if "error" not in llm_call_columns:
            db.execute("ALTER TABLE llm_calls ADD COLUMN error TEXT")
        variant_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(variants)").fetchall()
        }
        if "image_prompts_json" not in variant_columns:
            db.execute(
                "ALTER TABLE variants ADD COLUMN image_prompts_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "fact_ids_json" not in variant_columns:
            db.execute(
                "ALTER TABLE variants ADD COLUMN fact_ids_json TEXT NOT NULL DEFAULT '[]'"
            )
        asset_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(assets)").fetchall()
        }
        if "created_at" not in asset_columns:
            db.execute("ALTER TABLE assets ADD COLUMN created_at TEXT NOT NULL DEFAULT ''")
        if "error" not in asset_columns:
            db.execute("ALTER TABLE assets ADD COLUMN error TEXT")


def rows_as_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]


def decode(value: str) -> dict | list:
    return json.loads(value)
