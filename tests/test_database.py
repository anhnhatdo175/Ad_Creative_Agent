import sqlite3

from app import database


def test_initialize_adds_image_prompts_to_existing_database(tmp_path, monkeypatch):
    database_path = tmp_path / "existing.sqlite3"
    monkeypatch.setattr(database, "DATABASE_PATH", database_path)
    with sqlite3.connect(database_path) as db:
        db.execute(
            """CREATE TABLE variants (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                angle TEXT NOT NULL,
                caption TEXT NOT NULL,
                script_json TEXT NOT NULL,
                storyboard_json TEXT NOT NULL,
                flags_json TEXT NOT NULL
            )"""
        )

    database.initialize()

    with database.connect() as db:
        columns = {row["name"] for row in db.execute("PRAGMA table_info(variants)")}
    assert "image_prompts_json" in columns
    assert "fact_ids_json" in columns
    with database.connect() as db:
        llm_columns = {row["name"] for row in db.execute("PRAGMA table_info(llm_calls)")}
    assert "error" in llm_columns
