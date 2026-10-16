import json

from app import database
from app.jobs import prepare_retry_job, process_job, recover_interrupted_jobs
from app.schemas import ProductFacts, ProductFact


def test_startup_requeues_interrupted_jobs_and_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "recovery.sqlite3")
    database.initialize()
    with database.connect() as db:
        db.execute(
            "INSERT INTO products(id,facts_json,fetched_at) VALUES(?,?,?)",
            ("product", "{}", "2026-01-01T00:00:00+00:00"),
        )
        db.execute(
            """INSERT INTO jobs(id,product_id,status,prompt_version,created_at)
               VALUES(?,?,?,?,?)""",
            ("interrupted", "product", "running", "v2", "2026-01-01T00:00:00+00:00"),
        )
        db.execute(
            "INSERT INTO job_steps(id,job_id,name,status,started_at) VALUES(?,?,?,?,?)",
            ("step", "interrupted", "generate_variants", "running", "2026-01-01T00:00:01+00:00"),
        )

    queued = recover_interrupted_jobs()

    assert queued == ["interrupted"]
    with database.connect() as db:
        job = db.execute("SELECT status FROM jobs WHERE id='interrupted'").fetchone()
        step = db.execute("SELECT status,started_at FROM job_steps WHERE id='step'").fetchone()
    assert job["status"] == "queued"
    assert tuple(step) == ("queued", None)


def test_retry_requeues_only_failed_job_and_step(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "retry.sqlite3")
    database.initialize()
    with database.connect() as db:
        db.execute(
            "INSERT INTO products(id,facts_json,fetched_at) VALUES(?,?,?)",
            ("product", json.dumps({}), "2026-01-01T00:00:00+00:00"),
        )
        db.execute(
            """INSERT INTO jobs(id,product_id,status,prompt_version,created_at,error)
               VALUES(?,?,?,?,?,?)""",
            ("failed", "product", "failed", "v2", "2026-01-01T00:00:00+00:00", "model error"),
        )
        db.execute(
            """INSERT INTO job_steps(id,job_id,name,status,error,finished_at)
               VALUES(?,?,?,?,?,?)""",
            ("step", "failed", "generate_variants", "failed", "model error", "2026-01-01T00:00:01+00:00"),
        )

    assert prepare_retry_job("failed")
    assert not prepare_retry_job("missing")
    with database.connect() as db:
        job = db.execute("SELECT status,error FROM jobs WHERE id='failed'").fetchone()
        step = db.execute("SELECT status,error FROM job_steps WHERE id='step'").fetchone()
    assert tuple(job) == ("queued", None)
    assert tuple(step) == ("queued", None)


def test_failed_model_call_is_recorded_and_job_is_marked_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "failed-call.sqlite3")
    database.initialize()
    facts = ProductFacts(
        name="Bottle",
        category="Home goods",
        description="A steel bottle.",
        facts=[
            ProductFact(
                fact_id="fact_1",
                label="description",
                value="A steel bottle.",
                source_snippet="A steel bottle.",
            )
        ],
    )
    with database.connect() as db:
        db.execute(
            "INSERT INTO products(id,facts_json,fetched_at) VALUES(?,?,?)",
            ("product", facts.model_dump_json(), "2026-01-01T00:00:00+00:00"),
        )
        db.execute(
            """INSERT INTO jobs(id,product_id,status,prompt_version,created_at)
               VALUES(?,?,?,?,?)""",
            ("failed", "product", "queued", "v2", "2026-01-01T00:00:00+00:00"),
        )
    import app.jobs as jobs

    def fail_generation(_facts):
        raise RuntimeError("Ollama offline")

    monkeypatch.setattr(jobs, "generate_variants", fail_generation)
    process_job("failed")
    with database.connect() as db:
        job = db.execute("SELECT status,error FROM jobs WHERE id='failed'").fetchone()
        llm_call = db.execute(
            "SELECT model,error,valid FROM llm_calls WHERE job_id='failed'"
        ).fetchone()
    assert job["status"] == "failed"
    assert "Ollama offline" in job["error"]
    assert llm_call["model"]
    assert "Ollama offline" in llm_call["error"]
    assert llm_call["valid"] == 0
