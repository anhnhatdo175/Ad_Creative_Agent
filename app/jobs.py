import json
import logging
import time
import uuid
from datetime import datetime, timezone

from app import database
from app.config import OLLAMA_MODEL
from app.generation import PROMPT_VERSION, generate_variants
from app.ingestion import extract_product_facts, url_hash

logger = logging.getLogger("ad_creative")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(url: str | None, description: str | None, idempotency_key: str | None) -> str:
    with database.connect() as db:
        if idempotency_key:
            existing = db.execute("SELECT id FROM jobs WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            if existing:
                return existing["id"]
        facts = extract_product_facts(description, url)
        product_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
        db.execute(
            "INSERT INTO products(id,url,url_hash,facts_json,fetched_at) VALUES(?,?,?,?,?)",
            (product_id, url, url_hash(url), facts.model_dump_json(), now()),
        )
        db.execute(
            """INSERT INTO jobs(id,product_id,idempotency_key,status,prompt_version,created_at)
               VALUES(?,?,?,?,?,?)""",
            (job_id, product_id, idempotency_key, "queued", PROMPT_VERSION, now()),
        )
    return job_id


def process_job(job_id: str) -> None:
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        with database.connect() as db:
            job = db.execute(
                "SELECT jobs.*, products.facts_json FROM jobs JOIN products ON products.id=jobs.product_id WHERE jobs.id=?",
                (job_id,),
            ).fetchone()
            if not job or job["status"] != "queued":
                return
            db.execute(
                "UPDATE jobs SET status='running', error=NULL WHERE id=? AND status='queued'",
                (job_id,),
            )
            if db.execute("SELECT changes()").fetchone()[0] == 0:
                return
            facts_json = job["facts_json"]
        with database.connect() as db:
            step = db.execute(
                """SELECT id FROM job_steps WHERE job_id=? AND name='generate_variants'
                   ORDER BY started_at DESC LIMIT 1""",
                (job_id,),
            ).fetchone()
            if step:
                step_id = step["id"]
                db.execute(
                    """UPDATE job_steps SET status='running',error=NULL,started_at=?,finished_at=NULL
                       WHERE id=?""",
                    (now(), step_id),
                )
            else:
                step_id = str(uuid.uuid4())
                db.execute(
                    "INSERT INTO job_steps(id,job_id,name,status,started_at) VALUES(?,?,?,?,?)",
                    (step_id, job_id, "generate_variants", "running", now()),
                )
        facts = json.loads(facts_json)
        from app.schemas import ProductFacts

        variants, usage = generate_variants(ProductFacts.model_validate(facts))
        with database.connect() as db:
            invalid = []
            for variant in variants:
                if variant.flags:
                    invalid.extend(variant.flags)
                db.execute(
                    """INSERT INTO variants(
                           id,job_id,angle,caption,script_json,storyboard_json,flags_json,
                           image_prompts_json,fact_ids_json
                       ) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        variant.id,
                        job_id,
                        variant.angle,
                        variant.caption,
                        json.dumps(variant.script),
                        json.dumps([scene.model_dump() for scene in variant.storyboard]),
                        json.dumps(variant.flags),
                        json.dumps(variant.image_prompts),
                        json.dumps(variant.fact_ids),
                    ),
                )
            output = [variant.model_dump() for variant in variants]
            db.execute(
                "UPDATE job_steps SET status='succeeded', output_json=?, finished_at=? WHERE id=?",
                (json.dumps(output), now(), step_id),
            )
            db.execute(
                "UPDATE jobs SET status=?, total_cost=0 WHERE id=?",
                ("partial" if invalid else "succeeded", job_id),
            )
            db.execute(
                """INSERT INTO llm_calls(
                       id,job_id,step,model,prompt_version,input_tokens,output_tokens,
                       cost,latency_ms,valid
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()),
                    job_id,
                    "generate_variants",
                    usage.model,
                    PROMPT_VERSION,
                    usage.input_tokens,
                    usage.output_tokens,
                    0,
                    usage.latency_ms,
                    int(not invalid),
                ),
            )
    except Exception as exc:
        logger.exception("job failed request_id=%s job_id=%s step=generate_variants", request_id, job_id)
        with database.connect() as db:
            if db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
                db.execute(
                    """INSERT INTO llm_calls(
                           id,job_id,step,model,prompt_version,cost,latency_ms,error,valid
                       ) VALUES(?,?,?,?,?,?,?,?,0)""",
                    (
                        str(uuid.uuid4()),
                        job_id,
                        "generate_variants",
                        OLLAMA_MODEL,
                        PROMPT_VERSION,
                        0,
                        round((time.perf_counter() - started) * 1000),
                        str(exc)[:500],
                    ),
                )
            db.execute("UPDATE jobs SET status='failed', error=? WHERE id=?", (str(exc)[:500], job_id))
            db.execute(
                """UPDATE job_steps SET status='failed',error=?,finished_at=?
                   WHERE job_id=? AND status='running'""",
                (str(exc)[:500], now(), job_id),
            )


def job_status(job_id: str) -> dict | None:
    with database.connect() as db:
        job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not job:
            return None
        product = db.execute("SELECT facts_json FROM products WHERE id=?", (job["product_id"],)).fetchone()
        steps = database.rows_as_dicts(db.execute("SELECT * FROM job_steps WHERE job_id=?", (job_id,)).fetchall())
        variants = database.rows_as_dicts(db.execute("SELECT * FROM variants WHERE job_id=?", (job_id,)).fetchall())
        assets = database.rows_as_dicts(
            db.execute(
                """SELECT assets.* FROM assets JOIN variants ON variants.id=assets.variant_id
                   WHERE variants.job_id=? ORDER BY assets.created_at,assets.rowid""",
                (job_id,),
            ).fetchall()
        )
    latest_images: dict[str, dict] = {}
    for asset in assets:
        if asset["kind"] == "image":
            latest_images[asset["variant_id"]] = asset
    return {
        "id": job["id"],
        "status": job["status"],
        "error": job["error"],
        "product_facts": json.loads(product["facts_json"]),
        "steps": steps,
        "variants": [
            {
                **variant,
                "script": json.loads(variant["script_json"]),
                "storyboard": json.loads(variant["storyboard_json"]),
                "flags": json.loads(variant["flags_json"]),
                "image_prompts": json.loads(variant["image_prompts_json"]),
                "fact_ids": json.loads(variant["fact_ids_json"]),
                "image": (
                    {
                        "id": latest_images[variant["id"]]["id"],
                        "status": latest_images[variant["id"]]["status"],
                        "error": latest_images[variant["id"]]["error"],
                        "url": (
                            f"/assets/{latest_images[variant['id']]['id']}"
                            if latest_images[variant["id"]]["status"] == "succeeded"
                            else None
                        ),
                    }
                    if variant["id"] in latest_images
                    else None
                ),
            }
            for variant in variants
        ],
    }


def retry_job(job_id: str) -> bool:
    if not prepare_retry_job(job_id):
        return False
    process_job(job_id)
    return True


def prepare_retry_job(job_id: str) -> bool:
    with database.connect() as db:
        row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row or row["status"] != "failed":
            return False
        db.execute(
            """UPDATE job_steps SET status='queued',error=NULL
               WHERE job_id=? AND status='failed'""",
            (job_id,),
        )
        db.execute("UPDATE jobs SET status='queued', error=NULL WHERE id=?", (job_id,))
    return True


def recover_interrupted_jobs() -> list[str]:
    with database.connect() as db:
        db.execute(
            """UPDATE job_steps SET status='queued',error=NULL,started_at=NULL
               WHERE status='running' AND job_id IN
               (SELECT id FROM jobs WHERE status='running')"""
        )
        db.execute(
            """UPDATE jobs SET status='queued',error=NULL
               WHERE status='running'"""
        )
        return [
            row["id"]
            for row in db.execute(
                "SELECT id FROM jobs WHERE status='queued' ORDER BY created_at"
            ).fetchall()
        ]
