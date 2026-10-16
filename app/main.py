import hashlib
import asyncio
import io
import json
import logging
import secrets
import uuid
import zipfile
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from app.analytics import experiment_results
from app.config import (
    IMAGE_OUTPUT_DIRECTORY,
    MIN_EXPERIMENT_VIEWS,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)
from app.database import connect, initialize
from app.image_generation import (
    CloudflareImageError,
    generate_variant_image,
    is_configured as image_generation_configured,
    prepare_flux_prompt,
)
from app.ingestion import validate_public_http_url
from app.jobs import (
    create_job,
    job_status,
    prepare_retry_job,
    process_job,
    recover_interrupted_jobs,
)
from app.schemas import ProductInput

logging.basicConfig(level=logging.INFO)


async def resume_queued_jobs(job_ids: list[str]) -> None:
    for job_id in job_ids:
        await asyncio.to_thread(process_job, job_id)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize()
    queued_jobs = recover_interrupted_jobs()
    recovery_task = asyncio.create_task(resume_queued_jobs(queued_jobs))
    yield
    if not recovery_task.done():
        recovery_task.cancel()


app = FastAPI(title="Ad Creative Agent", version="0.1.0", lifespan=lifespan)
APP_DIRECTORY = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIRECTORY / "templates"))
app.mount("/static", StaticFiles(directory=str(APP_DIRECTORY / "static")), name="static")
request_timestamps: dict[str, deque[float]] = defaultdict(deque)


@app.get("/health")
def health() -> dict:
    with connect() as db:
        db.execute("SELECT 1")
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"error": None, "idempotency_key": str(uuid.uuid4())},
    )


@app.post("/jobs", response_class=HTMLResponse)
async def submit_job(
    request: Request,
    background_tasks: BackgroundTasks,
):
    is_json = "application/json" in request.headers.get("content-type", "")
    try:
        if is_json:
            payload = ProductInput.model_validate(await request.json())
        else:
            form = await request.form()
            payload = ProductInput(
                url=(str(form.get("url", "")).strip() or None),
                description=(str(form.get("description", "")).strip() or None),
                idempotency_key=(str(form.get("idempotency_key", "")).strip() or None),
            )
        job_id = create_job(
            str(payload.url) if payload.url else None,
            payload.description,
            payload.idempotency_key,
        )
    except ValidationError as exc:
        if is_json:
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
        message = "Please enter a valid product URL or description."
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": message, "idempotency_key": str(uuid.uuid4())},
            status_code=400,
        )
    except ValueError as exc:
        if is_json:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": str(exc), "idempotency_key": str(uuid.uuid4())},
            status_code=400,
        )
    status = job_status(job_id)
    if status is not None and status["status"] == "queued":
        background_tasks.add_task(process_job, job_id)
    if is_json:
        response_status = status["status"] if status else "failed"
        status_code = 202 if response_status in {"queued", "running"} else 200
        return JSONResponse({"id": job_id, "status": response_status}, status_code=status_code)
    return RedirectResponse(f"/jobs/{job_id}/page", status_code=303)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    result = job_status(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return result


@app.get("/jobs/{job_id}/page", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    result = job_status(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    product_context = _image_product_context(result["product_facts"])
    for variant in result["variants"]:
        if variant["image_prompts"]:
            variant["flux_prompt"] = prepare_flux_prompt(
                variant["image_prompts"][0],
                product_context,
            )
    return templates.TemplateResponse(
        request,
        "job.html",
        {
            "job": result,
            "image_generation_configured": image_generation_configured(),
        },
    )


@app.post("/variants/{variant_id}/image")
def create_variant_image(variant_id: str):
    if not image_generation_configured():
        raise HTTPException(
            status_code=503,
            detail="Image generation is not configured. Set Cloudflare credentials in .env and restart the app.",
        )
    with connect() as db:
        variant = db.execute(
            """SELECT variants.job_id,variants.image_prompts_json,products.facts_json
               FROM variants JOIN jobs ON jobs.id=variants.job_id
               JOIN products ON products.id=jobs.product_id WHERE variants.id=?""",
            (variant_id,),
        ).fetchone()
    if not variant:
        raise HTTPException(status_code=404, detail="Ad concept not found.")
    prompts = json.loads(variant["image_prompts_json"])
    if not prompts:
        raise HTTPException(status_code=409, detail="This concept has no image prompt.")
    try:
        generate_variant_image(
            variant_id,
            prompts[0],
            _image_product_context(json.loads(variant["facts_json"])),
        )
    except CloudflareImageError as exc:
        with connect() as db:
            asset = db.execute(
                """SELECT status FROM assets WHERE variant_id=? AND kind='image'
                   ORDER BY created_at DESC,rowid DESC LIMIT 1""",
                (variant_id,),
            ).fetchone()
        if not asset or asset["status"] != "failed":
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{variant['job_id']}/page", status_code=303)


@app.get("/assets/{asset_id}")
def get_image_asset(asset_id: str):
    with connect() as db:
        asset = db.execute(
            "SELECT path,status FROM assets WHERE id=? AND kind='image'",
            (asset_id,),
        ).fetchone()
    if not asset or asset["status"] != "succeeded" or not asset["path"]:
        raise HTTPException(status_code=404, detail="Generated image not found.")
    if Path(asset["path"]).name != asset["path"]:
        raise HTTPException(status_code=404, detail="Generated image not found.")
    image_path = IMAGE_OUTPUT_DIRECTORY / asset["path"]
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Generated image file is missing.")
    return FileResponse(image_path, media_type="image/jpeg")


@app.post("/jobs/{job_id}/retry")
def retry(job_id: str, request: Request, background_tasks: BackgroundTasks):
    if not prepare_retry_job(job_id):
        raise HTTPException(status_code=409, detail="Only failed jobs can be retried.")
    background_tasks.add_task(process_job, job_id)
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(
            f"/jobs/{job_id}/page",
            status_code=303,
            background=background_tasks,
        )
    return JSONResponse({"id": job_id, "status": "queued"}, status_code=202)


@app.get("/l/{variant_id}", response_class=HTMLResponse)
def landing(request: Request, variant_id: str):
    variant = _variant(variant_id)
    _record_event(request, variant_id, "view")
    with connect() as db:
        product = db.execute(
            """SELECT products.facts_json,jobs.id AS job_id
               FROM products JOIN jobs ON jobs.product_id=products.id
               JOIN variants ON variants.job_id=jobs.id WHERE variants.id=?""",
            (variant_id,),
        ).fetchone()
        image = db.execute(
            """SELECT assets.id FROM assets WHERE assets.variant_id=? AND assets.kind='image'
               AND assets.status='succeeded' ORDER BY assets.created_at DESC,rowid DESC LIMIT 1""",
            (variant_id,),
        ).fetchone()
    facts = json.loads(product["facts_json"])
    cited_facts = [
        fact for fact in facts.get("facts", [])
        if fact.get("fact_id") in variant.get("fact_ids", [])
    ]
    product_image_url = None
    if facts.get("source_url"):
        for candidate in facts.get("images", []):
            try:
                product_image_url = validate_public_http_url(candidate)
                break
            except ValueError:
                continue
    concept_image_url = (
        f"/assets/{image['id']}"
        if image and not facts.get("source_url") and not product_image_url
        else None
    )
    return templates.TemplateResponse(
        request,
        "landing.html",
        {
            "variant": variant,
            "facts": facts,
            "cited_facts": cited_facts[:3],
            "variant_id": variant_id,
            "job_id": product["job_id"],
            "product_image_url": product_image_url,
            "concept_image_url": concept_image_url,
        },
    )


@app.get("/r/{variant_id}")
def click_through(request: Request, variant_id: str):
    _variant(variant_id)
    _record_event(request, variant_id, "click")
    with connect() as db:
        row = db.execute(
            """SELECT products.url FROM products JOIN jobs ON jobs.product_id=products.id
               JOIN variants ON variants.job_id=jobs.id WHERE variants.id=?""",
            (variant_id,),
        ).fetchone()
    if not row["url"]:
        return templates.TemplateResponse(
            request,
            "destination.html",
            {"variant_id": variant_id},
        )
    return RedirectResponse(row["url"], status_code=307)


@app.get("/results/{job_id}")
def results(job_id: str):
    return _experiment_results(job_id)


@app.get("/jobs/{job_id}/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, job_id: str):
    result = job_status(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    metrics = _experiment_results(job_id)
    with connect() as db:
        usage = db.execute(
            """SELECT COALESCE(group_concat(DISTINCT model), 'none') AS model,
                      COALESCE(SUM(input_tokens),0) AS input_tokens,
                      COALESCE(SUM(output_tokens),0) AS output_tokens,
                      COALESCE(SUM(cost),0) AS cost,
                      COALESCE(SUM(latency_ms),0) AS latency_ms
               FROM llm_calls WHERE job_id=?""",
            (job_id,),
        ).fetchone()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "job": result,
            "metrics": metrics,
            "usage": dict(usage),
            "minimum_views": MIN_EXPERIMENT_VIEWS,
        },
    )


@app.get("/jobs/{job_id}/export.json")
def export_json(job_id: str):
    job = job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(
        content={"job": job, "experiment": _experiment_results(job_id)},
        headers={"Content-Disposition": f'attachment; filename="ad-pack-{job_id}.json"'},
    )


@app.get("/jobs/{job_id}/export.zip")
def export_zip(job_id: str):
    job = job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    experiment = _experiment_results(job_id)
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "ad-pack.json",
            json.dumps({"job": job, "experiment": experiment}, ensure_ascii=False, indent=2),
        )
        image_prompts = [
            f"Variant {variant['id']} ({variant['angle']}):\n"
            + "\n".join(variant["image_prompts"])
            for variant in job["variants"]
        ]
        archive.writestr("image-prompts.txt", "\n\n".join(image_prompts))
        for variant in job["variants"]:
            image = variant.get("image")
            if not image or image["status"] != "succeeded":
                continue
            with connect() as db:
                asset = db.execute(
                    "SELECT path FROM assets WHERE id=? AND kind='image'",
                    (image["id"],),
                ).fetchone()
            if asset and asset["path"] and Path(asset["path"]).name == asset["path"]:
                image_path = IMAGE_OUTPUT_DIRECTORY / asset["path"]
                if image_path.is_file():
                    archive.write(image_path, f"images/{image['id']}.jpg")
    content.seek(0)
    return StreamingResponse(
        content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="ad-pack-{job_id}.zip"'},
    )


def _experiment_results(job_id: str) -> dict:
    with connect() as db:
        variants = db.execute("SELECT id,angle FROM variants WHERE job_id=?", (job_id,)).fetchall()
        if not db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Job not found.")
        rows = []
        for variant in variants:
            counts = db.execute(
                "SELECT type,COUNT(*) AS count FROM events WHERE variant_id=? GROUP BY type",
                (variant["id"],),
            ).fetchall()
            totals = {item["type"]: item["count"] for item in counts}
            views = totals.get("view", 0)
            rows.append(
                {
                    "variant_id": variant["id"],
                    "angle": variant["angle"],
                    "views": views,
                    "clicks": min(totals.get("click", 0), views),
                }
            )
    return experiment_results(rows, MIN_EXPERIMENT_VIEWS)


def _image_product_context(facts: dict | None) -> str:
    if not facts:
        return ""
    parts = [facts.get("name", ""), *facts.get("features", [])[:2]]
    return " | ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())


def _variant(variant_id: str) -> dict:
    with connect() as db:
        row = db.execute("SELECT * FROM variants WHERE id=?", (variant_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Variant not found.")
    return {
        **dict(row),
        "script": json.loads(row["script_json"]),
        "storyboard": json.loads(row["storyboard_json"]),
        "flags": json.loads(row["flags_json"]),
        "image_prompts": json.loads(row["image_prompts_json"]),
        "fact_ids": json.loads(row["fact_ids_json"]),
    }


def _record_event(request: Request, variant_id: str, event_type: str) -> None:
    if "bot" in request.headers.get("user-agent", "").lower():
        return
    visitor = request.cookies.get("visitor_id") or secrets.token_urlsafe(24)
    visitor_hash = hashlib.sha256(visitor.encode()).hexdigest()
    now = datetime.now(timezone.utc)
    with connect() as db:
        if event_type == "click":
            prior_view = db.execute(
                """SELECT 1 FROM events WHERE variant_id=? AND type='view'
                   AND visitor_hash=? AND created_at>=?""",
                (variant_id, visitor_hash, (now - timedelta(minutes=30)).isoformat()),
            ).fetchone()
            if not prior_view:
                return
        exists = db.execute(
            """SELECT 1 FROM events WHERE variant_id=? AND type=? AND visitor_hash=? AND created_at>=?""",
            (
                variant_id,
                event_type,
                visitor_hash,
                (now - timedelta(minutes=30)).isoformat(),
            ),
        ).fetchone()
        if not exists:
            db.execute(
                "INSERT INTO events(id,variant_id,type,visitor_hash,created_at) VALUES(?,?,?,?,?)",
                (str(uuid.uuid4()), variant_id, event_type, visitor_hash, now.isoformat()),
            )
    request.state.visitor_cookie = visitor if not request.cookies.get("visitor_id") else None


@app.middleware("http")
async def set_visitor_cookie(request: Request, call_next):
    limited_paths = (
        request.url.path == "/jobs"
        or request.url.path.endswith("/retry")
        or request.url.path.endswith("/image")
    )
    if limited_paths and RATE_LIMIT_REQUESTS > 0:
        client_ip = request.client.host if request.client else "unknown"
        now_timestamp = datetime.now(timezone.utc).timestamp()
        timestamps = request_timestamps[client_ip]
        while timestamps and timestamps[0] <= now_timestamp - RATE_LIMIT_WINDOW_SECONDS:
            timestamps.popleft()
        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            response = JSONResponse(
                {"detail": "Request rate limit exceeded. Please wait before trying again."},
                status_code=429,
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
            )
            return response
        timestamps.append(now_timestamp)
    response = await call_next(request)
    visitor = getattr(request.state, "visitor_cookie", None)
    if visitor:
        response.set_cookie("visitor_id", visitor, httponly=True, samesite="lax", max_age=1800)
    return response
