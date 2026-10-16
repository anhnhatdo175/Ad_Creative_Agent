import io
import json
import uuid
import zipfile

from fastapi.testclient import TestClient


def test_job_creation_landing_and_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.sqlite3"))
    from app import config, database
    from app.database import connect

    config.DATABASE_PATH = tmp_path / "test.sqlite3"
    database.DATABASE_PATH = config.DATABASE_PATH
    from app import jobs

    jobs.database.DATABASE_PATH = config.DATABASE_PATH
    from app.generation import GenerationUsage
    from app.schemas import Variant

    def fake_generate(facts):
        return (
            [
                Variant(
                    id=str(uuid.uuid4()),
                    angle=angle,
                    caption=f"Explore {facts.name}. {facts.features[0]}",
                    script=f"Meet {facts.name}. {facts.features[0]} Explore its details.",
                    storyboard=[
                        {"visual": "Show the product.", "voiceover": f"Meet {facts.name}.", "duration": 4},
                        {"visual": "Show a feature.", "voiceover": facts.features[0], "duration": 5},
                        {"visual": "Show the product.", "voiceover": "Explore its details.", "duration": 5},
                    ],
                    image_prompts=[f"Product photo of {facts.name}."],
                    fact_ids=[facts.facts[0].fact_id],
                )
                for index, angle in enumerate(("benefit", "curiosity", "pain_point"))
            ],
            GenerationUsage("qwen3:4b", 100, 50, 1200),
        )

    monkeypatch.setattr(jobs, "generate_variants", fake_generate)
    from app.main import app

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post(
            "/jobs",
            data={"description": "A notebook made with recycled paper.", "idempotency_key": "test-key"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        job_id = response.headers["location"].split("/")[2]
        result = client.get(f"/jobs/{job_id}").json()
        assert result["status"] == "succeeded"
        assert len(result["variants"]) == 3
        assert result["variants"][0]["fact_ids"]
        assert result["variants"][0]["image_prompts"]
        job_page = client.get(f"/jobs/{job_id}/page")
        assert job_page.status_code == 200
        assert "Prompt sent to FLUX:" in job_page.text
        assert "not reinterpret its name or origin as scenery" in job_page.text
        assert 'http-equiv="refresh"' not in job_page.text
        assert 'id="job-progress"' not in job_page.text
        dashboard = client.get(f"/jobs/{job_id}/dashboard")
        assert dashboard.status_code == 200
        assert "Compare ad concepts" in dashboard.text
        assert "before estimating a winner" in dashboard.text.lower()
        json_export = client.get(f"/jobs/{job_id}/export.json")
        assert json_export.status_code == 200
        assert json_export.headers["content-disposition"].endswith(f"ad-pack-{job_id}.json\"")
        assert json_export.json()["job"]["id"] == job_id
        zip_export = client.get(f"/jobs/{job_id}/export.zip")
        assert zip_export.status_code == 200
        with zipfile.ZipFile(io.BytesIO(zip_export.content)) as archive:
            assert set(archive.namelist()) == {"ad-pack.json", "image-prompts.txt"}
            assert json.loads(archive.read("ad-pack.json"))["job"]["id"] == job_id
        variant_id = result["variants"][0]["id"]
        landing = client.get(f"/l/{variant_id}")
        assert landing.status_code == 200
        assert "notebook" in landing.text
        assert "PRODUCT DETAILS" in landing.text
        assert f'href="/jobs/{job_id}/page"' in landing.text
        assert "Back to ad pack" in landing.text
        assert "Product facts" in landing.text
        assert "No product website URL was provided." in landing.text
        assert "Product image" in landing.text
        assert landing.cookies.get("visitor_id")
        client.get(f"/l/{variant_id}")
        with connect() as db:
            view_count = db.execute(
                "SELECT COUNT(*) FROM events WHERE variant_id=? AND type='view'",
                (variant_id,),
            ).fetchone()[0]
        assert view_count == 1
        assert client.get(f"/results/{job_id}").json()["sufficient_data"] is False

        assert client.get("/static/style.css").status_code == 200
        click_response = client.get(f"/r/{variant_id}")
        assert click_response.status_code == 200
        assert "No product URL provided" in click_response.text
        assert client.get(f"/results/{job_id}").json()["variants"][0]["clicks"] == 1
        with connect() as db:
            db.execute(
                "UPDATE products SET url=? WHERE id=(SELECT product_id FROM jobs WHERE id=?)",
                ("https://shop.example/products/notebook", job_id),
            )
        redirect = client.get(f"/r/{variant_id}", follow_redirects=False)
        assert redirect.status_code == 307
        assert redirect.headers["location"] == "https://shop.example/products/notebook"
        with TestClient(app) as anonymous_client:
            anonymous_client.get(f"/r/{variant_id}")
        assert client.get(f"/results/{job_id}").json()["variants"][0]["clicks"] == 1

        api_response = client.post(
            "/jobs",
            json={"description": "A ceramic coffee mug.", "idempotency_key": "api-test"},
        )
        assert api_response.status_code == 202
        duplicate = client.post(
            "/jobs",
            json={"description": "Ignored duplicate", "idempotency_key": "api-test"},
        )
        assert duplicate.status_code == 200
        assert duplicate.json()["id"] == api_response.json()["id"]
        assert duplicate.json()["status"] == "succeeded"
        assert api_response.json()["status"] == "queued"
        with connect() as db:
            llm_call = db.execute(
                "SELECT model,input_tokens,output_tokens,cost,latency_ms FROM llm_calls WHERE job_id=?",
                (job_id,),
            ).fetchone()
        assert tuple(llm_call) == ("qwen3:4b", 100, 50, 0.0, 1200)


def test_retry_form_redirects_to_job_page(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "retry-route.sqlite3"))
    from app import config, database
    from app.database import connect

    config.DATABASE_PATH = tmp_path / "retry-route.sqlite3"
    database.DATABASE_PATH = config.DATABASE_PATH
    from app import main

    with TestClient(main.app) as client:
        with connect() as db:
            db.execute(
                "INSERT INTO products(id,facts_json,fetched_at) VALUES(?,?,?)",
                ("p", "{}", "2026-01-01T00:00:00+00:00"),
            )
            db.execute(
                """INSERT INTO jobs(id,product_id,status,prompt_version,created_at,error)
                   VALUES(?,?,?,?,?,?)""",
                ("j", "p", "failed", "v2", "2026-01-01T00:00:00+00:00", "model error"),
            )
            db.execute(
                """INSERT INTO job_steps(id,job_id,name,status,error)
                   VALUES(?,?,?,?,?)""",
                ("s", "j", "generate_variants", "failed", "model error"),
            )
        monkeypatch.setattr(main, "process_job", lambda job_id: None)
        form_response = client.post(
            "/jobs/j/retry",
            headers={"Accept": "text/html"},
            follow_redirects=False,
        )

    assert form_response.status_code == 303
    assert form_response.headers["location"] == "/jobs/j/page"

def test_job_page_polls_in_background_only_while_generation_is_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "pending.sqlite3"))
    from app import config, database

    config.DATABASE_PATH = tmp_path / "pending.sqlite3"
    database.DATABASE_PATH = config.DATABASE_PATH
    from app import main

    monkeypatch.setattr(
        main,
        "job_status",
        lambda job_id: {
            "id": job_id,
            "status": "running",
            "error": None,
            "product_facts": None,
            "steps": [],
            "variants": [],
        },
    )

    with TestClient(main.app) as client:
        response = client.get("/jobs/slow-job/page")

    assert response.status_code == 200
    assert 'id="job-progress"' in response.text
    assert "setTimeout(checkStatus, 2500)" in response.text
    assert 'http-equiv="refresh"' not in response.text


def test_job_creation_rate_limit_returns_retry_after(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "ratelimit.sqlite3"))
    from app import config, database

    config.DATABASE_PATH = tmp_path / "ratelimit.sqlite3"
    database.DATABASE_PATH = config.DATABASE_PATH
    from app import main

    main.request_timestamps.clear()
    monkeypatch.setattr(main, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(main, "RATE_LIMIT_WINDOW_SECONDS", 60)
    monkeypatch.setattr(main, "create_job", lambda *args: "queued-job")
    monkeypatch.setattr(
        main,
        "job_status",
        lambda job_id: {
            "id": job_id,
            "status": "queued",
            "error": None,
            "product_facts": None,
            "steps": [],
            "variants": [],
        },
    )
    monkeypatch.setattr(main, "process_job", lambda job_id: None)

    with TestClient(main.app) as client:
        first = client.post("/jobs", json={"description": "Product description"})
        second = client.post("/jobs", json={"description": "Another description"})

    assert first.status_code == 202
    assert second.status_code == 429
    assert second.headers["retry-after"] == "60"
    main.request_timestamps.clear()
