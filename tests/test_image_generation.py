import base64
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from app import database, image_generation
from app.image_generation import CloudflareImageError


def _seed_variant(database_path):
    database.DATABASE_PATH = database_path
    database.initialize()
    with database.connect() as db:
        db.execute(
            "INSERT INTO products(id,facts_json,fetched_at) VALUES(?,?,?)",
            ("product", json.dumps({"name": "Desk lamp"}), "2026-01-01T00:00:00+00:00"),
        )
        db.execute(
            """INSERT INTO jobs(id,product_id,status,prompt_version,created_at)
               VALUES(?,?,?,?,?)""",
            ("job", "product", "succeeded", "v2", "2026-01-01T00:00:00+00:00"),
        )
        db.execute(
            """INSERT INTO variants(
                   id,job_id,angle,caption,script_json,storyboard_json,flags_json,
                   image_prompts_json,fact_ids_json
               ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                "variant",
                "job",
                "benefit",
                "A lamp",
                json.dumps("A lamp"),
                json.dumps([]),
                json.dumps([]),
                json.dumps(["A desk lamp on a clean table."]),
                json.dumps(["fact_1"]),
            ),
        )


def test_cloudflare_response_is_saved_and_reused(tmp_path, monkeypatch):
    _seed_variant(tmp_path / "images.sqlite3")
    monkeypatch.setattr(image_generation, "IMAGE_CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setattr(image_generation, "IMAGE_CLOUDFLARE_API_TOKEN", "test-token")
    monkeypatch.setattr(image_generation, "IMAGE_OUTPUT_DIRECTORY", tmp_path / "images")
    calls = []
    jpeg = b"\xff\xd8\xfftest-image\xff\xd9"

    def fake_request(prompt):
        calls.append(prompt)
        return jpeg

    monkeypatch.setattr(image_generation, "_request_image", fake_request)
    asset_id = image_generation.generate_variant_image(
        "variant",
        "A desk lamp on a clean table.",
        "Desk lamp | adjustable reading lamp",
    )
    second_id = image_generation.generate_variant_image(
        "variant",
        "A desk lamp on a clean table.",
        "Desk lamp | adjustable reading lamp",
    )

    assert asset_id == second_id
    assert len(calls) == 1
    assert calls[0].startswith("Creative direction: A desk lamp on a clean table.")
    assert "Desk lamp | adjustable reading lamp" in calls[0]
    assert "do not reinterpret its name or origin as scenery" in calls[0]
    assert "soft directional light" in calls[0]
    assert "No visible text" in calls[0]
    with database.connect() as db:
        asset = db.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
    assert asset["status"] == "succeeded"
    assert (tmp_path / "images" / asset["path"]).read_bytes() == jpeg


def test_cloudflare_rest_request_uses_flux_model_and_base64_response(monkeypatch):
    monkeypatch.setattr(image_generation, "IMAGE_CLOUDFLARE_ACCOUNT_ID", "b" * 32)
    monkeypatch.setattr(image_generation, "IMAGE_CLOUDFLARE_API_TOKEN", "test-token")
    jpeg = b"\xff\xd8\xffmock\xff\xd9"
    response_body = json.dumps(
        {"success": True, "result": {"image": base64.b64encode(jpeg).decode("ascii")}}
    ).encode()
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            assert limit > len(response_body)
            return response_body

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(image_generation.urllib.request, "urlopen", fake_urlopen)
    assert image_generation._request_image("A product photo.") == jpeg
    assert captured["url"].endswith(
        "/accounts/" + "b" * 32 + "/ai/run/@cf/black-forest-labs/flux-1-schnell"
    )
    assert captured["authorization"] == "Bearer test-token"
    assert captured["body"] == {"prompt": "A product photo.", "steps": 4}


def test_image_generation_requires_cloudflare_credentials(monkeypatch):
    monkeypatch.setattr(image_generation, "IMAGE_CLOUDFLARE_ACCOUNT_ID", "")
    monkeypatch.setattr(image_generation, "IMAGE_CLOUDFLARE_API_TOKEN", "")
    with pytest.raises(CloudflareImageError, match="CLOUDFLARE_ACCOUNT_ID"):
        image_generation.generate_variant_image("unknown", "prompt")


def test_flux_prompt_adds_studio_direction_without_exceeding_limit():
    prompt = image_generation.prepare_flux_prompt(
        "  Stainless bottle   on a clean table. ",
        "Fuji apples | Fuji is an apple variety from Japan",
    )
    assert prompt.startswith("Creative direction: Stainless bottle on a clean table.")
    assert "Fuji apples | Fuji is an apple variety from Japan" in prompt
    assert "not reinterpret its name or origin as scenery" in prompt
    assert "commercial product photography" in prompt
    assert "No visible text" in prompt
    assert len(prompt) <= image_generation.MAX_PROMPT_LENGTH

    with pytest.raises(CloudflareImageError, match="characters"):
        image_generation.prepare_flux_prompt("x" * 2049)


def test_image_route_displays_generated_art_and_includes_it_in_zip(tmp_path, monkeypatch):
    database_path = tmp_path / "route.sqlite3"
    _seed_variant(database_path)
    monkeypatch.setattr(image_generation, "IMAGE_CLOUDFLARE_ACCOUNT_ID", "c" * 32)
    monkeypatch.setattr(image_generation, "IMAGE_CLOUDFLARE_API_TOKEN", "test-token")
    monkeypatch.setattr(image_generation, "IMAGE_OUTPUT_DIRECTORY", tmp_path / "generated")
    from app import config, main

    config.DATABASE_PATH = database_path
    database.DATABASE_PATH = database_path
    monkeypatch.setattr(main, "IMAGE_OUTPUT_DIRECTORY", tmp_path / "generated")
    monkeypatch.setattr(
        image_generation,
        "_request_image",
        lambda prompt: b"\xff\xd8\xffmock\xff\xd9",
    )

    with TestClient(main.app) as client:
        response = client.post("/variants/variant/image")
        assert response.status_code == 200
        assert "Generate image with FLUX" not in response.text
        assert "Regenerate image" in response.text
        assert "generated-image" in response.text
        with database.connect() as db:
            asset = db.execute("SELECT id FROM assets WHERE variant_id='variant'").fetchone()
        image = client.get(f"/assets/{asset['id']}")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert image.content == b"\xff\xd8\xffmock\xff\xd9"
        landing = client.get("/l/variant")
        assert landing.status_code == 200
        assert f'src="/assets/{asset["id"]}"' in landing.text
        archive_response = client.get("/jobs/job/export.zip")
        assert archive_response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
            assert f"images/{asset['id']}.jpg" in archive.namelist()


def test_cloudflare_failure_is_saved_and_shown_on_job_page(tmp_path, monkeypatch):
    database_path = tmp_path / "failed-image.sqlite3"
    _seed_variant(database_path)
    monkeypatch.setattr(image_generation, "IMAGE_CLOUDFLARE_ACCOUNT_ID", "d" * 32)
    monkeypatch.setattr(image_generation, "IMAGE_CLOUDFLARE_API_TOKEN", "test-token")
    from app import config, main

    config.DATABASE_PATH = database_path
    database.DATABASE_PATH = database_path

    def fail_request(prompt):
        raise CloudflareImageError("Cloudflare image generation failed (HTTP 401): invalid token")

    monkeypatch.setattr(image_generation, "_request_image", fail_request)
    with TestClient(main.app) as client:
        response = client.post("/variants/variant/image")
        assert response.status_code == 200
        assert "HTTP 401" in response.text
        assert "Retry image" in response.text
    with database.connect() as db:
        asset = db.execute("SELECT status,error FROM assets WHERE variant_id='variant'").fetchone()
    assert asset["status"] == "failed"
    assert "invalid token" in asset["error"]


def test_url_landing_uses_real_product_image_instead_of_generated_art(tmp_path, monkeypatch):
    database_path = tmp_path / "product-image.sqlite3"
    _seed_variant(database_path)
    image_directory = tmp_path / "generated"
    image_directory.mkdir()
    generated_path = image_directory / "generated.jpg"
    generated_path.write_bytes(b"\xff\xd8\xffgenerated\xff\xd9")
    from app import config, main

    config.DATABASE_PATH = database_path
    database.DATABASE_PATH = database_path
    monkeypatch.setattr(main, "IMAGE_OUTPUT_DIRECTORY", image_directory)
    monkeypatch.setattr(main, "validate_public_http_url", lambda url: url)
    with database.connect() as db:
        db.execute(
            "UPDATE products SET url=?,facts_json=? WHERE id='product'",
            (
                "https://shop.example/product",
                json.dumps(
                    {
                        "name": "Desk lamp",
                        "description": "An adjustable desk lamp.",
                        "features": [],
                        "images": ["https://cdn.example/images/desk-lamp.jpg"],
                        "source_url": "https://shop.example/product",
                        "facts": [{"fact_id": "fact_1", "value": "An adjustable desk lamp."}],
                    }
                ),
            ),
        )
        db.execute(
            """INSERT INTO assets(id,variant_id,kind,path,status,created_at)
               VALUES(?,?,'image',?,'succeeded',?)""",
            (
                "generated-asset",
                "variant",
                generated_path.name,
                "2026-01-01T00:00:00+00:00",
            ),
        )

    with TestClient(main.app) as client:
        response = client.get("/l/variant")
    assert response.status_code == 200
    assert 'src="https://cdn.example/images/desk-lamp.jpg"' in response.text
    assert "/assets/generated-asset" not in response.text
