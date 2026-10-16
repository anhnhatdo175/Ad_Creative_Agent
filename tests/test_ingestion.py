import json
from datetime import datetime, timedelta, timezone

from app import database
from app.ingestion import (
    ProductMetadataParser,
    _cached_facts,
    _json_ld_facts,
    _json_ld_product,
    _product_image_urls,
    extract_product_facts,
    fetch_product_page,
)


def test_extracts_product_json_ld_fields(monkeypatch):
    import app.ingestion as ingestion

    monkeypatch.setattr(ingestion, "validate_public_http_url", lambda url: url)
    parser = ProductMetadataParser()
    parser.feed(
        """
        <script type="application/ld+json">
        {"@context":"https://schema.org","@type":"Product","name":"Mug",
         "description":"A ceramic mug.","image":[{"@type":"ImageObject","url":"/images/mug.jpg"}],
         "offers":{"price":12,"priceCurrency":"USD"},
         "aggregateRating":{"ratingValue":4.5,"reviewCount":20},
         "review":{"reviewBody":"A useful mug."},
         "additionalProperty":{"name":"Capacity","value":"350 ml"}}
        </script>
        """
    )
    product = _json_ld_product(parser)
    facts = _json_ld_facts(product, "https://shop.example/mug")

    assert facts.name == "Mug"
    assert facts.price == "12"
    assert facts.rating == "4.5 / 20"
    assert facts.reviews == ["A useful mug."]
    assert facts.images == ["https://shop.example/images/mug.jpg"]
    assert "350 ml" in facts.features
    assert {fact.label for fact in facts.facts} >= {"price", "rating", "review", "Capacity"}


def test_fetch_obeys_robots_disallow(monkeypatch):
    import app.ingestion as ingestion

    monkeypatch.setattr(ingestion, "validate_public_http_url", lambda url: url)

    def fake_fetch(url, max_bytes, require_html=False):
        if url.endswith("/robots.txt"):
            return b"User-agent: *\nDisallow: /product", "utf-8"
        raise AssertionError("Disallowed page should not be fetched")

    monkeypatch.setattr(ingestion, "_fetch_url", fake_fetch)
    try:
        fetch_product_page("https://shop.example/product/item")
    except ValueError as exc:
        assert "disallows automated fetching" in str(exc)
    else:
        raise AssertionError("Expected robots.txt disallow to prevent fetching")


def test_product_url_facts_are_cached_for_24_hours(tmp_path, monkeypatch):
    import app.ingestion as ingestion

    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "cache.sqlite3")
    database.initialize()
    monkeypatch.setattr(ingestion, "validate_public_http_url", lambda url: url)
    fetch_count = 0

    def fake_fetch(url):
        nonlocal fetch_count
        fetch_count += 1
        return """
        <meta property="og:title" content="Reusable bottle">
        <meta property="og:description" content="Made from stainless steel.">
        <meta property="og:image" content="/media/bottle-front.jpg">
        """

    monkeypatch.setattr(ingestion, "fetch_product_page", fake_fetch)
    url = "https://shop.example/bottle"
    first = extract_product_facts(None, url)
    second = extract_product_facts(None, url)
    assert fetch_count == 1
    assert first.model_dump() == second.model_dump()
    cached = _cached_facts(url)
    assert cached is not None
    assert json.loads(cached)["name"] == "Reusable bottle"
    assert first.images == ["https://shop.example/media/bottle-front.jpg"]
    assert ingestion.PRODUCT_CACHE_TTL_SECONDS == 86400


def test_product_image_urls_resolve_relative_paths_and_skip_unsafe_urls(monkeypatch):
    import app.ingestion as ingestion

    def public_hosts_only(url):
        if "127.0.0.1" in url or not url.startswith(("http://", "https://")):
            raise ValueError("unsafe image URL")
        return url

    monkeypatch.setattr(ingestion, "validate_public_http_url", public_hosts_only)
    images = _product_image_urls(
        [
            "/images/item.jpg",
            {"url": "//cdn.example/item-large.jpg"},
            "http://127.0.0.1/private.jpg",
            "javascript:alert(1)",
        ],
        "https://shop.example/products/item",
    )
    assert images == [
        "https://shop.example/images/item.jpg",
        "https://cdn.example/item-large.jpg",
    ]


def test_description_fallback_is_not_cached_as_url_product_facts(tmp_path, monkeypatch):
    import app.ingestion as ingestion

    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "fallback.sqlite3")
    database.initialize()
    monkeypatch.setattr(ingestion, "validate_public_http_url", lambda url: url)

    def fail_fetch(url):
        raise ValueError("page unavailable")

    monkeypatch.setattr(ingestion, "fetch_product_page", fail_fetch)
    url = "https://shop.example/bottle"
    facts = extract_product_facts("A stainless steel bottle.", url)
    assert facts.description == "A stainless steel bottle."
    assert _cached_facts(url) is None


def test_expired_product_cache_entry_is_not_returned(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "expired.sqlite3")
    database.initialize()
    url = "https://shop.example/old-product"
    with database.connect() as db:
        db.execute(
            "INSERT INTO product_cache(url_hash,url,facts_json,fetched_at) VALUES(?,?,?,?)",
            (
                _url_hash(url),
                url,
                "{}",
                (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
            ),
        )
    assert _cached_facts(url) is None


def _url_hash(url: str) -> str:
    from app.ingestion import url_hash

    return url_hash(url)
