import hashlib
import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.robotparser import RobotFileParser

from app.config import (
    MAX_PAGE_BYTES,
    PRODUCT_CACHE_TTL_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)
from app.database import connect
from app.schemas import ProductFact, ProductFacts

PRODUCT_CACHE_VERSION = "product-images-v2"


class ProductMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.metadata: dict[str, str] = {}
        self.title_parts: list[str] = []
        self.in_title = False
        self.json_ld_scripts: list[str] = []
        self.in_json_ld = False
        self.current_script: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "meta":
            key = attributes.get("property") or attributes.get("name")
            content = attributes.get("content")
            if key and content:
                self.metadata[key.lower()] = content.strip()
        elif tag == "title":
            self.in_title = True
        elif tag == "script" and attributes.get("type", "").lower() == "application/ld+json":
            self.in_json_ld = True
            self.current_script = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        elif tag == "script" and self.in_json_ld:
            self.json_ld_scripts.append("".join(self.current_script))
            self.in_json_ld = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data.strip())
        if self.in_json_ld:
            self.current_script.append(data)


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_http_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_public_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public http:// and https:// product URLs are allowed.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError("The product URL host could not be resolved.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Private, local, and reserved network addresses are not allowed.")
    return url


def fetch_product_page(url: str) -> str:
    validate_public_http_url(url)
    _check_robots_policy(url)
    data, charset = _fetch_url(url, MAX_PAGE_BYTES, require_html=True)
    return data.decode(charset, errors="replace")


def _fetch_url(url: str, max_bytes: int, require_html: bool = False) -> tuple[bytes, str]:
    validate_public_http_url(url)
    opener = urllib.request.build_opener(SafeRedirectHandler())
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get_content_type()
            if require_html and content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError("The product URL did not return an HTML page.")
            data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError("The product page is larger than the configured size limit.")
            return data, response.headers.get_content_charset() or "utf-8"
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError(f"Could not fetch the product page: {exc}") from exc


def _check_robots_policy(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    robots_url = urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, "/robots.txt", "", "", "")
    )
    try:
        data, _ = _fetch_url(robots_url, min(MAX_PAGE_BYTES, 256_000))
    except ValueError as exc:
        if "HTTP Error 404" in str(exc) or "HTTP Error 410" in str(exc):
            return
        raise ValueError(f"Could not check robots.txt before fetching product page: {exc}") from exc
    parser = RobotFileParser(robots_url)
    parser.parse(data.decode("utf-8", errors="replace").splitlines())
    if not parser.can_fetch(USER_AGENT, url):
        raise ValueError("This product website disallows automated fetching in robots.txt.")


def _json_ld_product(parser: ProductMetadataParser) -> dict | None:
    candidates: list[dict] = []
    for script in parser.json_ld_scripts:
        try:
            payload = json.loads(script)
        except json.JSONDecodeError:
            continue
        _collect_product_objects(payload, candidates)
    return candidates[0] if candidates else None


def _collect_product_objects(value: object, candidates: list[dict]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_product_objects(item, candidates)
    elif isinstance(value, dict):
        types = value.get("@type", [])
        if isinstance(types, str):
            types = [types]
        if "Product" in types:
            candidates.append(value)
        for key, item in value.items():
            if key in {"@graph", "mainEntity", "mainEntityOfPage"}:
                _collect_product_objects(item, candidates)


def _as_text(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return _as_text(value.get("name") or value.get("value"))
    return None


def _image_candidates(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [candidate for item in value for candidate in _image_candidates(item)]
    if isinstance(value, dict):
        candidates: list[str] = []
        for key in ("url", "contentUrl"):
            if key in value:
                candidates.extend(_image_candidates(value[key]))
        return candidates
    return []


def _product_image_urls(values: list[object], page_url: str) -> list[str]:
    image_urls: list[str] = []
    for value in values:
        for candidate in _image_candidates(value):
            image_url = urllib.parse.urljoin(page_url, candidate.strip())
            try:
                validate_public_http_url(image_url)
            except ValueError:
                continue
            if image_url not in image_urls:
                image_urls.append(image_url)
            if len(image_urls) == 10:
                return image_urls
    return image_urls


def _json_ld_facts(product: dict, url: str) -> ProductFacts | None:
    name = _as_text(product.get("name"))
    description = _as_text(product.get("description"))
    if not name and not description:
        return None

    fact_values: list[tuple[str, str]] = []
    for label, value in (("name", name), ("description", description)):
        if value:
            fact_values.append((label, value))
    features = []
    properties = product.get("additionalProperty", [])
    if isinstance(properties, dict):
        properties = [properties]
    if isinstance(properties, list):
        for property_value in properties:
            if isinstance(property_value, dict):
                property_name = _as_text(property_value.get("name"))
                property_text = _as_text(property_value.get("value"))
                if property_text:
                    features.append(property_text)
                    fact_values.append((property_name or "feature", property_text))

    offers = product.get("offers", {})
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    price = _as_text(offers.get("price")) if isinstance(offers, dict) else None
    currency = _as_text(offers.get("priceCurrency")) if isinstance(offers, dict) else None
    if price:
        price_text = f"{price} {currency}".strip() if currency else price
        fact_values.append(("price", price_text))
    rating_data = product.get("aggregateRating", {})
    rating = None
    if isinstance(rating_data, dict):
        rating_value = _as_text(rating_data.get("ratingValue"))
        rating_count = _as_text(rating_data.get("reviewCount") or rating_data.get("ratingCount"))
        rating = " / ".join(value for value in (rating_value, rating_count) if value)
        if rating:
            fact_values.append(("rating", rating))
    reviews_data = product.get("review", [])
    if isinstance(reviews_data, dict):
        reviews_data = [reviews_data]
    reviews = [
        review_text
        for review in reviews_data if isinstance(reviews_data, list)
        if isinstance(review, dict)
        if (review_text := _as_text(review.get("reviewBody")))
    ]
    for review in reviews[:10]:
        fact_values.append(("review", review))
    images = _product_image_urls([product.get("image", [])], url)
    facts = [
        ProductFact(
            fact_id=f"fact_{index}",
            label=label,
            value=value,
            source_snippet=value,
        )
        for index, (label, value) in enumerate(fact_values[:20], start=1)
    ]
    return ProductFacts(
        name=name or "Product",
        category=_as_text(product.get("category")) or "Product",
        description=description or name or "Product",
        features=features[:10],
        price=price,
        rating=rating,
        reviews=reviews[:10],
        images=images,
        source_url=url,
        facts=facts,
    )


def extract_product_facts(description: str | None, url: str | None) -> ProductFacts:
    source = (description or "").strip()
    cacheable = False
    page_images: list[str] = []
    if url:
        validate_public_http_url(url)
        cached = _cached_facts(url)
        if cached:
            return ProductFacts.model_validate_json(cached)
        try:
            html = fetch_product_page(url)
            parser = ProductMetadataParser()
            parser.feed(html)
            page_images = _product_image_urls(
                [
                    parser.metadata.get("og:image"),
                    parser.metadata.get("twitter:image"),
                ],
                url,
            )
            page_title = " ".join(parser.title_parts).strip()
            structured_product = _json_ld_product(parser)
            if structured_product:
                structured_facts = _json_ld_facts(structured_product, url)
                if structured_facts:
                    structured_facts.images = _product_image_urls(
                        [
                            *structured_facts.images,
                            parser.metadata.get("og:image"),
                            parser.metadata.get("twitter:image"),
                        ],
                        url,
                    )
                    _cache_facts(url, structured_facts)
                    return structured_facts
            page_description = parser.metadata.get("og:description") or parser.metadata.get("description")
            source = page_description or source
            cacheable = bool(page_description)
            title = parser.metadata.get("og:title") or page_title
            if not source:
                raise ValueError("The page did not provide a product description.")
        except ValueError:
            if not source:
                raise
            title = source.splitlines()[0][:100]
    else:
        if not source:
            raise ValueError("Enter a product URL or paste a product description.")
        title = source.splitlines()[0][:100]

    name = title.split(" - ")[0].strip() or "Product"
    sentences = [line.strip(" -\t") for line in source.splitlines() if line.strip()]
    if len(sentences) == 1:
        sentences = [part.strip() for part in sentences[0].split(".") if part.strip()]
    facts = [
        ProductFact(
            fact_id=f"fact_{index}",
            label="description",
            value=sentence,
            source_snippet=sentence,
        )
        for index, sentence in enumerate(sentences[:20], start=1)
    ]
    if not facts:
        facts = [
            ProductFact(
                fact_id="fact_description",
                label="description",
                value=source[:500],
                source_snippet=source[:500],
            )
        ]
    facts = ProductFacts(
        name=name,
        category="Product",
        description=source[:5000],
        features=[fact.value for fact in facts[:5]],
        images=page_images,
        source_url=url,
        facts=facts,
    )
    if url and cacheable:
        _cache_facts(url, facts)
    return facts


def _cached_facts(url: str) -> str | None:
    digest = _product_cache_hash(url)
    with connect() as db:
        row = db.execute(
            "SELECT facts_json,fetched_at FROM product_cache WHERE url_hash=?",
            (digest,),
        ).fetchone()
    if not row:
        return None
    try:
        fetched_at = datetime.fromisoformat(row["fetched_at"])
    except ValueError:
        return None
    if (datetime.now(timezone.utc) - fetched_at).total_seconds() >= PRODUCT_CACHE_TTL_SECONDS:
        return None
    return row["facts_json"]


def _cache_facts(url: str, facts: ProductFacts) -> None:
    digest = _product_cache_hash(url)
    with connect() as db:
        db.execute(
            """INSERT INTO product_cache(url_hash,url,facts_json,fetched_at) VALUES(?,?,?,?)
               ON CONFLICT(url_hash) DO UPDATE SET
               url=excluded.url,facts_json=excluded.facts_json,fetched_at=excluded.fetched_at""",
            (digest, url, facts.model_dump_json(), datetime.now(timezone.utc).isoformat()),
        )


def url_hash(url: str | None) -> str | None:
    if not url:
        return None
    return hashlib.sha256(url.encode()).hexdigest()


def _product_cache_hash(url: str) -> str:
    return hashlib.sha256(f"{PRODUCT_CACHE_VERSION}:{url}".encode()).hexdigest()
