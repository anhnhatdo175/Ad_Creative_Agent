import base64
import binascii
import hashlib
import json
import re
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app import database
from app.config import (
    IMAGE_CLOUDFLARE_ACCOUNT_ID,
    IMAGE_CLOUDFLARE_API_TOKEN,
    IMAGE_GENERATION_TIMEOUT_SECONDS,
    IMAGE_OUTPUT_DIRECTORY,
)

MODEL = "@cf/black-forest-labs/flux-1-schnell"
MAX_PROMPT_LENGTH = 2048
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
IMAGE_GENERATION_LOCK = threading.Lock()
IMAGE_STYLE_GUIDANCE = (
    " Polished commercial product photography, one clear hero subject, intentional balanced "
    "composition, plausible uncluttered setting, soft directional light and natural shadows. "
    "Photorealistic. No visible text, lettering, logo, watermark, collage, or extra product features."
)


class CloudflareImageError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(
        re.fullmatch(r"[a-fA-F0-9]{32}", IMAGE_CLOUDFLARE_ACCOUNT_ID)
        and IMAGE_CLOUDFLARE_API_TOKEN
    )


def generate_variant_image(
    variant_id: str,
    prompt: str,
    product_context: str = "",
) -> str:
    with IMAGE_GENERATION_LOCK:
        return _generate_variant_image(variant_id, prompt, product_context)


def _generate_variant_image(
    variant_id: str,
    prompt: str,
    product_context: str,
) -> str:
    if not is_configured():
        raise CloudflareImageError(
            "Configure CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN to enable image generation."
        )
    if not prompt.strip() or len(prompt) > MAX_PROMPT_LENGTH:
        raise CloudflareImageError(
            f"Image prompts must contain 1–{MAX_PROMPT_LENGTH} characters."
        )

    flux_prompt = prepare_flux_prompt(prompt, product_context)
    asset_id, existing_path = _prepare_asset(variant_id, flux_prompt)
    if existing_path:
        return asset_id

    try:
        image_bytes = _request_image(flux_prompt)
        output_directory = IMAGE_OUTPUT_DIRECTORY.resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        image_path = output_directory / f"{asset_id}.jpg"
        temporary_path = output_directory / f".{asset_id}.tmp"
        temporary_path.write_bytes(image_bytes)
        temporary_path.replace(image_path)
        with database.connect() as db:
            db.execute(
                """UPDATE assets SET path=?,status='succeeded',error=NULL
                   WHERE id=? AND status='generating'""",
                (image_path.name, asset_id),
            )
        return asset_id
    except CloudflareImageError as exc:
        _mark_failed(asset_id, str(exc))
        raise
    except OSError as exc:
        message = f"Could not save the generated image: {exc}"
        _mark_failed(asset_id, message)
        raise CloudflareImageError(message) from exc


def prepare_flux_prompt(prompt: str, product_context: str = "") -> str:
    normalized = " ".join(prompt.split())
    normalized_context = " ".join(product_context.split())
    if not normalized or len(normalized) > MAX_PROMPT_LENGTH:
        raise CloudflareImageError(
            f"Image prompts must contain 1–{MAX_PROMPT_LENGTH} characters."
        )
    context_direction = (
        f" Product identity from verified facts: {normalized_context}. "
        "Depict this product or service itself; do not reinterpret its name or origin as scenery."
        if normalized_context
        else ""
    )
    context_direction = context_direction[:650]
    creative_direction = f"Creative direction: {normalized}"
    available_prompt_length = (
        MAX_PROMPT_LENGTH - len(context_direction) - len(IMAGE_STYLE_GUIDANCE) - 1
    )
    creative_direction = creative_direction[:available_prompt_length]
    return f"{creative_direction}{context_direction}{IMAGE_STYLE_GUIDANCE}"


def _prepare_asset(variant_id: str, prompt: str) -> tuple[str, str | None]:
    asset_id = str(uuid.uuid4())
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    created_at = datetime.now(timezone.utc).isoformat()
    with database.connect() as db:
        if not db.execute("SELECT 1 FROM variants WHERE id=?", (variant_id,)).fetchone():
            raise CloudflareImageError("Ad concept not found.")
        previous = db.execute(
            """SELECT id,path,status,prompt_hash FROM assets WHERE variant_id=? AND kind='image'
               ORDER BY created_at DESC,rowid DESC LIMIT 1""",
            (variant_id,),
        ).fetchone()
        if previous and previous["status"] == "generating":
            raise CloudflareImageError("Image generation is already in progress for this concept.")
        previous_path = previous["path"] if previous else None
        if (
            previous
            and previous["status"] == "succeeded"
            and previous["prompt_hash"] == prompt_hash
            and previous_path
            and Path(previous_path).name == previous_path
            and (IMAGE_OUTPUT_DIRECTORY / previous_path).is_file()
        ):
            return previous["id"], previous["path"]
        if previous:
            asset_id = previous["id"]
            db.execute(
                """UPDATE assets SET prompt_hash=?,status='generating',path=NULL,
                   error=NULL,created_at=? WHERE id=?""",
                (prompt_hash, created_at, asset_id),
            )
        else:
            db.execute(
                """INSERT INTO assets(id,variant_id,kind,prompt_hash,status,created_at)
                   VALUES(?,?,'image',?,'generating',?)""",
                (asset_id, variant_id, prompt_hash, created_at),
            )
    return asset_id, None


def _request_image(prompt: str) -> bytes:
    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{IMAGE_CLOUDFLARE_ACCOUNT_ID}/ai/run/{MODEL}"
    )
    payload = json.dumps({"prompt": prompt, "steps": 4}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {IMAGE_CLOUDFLARE_API_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=IMAGE_GENERATION_TIMEOUT_SECONDS
        ) as response:
            raw_response = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw_response) > MAX_RESPONSE_BYTES:
            raise CloudflareImageError("Cloudflare returned an oversized image response.")
        result = json.loads(raw_response)
    except urllib.error.HTTPError as exc:
        detail = _response_error(exc)
        raise CloudflareImageError(
            f"Cloudflare image generation failed (HTTP {exc.code}): {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise CloudflareImageError(f"Could not connect to Cloudflare: {exc.reason}") from exc
    except TimeoutError as exc:
        raise CloudflareImageError(
            f"Cloudflare image generation timed out after {IMAGE_GENERATION_TIMEOUT_SECONDS:g} seconds."
        ) from exc
    except json.JSONDecodeError as exc:
        raise CloudflareImageError("Cloudflare returned an invalid JSON response.") from exc

    if isinstance(result, dict) and result.get("success") is False:
        errors = result.get("errors", [])
        if errors and isinstance(errors[0], dict):
            message = str(errors[0].get("message") or "Image request was rejected.")
        else:
            message = "Image request was rejected."
        raise CloudflareImageError(
            f"Cloudflare image generation failed: "
            f"{message[:500].replace(IMAGE_CLOUDFLARE_API_TOKEN, '[redacted]')}"
        )
    result_payload = result.get("result") if isinstance(result, dict) else None
    image = result_payload.get("image") if isinstance(result_payload, dict) else None
    if not isinstance(image, str):
        raise CloudflareImageError("Cloudflare response did not contain an image.")
    try:
        image_bytes = base64.b64decode(image, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CloudflareImageError("Cloudflare returned invalid image data.") from exc
    if not image_bytes.startswith(b"\xff\xd8\xff"):
        raise CloudflareImageError("Cloudflare response was not a valid JPEG image.")
    return image_bytes


def _response_error(error: urllib.error.HTTPError) -> str:
    raw_detail = error.read(2048).decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw_detail)
        errors = payload.get("errors", [])
        if errors and isinstance(errors[0], dict):
            raw_detail = str(errors[0].get("message") or raw_detail)
    except json.JSONDecodeError:
        pass
    return raw_detail[:500].replace(IMAGE_CLOUDFLARE_API_TOKEN, "[redacted]")


def _mark_failed(asset_id: str, error: str) -> None:
    with database.connect() as db:
        db.execute(
            "UPDATE assets SET status='failed',error=? WHERE id=?",
            (error[:500], asset_id),
        )
