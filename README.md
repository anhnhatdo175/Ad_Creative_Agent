# Ad Creative Agent

A local-first product prototype that uses Ollama to turn product facts into three advertising concepts and measure landing-page views/clicks. It includes structured output validation, fact references, job recovery, experiment analytics, a results dashboard, exports, prompt evaluation tooling, and on-demand FLUX.1 schnell image generation through Cloudflare Workers AI.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Install and start Ollama, then pull the default local model:

```powershell
ollama pull qwen3:4b
ollama serve
```

If Ollama is already running in the system tray, leave it running; no need to start a second server. Copy `.env.example` to `.env` to configure another installed model. Open <http://127.0.0.1:8000>. The application makes requests only to the local Ollama endpoint by default; no hosted LLM API key is needed. Jobs are stored in SQLite and queued/running jobs are resumed after application restart. This lightweight worker implementation is intended for a single local application process, not a multi-worker production deployment.

Product URLs use a real product image from Product JSON-LD, Open Graph, or Twitter metadata when available. Image links are restricted to public HTTP(S) URLs and displayed without a referrer. If a URL page has no discoverable product image, the landing page indicates that rather than substituting generated artwork.

To enable image generation, create a Cloudflare API token with Workers AI execution permission, then set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` in `.env`. Never put the token in source code or commit `.env`. Images are generated only after clicking **Generate image with FLUX** on a description-only ad concept; the prompt is augmented with verified product identity and commercial-photo direction, then sent to Cloudflare Workers AI. This may consume account quota. Generated JPEGs are stored in `IMAGE_OUTPUT_DIRECTORY` and included in ZIP exports. Text-to-image models can invent visual details, so verify the product depiction before publishing.

If generation fails, check that Ollama is running and that `ollama list` includes the configured model. The app reports the provider error instead of returning template-generated ad copy. The app validates the model's JSON against a Pydantic schema, validates cited fact IDs and simple claim/evidence rules, and records local model/token/latency metadata. Local inference cost is recorded as zero. Small local models can still hallucinate; flagged content requires human review before publishing.

## Main user flow

1. Paste a product description or a public product URL.
2. Review three concepts, storyboards, and image prompts; product URLs use images extracted from the page when available, while description-only concepts can use FLUX-generated artwork.
3. Open a variant landing page and use its CTA.
4. Open the experiment dashboard for CTR and Wilson 95% intervals.
5. Download a JSON or ZIP export containing ad copy, storyboards, image prompts, generated images, and experiment results.

Useful endpoints: `/health`, `POST /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/retry`, `/l/{variant_id}`, `/r/{variant_id}`, `/results/{job_id}`, `/jobs/{job_id}/dashboard`, `/jobs/{job_id}/export.json`, `/jobs/{job_id}/export.zip`.

## Architecture

```text
FastAPI form/API
      |
      +-- ingestion.py --> ProductFacts with source snippets
      |
      +-- jobs.py -------> SQLite jobs and steps
      |                       |
      |                       +--> generation.py --> validator --> variants
      |
      +-- templates ------> fixed landing template --> anonymous events
                                                  --> analytics.py
```

SQLite and a single background worker are intended for local use, not multi-worker concurrent production traffic. URL ingestion rejects non-public addresses, checks `robots.txt`, reads JSON-LD/OpenGraph metadata, limits request time/size, and caches extracted page facts for 24 hours. Anonymous visitor identifiers are hashed before storage. Basic in-memory rate limiting protects job creation/retry routes; use an external shared limiter for multi-process deployment.

## Configuration

Copy `.env.example` to `.env` and adjust thresholds/model settings as needed. `.env` is ignored by Git. Product descriptions are sent to the Ollama service running at `OLLAMA_BASE_URL` (default: loopback on this computer). Image prompts are sent to Cloudflare only when a user requests an image and the Cloudflare settings are configured. `PROMPT_VERSION` chooses `v1` or `v2`; `PRODUCT_CACHE_TTL_SECONDS`, `RATE_LIMIT_REQUESTS`, and `RATE_LIMIT_WINDOW_SECONDS` configure cache and basic local rate limits. `IMAGE_OUTPUT_DIRECTORY` and `IMAGE_GENERATION_TIMEOUT_SECONDS` configure image storage and request timeout.

## Tests

```powershell
py -m pytest
```

## Prompt evaluation

Run `python scripts/run_eval.py` to execute both prompt versions over ten sample products. This makes 20 local model requests and writes `EVAL.md` plus `eval/results.json`. See [EVAL.md](./EVAL.md) for the measured results and interpretation.

## Current limitations

- Local model quality and speed depend on the model and available hardware. The default `qwen3:4b` model is configured for a modest local setup; review generated claims before publishing.
- Cloudflare Workers AI is a hosted service; image generation sends the selected image prompt outside the local machine and may consume account quota.
- The golden set is small, and measured grounding was 0% for both evaluated prompt versions; generated advertising claims require human review.
- In-memory rate limiting and SQLite-backed background tasks are local/single-process safeguards, not production-grade distributed infrastructure.
- User accounts, billing, advertising-network publishing, and MCP remain out of scope.
