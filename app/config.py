import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "data/ad_creative.sqlite3"))
MIN_EXPERIMENT_VIEWS = int(os.getenv("MIN_EXPERIMENT_VIEWS", "30"))
MAX_PAGE_BYTES = int(os.getenv("MAX_PAGE_BYTES", "1000000"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
ALLOWED_CAPTION_LENGTH = int(os.getenv("ALLOWED_CAPTION_LENGTH", "2200"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
PRODUCT_CACHE_TTL_SECONDS = int(os.getenv("PRODUCT_CACHE_TTL_SECONDS", "86400"))
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2")
IMAGE_CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
IMAGE_CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
IMAGE_OUTPUT_DIRECTORY = Path(os.getenv("IMAGE_OUTPUT_DIRECTORY", "data/generated_images"))
IMAGE_GENERATION_TIMEOUT_SECONDS = float(os.getenv("IMAGE_GENERATION_TIMEOUT_SECONDS", "120"))
USER_AGENT = "AdCreativeAgent/0.1 (+product-research; contact: admin@example.com)"
