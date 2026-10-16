import json
import re
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import (
    ALLOWED_CAPTION_LENGTH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TIMEOUT_SECONDS,
    PROMPT_VERSION,
)
from app.schemas import GeneratedAdPack, ProductFacts, Variant

ANGLES = {"pain_point", "social_proof", "curiosity", "benefit", "urgency", "comparison"}
FORBIDDEN_WORDS = {
    "guaranteed",
    "best",
    "perfect",
    "#1",
    "number one",
    "millions",
    "thousands",
    "hundreds",
}
SUPPORTED_PROMPT_VERSIONS = {"v1", "v2"}
PROMPTS_DIRECTORY = Path(__file__).resolve().parent.parent / "prompts"
OLLAMA_REQUEST_LOCK = threading.Lock()
STOP_WORDS = {
    "a", "about", "again", "all", "an", "and", "are", "as", "at", "be", "because",
    "been", "being", "but", "by", "can", "could", "discover", "do", "does", "for",
    "from", "get", "have", "has", "how", "if", "in", "into", "is", "it", "its",
    "just", "made", "make", "meet", "more", "of", "on", "or", "our", "out", "see",
    "that", "the", "their", "them", "there", "this", "to", "use", "using", "was",
    "we", "what", "when", "with", "your", "you",
}


@dataclass(frozen=True)
class GenerationUsage:
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


def generate_variants(
    facts: ProductFacts,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[list[Variant], GenerationUsage]:
    fact_ids = {fact.fact_id for fact in facts.facts}
    if not fact_ids:
        raise ValueError("Cannot generate ads without sourced product facts.")

    prompt = _build_prompt(facts, prompt_version)
    request_body = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Vietnamese-first product advertising assistant. "
                    "Use only the supplied product facts. Write in Vietnamese. Return valid JSON matching the schema. "
                    "Never invent product features, prices, ratings, reviews, guarantees, or statistics."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "format": GeneratedAdPack.model_json_schema(),
        "stream": False,
        "think": False,
        "options": {"temperature": 0.4, "num_ctx": OLLAMA_NUM_CTX},
    }
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with OLLAMA_REQUEST_LOCK:
            with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
                result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code == 404:
            raise RuntimeError(
                f"Ollama model '{OLLAMA_MODEL}' is unavailable. Pull it with: ollama pull {OLLAMA_MODEL}"
            ) from exc
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Start the Ollama app or run 'ollama serve'."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama model '{OLLAMA_MODEL}' exceeded the {OLLAMA_TIMEOUT_SECONDS:g}-second timeout."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned an invalid JSON response.") from exc

    message = result.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("Ollama response did not contain generated message content.")
    try:
        generated = GeneratedAdPack.model_validate_json(message["content"])
    except ValueError as exc:
        raise RuntimeError(f"Ollama output did not match the ad-pack schema: {exc}") from exc

    variants: list[Variant] = []
    seen_angles: set[str] = set()
    for item in generated.variants:
        if item.angle in seen_angles:
            raise RuntimeError(f"Ollama returned a duplicate ad angle: {item.angle}.")
        seen_angles.add(item.angle)
        unknown_fact_ids = set(item.fact_ids) - fact_ids
        if not item.fact_ids or unknown_fact_ids:
            raise RuntimeError(
                f"Ollama returned missing or unknown source fact IDs: {sorted(unknown_fact_ids)}."
            )

        cited_facts = [fact for fact in facts.facts if fact.fact_id in item.fact_ids]
        scenes = [scene.model_dump() for scene in item.storyboard]
        flags = validate_copy(
            item.caption,
            item.script,
            scenes,
            [fact.value for fact in cited_facts],
            item.image_prompts,
        )
        if item.angle == "social_proof" and not any(
            fact.label.lower() in {"review", "rating"} for fact in cited_facts
        ):
            flags.append("unsupported_claim")
        if item.angle == "urgency" and not re.search(
            r"\b(limited|today|ends|deadline|stock|sale|discount|offer|promotion|"
            r"khuyen mai|uu dai|sap het|co han)\b",
            " ".join(fact.value for fact in cited_facts),
            re.IGNORECASE,
        ):
            flags.append("unsupported_claim")
        if item.angle == "comparison" and not re.search(
            r"\b(than|versus|vs\.?|compared|comparison|unlike|better|cheaper|"
            r"hon|so voi|tot hon|re hon)\b",
            " ".join(fact.value for fact in cited_facts),
            re.IGNORECASE,
        ):
            flags.append("unsupported_claim")
        flags = list(dict.fromkeys(flags))
        variants.append(
            Variant(
                id=str(uuid.uuid4()),
                angle=item.angle,
                caption=item.caption,
                script=item.script,
                storyboard=item.storyboard,
                image_prompts=item.image_prompts,
                fact_ids=item.fact_ids,
                flags=flags,
            )
        )

    return variants, GenerationUsage(
        model=result.get("model", OLLAMA_MODEL),
        input_tokens=int(result.get("prompt_eval_count", 0)),
        output_tokens=int(result.get("eval_count", 0)),
        latency_ms=int(result.get("total_duration", 0) / 1_000_000),
    )


def _build_prompt(facts: ProductFacts, prompt_version: str = PROMPT_VERSION) -> str:
    if prompt_version not in SUPPORTED_PROMPT_VERSIONS:
        raise ValueError(f"Unsupported prompt version: {prompt_version}")
    prompt_path = PROMPTS_DIRECTORY / prompt_version / "ad_generation.txt"
    instructions = prompt_path.read_text(encoding="utf-8")
    return f"{instructions}\n\nProduct facts (source of truth):\n{facts.model_dump_json()}"


def validate_copy(
    caption: str,
    script: str,
    scenes: list[dict],
    supported_facts: list[str] | None = None,
    additional_claim_text: list[str] | None = None,
) -> list[str]:
    flags: list[str] = []
    if len(caption) > ALLOWED_CAPTION_LENGTH:
        flags.append("caption_too_long")
    total_duration = sum(scene["duration"] for scene in scenes)
    if not 13 <= total_duration <= 15:
        flags.append("script_duration_invalid")
    if len(scenes) not in {3, 4}:
        flags.append("storyboard_scene_count_invalid")
    if len(re.findall(r"\b[\w'-]+\b", script)) > 37:
        flags.append("script_too_long")
    all_copy = " ".join([caption, script, *(scene["voiceover"] for scene in scenes)]).lower()
    if any(re.search(rf"\b{re.escape(word)}\b", all_copy) for word in FORBIDDEN_WORDS):
        flags.append("forbidden_word")
    if not _claims_are_supported(
        caption,
        script,
        scenes,
        supported_facts or [],
        additional_claim_text or [],
    ):
        flags.append("unsupported_claim")
    return flags


def _claims_are_supported(
    caption: str,
    script: str,
    scenes: list[dict],
    supported_facts: list[str],
    additional_claim_text: list[str],
) -> bool:
    text = " ".join(
        [
            caption,
            script,
            *(scene["voiceover"] for scene in scenes),
            *(scene["visual"] for scene in scenes),
            *additional_claim_text,
        ]
    ).lower()
    claims = re.findall(
        r"(?<!\w)(?:\d+(?:[.,]\d+)?%?|\$[\d,.]+|#1)(?!\w)|"
        r"\b(?:free|guaranteed|best|perfect|number one|millions|thousands|hundreds)\b",
        text,
    )
    sources = [fact.lower() for fact in supported_facts]
    if not all(any(claim in source for source in sources) for claim in claims):
        return False
    source_terms = {
        term
        for source in sources
        for term in re.findall(r"[^\W_]+", source, flags=re.UNICODE)
        if term not in STOP_WORDS
    }
    for part in [
        caption,
        script,
        *(scene["voiceover"] for scene in scenes),
        *(scene["visual"] for scene in scenes),
        *additional_claim_text,
    ]:
        for sentence in re.split(r"[.!?\n]+", part.lower()):
            terms = {
                term
                for term in re.findall(r"[^\W_]+", sentence, flags=re.UNICODE)
                if term not in STOP_WORDS
            }
            if len(terms) >= 3:
                overlap = len(terms & source_terms)
                if overlap < 2 or overlap / len(terms) < 0.4:
                    return False
    return True
