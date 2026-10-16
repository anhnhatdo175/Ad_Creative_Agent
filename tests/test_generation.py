import json
import urllib.error
from unittest.mock import patch

import pytest

from app.generation import generate_variants
from app.ingestion import extract_product_facts
from app.schemas import ProductFact


def ollama_response(content: str) -> bytes:
    return json.dumps(
        {
            "model": "qwen3:4b",
            "message": {"role": "assistant", "content": content},
            "prompt_eval_count": 321,
            "eval_count": 123,
            "total_duration": 1_250_000_000,
        }
    ).encode()


def valid_ad_pack(fact_id: str) -> str:
    return json.dumps(
        {
            "variants": [
                {
                    "angle": angle,
                    "caption": f"{angle}: explore this reusable stainless steel bottle",
                    "script": "Meet the bottle. Made from stainless steel. Explore its details.",
                    "storyboard": [
                        {"visual": "Show the bottle.", "voiceover": "Meet the bottle.", "duration": 4},
                        {
                            "visual": "Show its material.",
                            "voiceover": "Made from stainless steel.",
                            "duration": 5,
                        },
                        {
                            "visual": "Show the bottle.",
                            "voiceover": "Explore its details.",
                            "duration": 5,
                        },
                    ],
                    "image_prompts": ["A reusable stainless steel bottle on a clean background."],
                    "fact_ids": [fact_id],
                }
                for angle in ("benefit", "curiosity", "pain_point")
            ]
        }
    )


def test_calls_local_ollama_with_schema_and_returns_validated_variants():
    facts = extract_product_facts("A reusable bottle made from stainless steel.", None)
    content = valid_ad_pack(facts.facts[0].fact_id)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return ollama_response(content)

    with patch("app.generation.urllib.request.urlopen", return_value=Response()) as urlopen:
        variants, usage = generate_variants(facts)

    request = urlopen.call_args.args[0]
    payload = json.loads(request.data)
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert payload["model"] == "qwen3:4b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["num_ctx"] == 2048
    assert "properties" in payload["format"]
    assert len(variants) == 3
    assert {variant.angle for variant in variants} == {"benefit", "curiosity", "pain_point"}
    assert all(variant.fact_ids == [facts.facts[0].fact_id] for variant in variants)
    assert all(not variant.flags for variant in variants)
    assert usage.model == "qwen3:4b"
    assert usage.input_tokens == 321
    assert usage.output_tokens == 123
    assert usage.latency_ms == 1250


def test_rejects_unknown_fact_ids_from_model():
    facts = extract_product_facts("A reusable bottle made from stainless steel.", None)
    content = valid_ad_pack("unknown-fact")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return ollama_response(content)

    with patch("app.generation.urllib.request.urlopen", return_value=Response()):
        with pytest.raises(RuntimeError, match="unknown source fact IDs"):
            generate_variants(facts)


def test_rejects_duplicate_angles_from_model():
    facts = extract_product_facts("A reusable bottle made from stainless steel.", None)
    content = valid_ad_pack(facts.facts[0].fact_id).replace(
        '"angle": "curiosity"', '"angle": "benefit"'
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return ollama_response(content)

    with patch("app.generation.urllib.request.urlopen", return_value=Response()):
        with pytest.raises(RuntimeError, match="duplicate ad angle"):
            generate_variants(facts)


def test_flags_unsupported_social_proof_and_unsubstantiated_copy():
    facts = extract_product_facts("A reusable water bottle made from stainless steel.", None)
    content = valid_ad_pack(facts.facts[0].fact_id).replace(
        "benefit: explore this reusable stainless steel bottle",
        "Join millions who choose this bottle",
    ).replace(
        "curiosity: explore this reusable stainless steel bottle",
        "The burden of single-use plastic bottles",
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return ollama_response(content)

    with patch("app.generation.urllib.request.urlopen", return_value=Response()):
        variants, _ = generate_variants(facts)

    social_proof = next(variant for variant in variants if variant.angle == "benefit")
    pain_point = next(variant for variant in variants if variant.angle == "curiosity")
    assert "unsupported_claim" in social_proof.flags
    assert "forbidden_word" in social_proof.flags
    assert "unsupported_claim" in pain_point.flags


def test_flags_social_proof_angle_when_product_has_no_reviews_or_rating():
    facts = extract_product_facts("A reusable bottle made from stainless steel.", None)
    pack = json.loads(valid_ad_pack(facts.facts[0].fact_id))
    pack["variants"][0]["angle"] = "social_proof"
    content = json.dumps(pack)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return ollama_response(content)

    with patch("app.generation.urllib.request.urlopen", return_value=Response()):
        variants, _ = generate_variants(facts)

    social_proof = next(variant for variant in variants if variant.angle == "social_proof")
    assert "unsupported_claim" in social_proof.flags


def test_social_proof_must_cite_a_review_or_rating_fact():
    facts = extract_product_facts("A reusable bottle made from stainless steel.", None)
    facts.rating = "5"
    facts.facts.append(
        ProductFact(
            fact_id="fact_rating",
            label="rating",
            value="Rated 5 by 10 reviews.",
            source_snippet="Rated 5 by 10 reviews.",
        )
    )
    pack = json.loads(valid_ad_pack(facts.facts[0].fact_id))
    pack["variants"][0]["angle"] = "social_proof"
    content = json.dumps(pack)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return ollama_response(content)

    with patch("app.generation.urllib.request.urlopen", return_value=Response()):
        variants, _ = generate_variants(facts)

    assert "unsupported_claim" in next(
        variant for variant in variants if variant.angle == "social_proof"
    ).flags


def test_reports_ollama_unavailable_without_falling_back_to_template_copy():
    facts = extract_product_facts("A reusable bottle made from stainless steel.", None)
    with patch(
        "app.generation.urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
            generate_variants(facts)
