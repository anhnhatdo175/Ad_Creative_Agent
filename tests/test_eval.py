from pathlib import Path

import pytest

from app.schemas import StoryboardScene, Variant
from app.generation import GenerationUsage
from scripts.run_eval import evaluate_product, mean_tfidf_cosine, render_report, summarize


ROOT = Path(__file__).resolve().parents[1]


def test_golden_set_contains_ten_products_and_prompt_versions():
    products = list((ROOT / "eval" / "products").glob("*.json"))
    assert len(products) == 10
    assert (ROOT / "prompts" / "v1" / "ad_generation.txt").is_file()
    assert (ROOT / "prompts" / "v2" / "ad_generation.txt").is_file()


def test_tfidf_similarity_distinguishes_identical_and_distinct_hooks():
    assert mean_tfidf_cosine(["green bottle", "green bottle"]) == pytest.approx(1.0)
    assert mean_tfidf_cosine(["green bottle", "garden service"]) < 0.1


def test_eval_summary_averages_metrics_and_latency():
    summary = summarize(
        [
            {
                "schema_valid": True,
                "grounding_rate": 1,
                "angle_diversity": 1,
                "mean_hook_similarity": 0.2,
                "constraint_rate": 1,
                "latency_ms": 1000,
                "input_tokens": 10,
                "output_tokens": 20,
            },
            {
                "schema_valid": False,
                "grounding_rate": 0,
                "angle_diversity": 0.5,
                "mean_hook_similarity": None,
                "constraint_rate": 0,
                "latency_ms": 3000,
                "input_tokens": 20,
                "output_tokens": 40,
            },
        ]
    )
    assert summary["schema_validity"] == 0.5
    assert summary["grounding_rate"] == 0.5
    assert summary["mean_hook_similarity"] == 0.2
    assert summary["mean_latency_ms"] == 2000


def test_eval_product_records_schema_grounding_and_constraints(tmp_path, monkeypatch):
    import scripts.run_eval as eval_runner

    product = ROOT / "eval" / "products" / "01-reusable-bottle.json"

    def fake_generate(facts, prompt_version):
        variants = [
            Variant(
                id=f"v{index}",
                angle=angle,
                caption=f"Hook {angle} stainless steel bottle",
                script="Made from stainless steel.",
                storyboard=[
                    StoryboardScene(visual="Show product", voiceover="Made from stainless steel.", duration=14)
                ],
                image_prompts=["Bottle on white background"],
                fact_ids=["fact_1"],
                flags=[],
            )
            for index, angle in enumerate(("benefit", "curiosity", "pain_point"))
        ]
        return variants, GenerationUsage("test-model", 10, 20, 500)

    monkeypatch.setattr(eval_runner, "generate_variants", fake_generate)
    result = evaluate_product(product, "v2")
    assert result["schema_valid"]
    assert result["grounding_rate"] == 1
    assert result["angle_diversity"] == 1
    assert result["constraint_rate"] == 1
    assert result["latency_ms"] == 500
    report = render_report({"v2": summarize([result])}, [result])
    assert "Grounding" in report
    assert "v2" in report
