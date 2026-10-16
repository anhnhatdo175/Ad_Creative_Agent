import pytest

from app.analytics import experiment_results, wilson_interval
from app.generation import ANGLES, validate_copy
from app.ingestion import extract_product_facts, validate_public_http_url


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1", "http://localhost"])
def test_rejects_non_public_or_unsupported_urls(url):
    with pytest.raises(ValueError):
        validate_public_http_url(url)


def test_manual_fallback_does_not_allow_unsafe_url_to_be_saved():
    with pytest.raises(ValueError):
        extract_product_facts("A safe product description.", "http://127.0.0.1")


def test_wilson_interval_for_zero_views():
    assert wilson_interval(0, 0) == (0.0, 0.0)


@pytest.mark.parametrize("clicks,views", [(-1, 10), (11, 10), (0, -1)])
def test_analytics_rejects_invalid_event_counts(clicks, views):
    with pytest.raises(ValueError, match="0 <= clicks <= views"):
        wilson_interval(clicks, views)


def test_experiment_rejects_click_count_above_views():
    with pytest.raises(ValueError, match="0 <= clicks <= views"):
        experiment_results(
            [{"variant_id": "a", "angle": "benefit", "views": 3, "clicks": 4}],
            minimum_views=1,
        )


def test_experiment_reports_insufficient_data_without_winner():
    result = experiment_results(
        [{"variant_id": "a", "angle": "benefit", "views": 10, "clicks": 2}],
        minimum_views=30,
    )
    assert not result["sufficient_data"]
    assert result["variants"][0]["probability_best"] is None


def test_experiment_reports_probability_after_minimum_sample():
    result = experiment_results(
        [
            {"variant_id": "a", "angle": "benefit", "views": 40, "clicks": 12},
            {"variant_id": "b", "angle": "curiosity", "views": 40, "clicks": 3},
        ],
        minimum_views=30,
        samples=1000,
    )
    assert result["sufficient_data"]
    assert result["variants"][0]["probability_best"] > result["variants"][1]["probability_best"]


def test_claim_guard_flags_unsupported_numbers_but_accepts_sourced_claims():
    scene = [{"visual": "Product", "voiceover": "A simple product.", "duration": 14}]
    assert "unsupported_claim" in validate_copy(
        "Only $9 today", "A simple product.", scene, ["A stainless steel bottle."]
    )
    assert "unsupported_claim" not in validate_copy(
        "Holds 500 ml", "A simple product.", scene, ["Holds 500 ml."]
    )
