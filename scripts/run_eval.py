import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.generation import generate_variants
from app.schemas import ProductFacts


def mean_tfidf_cosine(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    tokenized = [re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE) for text in texts]
    document_frequency = Counter(token for document in tokenized for token in set(document))
    document_count = len(tokenized)
    vectors: list[dict[str, float]] = []
    for document in tokenized:
        frequencies = Counter(document)
        vectors.append(
            {
                token: (frequency / max(len(document), 1))
                * (math.log((1 + document_count) / (1 + document_frequency[token])) + 1)
                for token, frequency in frequencies.items()
            }
        )
    similarities = []
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1:]:
            dot_product = sum(value * right.get(token, 0.0) for token, value in left.items())
            left_norm = math.sqrt(sum(value * value for value in left.values()))
            right_norm = math.sqrt(sum(value * value for value in right.values()))
            similarities.append(dot_product / (left_norm * right_norm) if left_norm and right_norm else 0.0)
    return sum(similarities) / len(similarities)


def evaluate_product(product_path: Path, prompt_version: str) -> dict:
    facts = ProductFacts.model_validate_json(product_path.read_text(encoding="utf-8"))
    started = time.monotonic()
    try:
        variants, usage = generate_variants(facts, prompt_version=prompt_version)
    except Exception as exc:
        return {
            "product": product_path.stem,
            "prompt_version": prompt_version,
            "schema_valid": False,
            "grounding_rate": 0.0,
            "angle_diversity": 0.0,
            "mean_hook_similarity": None,
            "constraint_rate": 0.0,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "input_tokens": 0,
            "output_tokens": 0,
            "error": str(exc),
        }

    valid_fact_ids = {fact.fact_id for fact in facts.facts}
    grounded_variants = [
        bool(variant.fact_ids)
        and set(variant.fact_ids) <= valid_fact_ids
        and "unsupported_claim" not in variant.flags
        for variant in variants
    ]
    return {
        "product": product_path.stem,
        "prompt_version": prompt_version,
        "schema_valid": len(variants) == 3,
        "grounding_rate": sum(grounded_variants) / len(variants) if variants else 0.0,
        "angle_diversity": len({variant.angle for variant in variants}) / 3,
        "mean_hook_similarity": mean_tfidf_cosine([variant.caption for variant in variants]),
        "constraint_rate": sum(not variant.flags for variant in variants) / len(variants) if variants else 0.0,
        "latency_ms": usage.latency_ms,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "error": None,
    }


def summarize(results: list[dict]) -> dict:
    count = len(results)
    successful = [result for result in results if result["schema_valid"]]
    return {
        "runs": count,
        "schema_validity": len(successful) / count if count else 0.0,
        "grounding_rate": sum(item["grounding_rate"] for item in results) / count if count else 0.0,
        "angle_diversity": sum(item["angle_diversity"] for item in results) / count if count else 0.0,
        "mean_hook_similarity": _mean_optional(
            item["mean_hook_similarity"] for item in results
        ),
        "constraint_rate": sum(item["constraint_rate"] for item in results) / count if count else 0.0,
        "mean_latency_ms": sum(item["latency_ms"] for item in results) / count if count else 0,
        "mean_input_tokens": sum(item["input_tokens"] for item in results) / count if count else 0,
        "mean_output_tokens": sum(item["output_tokens"] for item in results) / count if count else 0,
        "local_api_cost_usd": 0.0,
    }


def _mean_optional(values) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def render_report(summaries: dict[str, dict], results: list[dict]) -> str:
    lines = [
        "# Prompt evaluation report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "The same 10 product records were evaluated once per prompt version using the configured local Ollama model.",
        "Schema validity is successful structured generation. Grounding is the fraction of variants without an unsupported-claim flag and with valid fact IDs. Hook similarity is mean pairwise TF-IDF cosine of captions (lower means more distinct). Constraints count variants without any validator flag.",
        "",
        "| Prompt | Valid JSON | Grounding | Angle diversity | Mean hook similarity | Constraints | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for version, summary in summaries.items():
        similarity = summary["mean_hook_similarity"]
        similarity_text = f"{similarity:.3f}" if similarity is not None else "n/a"
        lines.append(
            f"| {version} | {summary['schema_validity']:.0%} | {summary['grounding_rate']:.0%} | "
            f"{summary['angle_diversity']:.0%} | {similarity_text} | "
            f"{summary['constraint_rate']:.0%} | {summary['mean_latency_ms'] / 1000:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Do not treat this small golden set as statistical proof. Compare the quality metrics together; do not declare a prompt winner when schema or grounding regress.",
            "",
            "## Failed runs",
            "",
        ]
    )
    failed = [result for result in results if result["error"]]
    if failed:
        lines.extend(f"- `{item['prompt_version']}` / `{item['product']}`: {item['error']}" for item in failed)
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "Generated from `eval/products/` by `python scripts/run_eval.py`. Raw metrics are saved to `eval/results.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt versions against the golden product set.")
    parser.add_argument("--products", type=Path, default=ROOT / "eval" / "products")
    parser.add_argument("--versions", nargs="+", default=["v1", "v2"])
    parser.add_argument("--report", type=Path, default=ROOT / "EVAL.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "eval" / "results.json")
    args = parser.parse_args()

    products = sorted(args.products.glob("*.json"))
    if len(products) < 10:
        raise SystemExit(f"Expected at least 10 golden product JSON files, found {len(products)}.")
    results = [
        evaluate_product(product, version)
        for version in args.versions
        for product in products
    ]
    summaries = {
        version: summarize([result for result in results if result["prompt_version"] == version])
        for version in args.versions
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps({"summaries": summaries, "runs": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.report.write_text(render_report(summaries, results), encoding="utf-8")
    print(f"Completed {len(results)} runs. Report: {args.report}; raw results: {args.json_output}")


if __name__ == "__main__":
    main()
