# Prompt evaluation report

Generated: 2026-10-16T15:53:40.920777+00:00

The same 10 product records were evaluated once per prompt version using the configured local Ollama model.
Schema validity is successful structured generation. Grounding is the fraction of variants without an unsupported-claim flag and with valid fact IDs. Hook similarity is mean pairwise TF-IDF cosine of captions (lower means more distinct). Constraints count variants without any validator flag.

| Prompt | Valid JSON | Grounding | Angle diversity | Mean hook similarity | Constraints | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 100% | 0% | 100% | 0.130 | 0% | 43.7 |
| v2 | 100% | 0% | 100% | 0.418 | 0% | 41.2 |

## Interpretation

Do not treat this small golden set as statistical proof. Compare the quality metrics together; do not declare a prompt winner when schema or grounding regress.

## Failed runs

None.

Generated from `eval/products/` by `python scripts/run_eval.py`. Raw metrics are saved to `eval/results.json`.
