import random


def wilson_interval(clicks: int, views: int, z: float = 1.96) -> tuple[float, float]:
    _validate_counts(clicks, views)
    if views <= 0:
        return 0.0, 0.0
    rate = clicks / views
    denominator = 1 + z * z / views
    center = (rate + z * z / (2 * views)) / denominator
    margin = z * ((rate * (1 - rate) / views + z * z / (4 * views**2)) ** 0.5) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def experiment_results(rows: list[dict], minimum_views: int, samples: int = 10000) -> dict:
    if minimum_views < 1:
        raise ValueError("minimum_views must be at least 1.")
    if samples < 1:
        raise ValueError("samples must be at least 1.")
    for row in rows:
        _validate_counts(row["clicks"], row["views"])
    results = []
    enough_data = bool(rows) and all(row["views"] >= minimum_views for row in rows)
    best_counts = [0 for _ in rows]
    if enough_data:
        rng = random.Random(0)
        for _ in range(samples):
            draws = [
                rng.betavariate(row["clicks"] + 1, row["views"] - row["clicks"] + 1)
                for row in rows
            ]
            best_counts[draws.index(max(draws))] += 1
    for index, row in enumerate(rows):
        low, high = wilson_interval(row["clicks"], row["views"])
        results.append(
            {
                **row,
                "ctr": row["clicks"] / row["views"] if row["views"] else 0.0,
                "wilson_95": [low, high],
                "probability_best": best_counts[index] / samples if enough_data else None,
            }
        )
    return {
        "sufficient_data": enough_data,
        "message": None if enough_data else "Not enough data to identify a winning variant.",
        "variants": results,
    }


def _validate_counts(clicks: int, views: int) -> None:
    if views < 0 or clicks < 0 or clicks > views:
        raise ValueError("Event counts must satisfy 0 <= clicks <= views.")
