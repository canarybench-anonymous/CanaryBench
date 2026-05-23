from typing import Any, Dict, Iterable, List, Sequence


def pareto_frontier(
    items: Sequence[Dict[str, Any]],
    privacy_key: str,
    utility_key: str,
    maximize_privacy: bool = True,
    maximize_utility: bool = True,
) -> List[Dict[str, Any]]:
    frontier: List[Dict[str, Any]] = []
    for candidate in items:
        dominated = False
        for other in items:
            if other is candidate:
                continue
            privacy_better_or_equal = (
                other[privacy_key] >= candidate[privacy_key]
                if maximize_privacy
                else other[privacy_key] <= candidate[privacy_key]
            )
            utility_better_or_equal = (
                other[utility_key] >= candidate[utility_key]
                if maximize_utility
                else other[utility_key] <= candidate[utility_key]
            )
            privacy_strict = other[privacy_key] != candidate[privacy_key]
            utility_strict = other[utility_key] != candidate[utility_key]
            if privacy_better_or_equal and utility_better_or_equal and (privacy_strict or utility_strict):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def add_pareto_membership(
    items: Sequence[Dict[str, Any]],
    privacy_key: str,
    utility_key: str,
) -> List[Dict[str, Any]]:
    frontier = pareto_frontier(items, privacy_key=privacy_key, utility_key=utility_key)
    frontier_ids = {id(item) for item in frontier}
    output: List[Dict[str, Any]] = []
    for item in items:
        enriched = dict(item)
        enriched["pareto_optimal"] = id(item) in frontier_ids
        output.append(enriched)
    return output

