"""Bounded prompt evidence from completed mutations still in the population."""

import json
from typing import Any, Dict, List, Optional

from openevolve.database import Program, lane_group_of


def recent_attempts(
    programs: Dict[str, Program],
    parent: Program,
    limit: int,
    lane_metric: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return distinct mutations of this parent source, across its group."""
    if limit <= 0:
        return []
    group = lane_group_of(parent, lane_metric)
    parent_ids = {
        p.id
        for p in programs.values()
        if p.code == parent.code and lane_group_of(p, lane_metric) == group
    }
    attempts = sorted(
        (
            p
            for p in programs.values()
            if p.parent_id in parent_ids and lane_group_of(p, lane_metric) == group
        ),
        key=lambda p: (p.iteration_found, p.timestamp, p.id),
        reverse=True,
    )
    result = []
    seen = set()
    for program in attempts:
        if program.code in seen:
            continue
        seen.add(program.code)
        result.append(
            {
                "id": program.id,
                "code": program.code,
                "metrics": program.metrics,
                "metadata": program.metadata,
                "iteration_found": program.iteration_found,
                "artifacts": json.loads(program.artifacts_json or "{}"),
            }
        )
        if len(result) == limit:
            break
    return result
