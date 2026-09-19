"""SAM 3.1 V2 orchestration boundary.

The runner owns identity-domain translation, not the algorithms behind the
existing pipeline stages.  Those stages are injected at their existing seam
so this module stays usable with fakes and does not duplicate the legacy
runner.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


Stage = Callable[[list[dict[str, Any]]], Any]


@dataclass(frozen=True)
class RunnerV2Seams:
    """Optional downstream callbacks using normalized player rows."""

    reid: Stage | None = None
    identity: Callable[[list[dict[str, Any]]], Mapping[str, str]] | None = None
    team_id: Stage | None = None
    tracknet: Stage | None = None
    court: Stage | None = None
    shot: Stage | None = None
    analytics: Stage | None = None
    statistics: Stage | None = None


@dataclass
class RunnerV2Result:
    observations: list[dict[str, Any]]
    downstream: dict[str, Any] = field(default_factory=dict)
    acceptance: dict[str, str] = field(default_factory=dict)


def _identity_map(
    rows: list[dict[str, Any]], resolver: Callable[[list[dict[str, Any]]], Mapping[str, str]] | None
) -> Mapping[str, str]:
    if resolver is None:
        return {}
    return resolver(rows)


def _copy_observations(observations: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for observation in observations:
        row = dict(observation)
        if "raw_track_id" not in row:
            raise ValueError("SAM31 observations require raw_track_id")
        row["raw_track_id"] = int(row["raw_track_id"])
        rows.append(row)
    return rows


class RunnerV2:
    """Feed SAM31 normalized observations through available downstream seams."""

    def __init__(self, seams: RunnerV2Seams | None = None) -> None:
        self.seams = seams or RunnerV2Seams()

    def run(
        self,
        observations: Iterable[Mapping[str, Any]],
        *,
        tracking_identity_accepted: bool,
    ) -> RunnerV2Result:
        rows = _copy_observations(observations)
        stable_ids = _identity_map(rows, self.seams.identity)
        for row in rows:
            key = str(row.get("observation_key", ""))
            if key in stable_ids:
                row["stable_player_id"] = stable_ids[key]

        if tracking_identity_accepted and any(row.get("stable_player_id") is None for row in rows):
            raise ValueError("accepted observations require stable_player_id")

        downstream: dict[str, Any] = {}
        for name in ("reid", "team_id", "tracknet", "court", "shot", "analytics"):
            stage = getattr(self.seams, name)
            if stage is not None:
                downstream[name] = stage(rows)

        acceptance = {
            "tracking_identity": "accepted" if tracking_identity_accepted else "rejected",
            "statistics": "deferred",
        }
        if tracking_identity_accepted and self.seams.statistics is not None:
            downstream["statistics"] = self.seams.statistics(rows)
            acceptance["statistics"] = "accepted"

        return RunnerV2Result(rows, downstream, acceptance)


def run_v2(
    observations: Iterable[Mapping[str, Any]],
    *,
    seams: RunnerV2Seams | None = None,
    tracking_identity_accepted: bool,
) -> RunnerV2Result:
    """Functional entrypoint for Modal or tests; legacy runner is untouched."""

    return RunnerV2(seams).run(
        observations,
        tracking_identity_accepted=tracking_identity_accepted,
    )
