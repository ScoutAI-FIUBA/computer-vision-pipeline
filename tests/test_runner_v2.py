from __future__ import annotations

import pytest

from pipeline.runner_v2 import RunnerV2, RunnerV2Seams


def test_runner_preserves_raw_ids_and_routes_stable_ids_to_downstream_seams():
    calls = []

    def resolve_identity(rows):
        return {row["observation_key"]: f"player-{row['raw_track_id']}" for row in rows}

    def downstream(rows):
        calls.append(rows)
        return {row["stable_player_id"] for row in rows}

    result = RunnerV2(
        seams=RunnerV2Seams(identity=resolve_identity, team_id=downstream, analytics=downstream)
    ).run(
        [
            {
                "observation_key": "60:3",
                "frame_index": 60,
                "raw_track_id": 3,
                "stable_player_id": None,
                "bbox_xyxy": [1, 2, 3, 4],
                "track_state": "tracked",
            }
        ],
        tracking_identity_accepted=True,
    )

    assert result.observations[0]["raw_track_id"] == 3
    assert result.observations[0]["stable_player_id"] == "player-3"
    assert result.downstream["team_id"] == {"player-3"}
    assert calls[0][0]["raw_track_id"] == 3


def test_runner_blocks_statistics_when_tracking_identity_gate_fails():
    statistics_calls = []
    result = RunnerV2(
        seams=RunnerV2Seams(
            analytics=lambda rows: {"accepted_rows": len(rows)},
            statistics=lambda rows: statistics_calls.append(rows),
        )
    ).run(
        [
            {
                "observation_key": "60:3",
                "frame_index": 60,
                "raw_track_id": 3,
                "stable_player_id": "player-3",
                "bbox_xyxy": [1, 2, 3, 4],
                "track_state": "tracked",
            }
        ],
        tracking_identity_accepted=False,
    )

    assert result.acceptance == {
        "tracking_identity": "rejected",
        "statistics": "deferred",
    }
    assert result.downstream["analytics"] == {"accepted_rows": 1}
    assert "statistics" not in result.downstream
    assert statistics_calls == []


def test_runner_rejects_rows_without_raw_identity():
    with pytest.raises(ValueError, match="raw_track_id"):
        RunnerV2().run(
            [{"observation_key": "60:3", "stable_player_id": "player-3"}],
            tracking_identity_accepted=True,
        )


def test_runner_accepts_statistics_only_after_identity_gate_passes():
    result = RunnerV2(
        seams=RunnerV2Seams(statistics=lambda rows: {"players": len(rows)})
    ).run(
        [
            {
                "observation_key": "61:8",
                "frame_index": 61,
                "raw_track_id": 8,
                "stable_player_id": "player-8",
                "bbox_xyxy": [4, 5, 6, 7],
                "track_state": "tracked",
            }
        ],
        tracking_identity_accepted=True,
    )

    assert result.downstream["statistics"] == {"players": 1}
    assert result.acceptance["statistics"] == "accepted"
